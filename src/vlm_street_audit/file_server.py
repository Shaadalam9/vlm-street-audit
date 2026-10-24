"""Videos and YOLO track CSVs from the CROWD file server (files.mobility-squad.com).

The server is an authenticated HTTP file server, not FTP, although its credentials
are called `ftp_username` / `ftp_password` in every CROWD project. Each drive is an
alias: a file is served at `v/<alias>/files/<path>` and a folder is listed at
`v/<alias>/browse/<path>`. Videos are `<video_id>.mp4` at the top of the video
aliases (tue1-tue5); track files are `<video_id>_<start>_<fps>.csv` in the bbox
folder below the `data` alias.

Lookup order for both: a local folder (the drives on lin4080), then the download
cache, then the server (the exact URL first, then a walk of the browse pages).
Downloads are written to a `.part` file and renamed only when complete. Adapted
from crowd-jaywalking's `scripts/crowd/crowd_source.py`.
"""

import logging
import os
import time
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urljoin, urlparse

from .config import load_config, load_secret, project_path

logger = logging.getLogger(__name__)

# The bbox folder lists about 86,000 files (13 MB of HTML): its page is slow to arrive.
LISTING_READ_TIMEOUT_SECONDS = 300
DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10
CHUNK_BYTES = 1024 * 1024
USER_AGENT = "vlm-street-audit/0.1"


class DownloadError(RuntimeError):
    """A file could not be downloaded."""


class TransientDownloadError(DownloadError):
    """A request failed or broke off; trying again may succeed."""


class AuthenticationError(DownloadError):
    """The server rejected the credentials in `secret`."""


def _safe_name(name: str) -> str:
    if not name or name != PurePosixPath(name).name or "\\" in name or "\x00" in name or name in {".", ".."}:
        raise ValueError(f"Not a plain file name: {name!r}")
    return name


def parse_track_filename(name: str) -> tuple[str, int, int]:
    """(video_id, start_second, fps) of `<video_id>_<start>_<fps>.csv`; video IDs may contain `_`."""
    stem = PurePosixPath(name).stem
    video_id, start, fps = stem.rsplit("_", 2)
    return video_id, int(start), int(fps)


def is_track_file_for(name: str, video_id: str, start_second: int) -> bool:
    if not name.lower().endswith(".csv"):
        return False
    try:
        found_id, found_start, _ = parse_track_filename(name)
    except ValueError:
        return False
    return found_id == video_id and found_start == start_second


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self.hrefs.extend(value.strip() for name, value in attrs if name.lower() == "href" and value)


class FileServer:
    """Fetch CROWD videos and track files, from local folders when present, else the server."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        token: str | None = None,
        *,
        video_aliases: Iterable[str],
        video_dirs: Iterable[Path] = (),
        video_download_dir: Path,
        bbox_aliases: Iterable[str] = ("data",),
        bbox_server_folder: str = "",
        bbox_dirs: Iterable[Path] = (),
        bbox_download_dir: Path,
        timeout_seconds: int = 120,
        crawl_page_limit: int = 500,
        session_factory: Callable[[], Any] | None = None,
    ) -> None:
        parsed = urlparse(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("file_server must be an absolute HTTP or HTTPS URL")
        if parsed.username or parsed.password:
            raise ValueError("Do not put credentials in file_server; they belong in 'secret'")
        self.base_url = base_url.strip().rstrip("/") + "/"
        self.origin = (parsed.scheme.lower(), parsed.netloc.lower())
        self.auth = (username, password) if username and password else None
        self.token = token or None
        self.video_aliases = self._aliases(video_aliases, "video_aliases")
        self.bbox_aliases = self._aliases(bbox_aliases, "bbox_aliases")
        folder = "/".join(part for part in bbox_server_folder.replace("\\", "/").split("/") if part)
        if ".." in folder.split("/"):
            raise ValueError("bbox_server_folder must be a relative folder on the server")
        self.bbox_server_folder = folder
        self.video_dirs = [Path(d) for d in video_dirs]
        self.bbox_dirs = [Path(d) for d in bbox_dirs]
        self.video_download_dir = Path(video_download_dir)
        self.bbox_download_dir = Path(bbox_download_dir)
        if timeout_seconds < 1 or crawl_page_limit < 1:
            raise ValueError("timeout_seconds and crawl_page_limit must be positive")
        self.timeout_seconds = timeout_seconds
        self.crawl_page_limit = crawl_page_limit
        if session_factory is None:
            import requests

            session_factory = requests.Session
        self.session_factory = session_factory
        self._listings: dict[tuple[str, ...], list[str]] = {}

    @classmethod
    def from_config(cls, settings: dict[str, Any] | None = None, secret: dict[str, str] | None = None) -> "FileServer":
        settings = settings or load_config()
        secret = secret if secret is not None else load_secret()
        if not secret.get("ftp_username") or not secret.get("ftp_password"):
            raise ValueError("Set ftp_username and ftp_password in 'secret'.")
        return cls(
            settings["file_server"],
            secret["ftp_username"],
            secret["ftp_password"],
            secret.get("ftp_token"),
            video_aliases=settings["video_aliases"],
            video_dirs=[project_path(d) for d in settings["video_dirs"]],
            video_download_dir=project_path(settings["video_download_dir"]),
            bbox_aliases=settings["bbox_aliases"],
            bbox_server_folder=settings["bbox_server_folder"],
            bbox_dirs=[project_path(d) for d in settings["bbox_dirs"]],
            bbox_download_dir=project_path(settings["bbox_download_dir"]),
            timeout_seconds=int(settings["timeout_seconds"]),
            crawl_page_limit=int(settings["crawl_page_limit"]),
        )

    @staticmethod
    def _aliases(values: Iterable[str], setting: str) -> tuple[str, ...]:
        aliases = tuple(str(value).strip().strip("/") for value in values)
        if not aliases or not all(aliases):
            raise ValueError(f"{setting} must hold at least one alias")
        return aliases

    # ---- public API ------------------------------------------------------------

    def video(self, video_id: str) -> Path:
        """Local path of `<video_id>.mp4`, downloading it if no local copy exists."""
        name = _safe_name(video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4")
        for folder in [*self.video_dirs, self.video_download_dir]:
            candidate = folder / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return self._retrying(lambda: self._download_video(name))

    def track_file(self, video_id: str, start_second: int, fps: float | None = None) -> Path | None:
        """Local path of the segment's track CSV, downloading it if needed; None if the server has none."""
        _safe_name(video_id)
        for folder in [*self.bbox_dirs, self.bbox_download_dir]:
            if folder.is_dir():
                found = sorted(p for p in folder.glob(f"{video_id}_{start_second}_*.csv")
                               if is_track_file_for(p.name, video_id, start_second))
                if found:
                    return found[0]
        if not self.bbox_server_folder:
            return None
        return self._retrying(lambda: self._download_track_file(video_id, start_second, fps))

    # ---- downloads -------------------------------------------------------------

    def _session(self) -> Any:
        session = self.session_factory()
        if self.auth:
            session.auth = self.auth
        session.headers.update({"User-Agent": USER_AGENT})
        return session

    def _file_url(self, alias: str, path: str) -> str:
        return urljoin(self.base_url, f"v/{quote(alias, safe='_-')}/files/{quote(path, safe='/._-')}")

    def _browse_url(self, alias: str, path: str = "") -> str:
        suffix = f"/{quote(path, safe='/._-')}" if path else ""
        return urljoin(self.base_url, f"v/{quote(alias, safe='_-')}/browse{suffix}")

    def _download_video(self, name: str) -> Path:
        destination = self.video_download_dir / name
        with self._session() as session:
            for alias in self.video_aliases:
                response = self._get(session, self._file_url(alias, name), stream=True, allow_not_found=True)
                if response is not None:
                    self._save(response, destination)
                    return destination
            target = name.lower()
            found = self._crawl(session, [self._browse_url(a) for a in self.video_aliases],
                                lambda file_name: file_name.lower() == target, first_only=True)
            if found:
                self._save(self._get(session, found[0], stream=True, allow_not_found=False), destination)
                return destination
        raise DownloadError(f"Video not on the file server: {name}")

    def _download_track_file(self, video_id: str, start_second: int, fps: float | None) -> Path | None:
        # CROWD names track files with the truncated frame rate; try it and the rounded one.
        names = list(dict.fromkeys(f"{video_id}_{start_second}_{value}.csv" for value in (int(fps), round(fps)))
                     ) if fps else []
        with self._session() as session:
            for alias in self.bbox_aliases:
                for name in names:
                    url = self._file_url(alias, f"{self.bbox_server_folder}/{name}")
                    response = self._get(session, url, stream=True, allow_not_found=True)
                    if response is not None:
                        destination = self.bbox_download_dir / name
                        self._save(response, destination)
                        return destination
            starts = tuple(self._browse_url(alias, self.bbox_server_folder) for alias in self.bbox_aliases)
            if starts not in self._listings:
                # One listing of the bbox folder serves every later segment.
                self._listings[starts] = self._crawl(session, list(starts), lambda n: n.lower().endswith(".csv"))
            for url in self._listings[starts]:
                name = PurePosixPath(urlparse(url).path).name
                if is_track_file_for(name, video_id, start_second):
                    response = self._get(session, url, stream=True, allow_not_found=True)
                    if response is not None:
                        destination = self.bbox_download_dir / name
                        self._save(response, destination)
                        return destination
        return None

    def _get(self, session: Any, url: str, *, stream: bool, allow_not_found: bool,
             read_timeout: int | None = None) -> Any | None:
        self._require_same_origin(url)
        timeout = self.timeout_seconds if read_timeout is None else (
            self.timeout_seconds, max(self.timeout_seconds, read_timeout))
        try:
            response = session.get(url, timeout=timeout, stream=stream,
                                   params={"token": self.token} if self.token else None)
        except Exception as error:
            raise TransientDownloadError("File server request failed") from error
        if response.status_code in {401, 403}:
            response.close()
            raise AuthenticationError("File server refused the login: check ftp_username and ftp_password in 'secret'.")
        if response.status_code == 404 and allow_not_found:
            response.close()
            return None
        if response.status_code >= 500:
            response.close()
            # 530 is Cloudflare's "origin unreachable": the machine serving the drives is off or offline.
            hint = ": the machine behind the server is unreachable" if response.status_code == 530 else ""
            raise TransientDownloadError(f"File server unavailable (HTTP {response.status_code}){hint}")
        if response.status_code >= 400:
            response.close()
            raise DownloadError(f"File server returned HTTP {response.status_code}")
        if not self._same_origin(response.url):
            response.close()
            raise DownloadError("File server redirected to another origin")
        return response

    def _crawl(self, session: Any, start_urls: list[str], matches: Callable[[str], bool],
               first_only: bool = False) -> list[str]:
        """File URLs whose name matches, found by walking the browse pages (downwards only)."""
        found: list[str] = []
        visited: set[str] = set()
        stack = list(reversed(start_urls))
        roots = [urlparse(url).path.rstrip("/") for url in start_urls]
        while stack:
            url = stack.pop()
            if url in visited:
                continue
            visited.add(url)
            if len(visited) > self.crawl_page_limit:
                raise DownloadError(f"File search passed {self.crawl_page_limit} browse pages")
            response = self._get(session, url, stream=False, allow_not_found=True,
                                 read_timeout=LISTING_READ_TIMEOUT_SECONDS)
            if response is None:
                continue
            try:
                parser = _Links()
                parser.feed(response.text)
            finally:
                response.close()
            for href in parser.hrefs:
                full = urljoin(url, href)
                if not self._same_origin(full):
                    continue
                path = urlparse(full).path
                if "/files/" in path and matches(PurePosixPath(path).name) and full not in found:
                    found.append(full)
                    if first_only:
                        return found
                # Every page links back to its parent: descend only below the start folders.
                if "/browse" in path and full not in visited and any(path.startswith(root + "/") for root in roots):
                    stack.append(full)
        return found

    def _save(self, response: Any, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        length = response.headers.get("content-length")
        expected = int(length) if length and length.isdigit() else None
        written = 0
        try:
            with partial.open("wb") as handle:
                try:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        handle.write(chunk)
                        written += len(chunk)
                except Exception as error:
                    raise TransientDownloadError("Download broke off") from error
                handle.flush()
                os.fsync(handle.fileno())
            if written == 0:
                raise DownloadError(f"File server sent an empty file for {destination.name}")
            if expected is not None and written != expected:
                raise TransientDownloadError(f"Incomplete download: {written} of {expected} bytes")
            partial.replace(destination)
            logger.info("Downloaded %s (%.1f MiB)", destination.name, written / 2**20)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        finally:
            response.close()

    def _retrying(self, attempt: Callable[[], Any]) -> Any:
        for number in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                return attempt()
            except TransientDownloadError as error:
                if number == DOWNLOAD_ATTEMPTS:
                    raise
                logger.warning("%s; attempt %d of %d", error, number + 1, DOWNLOAD_ATTEMPTS)
                time.sleep(RETRY_DELAY_SECONDS * number)
        raise AssertionError("unreachable")

    def _same_origin(self, url: str) -> bool:
        parsed = urlparse(url)
        return (parsed.scheme.lower(), parsed.netloc.lower()) == self.origin

    def _require_same_origin(self, url: str) -> None:
        if not self._same_origin(url):
            raise DownloadError("Refusing a URL outside the file server")
