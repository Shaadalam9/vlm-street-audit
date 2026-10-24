"""File server client against a fake HTTP session (no network)."""

from pathlib import Path

import pytest

from vlm_street_audit.file_server import (
    AuthenticationError,
    DownloadError,
    FileServer,
    parse_track_filename,
)

BASE = "https://files.example.org/"


class FakeResponse:
    def __init__(self, url: str, status: int = 200, body: bytes = b"", length: int | None = None) -> None:
        self.url = url
        self.status_code = status
        self._body = body
        self.headers = {"content-length": str(len(body) if length is None else length)}
        self.text = body.decode()
        self.closed = False

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """Serves `pages` (url -> body); every other URL is a 404. Records the requests."""

    def __init__(self, pages: dict[str, bytes], status: dict[str, int] | None = None) -> None:
        self.pages = pages
        self.status = status or {}
        self.requested: list[str] = []
        self.auth = None
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.requested.append(url)
        if url in self.status:
            return FakeResponse(url, self.status[url])
        if url in self.pages:
            return FakeResponse(url, 200, self.pages[url])
        return FakeResponse(url, 404)


def make_server(tmp_path: Path, session: FakeSession, **overrides) -> FileServer:
    settings = dict(
        video_aliases=["tue4", "tue5"],
        video_download_dir=tmp_path / "videos",
        bbox_aliases=["data"],
        bbox_server_folder="pedestrians_in-youtube/data/bbox",
        bbox_download_dir=tmp_path / "bbox",
        session_factory=lambda: session,
    )
    settings.update(overrides)
    return FileServer(BASE, "user", "pass", **settings)


def test_video_downloads_from_the_first_alias_that_has_it(tmp_path):
    session = FakeSession({f"{BASE}v/tue5/files/abc_1.mp4": b"video-bytes"})
    path = make_server(tmp_path, session).video("abc_1")
    assert path == tmp_path / "videos" / "abc_1.mp4"
    assert path.read_bytes() == b"video-bytes"
    assert session.requested == [f"{BASE}v/tue4/files/abc_1.mp4", f"{BASE}v/tue5/files/abc_1.mp4"]
    assert session.auth == ("user", "pass")
    assert not list((tmp_path / "videos").glob("*.part"))


def test_video_prefers_a_local_copy(tmp_path):
    local = tmp_path / "drive"
    local.mkdir()
    (local / "abc.mp4").write_bytes(b"local")
    session = FakeSession({})
    assert make_server(tmp_path, session, video_dirs=[local]).video("abc") == local / "abc.mp4"
    assert session.requested == []


def test_video_found_by_walking_the_browse_pages(tmp_path):
    session = FakeSession({
        f"{BASE}v/tue4/browse": b'<a href="/v/tue4/browse/sub">sub</a><a href="/v/tue4/browse/..">up</a>',
        f"{BASE}v/tue4/browse/sub": b'<a href="/v/tue4/files/sub/xyz.mp4">xyz</a>',
        f"{BASE}v/tue4/files/sub/xyz.mp4": b"deep",
    })
    assert make_server(tmp_path, session).video("xyz").read_bytes() == b"deep"


def test_missing_video_raises(tmp_path):
    with pytest.raises(DownloadError, match="not on the file server"):
        make_server(tmp_path, FakeSession({})).video("nothing")


def test_rejected_login_raises_authentication_error(tmp_path):
    session = FakeSession({}, status={f"{BASE}v/tue4/files/abc.mp4": 401})
    with pytest.raises(AuthenticationError):
        make_server(tmp_path, session).video("abc")


def test_incomplete_download_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("vlm_street_audit.file_server.RETRY_DELAY_SECONDS", 0)

    class Short(FakeSession):
        def get(self, url, **kwargs):
            self.requested.append(url)
            return FakeResponse(url, 200, b"half", length=100)

    with pytest.raises(DownloadError, match="Incomplete"):
        make_server(tmp_path, Short({})).video("abc")
    assert not any((tmp_path / "videos").iterdir())


def test_unsafe_names_and_foreign_links_are_refused(tmp_path):
    server = make_server(tmp_path, FakeSession({}))
    with pytest.raises(ValueError):
        server.video("../etc/passwd")
    with pytest.raises(DownloadError):
        server._get(FakeSession({}), "https://elsewhere.org/x", stream=False, allow_not_found=True)


def test_track_file_by_frame_rate_name(tmp_path):
    url = f"{BASE}v/data/files/pedestrians_in-youtube/data/bbox/a_b_120_29.csv"
    session = FakeSession({url: b"frame,x\n"})
    path = make_server(tmp_path, session).track_file("a_b", 120, fps=29.97)
    assert path == tmp_path / "bbox" / "a_b_120_29.csv"


def test_track_file_found_in_the_listing_is_listed_only_once(tmp_path):
    folder = f"{BASE}v/data/browse/pedestrians_in-youtube/data/bbox"
    files = f"/v/data/files/pedestrians_in-youtube/data/bbox"
    session = FakeSession({
        folder: f'<a href="{files}/a_120_30.csv"></a><a href="{files}/b_0_25.csv"></a>'.encode(),
        f"{BASE}{files[1:]}/a_120_30.csv": b"a",
        f"{BASE}{files[1:]}/b_0_25.csv": b"b",
    })
    server = make_server(tmp_path, session)
    assert server.track_file("a", 120).name == "a_120_30.csv"
    assert server.track_file("b", 0).name == "b_0_25.csv"
    assert server.track_file("c", 0) is None
    assert session.requested.count(folder) == 1


def test_parse_track_filename_keeps_underscores_in_the_video_id():
    assert parse_track_filename("_e0RQL7NJZg_1056_30.csv") == ("_e0RQL7NJZg", 1056, 30)


def test_credentials_in_the_url_are_refused(tmp_path):
    with pytest.raises(ValueError, match="secret"):
        FileServer("https://user:pw@files.example.org/", "u", "p", video_aliases=["tue4"],
                   video_download_dir=tmp_path, bbox_download_dir=tmp_path, session_factory=FakeSession)


def test_server_errors_are_retried_then_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("vlm_street_audit.file_server.RETRY_DELAY_SECONDS", 0)
    session = FakeSession({}, status={f"{BASE}v/tue4/files/abc.mp4": 530})
    with pytest.raises(DownloadError, match="unreachable"):
        make_server(tmp_path, session).video("abc")
    assert session.requested.count(f"{BASE}v/tue4/files/abc.mp4") == 3
