"""Cut the snippets the VLMs analyse from the sampled cities' videos.

For each chosen video (the same choice as fetch_data.py: per city, the videos with
the most qualifying footage), windows inside its mapped segments are ranked by
SigLIP2 keyframe semantics with packet energy as tie break, the best event per
cluster is kept, random windows are added, and each snippet is written as an
H.264 clip plus JPEG frames. See src/vlm_street_audit/snippets.py for the method.

Output: data/snippets/<video_id>/<snippet>.mp4, data/snippets/<video_id>/<snippet>/frame_NNN.jpg,
and data/snippets/snippets.csv (one row per snippet, with its scores). Cities that already have
snippets are skipped and their rows kept, so a run adds to the manifest (--redo cuts them again).

Run:  uv run python scripts/make_snippets.py --videos-per-city 1
      uv run python scripts/make_snippets.py --country Japan --download
      uv run python scripts/make_snippets.py --local-only   # only videos already on disk
      uv run python scripts/make_snippets.py --video path/to/clip.mp4   # one local file, whole length
"""

import argparse
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from vlm_street_audit.config import load_config, project_path
from vlm_street_audit.file_server import DownloadError, FileServer
from vlm_street_audit.mapping import pick_videos, read_segments
from vlm_street_audit.semantics import SemanticScorer, finite_or_blank, top_group
from vlm_street_audit.snippets import (
    MANIFEST_COLUMNS,
    extract_snippet,
    make_windows,
    normalise_intervals,
    probe_packets,
    select_windows,
    snippet_name,
    write_manifest,
)

logger = logging.getLogger("make_snippets")


def video_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                             str(path)], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-per-city", type=int, default=1)
    parser.add_argument("--city", action="append", help="locality name; repeat for several")
    parser.add_argument("--country", action="append", help="country name; repeat for several")
    parser.add_argument("--download", action="store_true", help="fetch videos missing locally from the file server")
    parser.add_argument("--local-only", action="store_true",
                        help="per city, choose only among videos already on disk (e.g. another project's downloads)")
    parser.add_argument("--no-semantics", action="store_true", help="rank by packet energy only (no SigLIP2)")
    parser.add_argument("--video", type=Path, help="one local video, used whole (for testing)")
    parser.add_argument("--redo", action="store_true", help="cut cities that already have snippets again")
    parser.add_argument("--output-dir", type=Path, help="default: snippet_dir from the config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Hugging Face request otherwise

    settings = load_config()
    length = float(settings["snippet_length_s"])
    out_dir = args.output_dir or project_path(settings["snippet_dir"])

    manifest = out_dir / "snippets.csv"
    existing = pd.read_csv(manifest, dtype=str, keep_default_na=False) if manifest.is_file() else pd.DataFrame(
        columns=MANIFEST_COLUMNS)
    done_cities = set(zip(existing.locality, existing.country)) - {("", "")}

    # (locality, country, video_id, intervals) for every video to cut; its file is found or fetched in turn.
    jobs: list[tuple[str, str, str, list[tuple[float, float]]]] = []
    server = None
    if args.video:
        jobs.append(("", "", args.video.stem, [(0.0, video_duration(args.video))]))
    else:
        segments = [s for s in read_segments(project_path(settings["mapping"]))
                    if (not args.city or s.locality in args.city) and (not args.country or s.country in args.country)]
        if not args.redo:
            skipped = {(s.locality, s.country) for s in segments} & done_cities
            segments = [s for s in segments if (s.locality, s.country) not in done_cities]
            if skipped:
                logger.info("%d cities already have snippets and are skipped (--redo to cut them again)", len(skipped))
        server = FileServer.from_config(settings, require_login=args.download)
        if args.local_only:
            # Choose among the videos already on disk: cities without one are skipped.
            local = {v for v in {s.video_id for s in segments} if server.local_video(v) is not None}
            segments = [s for s in segments if s.video_id in local]
        chosen = pick_videos(segments, args.videos_per_city)
        per_video: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
        for s in segments:
            if s.video_id in chosen:
                per_video[(s.locality, s.country, s.video_id)].append((s.start_second, s.end_second))
        jobs = [(locality, country, video_id, intervals)
                for (locality, country, video_id), intervals in sorted(per_video.items())]
    logger.info("%d videos to cut", len(jobs))

    scorer = None
    if settings["semantic_ranking"] and not args.no_semantics:
        scorer = SemanticScorer(project_path(settings["semantic_prompts"]), project_path(settings["embedding_cache_dir"]),
                                settings["semantic_model"], int(settings["semantic_batch_size"]),
                                float(settings["semantic_min_frame_gap_s"]), int(settings["semantic_top_k"]))

    rows: list[dict] = []
    cut = 0
    for number, (locality, country, video_id, raw_intervals) in enumerate(jobs, start=1):
        if args.video:
            path = args.video
        else:
            path = server.local_video(video_id)
            if path is None and args.download:
                try:
                    path = server.video(video_id)
                except DownloadError as error:
                    logger.error("%s (%s): %s", video_id, locality, error)
            if path is None:
                logger.warning("%s (%s): no local video; use --download", video_id, locality)
                continue
        intervals = normalise_intervals(raw_intervals)
        windows = make_windows(video_id, intervals, probe_packets(path, intervals), length,
                               float(settings["snippet_stride_s"]))
        if not windows:
            logger.warning("%s: no segment is %.0f s long", video_id, length)
            continue
        if scorer is not None:
            scorer.score(video_id, path, intervals, windows)
        selected = select_windows(windows, int(settings["snippets_ranked_per_video"]),
                                  int(settings["snippets_random_per_video"]), float(settings["snippet_min_gap_s"]),
                                  int(settings["snippet_seed"]))
        video_rows = []
        for window, kind in selected:
            name = snippet_name(video_id, window.start_s, length)
            clip = out_dir / video_id / f"{name}.mp4"
            frames = extract_snippet(path, window.start_s, length, clip, out_dir / video_id / name,
                                     float(settings["snippet_frames_fps"]))
            video_rows.append({
                "snippet": name, "locality": locality, "country": country, "video_id": video_id,
                "segment_start_s": window.segment_start_s, "segment_end_s": window.segment_end_s,
                "start_s": window.start_s, "end_s": window.end_s, "selection": kind, "rank": window.rank,
                "windows_in_video": len(windows), "semantic_score": finite_or_blank(window.semantic_score),
                "top_group": top_group(window), "packet_score": f"{window.packet_score:.4f}",
                "clip": clip.relative_to(out_dir).as_posix(),
                "frames_dir": (out_dir / video_id / name).relative_to(out_dir).as_posix(), "frame_count": len(frames),
            })
        rows.extend(video_rows)
        cut += 1
        # Save after every video, keeping earlier runs' rows for other videos.
        kept = existing[~existing.video_id.isin({r["video_id"] for r in rows})].to_dict("records")
        write_manifest(kept + rows, manifest)
        logger.info("[%d/%d] %s (%s): %d windows, %d snippets", number, len(jobs), video_id,
                    locality or path.name, len(windows), len(selected))

    print(f"{len(rows)} snippets from {cut} of {len(jobs)} videos -> {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
