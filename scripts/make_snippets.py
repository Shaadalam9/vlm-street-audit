"""Cut the snippets the VLMs analyse from the sampled cities' videos.

For each chosen video (the same choice as fetch_data.py: per city, the videos with
the most qualifying footage), windows inside its mapped segments are ranked by
SigLIP2 keyframe semantics with packet energy as tie break, the best event per
cluster is kept, random windows are added, and each snippet is written as an
H.264 clip plus JPEG frames. See src/vlm_street_audit/snippets.py for the method.

Output: data/snippets/<video_id>/<snippet>.mp4, data/snippets/<video_id>/<snippet>/frame_NNN.jpg,
and data/snippets/snippets.csv (one row per snippet, with its scores).

Run:  uv run python scripts/make_snippets.py --videos-per-city 1
      uv run python scripts/make_snippets.py --country Japan --download
      uv run python scripts/make_snippets.py --video path/to/clip.mp4   # one local file, whole length
"""

import argparse
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from vlm_street_audit.config import load_config, project_path
from vlm_street_audit.file_server import DownloadError, FileServer
from vlm_street_audit.mapping import pick_videos, read_segments
from vlm_street_audit.semantics import SemanticScorer, finite_or_blank, top_group
from vlm_street_audit.snippets import (
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
    parser.add_argument("--no-semantics", action="store_true", help="rank by packet energy only (no SigLIP2)")
    parser.add_argument("--video", type=Path, help="one local video, used whole (for testing)")
    parser.add_argument("--output-dir", type=Path, help="default: snippet_dir from the config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Hugging Face request otherwise

    settings = load_config()
    length = float(settings["snippet_length_s"])
    out_dir = args.output_dir or project_path(settings["snippet_dir"])

    # (locality, country, video_id, path, intervals) for every video to cut.
    jobs: list[tuple[str, str, str, Path, list[tuple[float, float]]]] = []
    if args.video:
        jobs.append(("", "", args.video.stem, args.video, [(0.0, video_duration(args.video))]))
    else:
        segments = [s for s in read_segments(project_path(settings["mapping"]))
                    if (not args.city or s.locality in args.city) and (not args.country or s.country in args.country)]
        chosen = pick_videos(segments, args.videos_per_city)
        per_video: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
        for s in segments:
            if s.video_id in chosen:
                per_video[(s.locality, s.country, s.video_id)].append((s.start_second, s.end_second))
        server = FileServer.from_config(settings, require_login=args.download)
        for (locality, country, video_id), intervals in sorted(per_video.items()):
            path = server.local_video(video_id)
            if path is None and args.download:
                try:
                    path = server.video(video_id)
                except DownloadError as error:
                    logger.error("%s: %s", video_id, error)
            if path is None:
                logger.warning("%s (%s): no local video; use --download", video_id, locality)
                continue
            jobs.append((locality, country, video_id, path, intervals))

    scorer = None
    if settings["semantic_ranking"] and not args.no_semantics:
        scorer = SemanticScorer(project_path(settings["semantic_prompts"]), project_path(settings["embedding_cache_dir"]),
                                settings["semantic_model"], int(settings["semantic_batch_size"]),
                                float(settings["semantic_min_frame_gap_s"]), int(settings["semantic_top_k"]))

    rows = []
    for locality, country, video_id, path, raw_intervals in jobs:
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
        for window, kind in selected:
            name = snippet_name(video_id, window.start_s, length)
            clip = out_dir / video_id / f"{name}.mp4"
            frames = extract_snippet(path, window.start_s, length, clip, out_dir / video_id / name,
                                     float(settings["snippet_frames_fps"]))
            rows.append({
                "snippet": name, "locality": locality, "country": country, "video_id": video_id,
                "segment_start_s": window.segment_start_s, "segment_end_s": window.segment_end_s,
                "start_s": window.start_s, "end_s": window.end_s, "selection": kind, "rank": window.rank,
                "windows_in_video": len(windows), "semantic_score": finite_or_blank(window.semantic_score),
                "top_group": top_group(window), "packet_score": f"{window.packet_score:.4f}",
                "clip": clip.relative_to(out_dir).as_posix(),
                "frames_dir": (out_dir / video_id / name).relative_to(out_dir).as_posix(), "frame_count": len(frames),
            })
        logger.info("%s (%s): %d windows, %d snippets", video_id, locality or path.name, len(windows), len(selected))

    write_manifest(rows, out_dir / "snippets.csv")
    print(f"{len(rows)} snippets from {len(jobs)} videos -> {out_dir / 'snippets.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
