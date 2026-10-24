"""Candidate snippets: fixed windows inside the mapped segments, ranked, de-duplicated, extracted.

Adapted from opticarvis-global (`src/candidates/candidate_index.py`,
`candidate_semantics.py`, and the clip extraction in `batch_corrected_pipeline.py`):

1. Windows of `length_s` every `stride_s` inside each qualifying mapping segment;
   no window crosses a segment boundary.
2. Packet energy: ffprobe demuxes the video (no pixel decoding) and each window gets
   the robust z-score of its encoded bytes per second (log, median/MAD). Busy, complex
   scenes encode to more bytes; this is only a tie break.
3. Semantic score (`semantics.py`): SigLIP2 similarity of the window's keyframes to
   problem prompts, minus their similarity to routine-scene prompts.
4. Event clusters: the best-ranked window claims its neighbours closer than
   `min_gap_s`, so one scene does not fill every slot.
5. Selection per video: the top `ranked_per_video` cluster representatives, plus
   `random_per_video` windows drawn uniformly (seeded) from the rest. The random ones
   are not biased toward problems, so they show what an average stretch looks like.
6. Extraction: ffmpeg re-encodes each snippet to H.264 (no audio) and writes frames at
   `frames_fps` as JPEGs for the VLMs.
"""

import csv
import math
import random
import shutil
import statistics
import subprocess
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Window:
    video_id: str
    segment_start_s: float
    segment_end_s: float
    start_s: float
    end_s: float
    packet_bytes_per_s: float = 0.0
    packet_score: float = 0.0
    semantic_score: float | None = None
    group_scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    cluster: int = 0
    is_representative: bool = False


def normalise_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted, merged, finite (start, end) ranges with end > start."""
    clean = sorted((float(s), float(e)) for s, e in intervals
                   if math.isfinite(float(s)) and math.isfinite(float(e)) and float(e) > float(s) >= 0)
    merged: list[tuple[float, float]] = []
    for start, end in clean:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"{name} is not on PATH")
    return path


def probe_packets(video: Path, intervals: list[tuple[float, float]]) -> list[tuple[float, int]]:
    """(time_s, bytes) of every video packet inside the intervals; no decoding."""
    read = ",".join(f"{start:.3f}%{end:.3f}" for start, end in intervals)
    command = [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,size",
               "-of", "csv=p=0", "-read_intervals", read, str(video)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed on {video}")
    packets = []
    for line in result.stdout.splitlines():
        parts = line.split(",")
        try:
            time_s, size = float(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            continue
        if math.isfinite(time_s) and any(start <= time_s < end for start, end in intervals):
            packets.append((time_s, size))
    return sorted(packets)


def robust_scores(values: list[float]) -> list[float]:
    """Within-video robust z-scores of log(1 + value)."""
    if not values:
        return []
    logged = [math.log1p(max(0.0, v)) for v in values]
    centre = statistics.median(logged)
    mad = statistics.median(abs(v - centre) for v in logged)
    if mad > 1e-12:
        return [(v - centre) / (1.4826 * mad) for v in logged]
    spread = statistics.pstdev(logged) if len(logged) > 1 else 0.0
    if spread <= 1e-12:
        return [0.0] * len(logged)
    mean = statistics.fmean(logged)
    return [(v - mean) / spread for v in logged]


def make_windows(video_id: str, intervals: list[tuple[float, float]], packets: list[tuple[float, int]],
                 length_s: float, stride_s: float) -> list[Window]:
    """Every full window inside each interval, with its packet energy score."""
    if length_s <= 0 or stride_s <= 0:
        raise ValueError("length_s and stride_s must be positive")
    times = [t for t, _ in packets]
    prefix = [0]
    for _, size in packets:
        prefix.append(prefix[-1] + size)
    windows = []
    for seg_start, seg_end in intervals:
        start = seg_start
        while start + length_s <= seg_end + 1e-9:
            left, right = bisect_left(times, start), bisect_left(times, start + length_s)
            if right > left:
                windows.append(Window(video_id, seg_start, seg_end, round(start, 3), round(start + length_s, 3),
                                      packet_bytes_per_s=(prefix[right] - prefix[left]) / length_s))
            start += stride_s
    for window, score in zip(windows, robust_scores([w.packet_bytes_per_s for w in windows])):
        window.packet_score = score
    return windows


def rank_windows(windows: list[Window]) -> list[Window]:
    """Rank by semantic score (when present), then packet energy, then time."""
    ordered = sorted(windows, key=lambda w: (-(w.semantic_score if w.semantic_score is not None else -math.inf),
                                             -w.packet_score, w.start_s))
    for rank, window in enumerate(ordered, start=1):
        window.rank = rank
    return ordered


def cluster_events(windows: list[Window], min_gap_s: float) -> None:
    """Best-ranked window claims neighbours closer than `min_gap_s` (not transitive)."""
    unassigned = sorted(windows, key=lambda w: (w.rank, w.start_s))
    cluster = 0
    while unassigned:
        head = unassigned[0]
        cluster += 1
        claimed = [w for w in unassigned if abs(w.start_s - head.start_s) < min_gap_s]
        for window in claimed:
            window.cluster = cluster
            window.is_representative = window is head
        unassigned = [w for w in unassigned if w.cluster == 0]


def select_windows(windows: list[Window], ranked_per_video: int, random_per_video: int, min_gap_s: float,
                   seed: int) -> list[tuple[Window, str]]:
    """Top-ranked event representatives, then random windows away from them."""
    rank_windows(windows)
    cluster_events(windows, min_gap_s)
    chosen = [(w, "ranked") for w in sorted((w for w in windows if w.is_representative), key=lambda w: w.rank)
              [:ranked_per_video]]
    rng = random.Random(f"{seed}:{windows[0].video_id}" if windows else seed)
    pool = [w for w in windows if all(abs(w.start_s - c.start_s) >= min_gap_s for c, _ in chosen)]
    rng.shuffle(pool)
    for window in pool:
        if sum(kind == "random" for _, kind in chosen) >= random_per_video:
            break
        if all(abs(window.start_s - c.start_s) >= min_gap_s for c, _ in chosen):
            chosen.append((window, "random"))
    return chosen


def snippet_name(video_id: str, start_s: float, length_s: float) -> str:
    return f"{video_id}_{int(round(start_s * 1000)):08d}_{int(round(length_s * 1000)):06d}"


def extract_snippet(video: Path, start_s: float, length_s: float, clip_path: Path, frames_dir: Path,
                    frames_fps: float) -> list[Path]:
    """Write the H.264 clip and its frames; returns the frame paths. Skips work already done."""
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    if not (clip_path.is_file() and clip_path.stat().st_size > 0):
        partial = clip_path.with_suffix(".part.mp4")
        command = [_tool("ffmpeg"), "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(video),
                   "-t", f"{length_s:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(partial)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            raise RuntimeError((result.stderr or "ffmpeg failed").strip()[:500])
        partial.replace(clip_path)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        command = [_tool("ffmpeg"), "-y", "-loglevel", "error", "-i", str(clip_path), "-vf", f"fps={frames_fps}",
                   "-q:v", "2", str(frames_dir / "frame_%03d.jpg")]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "ffmpeg failed").strip()[:500])
        frames = sorted(frames_dir.glob("frame_*.jpg"))
    return frames


MANIFEST_COLUMNS = [
    "snippet", "locality", "country", "video_id", "segment_start_s", "segment_end_s", "start_s", "end_s",
    "selection", "rank", "windows_in_video", "semantic_score", "top_group", "packet_score", "clip", "frames_dir",
    "frame_count",
]


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)
