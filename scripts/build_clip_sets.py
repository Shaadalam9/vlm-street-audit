"""Build the clip sets the VLMs run on.

cities  per sampled city video: its best-ranked snippet (role "main") and its random
        snippet (role "control", for invented problems), from data/snippets/snippets.csv.
jaad    a seeded sample of JAAD clips, half with a zebra crossing visible and half
        without, with JAAD's labels over the same frames (see vlm_street_audit/jaad.py).

Both send `vlm_frames_per_clip` frames spread over at most 10 s.
Output: data/clip_sets/{cities,jaad}.csv, data/clip_sets/jaad_labels.json, JAAD frames.

Run:  uv run python scripts/build_clip_sets.py
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

from vlm_street_audit.config import PROJECT_ROOT, load_config, project_path
from vlm_street_audit.jaad import FPS, frame_count, labels

COLUMNS = ["clip_id", "set", "role", "locality", "country", "seconds", "frames"]


def evenly(items: list, n: int) -> list:
    """n items spread over the list, starting at the second of each pair (1 fps frames -> every 2 s)."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[min(len(items) - 1, int(step * i + step / 2))] for i in range(n)]


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix() if path.resolve().is_relative_to(
        PROJECT_ROOT.resolve()) else str(path)


def city_rows(settings: dict) -> list[dict]:
    snippet_dir = project_path(settings["snippet_dir"])
    snippets = pd.read_csv(snippet_dir / "snippets.csv")
    rows = []
    for video_id, group in snippets.groupby("video_id"):
        picks = [("main", group[group.selection == "ranked"].sort_values("rank").head(1)),
                 ("control", group[group.selection == "random"].head(1))]
        for role, pick in picks:
            for snippet in pick.itertuples():
                frames = sorted((snippet_dir / snippet.frames_dir).glob("frame_*.jpg"))
                chosen = evenly(frames, int(settings["vlm_frames_per_clip"]))
                rows.append({"clip_id": snippet.snippet, "set": "cities", "role": role, "locality": snippet.locality,
                             "country": snippet.country, "seconds": snippet.end_s - snippet.start_s,
                             "frames": ";".join(relative(f) for f in chosen)})
    return rows


def extract_frames(video: Path, numbers: list[int], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"frame_{n:05d}.jpg" for n in numbers]
    if all(p.is_file() for p in paths):
        return paths
    select = "+".join(f"eq(n\\,{n})" for n in numbers)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"select={select}",
                    "-fps_mode", "vfr", "-q:v", "2", str(out_dir / "tmp_%02d.jpg")], check=True)
    for index, path in enumerate(paths, start=1):
        (out_dir / f"tmp_{index:02d}.jpg").replace(path)
    return paths


def jaad_rows(settings: dict, out_dir: Path) -> tuple[list[dict], dict]:
    jaad_dir = project_path(settings["jaad_dir"])
    n_frames = int(settings["vlm_frames_per_clip"])
    candidates = []
    for xml in sorted((jaad_dir / "annotations").glob("video_*.xml")):
        video = xml.stem
        size = frame_count(jaad_dir, video)
        span = min(10 * FPS, size)
        numbers = [int(i * span / n_frames) for i in range(n_frames)]
        candidates.append((video, numbers, {**labels(jaad_dir, video, numbers), "seconds": span / FPS}))
    rng = random.Random(int(settings["snippet_seed"]))
    half = int(settings["jaad_sample_size"]) // 2
    with_zebra = [c for c in candidates if c[2]["zebra_crossing_visible"]]
    without = [c for c in candidates if not c[2]["zebra_crossing_visible"]]
    sample = rng.sample(with_zebra, min(half, len(with_zebra))) + rng.sample(without, min(half, len(without)))
    rows, truth = [], {}
    for video, numbers, label in sorted(sample):
        frames = extract_frames(jaad_dir / "JAAD_clips" / f"{video}.mp4", numbers, out_dir / "jaad_frames" / video)
        rows.append({"clip_id": video, "set": "jaad", "role": "main", "locality": "", "country": "",
                     "seconds": round(label["seconds"], 1),
                     "frames": ";".join(relative(f) for f in frames)})
        truth[video] = {"frames": numbers, **label}
    return rows, truth


def write(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-jaad", action="store_true")
    args = parser.parse_args()
    settings = load_config()
    out_dir = project_path(settings["clip_sets_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cities = city_rows(settings)
    write(cities, out_dir / "cities.csv")
    print(f"cities: {len(cities)} clips ({sum(r['role'] == 'main' for r in cities)} main, "
          f"{sum(r['role'] == 'control' for r in cities)} control)")
    if not args.skip_jaad:
        rows, truth = jaad_rows(settings, out_dir)
        write(rows, out_dir / "jaad.csv")
        (out_dir / "jaad_labels.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
        print(f"jaad: {len(rows)} clips ({sum(t['zebra_crossing_visible'] for t in truth.values())} with a zebra "
              f"crossing, {sum(t['pedestrians_in_carriageway'] for t in truth.values())} with someone crossing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
