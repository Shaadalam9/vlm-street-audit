"""Fetch videos (and optionally their YOLO track CSVs) of the sampled cities.

The 75-city sample holds about 5,500 videos, far too many to copy whole, so this
takes the N videos with the most qualifying footage per city (default 1). Files
already on a local drive (lin4080) or in the download folder are not fetched again.

Run:  uv run python scripts/fetch_data.py --dry-run
      uv run python scripts/fetch_data.py --videos-per-city 2 --tracks
      uv run python scripts/fetch_data.py --country Japan --videos-per-city 1
"""

import argparse
import logging
import sys
from collections import defaultdict

from vlm_street_audit.config import load_config, project_path
from vlm_street_audit.file_server import DownloadError, FileServer
from vlm_street_audit.mapping import read_segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-per-city", type=int, default=1)
    parser.add_argument("--city", action="append", help="locality name; repeat for several")
    parser.add_argument("--country", action="append", help="country name; repeat for several")
    parser.add_argument("--tracks", action="store_true", help="also fetch the segments' track CSVs")
    parser.add_argument("--dry-run", action="store_true", help="list what would be fetched, fetch nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_config()
    segments = [
        s for s in read_segments(project_path(settings["mapping"]))
        if (not args.city or s.locality in args.city) and (not args.country or s.country in args.country)
    ]
    footage: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for segment in segments:
        footage[(segment.locality, segment.country)][segment.video_id] += segment.duration_seconds
    chosen = {
        video_id
        for videos in footage.values()
        for video_id, _ in sorted(videos.items(), key=lambda item: (-item[1], item[0]))[: args.videos_per_city]
    }
    plan = [s for s in segments if s.video_id in chosen]
    print(f"{len(footage)} cities, {len(chosen)} videos, {len(plan)} segments, "
          f"{sum(s.duration_seconds for s in plan) / 3600:.1f} h of qualifying footage")
    if args.dry_run:
        for (locality, country), videos in sorted(footage.items()):
            picked = [v for v in videos if v in chosen]
            print(f"  {locality}, {country}: {', '.join(picked)}")
        return 0

    server = FileServer.from_config(settings)
    failures = 0
    for video_id in sorted(chosen):
        try:
            print(f"{video_id}: {server.video(video_id)}")
        except DownloadError as error:
            failures += 1
            print(f"{video_id}: FAILED ({error})", file=sys.stderr)
    if args.tracks:
        for segment in plan:
            try:
                path = server.track_file(segment.video_id, segment.start_second)
            except DownloadError as error:
                failures += 1
                print(f"{segment.video_id} {segment.start_second}: FAILED ({error})", file=sys.stderr)
                continue
            print(f"{segment.video_id} {segment.start_second}: {path or 'no track file on the server'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
