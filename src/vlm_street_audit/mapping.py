"""Segments of a CROWD mapping file (one row per locality, nested lists per video)."""

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Segment:
    locality: str
    country: str
    video_id: str
    start_second: int
    end_second: int
    time_of_day: int
    vehicle_type: int

    @property
    def duration_seconds(self) -> int:
        return self.end_second - self.start_second


def _video_ids(value: str) -> list[str]:
    """The videos field is a bracketed list without quotes: [id_a,id_b]."""
    text = str(value).strip().removeprefix("[").removesuffix("]")
    return [item.strip() for item in text.split(",") if item.strip()]


def pick_videos(segments: list[Segment], per_city: int) -> set[str]:
    """Per city, the `per_city` videos with the most footage (ties by video ID)."""
    footage: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for segment in segments:
        footage[(segment.locality, segment.country)][segment.video_id] += segment.duration_seconds
    return {
        video_id
        for videos in footage.values()
        for video_id, _ in sorted(videos.items(), key=lambda item: (-item[1], item[0]))[:per_city]
    }


def read_segments(path: Path) -> Iterator[Segment]:
    for row in pd.read_csv(path).itertuples(index=False):
        per_video = zip(
            _video_ids(row.videos),
            json.loads(row.vehicle_type),
            json.loads(row.time_of_day),
            json.loads(row.start_time),
            json.loads(row.end_time),
            strict=True,
        )
        for video_id, vehicle, days, starts, ends in per_video:
            for day, start, end in zip(days, starts, ends, strict=True):
                yield Segment(row.locality, row.country, video_id, int(start), int(end), int(day), int(vehicle))
