"""Segments of a CROWD mapping file (one row per locality, nested lists per video)."""

import json
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
