"""Draw the cities (and so the countries) whose footage the VLMs analyse.

Same design as opticarvis-global's `scripts/city_sampler.py` (see its
docs/CITY_SAMPLING.md): a spatially balanced probability sample drawn with the
Local Pivotal Method from M-Colley/lpm-sampling.

    1. Frame: every locality in data/mapping_original.csv.
    2. Eligibility (applied BEFORE the draw, never after): footage of
       REQUIRED_VEHICLE_TYPE in REQUIRED_TIME_OF_DAY totalling at least
       MIN_FOOTAGE_HOURS, valid coordinates, positive population.
    3. Size measure: population ** ALPHA; inclusion probabilities pi from
       lpm_sampling.pi_from_size (iterative certainty-unit capping).
    4. Coordinates on the unit sphere, then LPM2 (or LPM1).
    5. 1 / pi is each city's Horvitz-Thompson design weight.

There is no one-city-per-country rule: large countries get several cities and
small ones at most one, as a PPS design does by itself. Countries are whatever
the drawn cities fall in. If a drawn city later has no usable footage, do not
swap in a neighbour; report it as missing.

Run:  uv run python scripts/sample_cities.py [--n 75] [--alpha 0.5] ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lpm_sampling import (
    lpm1,
    lpm2,
    nearest_distance_km,
    pi_from_size,
    spatial_balance,
    unit_sphere_xyz,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "mapping_original.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "city_sample"

# Design follows opticarvis-global except n (75, not 150) and alpha (0.5, not 0.75):
# at n=75, alpha 0.5 gives effective n 42 instead of 24 for ~8% longer distance to the nearest city.
SAMPLE_SIZE = 75
SEED = 20260818
ALPHA = 0.5  # 1 = strict PPS; lower pulls toward equal probability
METHOD = "lpm2"
POPULATION_COLUMN = "population_locality"
REQUIRED_VEHICLE_TYPE = 0
REQUIRED_TIME_OF_DAY = 0  # daytime
MIN_FOOTAGE_HOURS = 1.0

DESIGN_COLUMNS = [
    "source_row", "footage_hours_used", "size_measure", "pi", "design_weight",
    "sphere_x", "sphere_y", "sphere_z",
]


def parse_json_list(value):
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def parse_video_ids(value):
    """The videos field is a bracketed list without quotes: [id_a,id_b]."""
    if pd.isna(value):
        return []
    text = str(value).strip().removeprefix("[").removesuffix("]")
    return [item.strip() for item in text.split(",") if item.strip()]


def qualifying_segments(row, vehicle_type, time_of_day):
    """Per video, the (start, end) segments of the required vehicle type and time of day."""
    videos = parse_video_ids(row["videos"])
    per_video = zip(
        parse_json_list(row["vehicle_type"]),
        parse_json_list(row["time_of_day"]),
        parse_json_list(row["start_time"]),
        parse_json_list(row["end_time"]),
    )
    output = []
    for index, (vehicle, days, starts, ends) in enumerate(per_video):
        try:
            if int(vehicle) != vehicle_type:
                continue
        except (TypeError, ValueError):
            continue
        if not all(isinstance(group, list) for group in (days, starts, ends)):
            continue
        segments = []
        for day, start, end in zip(days, starts, ends):
            try:
                day, start_s, end_s = int(day), float(start), float(end)
            except (TypeError, ValueError):
                continue
            if day == time_of_day and np.isfinite(start_s) and np.isfinite(end_s) and end_s >= start_s:
                segments.append((start, end))
        if segments:
            output.append((videos[index] if index < len(videos) else "", segments))
    return output


def merged_seconds(segments):
    """Duration of one video's segments with overlaps merged."""
    spans = sorted((float(start), float(end)) for start, end in segments)
    total = 0.0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def prepare_frame(df, args):
    """Apply every eligibility rule before any probability is assigned."""
    frame = df.copy()
    for column in ("lat", "lon", POPULATION_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["locality", "country", "lat", "lon", POPULATION_COLUMN])
    frame = frame[
        frame["lat"].between(-90, 90)
        & frame["lon"].between(-180, 180)
        & ~(np.isclose(frame["lat"], 0) & np.isclose(frame["lon"], 0))
        & (frame[POPULATION_COLUMN] > 0)
    ].copy()
    frame["footage_hours_used"] = frame.apply(
        lambda row: sum(
            merged_seconds(segments)
            for _, segments in qualifying_segments(row, args.vehicle_type, args.time_of_day)
        ) / 3600,
        axis=1,
    )
    frame = frame[frame["footage_hours_used"] >= args.min_hours]
    frame = frame.reset_index().rename(columns={"index": "source_row"})
    if len(frame) < args.n:
        raise SystemExit(f"Only {len(frame)} eligible cities, fewer than n={args.n}.")
    return frame


def draw(frame, args):
    size_measure = np.power(frame[POPULATION_COLUMN].to_numpy(float), args.alpha)
    pi, certainty = pi_from_size(size_measure, n=args.n)
    xyz = unit_sphere_xyz(frame["lat"].to_numpy(float), frame["lon"].to_numpy(float))
    sampler = {"lpm1": lpm1, "lpm2": lpm2}[args.method]
    selected = np.asarray(sampler(pi, xyz, rng=np.random.default_rng(args.seed)))
    if len(selected) != args.n:
        raise RuntimeError(f"LPM selected {len(selected)} cities, expected {args.n}.")
    return selected, pi, certainty, xyz, size_measure


def clean_mapping(sample, args):
    """The drawn rows in mapping.csv format, holding only the qualifying footage."""
    clean = sample.drop(columns=DESIGN_COLUMNS + ["id"], errors="ignore").copy()
    for index in clean.index:
        records = qualifying_segments(clean.loc[index], args.vehicle_type, args.time_of_day)
        compact = {"separators": (",", ":")}
        clean.at[index, "videos"] = "[" + ",".join(video for video, _ in records) + "]"
        clean.at[index, "vehicle_type"] = json.dumps([args.vehicle_type] * len(records), **compact)
        clean.at[index, "time_of_day"] = json.dumps(
            [[args.time_of_day] * len(segments) for _, segments in records], **compact)
        clean.at[index, "start_time"] = json.dumps(
            [[start for start, _ in segments] for _, segments in records], **compact)
        clean.at[index, "end_time"] = json.dumps(
            [[end for _, end in segments] for _, segments in records], **compact)
    clean.insert(0, "id", np.arange(1, len(clean) + 1))
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE, help="number of cities")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=ALPHA, help="population damping exponent")
    parser.add_argument("--method", choices=["lpm1", "lpm2"], default=METHOD)
    parser.add_argument("--vehicle-type", type=int, default=REQUIRED_VEHICLE_TYPE)
    parser.add_argument("--time-of-day", type=int, default=REQUIRED_TIME_OF_DAY)
    parser.add_argument("--min-hours", type=float, default=MIN_FOOTAGE_HOURS)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    frame = prepare_frame(df, args)
    selected, pi, certainty, xyz, size_measure = draw(frame, args)

    frame["size_measure"] = size_measure
    frame["pi"] = pi
    frame["design_weight"] = np.where(pi > 0, 1 / pi, np.inf)
    frame[["sphere_x", "sphere_y", "sphere_z"]] = xyz
    sample = frame.iloc[np.sort(selected)].reset_index(drop=True)
    clean = clean_mapping(sample, args)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "eligible_frame_with_pi.csv", index=False)
    sample.to_csv(out / "design_manifest.csv", index=False)
    clean.to_csv(out / "mapping.csv", index=False)
    clean["country"].value_counts().rename_axis("country").reset_index(
        name="selected_city_count").to_csv(out / "selected_cities_by_country.csv", index=False)

    weights = 1 / pi[selected]
    population = frame[POPULATION_COLUMN].to_numpy(float)
    nearest_km = nearest_distance_km(xyz, xyz[selected])
    diagnostics = [
        "LPM CITY SAMPLING DIAGNOSTICS",
        "Implementation: M-Colley/lpm-sampling",
        f"Input rows: {len(df)}",
        f"Eligible cities: {len(frame)} in {frame['country'].nunique()} countries",
        f"Vehicle type: {args.vehicle_type}, time of day: {args.time_of_day}, "
        f"minimum footage: {args.min_hours} h",
        f"Sample size: {args.n}, method: {args.method.upper()}, seed: {args.seed}, alpha: {args.alpha}",
        f"sum(pi): {pi.sum():.12f}",
        f"Certainty cities: {len(certainty)}",
        f"Selected countries: {clean['country'].nunique()}",
        f"Spatial balance (lower is better): {spatial_balance(pi, xyz, selected):.6f}",
        "Population-weighted distance to nearest selected city (km): "
        f"{np.average(nearest_km, weights=population):.2f}",
        f"Kish effective sample size: {weights.sum() ** 2 / np.sum(weights ** 2):.2f}",
        f"Design weight range: {weights.min():.2f} - {weights.max():.2f}",
    ]
    (out / "diagnostics.txt").write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    print("\n".join(diagnostics))
    print(f"\nWrote {out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out}/")


if __name__ == "__main__":
    main()
