"""The analysis of the street audit (iRAP + MAPS-Mini items): what pedestrian infrastructure is
present and what is needed across regions, and whether the models audit every region alike.

Reads the "audit" run and its robustness runs (outputs/vlm/<tag>/{jaad,cities}/<model>/*.json),
the judges' ratings (outputs/vlm/judge/<judge>/<auditor>/*.json), JAAD's labels and the city
sample, and writes:
  outputs/vlm/analysis_audit/*.csv        every table
  outputs/vlm/analysis_audit/tables.md    the README tables, in Markdown
  docs/figures/audit_*.html and *.png

- Validity: the scene facts and the crossing items against JAAD's labels.
- Infrastructure present and needed, by continent: the consensus of the four models (at
  least 3 of 4) per clip, averaged per city; intervals from a bootstrap over cities, as each
  city gives two clips.
- Country indicators: Spearman's rho between a city's infrastructure and its country's road
  traffic mortality, median age, Gini and literacy (bootstrap intervals over cities).
- Model bias by region: agreement between the models (Fleiss' kappa), robustness and the
  judges' ratings in Europe and North America against the rest of the world.
- Robustness: Cohen's kappa between a model's main answers and its answers with the prompt
  reworded, one frame instead of five, half the image resolution, and between two sampled
  runs (temperature 0.7).
- Suggestions: the two judges' ratings, how far they agree, and whether a judge favours its
  own model.

Run:  uv run python scripts/analyse_audit.py [--no-png]
"""

import argparse
import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyse import BLUES, LABELS, MODEL_COLOURS, MODEL_NAMES, MODEL_ORDER, PAIR, markdown, pct, save
from vlm_street_audit.config import load_config, project_path
from vlm_street_audit.evaluation import SCENE_BOOLEANS, binary_scores, cohen_kappa, fleiss_kappa, wilson
from vlm_street_audit.vlm import CROSSING_ITEMS, INFORMAL_OR_NONE, NEEDED, ROAD_ITEMS, SIDE_ITEMS, expected_needs

MAIN = "audit"
# Robustness comparisons: (label, run, run it is compared with).
COMPARISONS = [("Reworded prompt", "audit-reworded", MAIN), ("One frame", "audit-1frame", MAIN),
               ("Half resolution", "audit-640px", MAIN), ("Resampled (T = 0.7)", "audit-t07-s1", "audit-t07-s2")]
ITEM_LABELS = {"area_type": "Area type", "land_use": "Land use", "number_of_lanes": "Lanes", "median": "Median",
               "street_lighting": "Street lighting", "traffic_calming": "Traffic calming",
               "pedestrian_fencing": "Pedestrian fencing", "transit_stops": "Transit stops",
               "bike_facility": "Bike facility", "footway": "Footway", "obstruction": "Footway obstruction",
               "trip_hazards": "Trip hazards", "crossing_demand": "Reason to cross",
               "crossing_facility": "Crossing facility", "crossing_quality": "Crossing quality",
               "kerb_ramps": "Kerb ramps"}
ITEMS = list(ROAD_ITEMS) + list(SIDE_ITEMS) + ["crossing_demand"] + list(CROSSING_ITEMS)
FORMAL = {"physical_barrier", "sidewalk_3m_or_more", "sidewalk_1_to_3m", "sidewalk_under_1m"}
# What is present, per clip and model (True/False, or NaN where it cannot be judged).
PRESENT = {
    "footway_both": "Formal footway on both sides",
    "footway_separated": "Footway set back ≥ 1 m or behind a barrier",
    "footway_clear": "Footways free of obstruction",
    "crossing": "Crossing facility",
    "crossing_protected": "Signalised or grade-separated crossing",
    "refuge": "Refuge island",
    "kerb_ramps": "Kerb ramps at the crossing",
    "lighting": "Street lighting",
    "traffic_calming": "Traffic calming",
    "fencing": "Pedestrian fencing",
    "bike": "Bike facility",
    "transit": "Transit stop",
}
NEED_LABELS = {"no_footway": "Footway (none or informal)", "blocked_footway": "Clear a blocked footway",
               "trip_hazards": "Repair trip hazards", "no_crossing": "Crossing where people need one",
               "poor_crossing": "Better crossing", "no_refuge_on_wide_road": "Refuge on a wide road",
               "no_kerb_ramps": "Kerb ramps", "no_street_lighting": "Street lighting"}
INDICATORS = {"traffic_mortality": "Road deaths per 100k", "med_age": "Median age", "gini": "Gini index",
              "literacy_rate": "Literacy rate"}
WEST = {"Europe", "North America"}
REGIONS = {True: "Europe & North America", False: "Rest of the world"}
CRITERIA = {"deficit_visible": "Deficit visible", "addresses_deficit": "Fixes the deficit",
            "specific": "Specific to the street", "feasible": "Feasible", "about_behaviour": "About behaviour"}
SEED = 20261028


def present(answer: dict) -> dict:
    sides = [answer[s] for s in ("left_side", "right_side") if answer[s]["footway"] != "not_visible"]
    footways = [s["footway"] for s in sides]
    facility = answer["crossing"]["crossing_facility"]
    road = answer["road"]
    in_view = facility != "none"
    return {
        "footway_both": len(footways) == 2 and all(f in FORMAL for f in footways) if footways else np.nan,
        "footway_separated": any(f in {"physical_barrier", "sidewalk_3m_or_more", "sidewalk_1_to_3m"}
                                 for f in footways) if footways else np.nan,
        "footway_clear": (all(s["obstruction"] == "none" for s in sides if s["footway"] in FORMAL)
                          if any(f in FORMAL for f in footways) else np.nan),
        "crossing": in_view,
        "crossing_protected": facility in {"grade_separated", "signalised_with_refuge", "signalised"},
        "refuge": facility in {"signalised_with_refuge", "marked_with_refuge", "refuge_only"},
        "kerb_ramps": (answer["crossing"]["kerb_ramps"] in {"both_kerbs", "one_kerb"}
                       if in_view and answer["crossing"]["kerb_ramps"] in {"both_kerbs", "one_kerb", "none"}
                       else np.nan),
        "lighting": road["street_lighting"] != "none",
        "traffic_calming": road["traffic_calming"] == "present",
        "fencing": road["pedestrian_fencing"] == "present",
        "bike": road["bike_facility"] != "none",
        "transit": road["transit_stops"] != "0",
    }


def load_run(root, tag: str, set_name: str) -> pd.DataFrame:
    """One row per valid answer: scene facts, items, what is present, what is needed, consistency."""
    rows = []
    for path in sorted((root / tag / set_name).glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        answer = record.get("answer")
        if not answer:
            continue
        # Crossing and road needs have one possible place, so only side needs are matched on "where".
        reported = {(n["need"], n["where"] if n["where"] in ("left_side", "right_side") and
                     n["need"] in {"no_footway", "blocked_footway", "trip_hazards"} else
                     "road" if n["need"] == "no_street_lighting" else "crossing") for n in answer["needed"]}
        expected = expected_needs(answer)
        rows.append({"model": record["model"], "clip_id": record["clip_id"], "role": record.get("role", "main"),
                     **{f"scene_{k}": v for k, v in answer["scene"].items()},
                     **{item: answer["road"][item] for item in ROAD_ITEMS},
                     **{f"{side}:{item}": answer[side][item] for side in ("left_side", "right_side")
                        for item in SIDE_ITEMS},
                     **{item: answer["crossing"][item] for item in ["crossing_demand", *CROSSING_ITEMS]},
                     **{f"has_{k}": v for k, v in present(answer).items()},
                     **{f"need_{k}": any(e[0] == k for e in expected) for k in NEEDED},
                     "needs": len({e[0] for e in expected}), "listed": len(reported),
                     "expected": len(expected), "matched": len(reported & expected),
                     "consistent": reported == expected})
    return pd.DataFrame(rows)


def item_frame(run: pd.DataFrame, item: str) -> pd.DataFrame:
    """(model, key, value) for one item; side items give one row per side, keyed clip|side."""
    if item in SIDE_ITEMS:
        parts = [run[["model", "clip_id", f"{side}:{item}"]].rename(columns={f"{side}:{item}": "value"})
                 .assign(key=lambda d, s=side: d.clip_id + "|" + s) for side in ("left_side", "right_side")]
        return pd.concat(parts)[["model", "key", "value"]]
    return run[["model", "clip_id", item]].rename(columns={item: "value", "clip_id": "key"})


def complete_clips(run: pd.DataFrame) -> pd.DataFrame:
    counts = run.groupby("clip_id").model.nunique()
    return run[run.clip_id.isin(counts[counts == len(MODEL_ORDER)].index)]


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman's rho: Pearson's r of the ranks (no SciPy needed)."""
    return a.rank().corr(b.rank())


def bootstrap(values: pd.DataFrame, statistic, reps: int = 2000) -> tuple[float, float]:
    """95% percentile interval of statistic(rows) over resamples of the rows (one row per city)."""
    rng = np.random.default_rng(SEED)
    stats = [statistic(values.iloc[rng.integers(0, len(values), len(values))]) for _ in range(reps)]
    stats = [s for s in stats if s == s]
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))) if stats else (np.nan, np.nan)


def kappa_by_item(run: pd.DataFrame) -> dict[str, float]:
    out = {}
    for item in ITEMS:
        wide = item_frame(run, item).pivot(index="key", columns="model", values="value").dropna()
        out[item] = fleiss_kappa(wide) if len(wide) else np.nan
    return out


def fmt(value: float, low: float | None = None, high: float | None = None, digits: int = 2) -> str:
    if value != value:
        return "–"
    text = f"{value:.{digits}f}"
    return text if low is None or low != low else f"{text} ({low:.{digits}f} to {high:.{digits}f})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-png", action="store_true", help="HTML figures only (no Kaleido/Chrome)")
    args = parser.parse_args()
    png = not args.no_png
    settings = load_config()
    root = project_path(settings["vlm_output_dir"])
    out = root / "analysis_audit"
    out.mkdir(parents=True, exist_ok=True)
    truth = json.loads((project_path(settings["clip_sets_dir"]) / "jaad_labels.json").read_text())
    clips = pd.read_csv(project_path(settings["clip_sets_dir"]) / "cities.csv")
    sample = pd.read_csv(project_path(settings["mapping"]))
    sample = sample[["locality", "country", "continent", "lat", "lon", *INDICATORS]].drop_duplicates(
        ["locality", "country"])
    for column in ("literacy_rate",):  # 0 means missing in the mapping
        sample[column] = sample[column].where(sample[column] > 0)
    tags = [MAIN] + sorted({t for _, a, b in COMPARISONS for t in (a, b)} - {MAIN})
    runs = {(tag, s): load_run(root, tag, s) for tag in tags for s in ("jaad", "cities")}
    md: list[str] = []
    main_cities = complete_clips(runs[(MAIN, "cities")]).merge(
        clips[["clip_id", "locality", "country"]], on="clip_id").merge(sample, on=["locality", "country"], how="left")
    main_cities["west"] = main_cities.continent.isin(WEST)
    main_jaad = runs[(MAIN, "jaad")]
    md.append(f"{main_jaad.clip_id.nunique()} JAAD clips and {main_cities.clip_id.nunique()} city clips "
              f"({main_cities.locality.nunique()} cities) answered by all four models.\n")

    # ---- Validity on JAAD ------------------------------------------------------------------
    fields = SCENE_BOOLEANS + ["crossing_marked"]
    rows = []
    for tag in tags:
        if runs[(tag, "jaad")].empty:  # the T = 0.7 runs cover the city clips only
            continue
        for model, group in runs[(tag, "jaad")].groupby("model"):
            labels = pd.DataFrame([truth[c] for c in group.clip_id])
            for field in fields:
                if field == "crossing_marked":  # a marked, signalised or raised crossing facility
                    predicted = ~group.crossing_facility.isin(["none", "refuge_only", "grade_separated"])
                    target = labels["zebra_crossing_visible"]
                else:
                    predicted, target = group[f"scene_{field}"], labels[field]
                rows.append({"run": tag, "model": model, "field": field,
                             **binary_scores(target, predicted.reset_index(drop=True))})
    validity = pd.DataFrame(rows)
    validity.to_csv(out / "jaad_validity.csv", index=False)
    field_labels = {**LABELS, "crossing_marked": "Crossing facility (iRAP item)"}
    table = validity[validity.run == MAIN].pivot(index="field", columns="model", values="balanced_accuracy")
    table = table.reindex(index=fields, columns=MODEL_ORDER)
    md.append("### JAAD: balanced accuracy against the labels\n")
    md.append(markdown(pd.DataFrame({"Item": [field_labels[f] for f in table.index],
                                     **{MODEL_NAMES[m]: [fmt(v) for v in table[m]] for m in table.columns}})))
    md.append("\nThe crossing-facility item counts a marked, signalised or raised crossing and is scored against "
              "JAAD's crossing label.\n")
    plot = validity[validity.run == MAIN].assign(field=lambda d: d.field.map(field_labels),
                                                 model=lambda d: d.model.map(MODEL_NAMES))
    figure = px.bar(plot, x="field", y="balanced_accuracy", color="model", barmode="group",
                    category_orders={"model": [MODEL_NAMES[m] for m in MODEL_ORDER],
                                     "field": [field_labels[f] for f in fields]},
                    color_discrete_map=MODEL_COLOURS, height=430,
                    labels={"field": "", "balanced_accuracy": "Balanced accuracy<br>(0.5 = chance)", "model": ""})
    figure.add_hline(y=0.5, line_dash="dot", line_color="#52514e")
    figure.update_yaxes(range=[0, 1.02], dtick=0.25, tickformat=".2f")
    figure.update_xaxes(ticktext=[field_labels[f].replace(" (", "<br>(") for f in fields],
                        tickvals=[field_labels[f] for f in fields])
    figure.update_layout(bargap=0.25, bargroupgap=0.08, legend=dict(orientation="h", y=1.1, x=0))
    figure.update_traces(hovertemplate="%{x}<br>%{y:.2f}")
    save(figure, "audit_jaad_validity", png)

    # ---- Present and needed, by continent (consensus of the four models) --------------------------
    keys = [f"has_{k}" for k in PRESENT] + [f"need_{k}" for k in NEEDED]
    consensus = (main_cities.groupby(["clip_id", "role", "locality", "country", "continent"])[keys]
                 .agg(lambda s: np.nan if s.notna().sum() < 3 else float(s.dropna().astype(float).sum()
                                                                           >= 3 * s.notna().sum() / 4))
                 .reset_index())
    consensus.to_csv(out / "cities_consensus.csv", index=False)
    city = consensus.groupby(["locality", "country", "continent"])[keys].mean().reset_index().merge(
        sample, on=["locality", "country", "continent"], how="left")
    city["needs"] = city[[f"need_{k}" for k in NEEDED]].sum(axis=1)
    city.to_csv(out / "cities_infrastructure.csv", index=False)
    continents = city.groupby("continent").locality.nunique().sort_values(ascending=False)
    rows = []
    for continent in list(continents.index) + ["All"]:
        group = city if continent == "All" else city[city.continent == continent]
        for key in keys:
            values = group[[key]].dropna()
            low, high = bootstrap(values, lambda d, k=key: d[k].mean()) if len(values) > 1 else (np.nan, np.nan)
            rows.append({"continent": continent, "cities": len(group), "key": key, "share": values[key].mean(),
                         "low": low, "high": high, "n": len(values)})
    by_continent = pd.DataFrame(rows)
    by_continent.to_csv(out / "by_continent.csv", index=False)
    order = list(continents.index) + ["All"]
    names = {c: f"{c} ({continents.get(c, len(city))})" for c in order}

    def continent_table(prefix: str, labels: dict) -> pd.DataFrame:
        t = by_continent[by_continent.key.str.startswith(prefix)].copy()
        t["cell"] = [f"{pct(r.share)} ({pct(r.low)}–{pct(r.high)})" if r.low == r.low else pct(r.share)
                     for r in t.itertuples()]
        t = t.pivot(index="key", columns="continent", values="cell").reindex(
            index=[prefix + k for k in labels], columns=order)
        return pd.DataFrame({"": [labels[k.removeprefix(prefix)] for k in t.index],
                            **{names[c]: t[c] for c in order}})

    md.append("\n### Infrastructure present, by continent (share of cities; at least 3 of 4 models agree)\n")
    md.append(markdown(continent_table("has_", PRESENT)))
    md.append("\n### Infrastructure needed, by continent (share of cities)\n")
    md.append(markdown(continent_table("need_", NEED_LABELS)))
    md.append("\nCities are the unit: each city's two clips are averaged, and intervals come from a bootstrap over "
              "cities. Oceania is one city.\n")
    for prefix, labels, name in (("has_", PRESENT, "audit_present_by_continent"),
                                 ("need_", NEED_LABELS, "audit_needed_by_continent")):
        grid = by_continent[by_continent.key.str.startswith(prefix)].pivot(index="key", columns="continent",
                                                                           values="share")
        grid = grid.reindex(index=[prefix + k for k in labels], columns=order)
        grid = grid[grid.fillna(0).gt(0).any(axis=1)]  # rows at 0% everywhere stay in the table only
        figure = go.Figure(go.Heatmap(
            z=grid.values, x=[names[c].replace(" (", "<br>(") for c in order],
            y=[labels[k.removeprefix(prefix)] for k in grid.index], zmin=0, zmax=1, colorscale=BLUES,
            text=[[pct(v) for v in row] for row in grid.values], texttemplate="%{text}", xgap=2, ygap=2,
            colorbar=dict(title=dict(text="Share of cities", side="right"), tickformat=".0%"),
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
        figure.update_yaxes(autorange="reversed")
        figure.add_vline(x=len(order) - 1.5, line_color="#52514e", line_width=1)
        figure.update_layout(height=120 + 38 * len(grid))
        save(figure, name, png)

    # ---- Country indicators -----------------------------------------------------------------------
    rows = []
    for key in keys + ["needs"]:
        for indicator in INDICATORS:
            pair = city[[key, indicator]].dropna()
            # A rho on an item that (almost) every city has or lacks rests on a few ties and its bootstrap
            # interval can miss the estimate (kerb ramps: present in 2 cities), so at least 5 cities each way.
            if len(pair) < 10 or (pair[key] > pair[key].min()).sum() < 5 or (pair[key] < pair[key].max()).sum() < 5:
                continue
            rho = spearman(pair[key], pair[indicator])
            low, high = bootstrap(pair, lambda d, k=key, i=indicator: spearman(d[k], d[i]))
            rows.append({"key": key, "indicator": indicator, "cities": len(pair), "rho": rho, "low": low,
                         "high": high})
    indicators = pd.DataFrame(rows)
    indicators.to_csv(out / "country_indicators.csv", index=False)
    all_labels = {**{f"has_{k}": v for k, v in PRESENT.items()},
                  **{f"need_{k}": f"Needed: {v}" for k, v in NEED_LABELS.items()}, "needs": "Number of needs"}
    md.append("\n### Infrastructure and country indicators (Spearman's rho over cities, 95% bootstrap interval)\n")
    t = indicators.assign(cell=[fmt(r.rho, r.low, r.high) for r in indicators.itertuples()]).pivot(
        index="key", columns="indicator", values="cell").reindex(index=[k for k in all_labels if k in
                                                                         set(indicators.key)]).fillna("–")
    md.append(markdown(pd.DataFrame({"": [all_labels[k] for k in t.index],
                                     **{f"{INDICATORS[i]} (n = {indicators[indicators.indicator == i].cities.max()})":
                                        t[i] for i in INDICATORS if i in t}})))
    md.append("\nCountry figures from the city sample (CROWD mapping); a positive rho means the item is more "
              "common where the indicator is higher.\n")
    grid = indicators.pivot(index="key", columns="indicator", values="rho").reindex(
        index=[k for k in all_labels if k in set(indicators.key)], columns=list(INDICATORS)).dropna(how="all")
    significant = indicators.assign(sig=(indicators.low > 0) | (indicators.high < 0)).pivot(
        index="key", columns="indicator", values="sig").reindex(index=grid.index, columns=grid.columns)
    # Diverging: orange for negative, blue for positive, grey at zero.
    diverging = [[0, "#b84a12"], [0.25, "#eb9a6e"], [0.5, "#f1f0ee"], [0.75, "#7fb0ea"], [1, "#184f95"]]
    figure = go.Figure(go.Heatmap(
        z=grid.values, x=[INDICATORS[i] for i in grid.columns], y=[all_labels[k] for k in grid.index],
        zmin=-0.6, zmax=0.6, colorscale=diverging, xgap=2, ygap=2,
        text=[[("" if v != v else f"{v:+.2f}" + ("*" if s is True else "")) for v, s in zip(row, srow)]
              for row, srow in zip(grid.values, significant.values)], texttemplate="%{text}",
        colorbar=dict(title=dict(text="Spearman's rho", side="right"), dtick=0.2),
        hovertemplate="%{y}<br>%{x}: rho %{text}<extra></extra>"))
    figure.update_yaxes(autorange="reversed")
    figure.add_hline(y=sum(k.startswith("has_") for k in grid.index) - 0.5, line_color="#52514e", line_width=1)
    figure.update_layout(height=140 + 30 * len(grid))
    save(figure, "audit_country_indicators", png)

    # ---- Model bias by region -----------------------------------------------------------------------
    rows = []
    for west, group in main_cities.groupby("west"):
        clips_here = np.array(sorted(group.clip_id.unique()))
        rng = np.random.default_rng(SEED)
        draws = [rng.choice(clips_here, len(clips_here)) for _ in range(500)]  # bootstrap over clips
        for item in ITEMS:
            wide = item_frame(group, item).pivot(index="key", columns="model", values="value").dropna()
            by_clip = pd.Series(range(len(wide))).groupby(wide.index.str.split("|").str[0].to_numpy()).apply(list)
            values = []
            for pick in draws:
                index = [i for c in pick if c in by_clip.index for i in by_clip[c]]
                kappa = fleiss_kappa(wide.iloc[index]) if index else np.nan
                if kappa == kappa:
                    values.append(kappa)
            rows.append({"region": REGIONS[west], "item": item, "clips": len(clips_here),
                         "fleiss_kappa": fleiss_kappa(wide) if len(wide) else np.nan,
                         "low": np.percentile(values, 2.5) if values else np.nan,
                         "high": np.percentile(values, 97.5) if values else np.nan})
    bias = pd.DataFrame(rows)
    bias.to_csv(out / "agreement_by_region.csv", index=False)
    overall = kappa_by_item(main_cities)
    md.append("\n### Agreement between the four models, by region (Fleiss' kappa, city clips, 95% interval)\n")
    t = bias.assign(cell=[fmt(r.fleiss_kappa, r.low, r.high) for r in bias.itertuples()]).pivot(
        index="item", columns="region", values="cell").reindex(ITEMS)
    clips_per = bias.drop_duplicates("region").set_index("region").clips
    md.append(markdown(pd.DataFrame({"Item": [ITEM_LABELS[i] + (" (per side)" if i in SIDE_ITEMS else "")
                                              for i in t.index],
                                     "All clips": [fmt(overall[i]) for i in t.index],
                                     **{f"{r} ({clips_per[r]} clips)": t[r] for r in REGIONS.values()}})))
    figure = go.Figure()
    for (west, region), colour in zip(REGIONS.items(), PAIR):
        b = bias[bias.region == region].set_index("item").reindex(ITEMS)
        figure.add_bar(name=f"{region} ({clips_per[region]} clips)", y=[ITEM_LABELS[i] for i in ITEMS],
                       x=b.fleiss_kappa, orientation="h", marker_color=colour,
                       error_x=dict(type="data", symmetric=False, color="#52514e", thickness=1.2, width=3,
                                    array=b.high - b.fleiss_kappa, arrayminus=b.fleiss_kappa - b.low),
                       hovertemplate="%{y}<br>kappa %{x:.2f}<extra>" + region + "</extra>")
    for boundary in (len(ROAD_ITEMS), len(ROAD_ITEMS) + len(SIDE_ITEMS)):
        figure.add_hline(y=boundary - 0.5, line_color="#52514e", line_width=1)
    figure.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, height=700,
                         yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.05, x=0),
                         xaxis=dict(title="Fleiss' kappa between the four models", range=[-0.2, 1], dtick=0.2,
                                    zeroline=True, zerolinecolor="#52514e"))
    save(figure, "audit_agreement_by_region", png)

    # Consistency of the needs with the items, by model and region.
    rows = []
    for (model, west), group in main_cities.groupby(["model", "west"]):
        expected, matched, listed = int(group.expected.sum()), int(group.matched.sum()), int(group.listed.sum())
        rows.append({"model": model, "region": REGIONS[west], "answers": len(group),
                     "needs_per_clip": group.needs.mean(), "consistent": group.consistent.mean(),
                     "expected_listed": matched / expected if expected else np.nan,
                     "listed_backed": matched / listed if listed else np.nan})
    consistency = pd.DataFrame(rows)
    consistency.to_csv(out / "needs_consistency.csv", index=False)
    md.append("\n### Listed needs against the item answers, by region\n")
    md.append(markdown(pd.DataFrame([{
        "Model": MODEL_NAMES[r.model], "Region": r.region, "Needs per clip": f"{r.needs_per_clip:.1f}",
        "Item needs listed": pct(r.expected_listed), "Listed needs backed by an item": pct(r.listed_backed),
        "Answers fully consistent": pct(r.consistent)} for r in consistency.sort_values(
            ["model", "region"], key=lambda s: s.map({m: i for i, m in enumerate(MODEL_ORDER)})
            if s.name == "model" else s).itertuples()])))

    # ---- Robustness -------------------------------------------------------------------------------
    rows = []
    for label, tag, base in COMPARISONS:
        for set_name in ("jaad", "cities"):
            a, b = runs[(tag, set_name)], runs[(base, set_name)]
            if a.empty or b.empty:
                continue
            for model in MODEL_ORDER:
                ma, mb = a[a.model == model], b[b.model == model]
                if ma.empty or mb.empty:
                    continue
                for item in ITEMS:
                    pair = item_frame(ma, item).merge(item_frame(mb, item), on=["model", "key"], suffixes=("_a", "_b"))
                    pair["clip_id"] = pair.key.str.split("|").str[0]
                    for west in (None, True, False):
                        part = pair
                        if west is not None:
                            west_clips = set(main_cities[main_cities.west == west].clip_id)
                            part = pair[pair.clip_id.isin(west_clips)]
                        if part.empty:
                            continue
                        rows.append({"comparison": label, "set": set_name, "model": model, "item": item,
                                     "region": "All" if west is None else REGIONS[west], "units": len(part),
                                     "same": float((part.value_a.astype(str) == part.value_b.astype(str)).mean()),
                                     "cohen_kappa": cohen_kappa(part.value_a, part.value_b)})
    robust = pd.DataFrame(rows)
    robust.to_csv(out / "robustness.csv", index=False)
    labels_here = [label for label, _, _ in COMPARISONS if not robust.empty and label in set(robust.comparison)]
    if labels_here:
        city_all = robust[(robust.set == "cities") & (robust.region == "All")]
        heat = city_all.groupby(["item", "comparison"]).cohen_kappa.mean().unstack("comparison")
        heat = heat.reindex(index=ITEMS, columns=labels_here)
        md.append("\n### Robustness: Cohen's kappa with the main answers (city clips, mean of the four models)\n")
        md.append(markdown(pd.DataFrame({"Item": [ITEM_LABELS[i] for i in heat.index],
                                         **{c: [fmt(v) for v in heat[c]] for c in heat.columns}})))
        by_region = (robust[(robust.set == "cities") & (robust.region != "All")]
                     .groupby(["comparison", "region"]).cohen_kappa.mean().unstack("region")
                     .reindex(index=labels_here, columns=list(REGIONS.values())))
        md.append("\n### Robustness by region (mean Cohen's kappa over items and models, city clips)\n")
        md.append(markdown(pd.DataFrame({"Change": by_region.index,
                                         **{r: [fmt(v) for v in by_region[r]] for r in by_region.columns}})))
        md.append("\n\"Resampled\" compares two runs at temperature 0.7 with each other, not with the main run.\n")
        figure = go.Figure(go.Heatmap(
            z=heat.clip(lower=0).values, x=[c.replace(" (", "<br>(") for c in heat.columns],
            y=[ITEM_LABELS[i] for i in heat.index], zmin=0, zmax=1, colorscale=BLUES,
            text=[[fmt(v) for v in row] for row in heat.values], texttemplate="%{text}", xgap=2, ygap=2,
            colorbar=dict(title=dict(text="Cohen's kappa", side="right"), dtick=0.25),
            hovertemplate="%{y}<br>%{x}: kappa %{text}<extra></extra>"))
        figure.update_yaxes(autorange="reversed")
        figure.update_layout(height=620)
        save(figure, "audit_robustness", png)

    # ---- Suggestions, rated by two judges -----------------------------------------------------------
    rows = []
    for path in sorted((root / "judge").glob("*/*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for i, (need, rating) in enumerate(zip(record["deficits"], record.get("ratings") or [])):
            rows.append({"judge": record["judge"], "auditor": record["auditor"], "clip_id": record["clip_id"],
                         "index": i, "need": need["need"],
                         "deficit_visible": rating["deficit_visible"] == "yes",
                         "addresses_deficit": rating["addresses_deficit"],
                         "specific": rating["specific"] == "specific", "feasible": rating["feasible"] == "yes",
                         "about_behaviour": rating["about_behaviour"]})
    ratings = pd.DataFrame(rows)
    if not ratings.empty:
        ratings = ratings[ratings.clip_id.isin(main_cities.clip_id)].merge(
            main_cities[["clip_id", "west"]].drop_duplicates(), on="clip_id")
        ratings.to_csv(out / "judge_ratings.csv", index=False)
        judges = [m for m in MODEL_ORDER if m in set(ratings.judge)]
        rows = []
        for (judge, auditor), group in ratings.groupby(["judge", "auditor"]):
            for criterion in CRITERIA:
                low, high = wilson(int(group[criterion].sum()), len(group))
                rows.append({"judge": judge, "auditor": auditor, "criterion": criterion, "n": len(group),
                             "share": group[criterion].mean(), "low": low, "high": high})
        quality = pd.DataFrame(rows)
        quality.to_csv(out / "suggestion_quality.csv", index=False)
        md.append("\n### Suggestions: share rated yes by each judge (city clips)\n")
        for judge in judges:
            q = quality[quality.judge == judge].pivot(index="criterion", columns="auditor", values="share")
            q = q.reindex(index=list(CRITERIA), columns=[m for m in MODEL_ORDER if m in q.columns])
            n = quality[quality.judge == judge].drop_duplicates("auditor").set_index("auditor").n
            md.append(f"\n#### Judge: {MODEL_NAMES[judge]}\n")
            md.append(markdown(pd.DataFrame({"Criterion": [CRITERIA[c] for c in q.index],
                                             **{f"{MODEL_NAMES[m]} (n = {n[m]})": [pct(v) for v in q[m]]
                                                for m in q.columns}})))
        region_quality = ratings.groupby("west")[list(CRITERIA)].mean().reindex(list(REGIONS))
        md.append("\n#### Judged quality by region (both judges, all auditors)\n")
        md.append(markdown(pd.DataFrame({"Criterion": [CRITERIA[c] for c in CRITERIA],
                                         **{f"{REGIONS[w]} (n = {int((ratings.west == w).sum())})":
                                            [pct(region_quality.loc[w, c]) for c in CRITERIA]
                                            for w in region_quality.index}})))
        if len(judges) == 2:
            both = ratings[ratings.judge == judges[0]].merge(ratings[ratings.judge == judges[1]],
                                                             on=["auditor", "clip_id", "index"], suffixes=("_a", "_b"))
            md.append(f"\n#### Agreement between the judges ({len(both)} suggestions rated by both)\n")
            md.append(markdown(pd.DataFrame([{
                "Criterion": CRITERIA[c], "Same rating": pct((both[f"{c}_a"] == both[f"{c}_b"]).mean()),
                "Cohen's kappa": fmt(cohen_kappa(both[f"{c}_a"], both[f"{c}_b"]))} for c in CRITERIA])))
            # Self-preference: how much more a judge likes its own model's suggestions than the other judge does,
            # against the same gap for the models that are neither judge.
            gap = quality.set_index(["judge", "auditor", "criterion"]).share.unstack("judge")
            rows = []
            for judge, other in (judges, judges[::-1]):
                for criterion in CRITERIA:
                    difference = gap.xs(criterion, level="criterion")[judge] - gap.xs(criterion,
                                                                                      level="criterion")[other]
                    rows.append({"judge": judge, "criterion": criterion, "own_model": difference.get(judge),
                                 "neutral_models": difference.drop(index=judges, errors="ignore").mean()})
            self_pref = pd.DataFrame(rows)
            self_pref.to_csv(out / "judge_self_preference.csv", index=False)
            md.append("\n#### Self-preference: a judge's yes-share minus the other judge's, in points\n")
            md.append(markdown(pd.DataFrame([{
                "Judge": MODEL_NAMES[r.judge], "Criterion": CRITERIA[r.criterion],
                "On its own model": f"{100 * r.own_model:+.0f}",
                "On the two neutral models": f"{100 * r.neutral_models:+.0f}"} for r in self_pref.itertuples()])))
        figure = make_subplots(rows=len(judges), cols=1, shared_xaxes=True, vertical_spacing=0.14,
                               subplot_titles=[f"Judge: {MODEL_NAMES[j]}" for j in judges])
        for row, judge in enumerate(judges, start=1):
            for auditor in MODEL_ORDER:
                q = quality[(quality.judge == judge) & (quality.auditor == auditor)].set_index("criterion")
                if q.empty:
                    continue
                # "About behaviour" was never rated yes, so it stays in the tables only.
                q = q.reindex([c for c in CRITERIA if quality[quality.criterion == c].share.max() > 0])
                figure.add_bar(name=MODEL_NAMES[auditor], legendgroup=auditor, showlegend=row == 1,
                               x=[CRITERIA[c].replace(" ", "<br>", 1) for c in q.index], y=q.share,
                               marker_color=MODEL_COLOURS[MODEL_NAMES[auditor]],
                               error_y=dict(type="data", symmetric=False, color="#52514e", thickness=1.2, width=3,
                                            array=q.high - q.share, arrayminus=q.share - q.low),
                               hovertemplate="%{x}<br>%{y:.0%}<extra>" + MODEL_NAMES[auditor] + "</extra>",
                               row=row, col=1)
        figure.update_yaxes(range=[0, 1.05], tickformat=".0%", title_text="Share rated yes")
        figure.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06, height=330 * len(judges) + 70,
                             legend=dict(orientation="h", y=1.1, x=0))
        save(figure, "audit_suggestion_quality", png)

    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
