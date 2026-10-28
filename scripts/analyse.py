"""The full analysis behind the README's Results: tables and Plotly figures for both prompts.

Reads outputs/vlm/<prompt>/{jaad,cities}/<model>/*.json, JAAD's labels and the city
sample, and writes:
  outputs/vlm/analysis/*.csv           every table
  outputs/vlm/analysis/tables.md       the README tables, in Markdown
  docs/figures/*.html and *.png (+ outputs/figures/*.html)

Proportions carry Wilson 95% intervals; "consensus" means at least 3 of the 4 models.

Run:  uv run python scripts/analyse.py [--no-png]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vlm_street_audit.config import PROJECT_ROOT, load_config, project_path
from vlm_street_audit.evaluation import (
    SCENE_BOOLEANS,
    agreement,
    clip_features,
    jaad_scores,
    load_answers,
    point_hits,
    wilson,
)
from vlm_street_audit.figures import save_figure, style
from vlm_street_audit.vlm import CATEGORY_DEFINITIONS

PROMPTS = ["open", "checklist"]
CATEGORIES = list(CATEGORY_DEFINITIONS)
MODEL_ORDER = ["qwen3.8-27b", "gemma4-31b", "molmo2-8b", "internvl3.5-8b"]
MODEL_NAMES = {"qwen3.8-27b": "Qwen3.8-27B", "gemma4-31b": "Gemma 4 31B", "molmo2-8b": "Molmo2-8B",
               "internvl3.5-8b": "InternVL3.5-8B"}
LABELS = {"zebra_crossing_visible": "Zebra crossing", "traffic_signal_visible": "Traffic signal",
          "pedestrians_visible": "Pedestrians visible", "pedestrians_in_carriageway": "Pedestrians in road",
          "parking_area": "Car park", "pavement_condition": "Pavement condition",
          "pavement_obstruction": "Pavement obstruction", "crossing_facilities": "Crossing facilities",
          "pedestrian_exposure": "Pedestrian exposure", "accessibility": "Accessibility",
          "vulnerable_users": "Vulnerable users"}
PROMPT_NAMES = {"open": "Open prompt", "checklist": "Checklist prompt"}
# Categorical colours in a fixed order (checked for colour-blind separation): one per model,
# the first two for two-way comparisons. Magnitudes use one blue ramp, light to dark.
MODEL_COLOURS = {"Qwen3.8-27B": "#2a78d6", "Gemma 4 31B": "#eb6834", "Molmo2-8B": "#1baf7a",
                 "InternVL3.5-8B": "#eda100"}
PAIR = ["#2a78d6", "#eb6834"]
BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FIGURES = PROJECT_ROOT / "docs" / "figures"


def pct(value: float) -> str:
    return "–" if value != value else f"{100 * value:.0f}%"


def markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def save(fig: go.Figure, name: str, png: bool) -> None:
    # No figure titles: the README (or a paper) gives each figure its caption.
    style(fig)
    fig.update_layout(title=None, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    save_figure(fig, name)
    # The README's PNGs link to these interactive copies.
    save_figure(fig, name, directory=FIGURES)
    if png:
        fig.write_image(FIGURES / f"{name}.png", width=1100, height=fig.layout.height or 500, scale=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-png", action="store_true", help="HTML figures only (no Kaleido/Chrome)")
    args = parser.parse_args()
    png = not args.no_png
    settings = load_config()
    root = project_path(settings["vlm_output_dir"])
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    truth = json.loads((project_path(settings["clip_sets_dir"]) / "jaad_labels.json").read_text())
    clips = pd.read_csv(project_path(settings["clip_sets_dir"]) / "cities.csv")
    sample = pd.read_csv(project_path(settings["mapping"]))[["locality", "country", "continent", "lat", "lon"]]
    md: list[str] = []

    # ---- JAAD: scene facts --------------------------------------------------------------
    jaad_rows = []
    for prompt in PROMPTS:
        answers = load_answers(root / prompt, "jaad")
        if answers.empty:
            continue
        scores = jaad_scores(answers, truth).assign(prompt=prompt)
        jaad_rows.append(scores)
    jaad = pd.concat(jaad_rows, ignore_index=True)
    jaad.to_csv(out / "jaad_scene_scores.csv", index=False)
    table = jaad[jaad.prompt == "checklist"].pivot(index="field", columns="model", values="balanced_accuracy")
    table = table.reindex(index=SCENE_BOOLEANS + ["parking_area"], columns=MODEL_ORDER)
    positives = jaad.drop_duplicates("field").set_index("field").positives
    md.append("### JAAD: balanced accuracy of the scene facts (checklist prompt)\n")
    md.append(markdown(pd.DataFrame({
        "Scene fact (clips with it / 120)": [f"{LABELS[f]} ({positives[f]})" for f in table.index],
        **{MODEL_NAMES[m]: [f"{v:.2f}" for v in table[m]] for m in table.columns}})))
    figure = px.bar(jaad.assign(field=jaad.field.map(LABELS), model=jaad.model.map(MODEL_NAMES),
                                prompt=jaad.prompt.map(PROMPT_NAMES)),
                    x="field", y="balanced_accuracy", color="model", barmode="group", facet_row="prompt",
                    category_orders={"model": [MODEL_NAMES[m] for m in MODEL_ORDER],
                                     "prompt": [PROMPT_NAMES[p] for p in PROMPTS]},
                    color_discrete_map=MODEL_COLOURS,
                    labels={"field": "", "balanced_accuracy": "Balanced accuracy<br>(0.5 = chance)", "model": ""},
                    height=650)
    figure.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    figure.add_hline(y=0.5, line_dash="dot", line_color="#52514e")
    figure.update_yaxes(range=[0, 1.02], dtick=0.25, tickformat=".2f")
    figure.update_layout(bargap=0.25, bargroupgap=0.08, legend=dict(orientation="h", y=1.06, x=0))
    figure.update_traces(hovertemplate="%{x}<br>%{y:.2f}")
    save(figure, "jaad_scene_facts", png)

    # ---- JAAD: points on pedestrians -------------------------------------------------
    hits = pd.concat([point_hits(load_answers(root / p, "jaad"), truth).assign(prompt=p) for p in PROMPTS
                      if (root / p / "jaad").is_dir()], ignore_index=True)
    hits.to_csv(out / "jaad_point_hits.csv", index=False)
    md.append("\n### JAAD: points given for pedestrian problems that land on a labelled pedestrian\n")
    rows = []
    for model in MODEL_ORDER:
        row = {"Model": MODEL_NAMES[model]}
        for prompt in PROMPTS:
            h = hits[(hits.model == model) & (hits.prompt == prompt)]
            if len(h):
                h = h.iloc[0]
                low, high = wilson(int(h.on_a_pedestrian), int(h.pedestrian_points))
                row[f"{prompt} prompt"] = (f"{pct(h.hit_rate)} of {int(h.pedestrian_points)} "
                                           f"({pct(low)}–{pct(high)})" if h.pedestrian_points else "no points")
        rows.append(row)
    md.append(markdown(pd.DataFrame(rows)))

    # ---- Cities: features per prompt -----------------------------------------------------
    city_features = []
    for prompt in PROMPTS:
        answers = load_answers(root / prompt, "cities")
        if answers.empty:
            continue
        features = clip_features(answers, CATEGORIES).assign(prompt=prompt)
        city_features.append(features)
    cities = pd.concat(city_features, ignore_index=True)
    cities = cities[cities.clip_id.isin(clips.clip_id)]
    cities = cities.merge(clips[["clip_id", "locality", "country"]], on="clip_id").merge(
        sample, on=["locality", "country"], how="left")
    complete = cities.groupby(["prompt", "clip_id"]).model.nunique()
    complete = complete[complete == len(MODEL_ORDER)].reset_index()[["prompt", "clip_id"]]
    cities = cities.merge(complete, on=["prompt", "clip_id"])
    md.append(f"\n{cities[cities.prompt == 'checklist'].clip_id.nunique()} city clips answered by all four models "
              f"with the checklist prompt, {cities[cities.prompt == 'open'].clip_id.nunique()} with the open prompt.\n")

    # Problem types reported, per model and prompt.
    long = cities.melt(id_vars=["prompt", "model", "clip_id", "role"],
                       value_vars=[f"issue_{c}" for c in CATEGORIES], var_name="category", value_name="flagged")
    long["category"] = long.category.str.removeprefix("issue_")
    share = long.groupby(["prompt", "model", "category"]).flagged.mean().reset_index()
    share.to_csv(out / "cities_share_flagged.csv", index=False)
    # One heat map per prompt, stacked so the column labels fit: rows are models, columns problem types, cells the share of clips flagged.
    prompts_here = [p for p in PROMPTS if p in set(share.prompt)]
    figure = make_subplots(rows=len(prompts_here), cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=[f"{PROMPT_NAMES[p]} ({cities[cities.prompt == p].clip_id.nunique()} clips)"
                                           for p in prompts_here])
    for i, prompt in enumerate(prompts_here, start=1):
        grid = (share[share.prompt == prompt].pivot(index="model", columns="category", values="flagged")
                .reindex(index=MODEL_ORDER, columns=CATEGORIES))
        figure.add_heatmap(z=grid.values, x=[LABELS[c].replace(" ", "<br>", 1) for c in CATEGORIES],
                           y=[MODEL_NAMES[m] for m in MODEL_ORDER], zmin=0, zmax=1, colorscale=BLUES,
                           text=[[pct(v) for v in row] for row in grid.values], texttemplate="%{text}",
                           xgap=2, ygap=2, showscale=i == len(prompts_here),
                           colorbar=dict(title=dict(text="Share of clips", side="right"), tickformat=".0%"),
                           hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>", row=i, col=1)
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(tickangle=0)
    figure.update_layout(height=640)
    save(figure, "problem_types_by_prompt", png)
    md.append("\n### City clips: share flagged with each problem type (checklist prompt, clips all four models answered)\n")
    pivot = share[share.prompt == "checklist"].pivot(index="category", columns="model", values="flagged")
    pivot = pivot.reindex(index=CATEGORIES, columns=MODEL_ORDER)
    md.append(markdown(pd.DataFrame({"Problem type": [LABELS[c] for c in pivot.index],
                                     **{MODEL_NAMES[m]: [pct(v) for v in pivot[m]] for m in pivot.columns}})))

    # Pavement answers, empty streets.
    rows = []
    for (prompt, model), group in cities.groupby(["prompt", "model"]):
        sides = pd.concat([group.scene_pavement_left, group.scene_pavement_right])
        empty = group[~group.scene_pedestrians_visible.astype(bool)]
        rows.append({"prompt": prompt, "model": model, "pavement_absent": (sides == "absent").mean(),
                     "pavement_unclear_or_not_visible": sides.isin(["unclear", "not_visible"]).mean(),
                     "types_used": int(sum(group[f"issue_{c}"].any() for c in CATEGORIES)),
                     "empty_clips": len(empty), "empty_with_problem": empty.any_issue.mean() if len(empty) else
                     float("nan")})
    behaviour = pd.DataFrame(rows)
    behaviour.to_csv(out / "prompt_behaviour.csv", index=False)
    md.append("\n### Open vs checklist prompt (city clips)\n")
    md.append(markdown(pd.DataFrame([{
        "Model": MODEL_NAMES[r.model], "Prompt": r.prompt, "Problem types used (of 6)": r.types_used,
        "Pavement sides judged absent": pct(r.pavement_absent),
        "Pavement sides unclear / not visible": pct(r.pavement_unclear_or_not_visible),
        "Clips without people that get a problem": f"{pct(r.empty_with_problem)} of {r.empty_clips}"}
        for r in behaviour.sort_values(["model", "prompt"], key=lambda s: s.map(
            {m: i for i, m in enumerate(MODEL_ORDER)} | {p: i for i, p in enumerate(PROMPTS)})).itertuples()])))

    # Ranked (main) vs random (control) clips, consensus of the four models.
    checklist = cities[cities.prompt == "checklist"]
    consensus = (checklist.groupby(["clip_id", "role", "locality", "country", "continent", "lat", "lon"])
                 [[f"issue_{c}" for c in CATEGORIES] + ["any_issue"]].sum().reset_index())
    for column in [f"issue_{c}" for c in CATEGORIES] + ["any_issue"]:
        consensus[column] = consensus[column] >= 3
    consensus.to_csv(out / "cities_consensus.csv", index=False)
    rows = []
    for category in CATEGORIES + ["any"]:
        column = "any_issue" if category == "any" else f"issue_{category}"
        row = {"category": category}
        for role in ("main", "control"):
            values = consensus[consensus.role == role][column]
            low, high = wilson(int(values.sum()), len(values))
            row.update({f"{role}_share": values.mean(), f"{role}_low": low, f"{role}_high": high, f"{role}_n": len(values)})
        rows.append(row)
    roles = pd.DataFrame(rows)
    roles.to_csv(out / "main_vs_control.csv", index=False)
    md.append("\n### Ranked vs random clips: problems the models agree on (at least 3 of 4, checklist prompt)\n")
    md.append(markdown(pd.DataFrame([{
        "Problem type": "Any problem" if r.category == "any" else LABELS[r.category],
        "Ranked clips": f"{pct(r.main_share)} ({pct(r.main_low)}–{pct(r.main_high)})",
        "Random clips": f"{pct(r.control_share)} ({pct(r.control_low)}–{pct(r.control_high)})"}
        for r in roles.itertuples()])))
    figure = go.Figure()
    for (role, name), colour in zip((("main", "Ranked clips"), ("control", "Random clips")), PAIR):
        n = int(roles[f"{role}_n"].iloc[0])
        figure.add_bar(name=f"{name} (n = {n})",
                       x=[("Any problem" if c == "any" else LABELS[c]).replace(" ", "<br>", 1) for c in roles.category],
                       y=roles[f"{role}_share"], marker_color=colour,
                       error_y=dict(type="data", symmetric=False, color="#52514e", thickness=1.5, width=4,
                                    array=roles[f"{role}_high"] - roles[f"{role}_share"],
                                    arrayminus=roles[f"{role}_share"] - roles[f"{role}_low"]),
                       hovertemplate="%{x}<br>%{y:.0%}<extra>" + name + "</extra>")
    figure.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, yaxis_title="Share of clips (95% CI)",
                         yaxis=dict(range=[0, 1], tickformat=".0%"), height=460,
                         legend=dict(orientation="h", y=1.08, x=0))
    save(figure, "ranked_vs_random", png)

    # Agreement between models.
    agree_rows = []
    for prompt in PROMPTS:
        features = cities[cities.prompt == prompt]
        if features.empty:
            continue
        columns = [f"scene_{f}" for f in SCENE_BOOLEANS] + ["any_issue"] + [f"issue_{c}" for c in CATEGORIES]
        agree_rows.append(agreement(features, columns).assign(prompt=prompt))
    agree = pd.concat(agree_rows, ignore_index=True)
    agree["label"] = agree["item"].str.replace("scene_", "").str.replace("issue_", "").map(
        lambda k: "Any problem" if k == "any_issue" else LABELS.get(k, k))
    agree.to_csv(out / "model_agreement.csv", index=False)
    order = [LABELS[f] for f in SCENE_BOOLEANS] + ["Any problem"] + [LABELS[c] for c in CATEGORIES]
    figure = go.Figure()
    for prompt, colour in zip(PROMPTS, PAIR):
        rows = agree[agree.prompt == prompt].set_index("label").reindex(order)
        n = cities[cities.prompt == prompt].clip_id.nunique()
        # kappa is undefined when no model ever flags the item; say so instead of drawing nothing.
        figure.add_bar(name=f"{PROMPT_NAMES[prompt]} ({n} clips)", y=order, x=rows.fleiss_kappa.fillna(0),
                       orientation="h", marker_color=colour,
                       text=["not defined (never flagged)" if v != v else f"{v:.2f}" for v in rows.fleiss_kappa],
                       textposition="outside", cliponaxis=False,
                       hovertemplate="%{y}<br>kappa %{text}<extra>" + PROMPT_NAMES[prompt] + "</extra>")
    figure.add_hline(y=len(SCENE_BOOLEANS) - 0.5, line_color="#52514e", line_width=1)
    figure.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, height=620,
                         xaxis=dict(title="Fleiss' kappa between the four models", range=[-0.1, 1.05], dtick=0.2,
                                    zeroline=True, zerolinecolor="#52514e"),
                         yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.06, x=0))
    save(figure, "model_agreement", png)
    md.append("\n### Agreement between the four models (Fleiss' kappa, city clips)\n")
    kappa = agree.pivot(index="label", columns="prompt", values="fleiss_kappa")
    order = [LABELS[f] for f in SCENE_BOOLEANS] + ["Any problem"] + [LABELS[c] for c in CATEGORIES]
    kappa = kappa.reindex(order)
    md.append(markdown(pd.DataFrame({"Item": kappa.index, **{p: [("–" if v != v else f"{v:.2f}") for v in kappa[p]]
                                                             for p in PROMPTS if p in kappa}})))

    # By continent and on the map (consensus, ranked and random clips together).
    continent = (consensus.groupby("continent")[[f"issue_{c}" for c in CATEGORIES] + ["any_issue"]].mean()
                 .assign(cities=consensus.groupby("continent").locality.nunique()))
    continent.to_csv(out / "by_continent.csv")
    heat = continent[[f"issue_{c}" for c in CATEGORIES]].rename(columns=lambda c: LABELS[c.removeprefix("issue_")])
    clips_per = consensus.groupby("continent").size()
    figure = go.Figure(go.Heatmap(
        z=heat.values, x=[c.replace(" ", "<br>", 1) for c in heat.columns], zmin=0, zmax=1, colorscale=BLUES,
        y=[f"{c} ({n} {'city' if n == 1 else 'cities'}, {k} {'clip' if k == 1 else 'clips'})"
           for c, n, k in zip(heat.index, continent.cities, clips_per.reindex(heat.index))],
        text=[[pct(v) for v in row] for row in heat.values], texttemplate="%{text}", xgap=2, ygap=2,
        colorbar=dict(title=dict(text="Share of clips", side="right"), tickformat=".0%"),
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(height=420)
    save(figure, "problems_by_continent", png)
    city = (consensus.groupby(["locality", "country", "lat", "lon"])[[f"issue_{c}" for c in CATEGORIES]].any()
            .sum(axis=1).rename("problem_types").reset_index())
    figure = px.scatter_geo(city, lat="lat", lon="lon", color="problem_types", hover_name="locality",
                            hover_data={"country": True, "lat": False, "lon": False}, projection="natural earth",
                            color_continuous_scale=BLUES[2:], range_color=[0, 6],
                            labels={"problem_types": "Problem types<br>(of 6)"}, height=560)
    figure.update_traces(marker=dict(size=10, line=dict(width=0.8, color="#0b0b0b")),
                         hovertemplate="%{hovertext}, %{customdata[0]}<br>%{marker.color} problem types<extra></extra>")
    figure.update_geos(showland=True, landcolor="#ecebe8", showcountries=True, countrycolor="#ffffff",
                       coastlinecolor="#c3c2b7", showframe=False, lataxis_range=[-58, 80])
    figure.update_coloraxes(colorbar=dict(dtick=1, title_side="right"))
    save(figure, "problems_map", png)

    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
