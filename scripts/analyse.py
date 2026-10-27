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
FIGURES = PROJECT_ROOT / "docs" / "figures"


def pct(value: float) -> str:
    return "–" if value != value else f"{100 * value:.0f}%"


def markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def save(fig: go.Figure, name: str, png: bool) -> None:
    style(fig)
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
    figure = px.bar(jaad.assign(field=jaad.field.map(LABELS), model=jaad.model.map(MODEL_NAMES)),
                    x="field", y="balanced_accuracy", color="model", barmode="group", facet_row="prompt",
                    category_orders={"model": [MODEL_NAMES[m] for m in MODEL_ORDER], "prompt": PROMPTS},
                    labels={"field": "", "balanced_accuracy": "Balanced accuracy", "model": ""},
                    title="Scene facts against JAAD's labels (120 clips; 0.5 = chance)", height=650)
    figure.add_hline(y=0.5, line_dash="dot", line_color="grey")
    figure.update_yaxes(range=[0, 1])
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
    figure = px.bar(share.assign(category=share.category.map(LABELS), model=share.model.map(MODEL_NAMES)),
                    x="model", y="flagged", color="category", facet_col="prompt", barmode="group",
                    category_orders={"model": [MODEL_NAMES[m] for m in MODEL_ORDER], "prompt": PROMPTS,
                                     "category": [LABELS[c] for c in CATEGORIES]},
                    labels={"flagged": "Share of city clips with the problem", "model": "", "category": ""},
                    title="Problem types reported on the city clips, by prompt", height=520)
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
    for role, name in (("main", "Ranked clips"), ("control", "Random clips")):
        figure.add_bar(name=name, x=[("Any problem" if c == "any" else LABELS[c]) for c in roles.category],
                       y=roles[f"{role}_share"], error_y=dict(type="data", symmetric=False,
                                                              array=roles[f"{role}_high"] - roles[f"{role}_share"],
                                                              arrayminus=roles[f"{role}_share"] - roles[f"{role}_low"]))
    figure.update_layout(barmode="group", title="Problems at least 3 of 4 models agree on: ranked vs random clips",
                         yaxis_title="Share of clips (95% CI)", height=480)
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
    figure = px.bar(agree, x="label", y="fleiss_kappa", color="prompt", barmode="group",
                    category_orders={"prompt": PROMPTS},
                    labels={"label": "", "fleiss_kappa": "Fleiss' kappa (4 models)", "prompt": "Prompt"},
                    title="Agreement between the four models on the city clips", height=480)
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
    figure = px.imshow(heat, text_auto=".0%", aspect="auto", color_continuous_scale="Reds", zmin=0, zmax=1,
                       labels={"color": "Share of clips", "y": ""},
                       y=[f"{c} ({n} {'city' if n == 1 else 'cities'})" for c, n in zip(heat.index, continent.cities)],
                       title="Problems at least 3 of 4 models agree on, by continent (checklist prompt)", height=420)
    save(figure, "problems_by_continent", png)
    city = (consensus.groupby(["locality", "country", "lat", "lon"])[[f"issue_{c}" for c in CATEGORIES]].any()
            .sum(axis=1).rename("problem_types").reset_index())
    figure = px.scatter_geo(city, lat="lat", lon="lon", color="problem_types", hover_name="locality",
                            hover_data={"country": True, "lat": False, "lon": False}, projection="natural earth",
                            color_continuous_scale="OrRd", range_color=[0, 6],
                            labels={"problem_types": "Problem types"},
                            title="Problem types the models agree on per city (checklist prompt, 2 clips per city)",
                            height=520)
    figure.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    save(figure, "problems_map", png)

    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
