"""Score the VLM answers: JAAD accuracy, pedestrian point hits, main vs control, model agreement.

Writes CSV tables to outputs/vlm/summary/ and Plotly figures to outputs/figures/, and
prints the tables.

Run:  uv run python scripts/score_vlm.py
"""

import json
import sys

import pandas as pd
import plotly.express as px

from vlm_street_audit.config import load_config, project_path
from vlm_street_audit.evaluation import (
    SCENE_BOOLEANS,
    agreement,
    clip_features,
    jaad_scores,
    load_answers,
    point_hits,
)
from vlm_street_audit.figures import save_figure, style
from vlm_street_audit.vlm import ISSUE_CATEGORIES


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n## {title}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> int:
    settings = load_config()
    root = project_path(settings["vlm_output_dir"])
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)

    jaad = load_answers(root, "jaad")
    if not jaad.empty:
        truth = json.loads((project_path(settings["clip_sets_dir"]) / "jaad_labels.json").read_text())
        validity = jaad.groupby("model").agg(clips=("clip_id", "size"), valid=("valid", "mean"),
                                             median_seconds=("seconds", "median")).reset_index()
        show("JAAD: answers", validity)
        scores = jaad_scores(jaad, truth)
        scores.to_csv(summary / "jaad_scene_scores.csv", index=False)
        show("JAAD: scene facts against the labels",
             scores[["model", "field", "n", "positives", "accuracy", "balanced_accuracy", "precision", "recall"]])
        hits = point_hits(jaad, truth)
        hits.to_csv(summary / "jaad_point_hits.csv", index=False)
        show("JAAD: points of pedestrian issues on a labelled pedestrian", hits)
        figure = px.bar(scores, x="field", y="balanced_accuracy", color="model", barmode="group",
                        title="JAAD: balanced accuracy of the scene facts (0.5 = chance)",
                        labels={"field": "", "balanced_accuracy": "Balanced accuracy"})
        figure.add_hline(y=0.5, line_dash="dot")
        save_figure(style(figure), "jaad_scene_balanced_accuracy")

    cities = load_answers(root, "cities")
    if not cities.empty:
        features = clip_features(cities, ISSUE_CATEGORIES)
        per_role = (features.assign(issue_count=features.issues.map(len))
                    .groupby(["model", "role"]).agg(clips=("clip_id", "size"), share_with_issue=("any_issue", "mean"),
                                                    issues_per_clip=("issue_count", "mean")).reset_index())
        per_role.to_csv(summary / "cities_main_vs_control.csv", index=False)
        show("Cities: problems reported on ranked (main) and random (control) clips", per_role)
        columns = [f"scene_{f}" for f in SCENE_BOOLEANS] + ["scene_road_type", "any_issue"] + [
            f"issue_{c}" for c in ISSUE_CATEGORIES if c != "other"]
        agree = agreement(features, columns)
        agree.to_csv(summary / "cities_model_agreement.csv", index=False)
        show("Cities: agreement between models", agree)
        figure = px.bar(per_role, x="model", y="share_with_issue", color="role", barmode="group",
                        title="Share of clips with at least one reported problem",
                        labels={"model": "", "share_with_issue": "Share of clips", "role": "Clip"})
        save_figure(style(figure), "cities_main_vs_control")
    return 0


if __name__ == "__main__":
    sys.exit(main())
