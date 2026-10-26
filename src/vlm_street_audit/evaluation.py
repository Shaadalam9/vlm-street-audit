"""Scores for the VLM answers without human raters.

- JAAD: the scene facts against JAAD's labels (accuracy, balanced accuracy, precision,
  recall, F1), and whether points given for pedestrian issues land on a labelled pedestrian.
- Cities: how often each model reports problems on ranked ("main") clips against random
  ("control") clips, and how far the models agree with each other (Cohen's kappa per
  pair, Fleiss' kappa over all models).
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SCENE_BOOLEANS = ["zebra_crossing_visible", "traffic_signal_visible", "pedestrians_visible",
                  "pedestrians_in_carriageway"]
PEDESTRIAN_CATEGORIES = {"pedestrian_exposure", "vulnerable_users"}


def load_answers(root: Path, set_name: str) -> pd.DataFrame:
    """One row per (model, clip) with the scene facts flattened and the issues kept as a list."""
    rows = []
    for path in sorted((root / set_name).glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        answer = record.get("answer") or {}
        row = {"model": record["model"], "clip_id": record["clip_id"], "role": record.get("role", "main"),
               "valid": bool(answer), "seconds": record.get("seconds"), "issues": answer.get("issues", [])}
        row.update({f"scene_{k}": v for k, v in (answer.get("scene") or {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def binary_scores(truth: pd.Series, predicted: pd.Series) -> dict:
    t, p = truth.astype(bool).to_numpy(), predicted.astype(bool).to_numpy()
    tp, tn = int(np.sum(t & p)), int(np.sum(~t & ~p))
    fp, fn = int(np.sum(~t & p)), int(np.sum(t & ~p))
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {"n": len(t), "positives": int(t.sum()), "accuracy": (tp + tn) / len(t) if len(t) else float("nan"),
            "balanced_accuracy": np.nanmean([recall, specificity]), "precision": precision, "recall": recall,
            "f1": f1}


def jaad_scores(answers: pd.DataFrame, truth: dict) -> pd.DataFrame:
    rows = []
    valid = answers[answers.valid]
    for model, group in valid.groupby("model"):
        labels = pd.DataFrame([truth[c] for c in group.clip_id])
        for field in SCENE_BOOLEANS:
            rows.append({"model": model, "field": field,
                         **binary_scores(labels[field], group[f"scene_{field}"].reset_index(drop=True))})
        rows.append({"model": model, "field": "parking_area",
                     **binary_scores(labels["parking_area"],
                                     (group["scene_road_type"] == "parking_area").reset_index(drop=True))})
    return pd.DataFrame(rows)


def point_hits(answers: pd.DataFrame, truth: dict, margin: int = 20) -> pd.DataFrame:
    """Share of points for pedestrian issues that land on a labelled pedestrian box (± margin, 0-1000 units)."""
    rows = []
    for model, group in answers[answers.valid].groupby("model"):
        points = hits = 0
        for clip_id, issues in zip(group.clip_id, group.issues):
            frames = truth[clip_id]["frames"]
            for issue in issues:
                if issue["category"] not in PEDESTRIAN_CATEGORIES or not 1 <= issue["frame"] <= len(frames):
                    continue
                boxes = truth[clip_id]["boxes"][str(frames[issue["frame"] - 1])]
                points += 1
                hits += any(b["x1"] - margin <= issue["x"] <= b["x2"] + margin
                            and b["y1"] - margin <= issue["y"] <= b["y2"] + margin for b in boxes)
        rows.append({"model": model, "pedestrian_points": points, "on_a_pedestrian": hits,
                     "hit_rate": hits / points if points else float("nan")})
    return pd.DataFrame(rows)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    a, b = a.astype(str).to_numpy(), b.astype(str).to_numpy()
    categories = sorted(set(a) | set(b))
    observed = float(np.mean(a == b))
    expected = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in categories)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def fleiss_kappa(table: pd.DataFrame) -> float:
    """Rows are items, columns are raters, values are categories."""
    categories = sorted(set(table.astype(str).to_numpy().ravel()))
    counts = np.array([[int((row == c).sum()) for c in categories] for row in table.astype(str).to_numpy()])
    raters = counts.sum(axis=1)
    if len(counts) == 0 or raters.min() < 2:
        return float("nan")
    per_item = (np.sum(counts ** 2, axis=1) - raters) / (raters * (raters - 1))
    shares = counts.sum(axis=0) / counts.sum()
    expected = float(np.sum(shares ** 2))
    return (float(per_item.mean()) - expected) / (1 - expected) if expected < 1 else float("nan")


def clip_features(answers: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Per (model, clip): the scene facts plus whether each issue category was reported."""
    features = answers[answers.valid].copy()
    features["any_issue"] = features.issues.map(bool)
    for category in categories:
        features[f"issue_{category}"] = features.issues.map(lambda issues, c=category: any(
            i["category"] == c for i in issues))
    return features


def agreement(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fleiss' kappa over all models and mean pairwise Cohen's kappa, per column, on clips every model answered."""
    models = sorted(features.model.unique())
    rows = []
    for column in columns:
        wide = features.pivot(index="clip_id", columns="model", values=column).dropna()
        pairs = [cohen_kappa(wide[a], wide[b]) for a, b in combinations(models, 2) if a in wide and b in wide]
        pairs = [p for p in pairs if not np.isnan(p)]  # undefined when both models never use a category
        rows.append({"item": column, "clips": len(wide), "fleiss_kappa": fleiss_kappa(wide),
                     "mean_pairwise_cohen_kappa": float(np.nanmean(pairs)) if pairs else float("nan"),
                     "share_all_agree": float((wide.astype(str).nunique(axis=1) == 1).mean()) if len(wide) else
                     float("nan")})
    return pd.DataFrame(rows)
