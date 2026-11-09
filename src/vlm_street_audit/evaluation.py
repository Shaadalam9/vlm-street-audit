"""Statistics for the audit: binary scores against labels, Cohen's and Fleiss' kappa, Wilson intervals."""

import math

import numpy as np
import pandas as pd

SCENE_BOOLEANS = ["zebra_crossing_visible", "traffic_signal_visible", "pedestrians_visible",
                  "pedestrians_in_carriageway"]


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


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
