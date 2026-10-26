import math

import pandas as pd

from vlm_street_audit.evaluation import agreement, binary_scores, cohen_kappa, fleiss_kappa, point_hits


def test_binary_scores():
    s = binary_scores(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 0, 0]))
    assert s["accuracy"] == 0.75 and s["recall"] == 0.5 and s["precision"] == 1.0
    assert s["balanced_accuracy"] == 0.75


def test_kappas_are_one_for_perfect_and_zero_or_below_for_chance():
    a = pd.Series([True, False, True, False])
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, ~a) < 0
    assert math.isclose(fleiss_kappa(pd.DataFrame({"m1": a, "m2": a, "m3": a})), 1.0)


def test_agreement_uses_clips_all_models_answered():
    features = pd.DataFrame({"model": ["a", "a", "b", "b", "b"], "clip_id": ["1", "2", "1", "2", "3"],
                             "any_issue": [True, False, True, False, True]})
    row = agreement(features, ["any_issue"]).iloc[0]
    assert row.clips == 2 and row.share_all_agree == 1.0


def test_point_hits_counts_only_pedestrian_issues():
    truth = {"c": {"frames": [0, 30], "boxes": {"0": [], "30": [{"x1": 100, "y1": 100, "x2": 200, "y2": 400}]}}}
    issue = {"category": "pedestrian_exposure", "frame": 2, "x": 150, "y": 300}
    answers = pd.DataFrame({"model": ["m"], "clip_id": ["c"], "valid": [True], "issues": [[
        issue, {**issue, "x": 900}, {**issue, "category": "pavement_condition"}]]})
    row = point_hits(answers, truth).iloc[0]
    assert row.pedestrian_points == 2 and row.on_a_pedestrian == 1


def test_wilson_interval():
    from vlm_street_audit.evaluation import wilson

    low, high = wilson(5, 10)
    assert round(low, 3) == 0.237 and round(high, 3) == 0.763
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert all(v != v for v in wilson(0, 0))
