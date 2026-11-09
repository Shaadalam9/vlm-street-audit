import math

import pandas as pd

from vlm_street_audit.evaluation import binary_scores, cohen_kappa, fleiss_kappa, wilson


def test_binary_scores():
    s = binary_scores(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 0, 0]))
    assert s["accuracy"] == 0.75 and s["recall"] == 0.5 and s["precision"] == 1.0
    assert s["balanced_accuracy"] == 0.75


def test_kappas_are_one_for_perfect_and_zero_or_below_for_chance():
    a = pd.Series([True, False, True, False])
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, ~a) < 0
    assert math.isclose(fleiss_kappa(pd.DataFrame({"m1": a, "m2": a, "m3": a})), 1.0)


def test_wilson_interval():
    low, high = wilson(5, 10)
    assert round(low, 3) == 0.237 and round(high, 3) == 0.763
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert all(v != v for v in wilson(0, 0))
