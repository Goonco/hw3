import math

import pytest

from posttrain.test_time import majority_vote, pass_at_k, select_by_confidence


def _brute_pass_at_k(n, c, k):
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@pytest.mark.parametrize("n,c,k", [(10, 0, 1), (10, 3, 1), (10, 3, 4), (16, 5, 8), (16, 15, 2), (20, 20, 20), (7, 2, 7)])
def test_pass_at_k_matches_definition(n, c, k):
    assert pass_at_k(n, c, k) == pytest.approx(_brute_pass_at_k(n, c, k), abs=1e-9)


def test_pass_at_k_large_n_is_stable():
    val = pass_at_k(1000, 1, 500)
    assert val == pytest.approx(0.5, abs=1e-9)
    assert 0.0 <= pass_at_k(1000, 37, 3) <= 1.0


@pytest.mark.parametrize("n,c,k", [(5, 6, 1), (5, 2, 0), (5, 2, 6), (5, -1, 1)])
def test_pass_at_k_invalid(n, c, k):
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_majority_vote():
    assert majority_vote(["3", "5", "3", None, "5", "3"]) == "3"
    assert majority_vote(["5", "3", "3", "5"]) == "5"  # tie -> earliest first occurrence
    assert majority_vote([None, "7", None]) == "7"
    assert majority_vote([None, None]) is None
    assert majority_vote([]) is None
    assert majority_vote([1, 2, 2, 1, 2]) == 2


def test_select_by_confidence():
    assert select_by_confidence([-1.0, -0.2, -0.5]) == 1
    assert select_by_confidence([-0.3, -0.3, -1.0]) == 0  # tie -> smallest index
    assert select_by_confidence([None, float("nan"), -9.0]) == 2
    with pytest.raises(ValueError):
        select_by_confidence([])


def test_snapshot_cases():
    """Randomised cases (the autograder swaps in a hidden set with different numbers)."""
    from tests.helpers import load_snapshot

    s = load_snapshot("test_time")
    for case in s["pass_at_k"]:
        assert pass_at_k(case["n"], case["c"], case["k"]) == pytest.approx(case["expected"], abs=1e-9), case
    for case in s["majority_vote"]:
        assert majority_vote(case["answers"]) == case["expected"], case
    for case in s["select_by_confidence"]:
        assert select_by_confidence(case["scores"]) == case["expected"], case
