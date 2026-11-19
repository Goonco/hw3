"""Part 5: test-time scaling (pure Python, no torch).

Implement every function marked TODO.  Tests: tests/test_test_time.py
"""

from __future__ import annotations

import math
from typing import Hashable, Optional, Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    Given n samples of which c are correct, return the probability that at least one
    of k samples drawn WITHOUT replacement from the n is correct:
        1 - C(n - c, k) / C(n, k)
    Requirements: raise ValueError if not (0 <= c <= n and 1 <= k <= n).
    Must be numerically stable for n up to 1000 (do not form huge binomials).
    """
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("Expected 0 <= c <= n and 1 <= k <= n")
    if k > n - c:
        return 1.0
    miss_probability = math.prod((n - c - i) / (n - i) for i in range(k))
    return 1.0 - miss_probability


def majority_vote(answers: Sequence[Optional[Hashable]]) -> Optional[Hashable]:
    """Most common non-None answer.

    Ties are broken in favour of the answer whose FIRST occurrence is earliest.
    Returns None if every answer is None (or the list is empty).
    """
    counts = {}
    for answer in answers:
        if answer is not None:
            counts[answer] = counts.get(answer, 0) + 1
    return max(counts, key=counts.__getitem__, default=None)


def select_by_confidence(scores: Sequence[Optional[float]]) -> int:
    """Index of the candidate with the highest confidence score (e.g. mean token log-prob).

    None or NaN scores count as -infinity.  Ties go to the smallest index.
    Raise ValueError on an empty list.
    """
    def score_at(index: int) -> float:
        score = scores[index]
        if score is None or math.isnan(score):
            return -math.inf
        return score

    return max(range(len(scores)), key=score_at)
