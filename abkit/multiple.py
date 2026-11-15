"""Multiple-comparison corrections."""

from __future__ import annotations


def bh(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg rejection decisions, preserving input order."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    cutoff = 0
    for rank, (_, p_value) in enumerate(ranked, 1):
        if p_value <= alpha * rank / max(len(p_values), 1):
            cutoff = rank
    return [rank <= cutoff for rank in _input_ranks(ranked, len(p_values))]


def _input_ranks(ranked: list[tuple[int, float]], size: int) -> list[int]:
    rank_by_index = {index: rank for rank, (index, _) in enumerate(ranked, 1)}
    return [rank_by_index[index] for index in range(size)]


def holm(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down decisions."""
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    rejected = [False] * len(p_values)
    for rank, (index, p_value) in enumerate(ranked):
        if p_value > alpha / (len(p_values) - rank):
            break
        rejected[index] = True
    return rejected
