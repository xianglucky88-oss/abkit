"""Multiple-comparison corrections.

Two procedures are supported:

* ``bh``   -- Benjamini-Hochberg step-up, controlling the false discovery rate.
* ``holm`` -- Holm step-down, controlling the family-wise error rate.

:func:`bh` and :func:`holm` keep their original boolean-array interface
(input order preserved).  :func:`correct` returns the full structured result
(ranks, adjusted p-values, rejection decisions, tied-p groups and invalid
p-value reasons) that the report layer renders into JSON and Markdown, so a
reviewer can reconstruct every decision from the report.
"""

from __future__ import annotations

import math

METHODS = ("bh", "holm")


def _validate_alpha(alpha: float) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a number in (0, 1)")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be in (0, 1)")


def _classify(raw: object) -> tuple[bool, float | None, str | None]:
    """Return (is_valid, float_value, reason) for one raw p-value."""
    if raw is None:
        return False, None, "missing p-value"
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return False, None, "p-value is not numeric"
    value = float(raw)
    if math.isnan(value):
        return False, None, "p-value is NaN"
    if math.isinf(value) or not 0.0 <= value <= 1.0:
        return False, None, "p-value outside [0, 1]"
    return True, value, None


def _adjusted_p_values(ranked: list[tuple[int, float]], method: str) -> list[float]:
    """Adjusted p-values in rank order (rank 1 first), capped at 1.0."""
    m = len(ranked)
    adjusted = [0.0] * m
    if method == "bh":
        # BH: adj_p at rank j = min over k >= j of m/k * p_(k).
        running = math.inf
        for j in range(m - 1, -1, -1):
            running = min(running, m / (j + 1) * ranked[j][1])
            adjusted[j] = min(1.0, running)
    else:
        # Holm: adj_p at rank j = max over k <= j of (m - k + 1) * p_(k).
        running = 0.0
        for j in range(m):
            running = max(running, (m - j) * ranked[j][1])
            adjusted[j] = min(1.0, running)
    return adjusted


def correct(
    p_values: list[float | None],
    *,
    method: str = "bh",
    alpha: float = 0.05,
) -> dict:
    """Run a multiple-comparison correction and return structured evidence.

    Invalid entries (``None``, NaN, non-numeric or outside ``[0, 1]``) are
    excluded from the correction but kept in ``items`` at their input index
    with an explanatory ``reason``. Tied p-values are ordered by input index
    and reported under ``tie_groups``.

    Returns a JSON-safe dict (no NaN/Infinity) containing the method, alpha,
    counts, per-index items and the rejection booleans in input order.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    _validate_alpha(alpha)

    items: list[dict] = []
    valid: list[tuple[int, float]] = []
    for index, raw in enumerate(p_values):
        ok, value, reason = _classify(raw)
        items.append(
            {
                "index": index,
                "p_value": value,
                "valid": ok,
                "reason": reason,
                "rank": None,
                "adjusted_p_value": None,
                "rejected": False,
            }
        )
        if ok:
            valid.append((index, value))

    m = len(valid)
    # Explicit tie-break: ascending p-value, then original input index.
    ranked = sorted(valid, key=lambda pair: (pair[1], pair[0]))
    adjusted = _adjusted_p_values(ranked, method)
    for j, (index, _) in enumerate(ranked):
        items[index]["rank"] = j + 1
        items[index]["adjusted_p_value"] = adjusted[j]
        items[index]["rejected"] = adjusted[j] <= alpha

    tie_groups: list[list[int]] = []
    current: list[int] = []
    previous: float | None = None
    for index, value in ranked:
        if current and value != previous:
            if len(current) > 1:
                tie_groups.append(current)
            current = []
        current.append(index)
        previous = value
    if len(current) > 1:
        tie_groups.append(current)

    return {
        "method": method,
        "alpha": float(alpha),
        "input_count": len(p_values),
        "valid_count": m,
        "items": items,
        "rejected": [item["rejected"] for item in items],
        "tie_groups": tie_groups,
    }


def bh(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg rejection decisions, preserving input order."""
    return correct(p_values, method="bh", alpha=alpha)["rejected"]


def holm(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down decisions, preserving input order."""
    return correct(p_values, method="holm", alpha=alpha)["rejected"]
