"""Variance reduction helpers used by the experiment engine."""

from __future__ import annotations

import math

from .stats import mean, variance


def cuped(y: list[float], x: list[float]) -> tuple[list[float], float]:
    """Return CUPED-adjusted outcomes and the fitted theta."""
    if len(y) != len(x) or not y:
        raise ValueError("y and x must have the same non-zero length")
    mx, my = mean(x), mean(y)
    vx = sum((value - mx) ** 2 for value in x)
    theta = 0.0 if vx == 0 else sum((yi - my) * (xi - mx) for yi, xi in zip(y, x)) / vx
    return [yi - theta * (xi - mx) for yi, xi in zip(y, x)], theta


def ratio_delta(numerator: list[float], denominator: list[float]) -> tuple[float, float]:
    """Estimate a ratio metric and its first-order delta-method standard error."""
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("numerator and denominator must have the same non-zero length")
    total_denominator = sum(denominator)
    if total_denominator == 0:
        return 0.0, 0.0
    ratio = sum(numerator) / total_denominator
    terms = [n - ratio * d for n, d in zip(numerator, denominator)]
    se = math.sqrt(variance(terms) / len(terms)) / abs(mean(denominator) or 1.0)
    return ratio, se
