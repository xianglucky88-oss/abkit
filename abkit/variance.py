"""Variance reduction helpers used by the experiment engine."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .stats import mean, variance


def cuped(y: list[float], x: list[float]) -> tuple[list[float], float]:
    """Return CUPED-adjusted outcomes and the fitted theta.

    theta = Cov(y, x) / Var(x)。协变量无方差时 theta = 0，调整退化为原值。
    """
    if len(y) != len(x) or not y:
        raise ValueError("y and x must have the same non-zero length")
    mx, my = mean(x), mean(y)
    vx = sum((value - mx) ** 2 for value in x)
    theta = 0.0 if vx == 0 else sum((yi - my) * (xi - mx) for yi, xi in zip(y, x)) / vx
    return [yi - theta * (xi - mx) for yi, xi in zip(y, x)], theta


def cuped_details(y: list[float], x: list[float]) -> dict:
    """CUPED 的诊断信息：theta、调整前后方差与缩减比例。"""
    adjusted, theta = cuped(y, x)
    before = variance(y)
    after = variance(adjusted)
    return {
        "adjusted": adjusted,
        "theta": theta,
        "variance_before": before,
        "variance_after": after,
        "variance_reduction": 0.0 if before == 0 else 1.0 - after / before,
        "covariate_has_variance": theta != 0.0,
    }


@dataclass
class RatioEstimate:
    """比率指标的 delta 方法估计，供规划层直接使用。

    ok=False 时 ratio/se/unit_variance 为 None，reason 给出可解释的原因
    （例如分母全为零），调用方应把原因透传到报告而不是继续规划。
    """

    ratio: float | None
    se: float | None
    n: int
    mean_denominator: float
    unit_variance: float | None  # 线性化项的每观测方差 Var(n_i - r·d_i) / mean(d)^2
    ok: bool
    reason: str | None


def ratio_estimate(numerator: list[float], denominator: list[float],
                   min_mean_denominator: float = 1e-12) -> RatioEstimate:
    """Estimate a ratio metric with a first-order delta-method standard error.

    分母中允许出现 0（比率按总量定义，单个零值只是该用户没有分母事件）；
    但分母之和为 0 或均值接近 0 时，比率与标准误都没有可解释的定义，
    返回 ok=False 并说明原因，而不是返回看似正常的 (0.0, 0.0)。
    """
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("numerator and denominator must have the same non-zero length")
    n = len(numerator)
    total_denominator = sum(denominator)
    mean_denominator = total_denominator / n
    if total_denominator == 0:
        return RatioEstimate(None, None, n, 0.0, None, False,
                             "分母之和为 0，比率未定义（分母全为零或正负抵消）")
    if abs(mean_denominator) < min_mean_denominator:
        return RatioEstimate(None, None, n, mean_denominator, None, False,
                             f"分母均值 {mean_denominator:.3g} 接近 0，delta 方法标准误不稳定")
    if n < 2:
        return RatioEstimate(None, None, n, mean_denominator, None, False,
                             "样本量 < 2，无法估计线性化项的方差")
    ratio = sum(numerator) / total_denominator
    terms = [value - ratio * d for value, d in zip(numerator, denominator)]
    unit_variance = variance(terms) / (mean_denominator ** 2)
    se = math.sqrt(unit_variance / n)
    return RatioEstimate(ratio, se, n, mean_denominator, unit_variance, True, None)


def ratio_delta(numerator: list[float], denominator: list[float]) -> tuple[float, float]:
    """Backward-compatible wrapper returning (ratio, se).

    退化情形（分母全零等）保持旧行为返回 (0.0, 0.0)；新代码请用
    `ratio_estimate`，它能区分"比率为 0"与"比率未定义"。
    """
    estimate = ratio_estimate(numerator, denominator)
    if not estimate.ok:
        return 0.0, 0.0
    return estimate.ratio, estimate.se
