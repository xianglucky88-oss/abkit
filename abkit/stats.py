"""统计分布工具（标准库实现，避免引入 SciPy）。"""

from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_ppf(p: float) -> float:
    """标准正态分位数。用二分法反解 CDF，精度足够工程使用。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def variance(xs: list[float]) -> float:
    """样本方差（无偏）。"""
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def stderr(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return math.sqrt(variance(xs) / n)


def welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch t 检验，返回 (t, 双侧 p)。p 用正态近似。"""
    va, vb = variance(a), variance(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (mean(a) - mean(b)) / se
    p = 2 * (1 - normal_cdf(abs(t)))
    return t, p


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间。

    trials <= 0 时返回全区间 (0.0, 1.0)，避免空样本产生 NaN。
    """
    if trials <= 0:
        return 0.0, 1.0
    z = normal_ppf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)
