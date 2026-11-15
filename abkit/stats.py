"""统计分布工具（标准库实现，避免引入 SciPy）。"""

from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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
