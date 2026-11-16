"""统计分布工具（标准库实现，避免引入 SciPy）。"""

from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi2_sf(x: float, df: int) -> float:
    """卡方分布的右尾概率 P(X > x)，即拟合优度检验的 p 值。

    基于正则化上不完全伽马函数 Q(df/2, x/2)，用级数展开 / 连分数
    （Numerical Recipes 的 gammp/gammq 思路）纯标准库实现，df >= 1。
    """
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if x <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    if z < a + 1.0:
        # 级数展开：1 - P(a, z)
        term = 1.0 / a
        total = term
        for k in range(1, 1000):
            term *= z / (a + k)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        q = 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a))
    else:
        # 连分数：Q(a, z)
        tiny = 1e-300
        b = z + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, 1000):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        q = h * math.exp(-z + a * math.log(z) - math.lgamma(a))
    return min(1.0, max(0.0, q))


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
