"""序贯检验：alpha 消耗、成组序贯设计与逐次查看流程。

实验团队会在数据积累到不同阶段查看结果。若每次查看都套用固定 alpha 的
单次检验，一类错误会随查看次数膨胀（5 次等距查看、每次 0.05 时累计约
0.142）。本模块提供统计上自洽的替代流程，把信息比例、查看次数、早停
原因和最终结论联系起来：

- ``alpha_spending``：Lan-DeMets 消耗函数，给出信息比例 t 处累计可消耗
  的 alpha，且任意 family 都满足 spent(1) == alpha；
- ``GroupSequentialDesign``：由消耗函数推出每次查看的双侧临界值（递归
  数值积分，Jennison-Turnbull 口径），零假设下累计越界概率恰好等于累计
  消耗的 alpha；``spending="unadjusted"`` 是未校正对照设计，其 ``spent``
  为数值积分算出的真实累计一类错误，用于校准中演示过早查看的膨胀；
- ``monitor``：按设计逐次查看累计样本——未越界则继续观察（continue），
  越界则以 efficacy 早停，达到查看上限仍未显著则以 max_looks 收尾并
  给出 inconclusive 结论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .stats import mean, normal_cdf, normal_pdf, normal_ppf, welch_t

SPENDING_FAMILIES = ("obrien_fleming", "pocock", "linear")

# 逐次查看的决定原因
REASON_EFFICACY = "efficacy"    # 越过疗效边界，提前显著
REASON_CONTINUE = "continue"    # 未越界，继续观察
REASON_MAX_LOOKS = "max_looks"  # 达到查看上限仍未显著


def alpha_spending(info_fraction: float, alpha: float = 0.05,
                   family: str = "obrien_fleming") -> float:
    """信息比例 ``info_fraction`` 处累计消耗的 alpha（双侧口径）。

    family:
        obrien_fleming  Lan-DeMets O'Brien-Fleming 型，早期保守（默认）；
        pocock          Lan-DeMets Pocock 型，各次查看临界值接近恒定；
        linear          alpha * t，线性消耗（旧实现的口径）。

    任一 family 都满足 spent(1) == alpha，保证整个序贯过程的总一类错误
    恰好为 alpha。
    """
    if not 0 < info_fraction <= 1:
        raise ValueError("info_fraction must be in (0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if family == "obrien_fleming":
        z_alpha = normal_ppf(1 - alpha / 2)
        return min(alpha, 2 * (1 - normal_cdf(z_alpha / math.sqrt(info_fraction))))
    if family == "pocock":
        return alpha * math.log(1 + (math.e - 1) * info_fraction)
    if family == "linear":
        return alpha * info_fraction
    raise ValueError(f"unknown spending family: {family!r}; expected one of {SPENDING_FAMILIES}")


def _propagate(density: list[float], grid: list[float], t_prev: float, t_next: float,
               u_prev: float, step: float) -> list[float]:
    """把上一查看的继续密度（截断于 ±u_prev）传播到下一查看。

    利用密度关于 0 对称，只在半直线上计算：
    f_k(z) = ∫_0^{u_prev} f_{k-1}(x) [φ((z-ρx)/s) + φ((z+ρx)/s)] / s dx，
    其中 ρ = sqrt(t_prev/t_next)，s = sqrt(1-ρ²)。截断点用梯形权重并
    对 u_prev 处做线性插值，保证概率质量守恒。
    """
    rho = math.sqrt(t_prev / t_next)
    s = math.sqrt(1.0 - rho * rho)
    m = 0
    for i, x in enumerate(grid):
        if x <= u_prev:
            m = i
        else:
            break
    rem = u_prev - grid[m]
    nodes = grid[:m + 1]
    vals = density[:m + 1]
    weights = [step] * (m + 1)
    weights[0] = 0.5 * step
    weights[m] = 0.5 * step
    if rem > 1e-12 and m + 1 < len(grid):
        nodes = nodes + [u_prev]
        vals = vals + [vals[m] + (density[m + 1] - density[m]) * rem / step]
        weights[m] += 0.5 * rem
        weights = weights + [0.5 * rem]
    out = []
    for z in grid:
        acc = 0.0
        for x, fx, w in zip(nodes, vals, weights):
            acc += w * fx * (normal_pdf((z - rho * x) / s) + normal_pdf((z + rho * x) / s))
        out.append(acc / s)
    return out


def _cumulative(density: list[float], step: float) -> list[float]:
    """半直线上密度的梯形累计积分。"""
    cum = [0.0] * len(density)
    for i in range(1, len(density)):
        cum[i] = cum[i - 1] + 0.5 * (density[i - 1] + density[i]) * step
    return cum


def _interpolate(cum: list[float], grid: list[float], step: float, z: float) -> float:
    i = min(int(z / step), len(grid) - 2)
    frac = (z - grid[i]) / step
    return cum[i] + frac * (cum[i + 1] - cum[i])


def _solve_boundary(cum: list[float], grid: list[float], step: float, target: float) -> float:
    """求 u 使 ∫_0^u f = target（f 为半直线继续密度，单调可二分）。"""
    lo, hi = 0.0, grid[-1]
    for _ in range(80):
        mid = (lo + hi) / 2
        if _interpolate(cum, grid, step, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class GroupSequentialDesign:
    """K 次查看的成组序贯双侧设计。

    boundaries_[k] 是第 k+1 次查看的临界值：零假设下 |Z| 首次越界的累计
    概率等于 spent[k]。``spent`` 对消耗函数 family 而言是构造值（恰好等于
    alpha_spending(t_k)），对 "unadjusted" 而言是数值积分得到的真实累计
    一类错误（会大于 alpha，用于演示过早查看的代价）。
    """

    def __init__(self, looks: int = 5, alpha: float = 0.05,
                 spending: str = "obrien_fleming",
                 info_fractions: list[float] | None = None,
                 grid_step: float = 0.02):
        if looks < 1:
            raise ValueError("looks must be >= 1")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if info_fractions is None:
            fractions = [(k + 1) / looks for k in range(looks)]
        else:
            fractions = [float(t) for t in info_fractions]
            if not fractions or any(not 0 < t <= 1 for t in fractions):
                raise ValueError("info_fractions must be non-empty with values in (0, 1]")
            if any(b <= a for a, b in zip(fractions, fractions[1:])):
                raise ValueError("info_fractions must be strictly increasing")
            if fractions[-1] < 1.0:
                fractions.append(1.0)  # 末次分析强制落在信息比例 1，花完全部 alpha
        self.looks = len(fractions)
        self.alpha = alpha
        self.spending = spending
        self.info_fractions = tuple(fractions)
        if spending == "unadjusted":
            u = normal_ppf(1 - alpha / 2)
            self.boundaries_ = (u,) * self.looks
            self.spent = self._actual_spent(grid_step)
        elif spending in SPENDING_FAMILIES:
            self.spent = tuple(alpha_spending(t, alpha, spending) for t in self.info_fractions)
            self.boundaries_ = self._solve_boundaries(grid_step)
        else:
            raise ValueError(
                f"unknown spending: {spending!r}; expected one of {SPENDING_FAMILIES} or 'unadjusted'")

    def critical_value(self, look: int) -> float:
        """第 look 次查看（1 起）的双侧临界值。"""
        if not 1 <= look <= self.looks:
            raise ValueError(f"look must be in [1, {self.looks}]")
        return self.boundaries_[look - 1]

    def decide(self, z: float, look: int) -> tuple[bool, str]:
        """第 look 次查看观察到 z 后的决定：(是否停止, 原因)。"""
        critical = self.critical_value(look)
        if abs(z) >= critical:
            return True, REASON_EFFICACY
        if look == self.looks:
            return True, REASON_MAX_LOOKS
        return False, REASON_CONTINUE

    def _solve_boundaries(self, grid_step: float) -> tuple[float, ...]:
        first = normal_ppf(max(1e-15, min(1 - 1e-15, 1 - self.spent[0] / 2)))
        if self.looks == 1:
            return (first,)
        grid = [i * grid_step for i in range(int(max(6.0, first + 1.0) / grid_step) + 1)]
        density = [normal_pdf(x) for x in grid]
        boundaries = [first]
        for k in range(1, self.looks):
            density = _propagate(density, grid, self.info_fractions[k - 1],
                                 self.info_fractions[k], boundaries[-1], grid_step)
            if self.spent[k] - self.spent[k - 1] < 1e-12:
                boundaries.append(boundaries[-1])  # 消耗增量可忽略时沿用上一临界值
                continue
            cum = _cumulative(density, grid_step)
            boundaries.append(_solve_boundary(cum, grid, grid_step, (1 - self.spent[k]) / 2))
        return tuple(boundaries)

    def _actual_spent(self, grid_step: float) -> tuple[float, ...]:
        """任意边界序列在零假设下的真实累计越界概率（递归积分）。"""
        grid = [i * grid_step for i in range(int(max(6.0, max(self.boundaries_) + 1.0) / grid_step) + 1)]
        density = [normal_pdf(x) for x in grid]
        spent = []
        for k, u in enumerate(self.boundaries_):
            cum = _cumulative(density, grid_step)
            spent.append(1.0 - 2.0 * _interpolate(cum, grid, grid_step, u))
            if k + 1 < self.looks:
                density = _propagate(density, grid, self.info_fractions[k],
                                     self.info_fractions[k + 1], u, grid_step)
        return tuple(spent)


@dataclass(frozen=True)
class LookRecord:
    """单次查看的记录：信息比例、样本量、统计量、临界值与决定原因。"""

    look: int
    info_fraction: float
    n_control: int
    n_treatment: int
    estimate: float
    z: float
    critical: float
    alpha_spent: float
    stop: bool
    reason: str


@dataclass(frozen=True)
class SequentialReport:
    """一次序贯实验的完整结论：逐次记录 + 早停原因 + 最终结论。"""

    design: GroupSequentialDesign
    records: tuple[LookRecord, ...]
    significant: bool
    stop_reason: str
    looks_used: int
    estimate: float
    z: float
    conclusion: str


def monitor(design: GroupSequentialDesign, control: list[float],
            treatment: list[float]) -> SequentialReport:
    """按设计逐次查看累计样本（模拟数据随时间到达的序贯实验）。

    每次查看取两臂样本的前 info_fraction 部分，未越界则继续观察；
    越界以 efficacy 收尾，达到查看上限仍未显著以 max_looks 收尾。
    空样本或样本不足时 z 恒为 0，自然走到 max_looks 并给出
    inconclusive，不会产生 NaN。
    """
    records = []
    for look in range(1, design.looks + 1):
        t = design.info_fractions[look - 1]
        control_slice = control[:math.ceil(t * len(control))]
        treatment_slice = treatment[:math.ceil(t * len(treatment))]
        z, _ = welch_t(treatment_slice, control_slice)
        estimate = mean(treatment_slice) - mean(control_slice)
        stop, reason = design.decide(z, look)
        records.append(LookRecord(
            look=look,
            info_fraction=t,
            n_control=len(control_slice),
            n_treatment=len(treatment_slice),
            estimate=estimate,
            z=z,
            critical=design.critical_value(look),
            alpha_spent=design.spent[look - 1],
            stop=stop,
            reason=reason,
        ))
        if stop:
            break
    final = records[-1]
    significant = final.reason == REASON_EFFICACY
    if significant:
        conclusion = "positive effect" if final.z > 0 else "negative effect"
    else:
        conclusion = "inconclusive"
    return SequentialReport(
        design=design,
        records=tuple(records),
        significant=significant,
        stop_reason=final.reason,
        looks_used=len(records),
        estimate=final.estimate,
        z=final.z,
        conclusion=conclusion,
    )


def z_test(effect: float, standard_error: float, alpha: float = 0.05) -> dict[str, float | bool]:
    """单次（固定 horizon）双侧 z 检验。

    等价于 GroupSequentialDesign(looks=1, alpha=alpha) 的唯一一次查看。
    序贯场景下反复调用会膨胀一类错误，请改用 GroupSequentialDesign +
    monitor。
    """
    if standard_error <= 0:
        return {"z": 0.0, "p_value": 1.0, "stop": False}
    z = effect / standard_error
    p_value = 2 * (1 - normal_cdf(abs(z)))
    critical = normal_ppf(1 - alpha / 2)
    return {"z": z, "p_value": p_value, "stop": abs(z) >= critical}
