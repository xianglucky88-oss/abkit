"""Normal-approximation power and sample-size helpers.

全模块统一口径（双侧、等臂、正态近似）：

    n  = (z_{1-alpha/2} + z_power)^2 * (var_control + var_treatment) / effect^2
    MDE = (z_{1-alpha/2} + z_power) * sqrt((var_control + var_treatment) / n)

`required_per_arm` / `detectable_effect_from_variances` 直接接受每臂方差，
使 CUPED / delta 方法调整后的不确定性可以原样进入规划，而不必折算成
单一标准差。旧的 `required_sample_size` / `detectable_effect` 委托到同一
实现，保证两种入口结果一致。
"""

from __future__ import annotations

import math

from .stats import normal_ppf

# 天花板取整的容差：n_exact 理论上恰为整数时，浮点误差可能落在整数
# 上方几个 ulp，直接 ceil 会多算一个样本；减一个微小 epsilon 再 ceil。
_CEIL_EPS = 1e-9


def _check_alpha_power(alpha: float, power: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")


def _z_sum(alpha: float, power: float) -> float:
    return normal_ppf(1 - alpha / 2) + normal_ppf(power)


def required_per_arm(effect: float, var_control: float, var_treatment: float,
                     alpha: float = 0.05, power: float = 0.8) -> int:
    """达到目标功效所需的每臂样本量（双侧、等臂）。

    effect 按双侧检验取绝对值；effect 为 0 时任何有限样本量都无法达到
    目标功效，抛出 ValueError。取整后用 `detectable_effect_from_variances`
    校验，若因四舍五入边界差一则进一，保证返回值确实达到目标功效。
    """
    _check_alpha_power(alpha, power)
    if var_control < 0 or var_treatment < 0:
        raise ValueError("variances must be non-negative")
    total_var = var_control + var_treatment
    if total_var <= 0:
        raise ValueError("var_control + var_treatment must be positive")
    if effect == 0:
        raise ValueError("effect must be non-zero; a zero effect requires infinite n")
    target = abs(effect)
    z = _z_sum(alpha, power)
    n_exact = z * z * total_var / (target * target)
    n = max(2, int(math.ceil(n_exact - _CEIL_EPS)))
    while detectable_effect_from_variances(n, var_control, var_treatment, alpha, power) > target:
        n += 1
    return n


def detectable_effect_from_variances(n_per_arm: int, var_control: float, var_treatment: float,
                                     alpha: float = 0.05, power: float = 0.8) -> float:
    """等臂 n_per_arm 下达到目标功效的最小 |effect|（MDE）。"""
    _check_alpha_power(alpha, power)
    if n_per_arm < 1:
        raise ValueError("n_per_arm must be positive")
    if var_control < 0 or var_treatment < 0:
        raise ValueError("variances must be non-negative")
    total_var = var_control + var_treatment
    if total_var <= 0:
        return 0.0
    return _z_sum(alpha, power) * math.sqrt(total_var / n_per_arm)


def required_sample_size(effect: float, standard_deviation: float = 1.0,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """Approximate observations per arm for a two-sided test."""
    if effect <= 0 or standard_deviation <= 0:
        raise ValueError("effect and standard_deviation must be positive")
    var = standard_deviation ** 2
    return required_per_arm(effect, var, var, alpha, power)


def detectable_effect(n_per_arm: int, standard_deviation: float = 1.0,
                      alpha: float = 0.05, power: float = 0.8) -> float:
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    var = standard_deviation ** 2
    return detectable_effect_from_variances(n_per_arm, var, var, alpha, power)
