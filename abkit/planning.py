"""实验规划流程：方差缩减 → 不确定性估计 → 样本量 / 可检测效应。

把 CUPED、比率指标的 delta 方法与功效计算串成一条口径一致的链路：

    effect = mean_t - mean_c            （均值指标）
           = ratio_t - ratio_c          （比率指标，delta 方法）
    SE     = sqrt(var_c / n_c + var_t / n_t)     # 调整后口径替换为调整后每臂方差
    n      = (z_{1-alpha/2} + z_power)^2 * (var_c + var_t) / effect^2   # 等臂规划
    MDE    = (z_{1-alpha/2} + z_power) * SE

与"把 CUPED 输出直接塞进 required_sample_size"不同，这里样本量建议
显式记录所依赖的不确定性来源：返回字典的 sample_size.basis /
basis_detail 会说明推荐值基于调整后还是未调整的每臂方差，并给出
调整前后两套 effect / SE / 样本量供对照。

退化情形（样本不足、分母全零、效应为零、方差为零）不会抛出异常或
返回看似正常的数字，而是以 status + reason 的形式交给计划层解释。
"""

from __future__ import annotations

import math

from .power import detectable_effect_from_variances, required_per_arm
from .stats import mean, normal_ppf, variance
from .variance import cuped, ratio_estimate

MIN_ARM_SIZE = 2
# 方差缩减比例低于该值时视为"没有实质缩减"，按未调整口径规划，
# 避免报告声称依赖了一个数值上并不存在的"调整后不确定性"。
_REDUCTION_EPS = 1e-12


def _z_sum(alpha: float, power: float) -> float:
    return normal_ppf(1 - alpha / 2) + normal_ppf(power)


def _check_alpha_power(alpha: float, power: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")


def _new_plan(metric_kind: str, alpha: float, power: float) -> dict:
    return {
        "status": "ok",
        "metric_kind": metric_kind,
        "alpha": alpha,
        "power": power,
        "n_observed": {},
        "effect": {"raw": None, "adjusted": None, "used_for_planning": None},
        "variance": {},
        "standard_error": {},
        "variance_reduction": None,
        "mde_at_observed_n": None,
        "sample_size": {
            "per_arm_unadjusted": None,
            "per_arm_adjusted": None,
            "recommended": None,
            "basis": None,
            "basis_detail": None,
            "se_at_recommended_n": None,
            "detectable_effect_at_recommended_n": None,
        },
        "warnings": [],
        "reason": None,
    }


def plan_mean_metric(control: list[float], treatment: list[float],
                     control_baseline: list[float] | None = None,
                     treatment_baseline: list[float] | None = None,
                     alpha: float = 0.05, power: float = 0.8) -> dict:
    """均值指标规划：可选 CUPED，输出调整前后口径与样本量建议。

    baseline 为实验前协变量，按位置与观测对齐；长度不一致或只有一臂
    提供时回退为未调整分析，并在 warnings 中说明。
    """
    _check_alpha_power(alpha, power)
    control, treatment = list(control), list(treatment)
    plan = _new_plan("mean", alpha, power)
    nc, nt = len(control), len(treatment)
    plan["n_observed"] = {"control": nc, "treatment": nt}
    if nc < MIN_ARM_SIZE or nt < MIN_ARM_SIZE:
        plan["status"] = "insufficient_data"
        plan["reason"] = (f"每臂至少需要 {MIN_ARM_SIZE} 个观测才能估计方差"
                          f"（control={nc}, treatment={nt}）")
        return plan
    var_c, var_t = variance(control), variance(treatment)
    plan["effect"]["raw"] = mean(treatment) - mean(control)
    plan["variance"]["control_raw"] = var_c
    plan["variance"]["treatment_raw"] = var_t
    plan["standard_error"]["raw"] = math.sqrt(var_c / nc + var_t / nt)

    adjusted = _mean_cuped(control, treatment, control_baseline, treatment_baseline,
                           plan["warnings"])
    if adjusted is not None:
        adj_c, adj_t, theta = adjusted
        var_c_adj, var_t_adj = variance(adj_c), variance(adj_t)
        plan["effect"]["adjusted"] = mean(adj_t) - mean(adj_c)
        plan["variance"]["control_adjusted"] = var_c_adj
        plan["variance"]["treatment_adjusted"] = var_t_adj
        plan["variance"]["theta"] = theta
        plan["standard_error"]["adjusted"] = math.sqrt(var_c_adj / nc + var_t_adj / nt)
        raw_sum = var_c + var_t
        if raw_sum > 0:
            plan["variance_reduction"] = 1.0 - (var_c_adj + var_t_adj) / raw_sum
    return _finish(plan, nc, nt)


def plan_ratio_metric(control_numerator: list[float], control_denominator: list[float],
                      treatment_numerator: list[float], treatment_denominator: list[float],
                      control_baseline: tuple[list[float], list[float]] | None = None,
                      treatment_baseline: tuple[list[float], list[float]] | None = None,
                      alpha: float = 0.05, power: float = 0.8) -> dict:
    """比率指标规划：delta 方法估计不确定性，可选对线性化项做 CUPED。

    baseline 为 (分子, 分母) 二元组，与实验期数据按位置对齐；提供时对
    线性化项 n_i - r·d_i 做 CUPED（r 为各臂自身比率，与 delta 方法同口径；
    协变量取实验前的线性化项）。分母全零 / 接近零时返回
    status="invalid_metric" 并给出可供计划层展示的原因。
    """
    _check_alpha_power(alpha, power)
    plan = _new_plan("ratio", alpha, power)
    est_c = ratio_estimate(control_numerator, control_denominator)
    est_t = ratio_estimate(treatment_numerator, treatment_denominator)
    plan["n_observed"] = {"control": est_c.n, "treatment": est_t.n}
    problems = []
    if not est_c.ok:
        problems.append(f"control 臂：{est_c.reason}")
    if not est_t.ok:
        problems.append(f"treatment 臂：{est_t.reason}")
    if problems:
        plan["status"] = "invalid_metric"
        plan["reason"] = "；".join(problems)
        return plan
    nc, nt = est_c.n, est_t.n
    plan["ratio"] = {"control": est_c.ratio, "treatment": est_t.ratio}
    plan["effect"]["raw"] = est_t.ratio - est_c.ratio
    var_c, var_t = est_c.unit_variance, est_t.unit_variance
    plan["variance"]["control_raw"] = var_c
    plan["variance"]["treatment_raw"] = var_t
    plan["standard_error"]["raw"] = math.sqrt(var_c / nc + var_t / nt)
    plan["variance_note"] = "方差为 delta 方法线性化项的每观测方差 Var(n_i - r·d_i) / mean(d)^2"

    adjusted = _ratio_cuped((control_numerator, control_denominator),
                            (treatment_numerator, treatment_denominator),
                            control_baseline, treatment_baseline, plan["warnings"])
    if adjusted is not None:
        var_c_adj, var_t_adj, effect_adj, theta = adjusted
        plan["effect"]["adjusted"] = effect_adj
        plan["variance"]["control_adjusted"] = var_c_adj
        plan["variance"]["treatment_adjusted"] = var_t_adj
        plan["variance"]["theta"] = theta
        plan["standard_error"]["adjusted"] = math.sqrt(var_c_adj / nc + var_t_adj / nt)
        raw_sum = var_c + var_t
        if raw_sum > 0:
            plan["variance_reduction"] = 1.0 - (var_c_adj + var_t_adj) / raw_sum
    return _finish(plan, nc, nt)


def _mean_cuped(control, treatment, control_baseline, treatment_baseline, warnings):
    """pooled theta 的 CUPED；返回 (adj_c, adj_t, theta)，不可用时返回 None。"""
    if control_baseline is None and treatment_baseline is None:
        return None
    if control_baseline is None or treatment_baseline is None:
        warnings.append("只有一臂提供 baseline，无法应用 CUPED，按未调整方差规划")
        return None
    if len(control_baseline) != len(control) or len(treatment_baseline) != len(treatment):
        warnings.append(
            f"baseline 长度与观测不一致（control: {len(control_baseline)} vs {len(control)}，"
            f"treatment: {len(treatment_baseline)} vs {len(treatment)}），回退为未调整分析")
        return None
    pooled_y = control + treatment
    pooled_x = list(control_baseline) + list(treatment_baseline)
    adjusted, theta = cuped(pooled_y, pooled_x)
    if theta == 0.0:
        warnings.append("协变量无方差（或与结局不相关），CUPED 未带来方差缩减")
    return adjusted[:len(control)], adjusted[len(control):], theta


def _ratio_cuped(control, treatment, control_baseline, treatment_baseline, warnings):
    """对线性化项做 CUPED；返回 (var_c_adj, var_t_adj, effect_adj, theta) 或 None。"""
    if control_baseline is None and treatment_baseline is None:
        return None
    if control_baseline is None or treatment_baseline is None:
        warnings.append("只有一臂提供 baseline，无法应用 CUPED，按未调整方差规划")
        return None
    (c_num, c_den), (t_num, t_den) = control, treatment
    c_num0, c_den0 = control_baseline
    t_num0, t_den0 = treatment_baseline
    if (len(c_num0) != len(c_num) or len(c_den0) != len(c_num)
            or len(t_num0) != len(t_num) or len(t_den0) != len(t_num)):
        warnings.append(
            f"baseline 长度与观测不一致（control: {len(c_num0)}/{len(c_den0)} vs {len(c_num)}，"
            f"treatment: {len(t_num0)}/{len(t_den0)} vs {len(t_num)}），回退为未调整分析")
        return None
    # 口径与 ratio_estimate 一致：线性化项按各臂自身比率构造，
    # 保证调整前后方差可比；协变量完全来自实验前数据（避免使用实验期信息）。
    if sum(c_den0) == 0 or sum(t_den0) == 0:
        warnings.append("baseline 分母之和为 0，无法构造线性化协变量，回退为未调整分析")
        return None
    r_c = sum(c_num) / sum(c_den)
    r_t = sum(t_num) / sum(t_den)
    r0_c = sum(c_num0) / sum(c_den0)
    r0_t = sum(t_num0) / sum(t_den0)
    terms_c = [n - r_c * d for n, d in zip(c_num, c_den)]
    terms_t = [n - r_t * d for n, d in zip(t_num, t_den)]
    x_c = [n - r0_c * d for n, d in zip(c_num0, c_den0)]
    x_t = [n - r0_t * d for n, d in zip(t_num0, t_den0)]
    adjusted, theta = cuped(terms_c + terms_t, x_c + x_t)
    if theta == 0.0:
        warnings.append("协变量无方差（或与线性化项不相关），CUPED 未带来方差缩减")
    adj_c, adj_t = adjusted[:len(terms_c)], adjusted[len(terms_c):]
    md_c, md_t = mean(c_den), mean(t_den)
    var_c_adj = variance(adj_c) / (md_c * md_c)
    var_t_adj = variance(adj_t) / (md_t * md_t)
    # 调整后的比率估计：r_adj = r - theta·(mean(x_arm) - mean(x_pool)) / mean(d_arm)
    mx_pool = mean(x_c + x_t)
    r_c_adj = sum(c_num) / sum(c_den) - theta * (mean(x_c) - mx_pool) / md_c
    r_t_adj = sum(t_num) / sum(t_den) - theta * (mean(x_t) - mx_pool) / md_t
    return var_c_adj, var_t_adj, r_t_adj - r_c_adj, theta


def _finish(plan: dict, nc: int, nt: int) -> dict:
    """统一的规划收尾：选口径、算 MDE、算每臂样本量并记录依据。"""
    alpha, power = plan["alpha"], plan["power"]
    reduction = plan["variance_reduction"]
    use_adjusted = reduction is not None and reduction > _REDUCTION_EPS
    if use_adjusted:
        var_c = plan["variance"]["control_adjusted"]
        var_t = plan["variance"]["treatment_adjusted"]
        effect = plan["effect"]["adjusted"]
        basis = "adjusted"
    else:
        var_c = plan["variance"]["control_raw"]
        var_t = plan["variance"]["treatment_raw"]
        effect = plan["effect"]["raw"]
        basis = "unadjusted"

    se_observed = math.sqrt(var_c / nc + var_t / nt)
    plan["mde_at_observed_n"] = _z_sum(alpha, power) * se_observed

    ss = plan["sample_size"]
    # 调整前后两套口径都给出每臂样本量，便于报告说明二者关系。
    effect_raw = plan["effect"]["raw"]
    raw_c, raw_t = plan["variance"]["control_raw"], plan["variance"]["treatment_raw"]
    if effect_raw != 0 and raw_c + raw_t > 0:
        ss["per_arm_unadjusted"] = required_per_arm(effect_raw, raw_c, raw_t, alpha, power)
    effect_adj = plan["effect"]["adjusted"]
    if effect_adj is not None and effect_adj != 0:
        adj_c = plan["variance"]["control_adjusted"]
        adj_t = plan["variance"]["treatment_adjusted"]
        if adj_c + adj_t > 0:
            ss["per_arm_adjusted"] = required_per_arm(effect_adj, adj_c, adj_t, alpha, power)

    if var_c + var_t <= 0:
        plan["status"] = "zero_variance"
        plan["reason"] = "用于规划的方差为 0，无法估计不确定性（数据退化或与协变量完全相关）"
        return plan
    if effect == 0:
        plan["status"] = "zero_effect"
        plan["reason"] = ("观测效应为 0：任何有限样本量都无法达到目标功效。"
                          "可参考 mde_at_observed_n（当前样本量下的可检测效应）重新设定目标效应")
        return plan

    plan["effect"]["used_for_planning"] = effect
    if effect < 0:
        plan["warnings"].append("观测效应为负，按双侧检验以其绝对值规划样本量")
    target = abs(effect)
    n = required_per_arm(target, var_c, var_t, alpha, power)
    ss["recommended"] = n
    ss["basis"] = basis
    ss["se_at_recommended_n"] = math.sqrt((var_c + var_t) / n)
    ss["detectable_effect_at_recommended_n"] = detectable_effect_from_variances(
        n, var_c, var_t, alpha, power)
    label = "调整后" if basis == "adjusted" else "未调整（原始）"
    ss["basis_detail"] = (
        f"{label}每臂方差 control={var_c:.6g}、treatment={var_t:.6g}，"
        f"SE = sqrt({var_c:.6g}/n + {var_t:.6g}/n)；"
        f"alpha={alpha}、power={power}、目标 |effect|={target:.6g}")
    return plan
