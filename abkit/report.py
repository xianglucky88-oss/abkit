"""Human-readable result reporting."""

from __future__ import annotations


def summarize(metric: str, estimate: float, p_value: float, alpha: float = 0.05) -> dict:
    return {
        "metric": metric,
        "estimate": estimate,
        "p_value": p_value,
        "significant": p_value < alpha,
        "conclusion": "evidence of a difference" if p_value < alpha else "inconclusive",
    }


def markdown(result: dict) -> str:
    decision = "significant" if result["significant"] else "not significant"
    return (f"### {result['metric']}\n\n"
            f"- Estimate: `{result['estimate']:.6g}`\n"
            f"- p-value: `{result['p_value']:.6g}`\n"
            f"- Decision: **{decision}** ({result['conclusion']})")


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def plan_markdown(plan: dict) -> str:
    """把 planning 模块返回的规划字典渲染成中文 markdown 报告。

    报告显式呈现调整前后的效应 / 标准误 / 每臂样本量，以及推荐样本量
    所依赖的不确定性来源（sample_size.basis_detail）。
    """
    kind = "比率指标" if plan["metric_kind"] == "ratio" else "均值指标"
    lines = [f"### 实验规划报告（{kind}）", ""]
    if plan["status"] == "ok":
        lines.append("- 状态：✅ 可规划")
    else:
        lines.append(f"- 状态：⚠️ `{plan['status']}`")
        if plan.get("reason"):
            lines.append(f"- 原因：{plan['reason']}")
    lines.append(f"- 检验口径：双侧 alpha = {plan['alpha']:g}，目标功效 = {plan['power']:g}")
    n_obs = plan.get("n_observed") or {}
    if n_obs:
        lines.append(f"- 当前样本量：control = {n_obs.get('control')}，treatment = {n_obs.get('treatment')}")
    if "ratio" in plan:
        lines.append(f"- 比率：control = {_fmt(plan['ratio']['control'])}，"
                     f"treatment = {_fmt(plan['ratio']['treatment'])}")
    if plan.get("variance_note"):
        lines.append(f"- 注：{plan['variance_note']}")

    var = plan["variance"]
    se = plan["standard_error"]
    ss = plan["sample_size"]
    lines += ["", "**调整前**"]
    lines.append(f"- 效应（treatment - control）：{_fmt(plan['effect']['raw'])}")
    lines.append(f"- 每臂方差：control {_fmt(var.get('control_raw'))} / treatment {_fmt(var.get('treatment_raw'))}")
    lines.append(f"- 标准误（当前样本量）：{_fmt(se.get('raw'))}")
    lines.append(f"- 达到目标功效所需每臂样本量：{_fmt(ss['per_arm_unadjusted'])}")

    if plan["effect"]["adjusted"] is not None:
        lines += ["", "**调整后（CUPED）**"]
        lines.append(f"- 效应：{_fmt(plan['effect']['adjusted'])}")
        lines.append(f"- 每臂方差：control {_fmt(var.get('control_adjusted'))} / "
                     f"treatment {_fmt(var.get('treatment_adjusted'))}")
        lines.append(f"- 标准误（当前样本量）：{_fmt(se.get('adjusted'))}")
        reduction = plan["variance_reduction"]
        lines.append("- 方差缩减：" + ("—" if reduction is None else f"{reduction * 100:.1f}%"))
        lines.append(f"- 达到目标功效所需每臂样本量：{_fmt(ss['per_arm_adjusted'])}")

    lines += ["", "**样本量建议**"]
    if ss["recommended"] is not None:
        basis = "调整后不确定性" if ss["basis"] == "adjusted" else "未调整不确定性"
        lines.append(f"- 推荐每臂样本量：**{ss['recommended']}**（基于{basis}）")
        lines.append(f"- 依据：{ss['basis_detail']}")
        target = abs(plan["effect"]["used_for_planning"])
        lines.append(f"- 校验：n = {ss['recommended']} 时可检测效应 "
                     f"{_fmt(ss['detectable_effect_at_recommended_n'])} ≤ 目标 |effect| = {_fmt(target)}")
    else:
        lines.append("- 无法给出样本量建议（见上方原因）")
    if plan["mde_at_observed_n"] is not None:
        lines.append(f"- 当前样本量下的可检测效应（MDE）：{_fmt(plan['mde_at_observed_n'])}")
    for warning in plan["warnings"]:
        lines.append(f"- ⚠️ {warning}")
    return "\n".join(lines)
