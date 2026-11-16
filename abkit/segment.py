"""分层与异质性分析。

总体平均提升为正不代表每个分层都受益，因此本模块把每个分层的估计结果
连同样本量与估计状态一起返回，避免业务方把有偏的局部均值当成全局推荐依据。

估计状态（每个分层、每个处理水平对比各有一份）：
    "ok"             有效分层：对照臂与该处理臂样本量都达到 min_n_per_arm
    "not_estimable"  不可估计分层：对照臂或该处理臂没有任何样本（单臂分层）
    "insufficient_n" 样本不足分层：两臂都有样本，但至少一臂低于 min_n_per_arm

数据质量问题的处理策略（哪些跳过、哪些标记、哪些拒绝整批）：
    行级 —— 跳过该行并计入返回值的 skipped_rows：
        分层键缺失/空白           -> missing_segment
        处理臂标签缺失/空白       -> missing_treatment
        metric 缺失/空白/不可解析 -> missing_metric
        metric 为 inf 或 nan      -> non_finite_metric
    分层级 —— 保留该分层，effect 置 None 并标注 status（不静默丢弃）：
        某臂没有样本              -> not_estimable
        两臂都有但样本不足        -> insufficient_n
    批级 —— 抛出 ValueError 拒绝整批分析（与 cuped/ratio_delta 对退化
    输入抛错的口径一致）：
        清洗后没有可用行
        处理水平不足两个
        显式指定的 control 不在观测到的处理水平中
        min_n_per_arm < 1

输入约束与总体分析对齐：welch_t 要求每臂至少 2 个样本才能估计方差，
因此 min_n_per_arm 默认取 2，只有达到该下限的对比才给出效应估计值。
有限的极端值（哪怕很大）是合法数据，不做截断；只有非有限值会被丢弃。
"""

from __future__ import annotations

import math
from collections import defaultdict

from .stats import mean

STATUS_OK = "ok"
STATUS_NOT_ESTIMABLE = "not_estimable"
STATUS_INSUFFICIENT_N = "insufficient_n"

_SKIP_REASONS = ("missing_segment", "missing_treatment", "missing_metric", "non_finite_metric")


def _category_key(text: str) -> tuple[int, float, str]:
    """类别排序键：可解析为数字的按数值排序，其余按字典序，数字类排在前面。"""
    try:
        return (0, float(text), text)
    except ValueError:
        return (1, 0.0, text)


def _label(raw: object) -> str | None:
    """把分层键/处理臂标签规整成非空字符串；缺失或空白返回 None。"""
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _metric_value(raw: object) -> float | None:
    """解析 metric；缺失、空白或不可解析返回 None（inf/nan 由调用方另行判断）。"""
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _clean_rows(
    rows: list[dict], segment: str, treatment: str, metric: str
) -> tuple[list[tuple[str, str, float]], dict[str, int]]:
    """逐行清洗，返回 (可用行, 跳过计数)。可用行是 (分层, 处理臂, metric) 三元组。"""
    usable: list[tuple[str, str, float]] = []
    skipped = {reason: 0 for reason in _SKIP_REASONS}
    for row in rows:
        seg = _label(row.get(segment))
        if seg is None:
            skipped["missing_segment"] += 1
            continue
        arm = _label(row.get(treatment))
        if arm is None:
            skipped["missing_treatment"] += 1
            continue
        value = _metric_value(row.get(metric))
        if value is None:
            skipped["missing_metric"] += 1
            continue
        if not math.isfinite(value):
            skipped["non_finite_metric"] += 1
            continue
        usable.append((seg, arm, value))
    return usable, skipped


def _contrast_status(n_control: int, n_arm: int, min_n_per_arm: int) -> str:
    if n_control == 0 or n_arm == 0:
        return STATUS_NOT_ESTIMABLE
    if n_control < min_n_per_arm or n_arm < min_n_per_arm:
        return STATUS_INSUFFICIENT_N
    return STATUS_OK


def _segment_status(counts: dict[str, int], levels: list[str], min_n_per_arm: int) -> str:
    """分层级状态取最保守口径：任一臂为空则不可估计，任一臂不足则样本不足。"""
    if any(counts[level] == 0 for level in levels):
        return STATUS_NOT_ESTIMABLE
    if any(counts[level] < min_n_per_arm for level in levels):
        return STATUS_INSUFFICIENT_N
    return STATUS_OK


def group_means(rows: list[dict], segment: str, metric: str) -> dict[str, dict[str, float | int]]:
    """每个分层的样本量与均值，分层按类别排序。

    行级数据问题（分层键缺失/空白、metric 缺失/不可解析、metric 非有限）
    一律跳过该行，与 interaction_effect 的行级策略一致。
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        seg = _label(row.get(segment))
        value = _metric_value(row.get(metric))
        if seg is None or value is None or not math.isfinite(value):
            continue
        groups[seg].append(value)
    return {
        seg: {"n": len(values), "mean": mean(values)}
        for seg, values in sorted(groups.items(), key=lambda item: _category_key(item[0]))
    }


def interaction_effect(
    rows: list[dict],
    segment: str,
    treatment: str,
    metric: str,
    *,
    control: str | None = None,
    min_n_per_arm: int = 2,
) -> dict:
    """按分层估计处理效应（支持两个及以上处理水平）。

    control 默认为类别排序后的第一个处理水平；每个非对照水平都给出一条
    与对照的对比。效应值只对 status 为 "ok" 的对比给出，其余为 None。

    返回结构：
        {
            "metric": ..., "segment": ..., "control": ...,
            "treatment_levels": [...],   # 类别排序后的全部处理水平
            "min_n_per_arm": ...,
            "skipped_rows": {...},       # 行级跳过计数
            "segments": [
                {
                    "segment": ...,
                    "status": "ok" | "not_estimable" | "insufficient_n",
                    "n": {水平: 样本量},   # 所有水平都在，缺失臂为 0
                    "means": {水平: 均值},  # 只包含有样本的臂
                    "contrasts": {非对照水平: {"effect": float | None, "status": ...}},
                },
                ...
            ],
        }
    """
    if min_n_per_arm < 1:
        raise ValueError("min_n_per_arm must be >= 1")
    usable, skipped = _clean_rows(rows, segment, treatment, metric)
    if not usable:
        raise ValueError("no usable rows after cleaning; refusing to analyze an empty batch")
    levels = sorted({arm for _, arm, _ in usable}, key=_category_key)
    if len(levels) < 2:
        raise ValueError(f"need at least 2 treatment levels, got {levels}")
    if control is None:
        control_level = levels[0]
    else:
        control_level = _label(control)
        if control_level is None:
            raise ValueError("control must be a non-empty label")
        if control_level not in levels:
            raise ValueError(f"control level {control_level!r} not among observed levels {levels}")

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seg, arm, value in usable:
        grouped[seg][arm].append(value)

    segments_out = []
    for seg in sorted(grouped, key=_category_key):
        counts = {level: len(grouped[seg].get(level, [])) for level in levels}
        means = {level: mean(values) for level, values in grouped[seg].items()}
        contrasts = {}
        for level in levels:
            if level == control_level:
                continue
            status = _contrast_status(counts[control_level], counts[level], min_n_per_arm)
            effect = means[level] - means[control_level] if status == STATUS_OK else None
            contrasts[level] = {"effect": effect, "status": status}
        segments_out.append(
            {
                "segment": seg,
                "status": _segment_status(counts, levels, min_n_per_arm),
                "n": counts,
                "means": means,
                "contrasts": contrasts,
            }
        )
    return {
        "metric": metric,
        "segment": segment,
        "control": control_level,
        "treatment_levels": levels,
        "min_n_per_arm": min_n_per_arm,
        "skipped_rows": skipped,
        "segments": segments_out,
    }
