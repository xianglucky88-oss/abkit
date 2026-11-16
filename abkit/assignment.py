"""分流与样本比例失衡（SRM）检测，支持任意组数与指定期望比例。

设计要点：
    - ``bucket`` 保留底层稳定哈希原语：salt / 桶数 / 用户 ID 不变时落桶确定；
    - 不等比例分流通过「大空间哈希槽 -> 臂」的确定性映射实现，配置不变则
      映射不变，同一用户永远落入同一臂；
    - SRM 检验用皮尔逊卡方拟合优度，天然支持任意组数（两组时与双侧 z 检验
      等价），并返回逐臂的调整后标准化残差用于定位具体偏离的臂；
    - 检测结论区分三类问题：分流异常（imbalance）、样本不足
      （insufficient_sample）、期望比例配置错误（config_error）。
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from math import sqrt

from .stats import chi2_sf

# 哈希取模空间：64 位。底层原语的桶数上限。
MAX_BUCKETS = 1 << 32
# 不等比例映射使用的虚拟槽数量，量化误差约为 1/SLOT_COUNT。
SLOT_COUNT = 1_000_000
# 卡方检验对单格期望频数的经验下限，低于该值时检验不可靠。
MIN_EXPECTED = 5.0
# 调整后标准化残差的 0.001 显著临界值（|z| 超过它说明该臂明显偏离）。
RESIDUAL_CRIT = 3.29


# --------------------------------------------------------------------------- #
# 期望比例归一化
# --------------------------------------------------------------------------- #
def normalize_ratios(expected_ratio: list[float] | tuple[float, ...] | None,
                     buckets: int) -> list[float]:
    """把期望比例归一化为长度等于桶数、和为 1 的概率向量。

    允许正整数权重（如 [1, 2, 1]）或任意非负比例。全零、负数、长度不匹配、
    NaN/Inf 都属于期望比例配置错误，抛 ``ValueError``。
    """
    if expected_ratio is None:
        return [1.0 / buckets] * buckets
    ratios = list(expected_ratio)
    if len(ratios) != buckets:
        raise ValueError(
            f"expected_ratio length {len(ratios)} does not match buckets {buckets}"
        )
    cleaned: list[float] = []
    for value in ratios:
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError("expected_ratio contains NaN or infinite values")
        if number < 0:
            raise ValueError("expected_ratio must be non-negative")
        cleaned.append(number)
    total = sum(cleaned)
    if total <= 0:
        raise ValueError("expected_ratio must have a positive total")
    return [value / total for value in cleaned]


# --------------------------------------------------------------------------- #
# 稳定分桶
# --------------------------------------------------------------------------- #
def _validate_buckets(buckets: int) -> int:
    if isinstance(buckets, bool) or not isinstance(buckets, int):
        raise TypeError("buckets must be an int")
    if not 1 <= buckets <= MAX_BUCKETS:
        raise ValueError(f"buckets must be in [1, {MAX_BUCKETS}]")
    return buckets


def bucket(user_id: str, salt: str, buckets: int) -> int:
    """把用户稳定地分到某个桶。同样的输入永远得到同样的桶。

    直接对 ``buckets`` 取模，因此等比例分流时各桶期望频数完全相等，
    不存在量化误差。``buckets`` 必须在 ``[1, 2^32]`` 内。
    """
    _validate_buckets(buckets)
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def _slot_boundaries(probabilities: list[float]) -> list[int]:
    """概率向量 -> 每个臂独占的哈希槽区间边界（长度为 buckets+1）。

    每个正概率臂至少保证 1 个槽，最后一个臂吞掉取整余量，
    因而 ``boundaries[-1] == SLOT_COUNT`` 恒成立。
    """
    boundaries = [0]
    allocated = 0
    positive = sum(1 for p in probabilities if p > 0.0)
    if positive > SLOT_COUNT:
        raise ValueError(
            f"too many positive-probability arms ({positive}) for {SLOT_COUNT} slots"
        )
    for index, probability in enumerate(probabilities[:-1]):
        if probability > 0.0:
            width = max(1, round(probability * SLOT_COUNT))
            # 给后面的正概率臂各留至少一个槽
            remaining_positive = sum(1 for p in probabilities[index + 1:] if p > 0.0)
            width = min(width, SLOT_COUNT - allocated - remaining_positive)
        else:
            width = 0
        allocated += width
        boundaries.append(allocated)
    boundaries.append(SLOT_COUNT)
    return boundaries


def arm_bucket(user_id: str, salt: str, buckets: int,
               expected_ratio: list[float] | None = None) -> int:
    """按指定期望比例把用户稳定分到某个臂。

    - 不给 ``expected_ratio``（或所有臂等概率）时退化为 :func:`bucket`，
      与历史行为完全一致；
    - 不等比例时用户先哈希到 :data:`SLOT_COUNT` 个虚拟槽，再通过确定性的
      槽区间映射到臂。salt、桶数、配置不变时结果稳定。
    """
    _validate_buckets(buckets)
    probabilities = normalize_ratios(expected_ratio, buckets)
    return _resolve_arm(user_id, salt, buckets, probabilities)


def _resolve_arm(user_id: str, salt: str, buckets: int,
                 probabilities: list[float]) -> int:
    """内部：基于已归一化概率向量的稳定落臂。"""
    if all(abs(p - 1.0 / buckets) <= 1e-12 for p in probabilities):
        return bucket(user_id, salt, buckets)
    slot = bucket(user_id, f"{salt}::ratio::{SLOT_COUNT}", SLOT_COUNT)
    boundaries = _slot_boundaries(probabilities)
    # boundaries 是各臂起点（长度 k+1），bisect_right - 1 即 slot 所属臂
    return max(0, min(bisect_right(boundaries, slot) - 1, buckets - 1))


def assign(user_ids: list[str], salt: str, buckets: int = 2,
           expected_ratio: list[float] | None = None,
           dedupe: bool = True) -> dict[int, int]:
    """返回每个桶的人数（桶编号 0..buckets-1 一定齐全，零样本桶计数为 0）。

    ``dedupe=True`` 时同一用户 ID 只计一次（实验平台的分流对象是去重用户）；
    重复 ID 的数量可通过 :func:`assign_detail` 获取。
    """
    return assign_detail(user_ids, salt, buckets, expected_ratio, dedupe)["counts"]


def assign_detail(user_ids: list[str], salt: str, buckets: int = 2,
                  expected_ratio: list[float] | None = None,
                  dedupe: bool = True) -> dict:
    """分桶并附带可解释的过程信息。

    返回字段：
        counts            各桶去重后人数
        duplicates        被去重丢弃的重复用户 ID 次数
        unique_users      去重后的用户数
        buckets           桶数
        expected_ratio    归一化后的期望比例
        assignments       每个（去重后）用户的落桶映射
    """
    _validate_buckets(buckets)
    probabilities = normalize_ratios(expected_ratio, buckets)
    equal_split = all(abs(p - 1.0 / buckets) <= 1e-12 for p in probabilities)
    # 槽区间只依赖配置，预计算一次；每个用户用二分查找 O(log buckets)
    boundaries = None if equal_split else _slot_boundaries(probabilities)
    counts = {i: 0 for i in range(buckets)}
    assignments: dict[str, int] = {}
    duplicates = 0
    seen: set[str] = set()
    for uid in user_ids:
        key = str(uid)
        if dedupe:
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
        if equal_split:
            arm = bucket(key, salt, buckets)
        else:
            slot = bucket(key, f"{salt}::ratio::{SLOT_COUNT}", SLOT_COUNT)
            # boundaries 形如 [0, b1, ..., S]，定位 slot 所属区间
            arm = max(0, min(bisect_right(boundaries, slot) - 1, buckets - 1))
        assignments[key] = arm
        counts[arm] += 1
    return {
        "counts": counts,
        "duplicates": duplicates,
        "unique_users": len(assignments),
        "buckets": buckets,
        "expected_ratio": probabilities,
        "assignments": assignments,
    }


# --------------------------------------------------------------------------- #
# SRM 统计检验
# --------------------------------------------------------------------------- #
def _aligned_counts(counts: dict[int, int], buckets: int) -> list[int]:
    """把计数字典对齐成长度为 buckets 的有序向量，并做输入一致性校验。"""
    observed: list[int] = []
    for arm in range(buckets):
        value = counts.get(arm, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"count for bucket {arm} must be a non-negative int")
        observed.append(value)
    extra = set(counts) - set(range(buckets))
    if extra:
        raise ValueError(f"counts contain buckets outside [0, {buckets}): {sorted(extra)}")
    return observed


def srm_p_value(counts: dict[int, int],
                expected_ratio: list[float] | None = None) -> float:
    """多臂样本比例失衡的卡方拟合优度 p 值。

    组数任意：统计量 ``sum((O-E)^2/E)`` 服从 ``df = k-1`` 的卡方分布
    （两组等比例时与历史的双侧 z 检验完全等价）。零概率臂若出现样本，
    直接判定 p=0。
    """
    if not counts:
        raise ValueError("counts must not be empty")
    buckets = max(counts) + 1
    observed = _aligned_counts(counts, buckets)
    probabilities = normalize_ratios(expected_ratio, buckets)
    total = sum(observed)
    if total == 0 or buckets < 2:
        return 1.0
    # 零概率臂不允许出现任何样本
    for value, probability in zip(observed, probabilities):
        if probability == 0.0 and value > 0:
            return 0.0
    statistic = 0.0
    for value, probability in zip(observed, probabilities):
        if probability <= 0.0:
            continue
        expected = total * probability
        statistic += (value - expected) ** 2 / expected
    return chi2_sf(statistic, buckets - 1)


def is_imbalanced(counts: dict[int, int], threshold: float = 0.001,
                  expected_ratio: list[float] | None = None) -> bool:
    """p 值小于阈值时认为分流有问题。"""
    return srm_p_value(counts, expected_ratio) < threshold


# --------------------------------------------------------------------------- #
# 端到端检测与可解释告警
# --------------------------------------------------------------------------- #
def _config_error(reason: str, buckets: int | None = None,
                  expected_ratio: list[float] | None = None) -> dict:
    return {
        "status": "config_error",
        "reason": reason,
        "buckets": buckets,
        "expected_ratio": expected_ratio,
        "total": 0,
        "statistic": None,
        "p_value": None,
        "alpha": None,
        "min_expected": MIN_EXPECTED,
        "expected_ok": None,
        "zero_sample_arms": [],
        "zero_probability_arms": [],
        "deviating_arms": [],
        "per_arm": [],
        "alert": f"期望比例配置错误：{reason}，请先修正实验配置后再做分流诊断。",
    }


def srm_report(counts: dict[int, int],
               expected_ratio: list[float] | None = None,
               alpha: float = 0.001,
               min_expected: float = MIN_EXPECTED) -> dict:
    """从分桶计数生成结构化 SRM 诊断报告。

    状态（``status``）四选一：
        config_error          期望比例 / 计数输入本身不合法
        insufficient_sample   样本量不足以支撑卡方检验（含零样本桶）
        imbalance             检验显著，存在真实分流异常，并定位到具体臂
        ok                    未发现失衡
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if min_expected <= 0:
        raise ValueError("min_expected must be positive")
    if not counts:
        return _config_error("counts 为空")

    buckets = max(counts) + 1
    try:
        probabilities = normalize_ratios(expected_ratio, buckets)
        observed = _aligned_counts(counts, buckets)
    except (ValueError, TypeError) as exc:
        return _config_error(str(exc), buckets,
                             list(expected_ratio) if expected_ratio is not None else None)

    total = sum(observed)
    zero_probability_arms = [arm for arm, p in enumerate(probabilities) if p == 0.0]

    # 零概率臂出现样本：分流与配置直接冲突
    violated = [arm for arm in zero_probability_arms if observed[arm] > 0]
    if violated:
        reason = f"零概率臂 {violated} 出现了样本，与期望比例互斥"
        return {
            "status": "config_error",
            "reason": reason,
            "buckets": buckets,
            "expected_ratio": probabilities,
            "total": total,
            "statistic": None,
            "p_value": 0.0,
            "alpha": alpha,
            "min_expected": min_expected,
            "expected_ok": False,
            "zero_sample_arms": [arm for arm, value in enumerate(observed) if value == 0],
            "zero_probability_arms": zero_probability_arms,
            "deviating_arms": violated,
            "per_arm": [
                {
                    "arm": arm,
                    "observed": observed[arm],
                    "expected": total * probabilities[arm],
                    "expected_ratio": probabilities[arm],
                    "observed_ratio": observed[arm] / total if total else 0.0,
                    "adjusted_residual": None,
                    "deviates": arm in violated,
                    "zero_sample": observed[arm] == 0,
                }
                for arm in range(buckets)
            ],
            "alert": f"期望比例配置错误：{reason}。",
        }

    if total == 0:
        return {
            "status": "insufficient_sample",
            "reason": "所有桶计数均为 0，没有任何样本可用于检验",
            "buckets": buckets,
            "expected_ratio": probabilities,
            "total": 0,
            "statistic": 0.0,
            "p_value": None,
            "alpha": alpha,
            "min_expected": min_expected,
            "expected_ok": True,
            "zero_sample_arms": list(range(buckets)),
            "zero_probability_arms": zero_probability_arms,
            "deviating_arms": [],
            "per_arm": [
                {
                    "arm": arm,
                    "observed": 0,
                    "expected": 0.0,
                    "expected_ratio": probabilities[arm],
                    "observed_ratio": 0.0,
                    "adjusted_residual": None,
                    "deviates": False,
                    "zero_sample": True,
                }
                for arm in range(buckets)
            ],
            "alert": "样本不足：输入中没有任何用户，暂不能做分流诊断。",
        }

    expected_values = [total * p for p in probabilities]
    zero_sample_arms = [arm for arm, value in enumerate(observed) if value == 0]
    sparse_arms = [
        arm for arm, expected in enumerate(expected_values)
        if probabilities[arm] > 0.0 and expected < min_expected
    ]

    per_arm: list[dict] = []
    statistic = 0.0
    for arm, (value, probability, expected) in enumerate(
            zip(observed, probabilities, expected_values)):
        if probability > 0.0:
            statistic += (value - expected) ** 2 / expected
            # 调整后标准化残差：跨臂可比，近似标准正态
            residual = ((value - expected)
                        / sqrt(total * probability * (1.0 - probability)))
        else:
            residual = None
        per_arm.append({
            "arm": arm,
            "observed": value,
            "expected": expected,
            "expected_ratio": probability,
            "observed_ratio": value / total,
            "adjusted_residual": residual,
            "deviates": False,
            "zero_sample": value == 0,
        })

    # 样本不足：零样本桶或正概率臂期望频数过低，卡方近似不可靠
    if zero_sample_arms or sparse_arms:
        arms = sorted(set(zero_sample_arms) | set(sparse_arms))
        return {
            "status": "insufficient_sample",
            "reason": (f"桶 {arms} 为零样本或期望频数 < {min_expected:g}，"
                       "卡方检验不可靠，需继续积累样本"),
            "buckets": buckets,
            "expected_ratio": probabilities,
            "total": total,
            "statistic": statistic,
            "p_value": None,
            "alpha": alpha,
            "min_expected": min_expected,
            "expected_ok": True,
            "zero_sample_arms": zero_sample_arms,
            "zero_probability_arms": zero_probability_arms,
            "deviating_arms": [],
            "per_arm": per_arm,
            "alert": (f"样本不足：共 {total} 个样本，桶 {arms} 的期望频数过低"
                      f"（阈值 {min_expected:g}），暂不能下分流结论。"),
        }

    if buckets < 2:
        p_value = 1.0
    else:
        p_value = chi2_sf(statistic, buckets - 1)

    ranked = sorted(
        (entry for entry in per_arm if entry["adjusted_residual"] is not None),
        key=lambda entry: abs(entry["adjusted_residual"]),
        reverse=True,
    )
    deviating = [
        entry["arm"] for entry in ranked
        if abs(entry["adjusted_residual"]) >= RESIDUAL_CRIT
    ]
    for entry in per_arm:
        entry["deviates"] = entry["arm"] in deviating

    if p_value < alpha:
        status = "imbalance"
        detail = "、".join(
            f"臂 {entry['arm']}（观测 {entry['observed']} / 期望 "
            f"{entry['expected']:.1f}，残差 {entry['adjusted_residual']:+.2f}）"
            for entry in ranked if entry["arm"] in deviating
        ) or "各臂残差均未超临界值，但整体卡方显著"
        alert = (f"分流异常：卡方={statistic:.4f}，p={p_value:.3g} < alpha={alpha:g}，"
                 f"共 {buckets} 个臂、{total} 个样本；偏离最大的臂：{detail}。")
    else:
        status = "ok"
        top = ranked[0] if ranked else None
        alert = (
            f"分流正常：卡方={statistic:.4f}，p={p_value:.3g} >= alpha={alpha:g}，"
            f"共 {buckets} 个臂、{total} 个样本。"
            + ("" if top is None else
               f" 波动最大的是臂 {top['arm']}（残差 {top['adjusted_residual']:+.2f}）。")
        )

    return {
        "status": status,
        "reason": None if status == "ok" else "卡方拟合优度检验显著",
        "buckets": buckets,
        "expected_ratio": probabilities,
        "total": total,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "min_expected": min_expected,
        "expected_ok": True,
        "zero_sample_arms": [],
        "zero_probability_arms": zero_probability_arms,
        "deviating_arms": deviating,
        "per_arm": per_arm,
        "alert": alert,
    }


def run_srm_check(user_ids: list[str], salt: str, buckets: int = 2,
                  expected_ratio: list[float] | None = None,
                  alpha: float = 0.001,
                  min_expected: float = MIN_EXPECTED,
                  dedupe: bool = True) -> dict:
    """端到端流程：用户 ID -> 稳定分桶 -> 卡方 SRM 诊断 -> 可解释告警。

    配置类错误（桶数非法、比例长度不匹配、比例全零等）不会抛出异常，
    而是返回 ``status == "config_error"`` 的报告，与分流异常、样本不足
    三类结论在同一结构中区分开。
    """
    try:
        detail = assign_detail(user_ids, salt, buckets, expected_ratio, dedupe)
    except (ValueError, TypeError) as exc:
        return _config_error(str(exc),
                             buckets if isinstance(buckets, int) else None,
                             list(expected_ratio) if expected_ratio is not None else None)
    report = srm_report(detail["counts"], detail["expected_ratio"], alpha, min_expected)
    report["salt"] = salt
    report["duplicates"] = detail["duplicates"]
    report["unique_users"] = detail["unique_users"]
    report["assignments"] = detail["assignments"]
    if detail["duplicates"]:
        report["alert"] += f" 注意：检测到 {detail['duplicates']} 条重复用户 ID，已按去重口径统计。"
    return report
