"""分流与样本比例失衡（SRM）检测。"""

from __future__ import annotations

import hashlib

from .stats import normal_cdf


def bucket(user_id: str, salt: str, buckets: int) -> int:
    """把用户稳定地分到某个桶。同样的输入永远得到同样的桶。"""
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def assign(user_ids: list[str], salt: str, buckets: int = 2) -> dict[str, int]:
    """返回每个桶的人数。"""
    counts = {i: 0 for i in range(buckets)}
    for uid in user_ids:
        counts[bucket(uid, salt, buckets)] += 1
    return counts


def srm_p_value(counts: dict[int, int], expected_ratio: list[float] | None = None) -> float:
    """样本比例失衡的 p 值。

    当前实现用两组的比例 z 检验；三组及以上时只用前两组，结果不可靠。
    """
    groups = sorted(counts)
    if len(groups) < 2:
        return 1.0

    a, b = counts[groups[0]], counts[groups[1]]
    n = a + b
    if n == 0:
        return 1.0

    if expected_ratio is None:
        p0 = 0.5
    else:
        p0 = expected_ratio[0]

    se = (p0 * (1 - p0) / n) ** 0.5
    if se == 0:
        return 1.0
    z = (a / n - p0) / se
    return 2 * (1 - normal_cdf(abs(z)))


def is_imbalanced(counts: dict[int, int], threshold: float = 0.001) -> bool:
    """p 值小于阈值时认为分流有问题。"""
    return srm_p_value(counts) < threshold
