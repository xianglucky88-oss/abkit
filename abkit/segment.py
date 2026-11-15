"""Segment-level summaries and a small interaction helper."""

from __future__ import annotations

from collections import defaultdict

from .stats import mean


def group_means(rows: list[dict], segment: str, metric: str) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[segment])].append(float(row[metric]))
    return {key: mean(values) for key, values in groups.items()}


def interaction_effect(rows: list[dict], segment: str, treatment: str, metric: str) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row[segment]), str(row[treatment]))].append(float(row[metric]))
    levels = sorted({key[1] for key in grouped})
    if len(levels) < 2:
        return {}
    control, treated = levels[:2]
    return {
        seg: mean(grouped[(seg, treated)]) - mean(grouped[(seg, control)])
        for seg in sorted({key[0] for key in grouped})
        if (seg, treated) in grouped and (seg, control) in grouped
    }
