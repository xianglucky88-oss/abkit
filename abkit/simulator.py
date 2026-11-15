"""Synthetic experiments with a known treatment effect."""

from __future__ import annotations

import random


def simulate(n_per_arm: int = 200, effect: float = 0.2, seed: int = 7) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    control = [rng.gauss(0.0, 1.0) for _ in range(n_per_arm)]
    treatment = [rng.gauss(effect, 1.0) for _ in range(n_per_arm)]
    return control, treatment


def calibration(repetitions: int = 20, n_per_arm: int = 200, effect: float = 0.2) -> dict[str, float]:
    from .stats import welch_t

    hits = 0
    estimates = []
    for seed in range(repetitions):
        control, treatment = simulate(n_per_arm, effect, seed)
        _, p_value = welch_t(treatment, control)
        hits += p_value < 0.05
        estimates.append(sum(treatment) / len(treatment) - sum(control) / len(control))
    return {"rejection_rate": hits / max(repetitions, 1), "mean_estimate": sum(estimates) / max(len(estimates), 1)}
