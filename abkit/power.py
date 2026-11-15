"""Normal-approximation power and sample-size helpers."""

from __future__ import annotations

from .stats import normal_ppf


def required_sample_size(effect: float, standard_deviation: float = 1.0,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """Approximate observations per arm for a two-sided test."""
    if effect <= 0 or standard_deviation <= 0 or not 0 < power < 1:
        raise ValueError("effect, standard_deviation must be positive; power in (0, 1)")
    z_alpha = normal_ppf(1 - alpha / 2)
    z_power = normal_ppf(power)
    n = ((z_alpha + z_power) * standard_deviation / effect) ** 2 * 2
    return max(2, int(n + 0.999999))


def detectable_effect(n_per_arm: int, standard_deviation: float = 1.0,
                      alpha: float = 0.05, power: float = 0.8) -> float:
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    return (normal_ppf(1 - alpha / 2) + normal_ppf(power)) * standard_deviation * (2 / n_per_arm) ** 0.5
