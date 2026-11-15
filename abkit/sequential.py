"""Small sequential-testing primitives."""

from __future__ import annotations

from .stats import normal_cdf, normal_ppf


def alpha_spending(info_fraction: float, alpha: float = 0.05) -> float:
    """Pocock-like spending used by the initial implementation."""
    if not 0 < info_fraction <= 1 or not 0 < alpha < 1:
        raise ValueError("info_fraction must be in (0, 1] and alpha in (0, 1)")
    return alpha * info_fraction


def z_test(effect: float, standard_error: float, alpha: float = 0.05) -> dict[str, float | bool]:
    """Return a two-sided normal test with a simple alpha-spending rule."""
    if standard_error <= 0:
        return {"z": 0.0, "p_value": 1.0, "stop": False}
    z = effect / standard_error
    p_value = 2 * (1 - normal_cdf(abs(z)))
    critical = normal_ppf(1 - alpha / 2)
    return {"z": z, "p_value": p_value, "stop": abs(z) >= critical}
