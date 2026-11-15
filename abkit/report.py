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
