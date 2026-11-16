"""Human-readable result reporting.

The single-metric :func:`summarize` / :func:`markdown` pair is retained.
For multi-metric experiments use :func:`summarize_metrics`, which runs a
multiple-comparison correction (``abkit.multiple``) and returns one
JSON-safe dict carrying:

* metric identity and original input order,
* raw p-values, adjusted p-values, ranks and invalid-p reasons,
* the correction method and alpha,
* per-metric uncorrected vs. corrected decisions and which metrics flipped,
* both overall decision rules -- "any metric significant" and
  "all key metrics pass".

:func:`markdown` renders that same dict, so JSON and Markdown can never
disagree: every field printed in Markdown is read straight from the dict.
"""

from __future__ import annotations

import json

from .multiple import METHODS, correct

DECISION_RULES = ("any", "all_key")


def summarize(metric: str, estimate: float, p_value: float, alpha: float = 0.05) -> dict:
    return {
        "metric": metric,
        "estimate": estimate,
        "p_value": p_value,
        "significant": p_value < alpha,
        "conclusion": "evidence of a difference" if p_value < alpha else "inconclusive",
    }


def _check_entries(entries: list[dict]) -> None:
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty list of metric records")
    seen: set[str] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict) or "metric" not in entry or "p_value" not in entry:
            raise ValueError(
                f"entry #{position} must be a dict with 'metric' and 'p_value'"
            )
        name = entry["metric"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"entry #{position} has an invalid metric name: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate metric name: {name!r}")
        seen.add(name)


def _format_p(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def summarize_metrics(
    entries: list[dict],
    *,
    method: str = "bh",
    alpha: float = 0.05,
    key_metrics: list[str] | None = None,
    decision_rule: str = "any",
) -> dict:
    """Build the structured multi-metric report.

    Each entry is ``{"metric": <name>, "p_value": <float|None>,
    "estimate": <float, optional>}``.  ``p_value`` may be missing/``None``
    or otherwise invalid; such metrics are excluded from the correction but
    retained with an explanatory reason.

    ``key_metrics`` names the metrics required by the ``all_key`` rule
    (default: every metric).  ``decision_rule`` selects which rule drives
    the top-level ``decision``: ``"any"`` (at least one corrected rejection)
    or ``"all_key"`` (every key metric rejected; invalid key metrics fail).
    """
    _check_entries(entries)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if decision_rule not in DECISION_RULES:
        raise ValueError(
            f"unknown decision_rule {decision_rule!r}; choose from {DECISION_RULES}"
        )

    names = [entry["metric"] for entry in entries]
    if key_metrics is None:
        key_metrics = list(names)
    unknown = [name for name in key_metrics if name not in names]
    if unknown:
        raise ValueError(f"key_metrics not found in entries: {unknown}")

    correction = correct([entry["p_value"] for entry in entries], method=method, alpha=alpha)

    metrics: list[dict] = []
    changed: list[dict] = []
    for entry, item in zip(entries, correction["items"]):
        name = entry["metric"]
        if item["valid"]:
            uncorrected = item["p_value"] < alpha
            uncorrected_label = "significant" if uncorrected else "not significant"
        else:
            uncorrected = False
            uncorrected_label = "unevaluable"
        corrected = item["rejected"]
        is_key = name in key_metrics
        change = "unchanged"
        if not item["valid"]:
            change = "unevaluable"
        elif uncorrected != corrected:
            change = "lost_significance" if uncorrected else "gained_significance"
        record = {
            "metric": name,
            "is_key_metric": is_key,
            "input_order": item["index"],
            "estimate": entry.get("estimate"),
            "p_value": item["p_value"],
            "rank": item["rank"],
            "adjusted_p_value": item["adjusted_p_value"],
            "valid": item["valid"],
            "reason": item["reason"],
            "uncorrected_significant": uncorrected,
            "uncorrected_decision": uncorrected_label,
            "corrected_significant": corrected,
            "corrected_decision": "significant" if corrected else "not significant",
            "conclusion_change": change,
        }
        metrics.append(record)
        if change in ("lost_significance", "gained_significance"):
            changed.append(
                {"metric": name, "from": uncorrected_label, "to": record["corrected_decision"]}
            )

    rejected_names = [record["metric"] for record in metrics if record["corrected_significant"]]
    key_records = [record for record in metrics if record["is_key_metric"]]
    all_key_pass = bool(key_records) and all(
        record["corrected_significant"] for record in key_records
    )
    any_rejected = bool(rejected_names)
    rule_results = {
        "any": {
            "label": "at least one metric significant after correction",
            "satisfied": any_rejected,
            "basis": "one or more corrected rejections"
            if any_rejected
            else "no metric rejected after correction",
        },
        "all_key": {
            "label": "all key metrics pass after correction",
            "satisfied": all_key_pass,
            "basis": f"{sum(r['corrected_significant'] for r in key_records)}/"
            f"{len(key_records)} key metric(s) rejected",
        },
    }

    return {
        "report_type": "multi_metric_correction",
        "method": method,
        "alpha": correction["alpha"],
        "decision_rule": decision_rule,
        "input_order": names,
        "key_metrics": list(key_metrics),
        "counts": {
            "total": correction["input_count"],
            "valid": correction["valid_count"],
            "invalid": correction["input_count"] - correction["valid_count"],
            "rejected": len(rejected_names),
        },
        "tie_groups": correction["tie_groups"],
        "metrics": metrics,
        "rejected_metrics": rejected_names,
        "changed_metrics": changed,
        "decision_rules": rule_results,
        "decision": rule_results[decision_rule]["satisfied"],
    }


def to_json(result: dict) -> str:
    """Serialize a report dict to strict JSON (NaN/Infinity are rejected)."""
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)


def _markdown_multi(result: dict) -> str:
    lines = [
        f"## Multi-metric correction — {result['method'].upper()} (alpha={result['alpha']:g})",
        "",
        "| # | Metric | Key | Raw p | Rank | Adjusted p | Uncorrected | Corrected | Change |",
        "| - | ------ | --- | ----- | ---- | ---------- | ----------- | --------- | ------ |",
    ]
    for record in result["metrics"]:
        reason = f" ({record['reason']})" if record["reason"] else ""
        lines.append(
            "| {order} | {metric} | {key} | {raw} | {rank} | {adj} | {unc} | {cor} | {change}{reason} |".format(
                order=record["input_order"],
                metric=record["metric"],
                key="yes" if record["is_key_metric"] else "no",
                raw=_format_p(record["p_value"]),
                rank="-" if record["rank"] is None else record["rank"],
                adj=_format_p(record["adjusted_p_value"]),
                unc=record["uncorrected_decision"],
                cor=record["corrected_decision"],
                change=record["conclusion_change"],
                reason=reason,
            )
        )
    if result["tie_groups"]:
        groups = ["[" + ", ".join(str(i) for i in group) + "]" for group in result["tie_groups"]]
        lines += ["", f"Tied p-value input indices (rank tie broken by input order): {', '.join(groups)}"]

    lines += [
        "",
        "### Overall decisions",
    ]
    for rule_id, rule in result["decision_rules"].items():
        marker = " (selected)" if rule_id == result["decision_rule"] else ""
        lines.append(
            f"- `{rule_id}`{marker}: {rule['label']} -> "
            f"**{'PASS' if rule['satisfied'] else 'FAIL'}** ({rule['basis']})"
        )
    lines.append(
        f"- Top-level decision (`{result['decision_rule']}`): "
        f"**{'reject' if result['decision'] else 'do not reject'}**"
    )

    if result["changed_metrics"]:
        lines += ["", "### Metrics whose conclusion changed under correction"]
        for change in result["changed_metrics"]:
            lines.append(
                f"- `{change['metric']}`: {change['from']} -> {change['to']}"
            )
    else:
        lines += ["", "### Metrics whose conclusion changed under correction", "- none"]
    return "\n".join(lines)


def markdown(result: dict) -> str:
    """Render either a single-metric :func:`summarize` or multi-metric report."""
    if result.get("report_type") == "multi_metric_correction":
        return _markdown_multi(result)

    decision = "significant" if result["significant"] else "not significant"
    return (f"### {result['metric']}\n\n"
            f"- Estimate: `{result['estimate']:.6g}`\n"
            f"- p-value: `{result['p_value']:.6g}`\n"
            f"- Decision: **{decision}** ({result['conclusion']})")
