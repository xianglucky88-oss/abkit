import json
import math
import unittest

from abkit.multiple import bh, correct, holm
from abkit.report import markdown, summarize, summarize_metrics, to_json


class MultipleCorrectionTests(unittest.TestCase):
    def test_legacy_boolean_interface_preserved(self):
        self.assertEqual(bh([0.001, 0.04, 0.9]), [True, False, False])
        self.assertEqual(holm([0.001, 0.04, 0.9]), [True, False, False])

    def test_rejection_arrays_stay_in_input_order(self):
        p_values = [0.9, 0.001, 0.04]
        result = correct(p_values, method="bh")
        self.assertEqual(result["rejected"], [False, True, False])
        self.assertEqual([item["p_value"] for item in result["items"]], p_values)
        self.assertEqual([item["index"] for item in result["items"]], [0, 1, 2])

    def test_tied_p_values_rank_by_input_index(self):
        # Three ties: ranks 1..3 assigned by input index, all rejected together.
        # Three ties at 0.01: BH adjusted p is min(3/1*0.01) = 0.01 for all.
        result = correct([0.01, 0.01, 0.01], method="bh", alpha=0.05)
        self.assertEqual(result["tie_groups"], [[0, 1, 2]])
        self.assertEqual([item["rank"] for item in result["items"]], [1, 2, 3])
        self.assertEqual(result["rejected"], [True, True, True])
        self.assertTrue(math.isclose(result["items"][0]["adjusted_p_value"], 0.01))

    def test_ties_holm_vs_bh_adjusted_values(self):
        result = correct([0.02, 0.02, 0.5], method="holm", alpha=0.05)
        self.assertEqual(result["tie_groups"], [[0, 1]])
        # Holm adjusted: min(3*0.02, 3*0.02)=0.06 for both tied ranks.
        self.assertTrue(math.isclose(result["items"][0]["adjusted_p_value"], 0.06))
        self.assertEqual(result["rejected"], [False, False, False])

    def test_invalid_p_values_are_kept_and_explained(self):
        result = correct([0.001, None, float("nan"), 1.4, "0.02", 0.03], method="bh")
        self.assertEqual(result["valid_count"], 2)
        self.assertEqual(result["input_count"], 6)
        self.assertFalse(result["items"][1]["valid"])
        self.assertEqual(result["items"][1]["reason"], "missing p-value")
        self.assertEqual(result["items"][2]["reason"], "p-value is NaN")
        self.assertEqual(result["items"][3]["reason"], "p-value outside [0, 1]")
        self.assertEqual(result["items"][4]["reason"], "p-value is not numeric")
        # Invalid entries are never rejected and never break input order.
        self.assertEqual(len(result["items"]), 6)
        self.assertEqual(result["rejected"][1:5], [False, False, False, False])

    def test_empty_list(self):
        result = correct([], method="bh")
        self.assertEqual(result["items"], [])
        self.assertEqual(result["rejected"], [])
        self.assertEqual(bh([]), [])
        self.assertEqual(holm([]), [])

    def test_alpha_validation(self):
        for bad_alpha in (0.0, 1.0, -0.1, 1.2, "0.05", True):
            with self.assertRaises(ValueError):
                correct([0.01], alpha=bad_alpha)
            with self.assertRaises(ValueError):
                bh([0.01], alpha=bad_alpha)

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            correct([0.01], method="bonferroni")

    def test_bh_and_holm_can_disagree(self):
        # BH (FDR) rejects three, Holm (FWER) stops after the first rank.
        p_values = [0.005, 0.02, 0.03, 0.08]
        bh_result = correct(p_values, method="bh", alpha=0.05)
        holm_result = correct(p_values, method="holm", alpha=0.05)
        self.assertEqual(bh_result["rejected"], [True, True, True, False])
        self.assertEqual(holm_result["rejected"], [True, False, False, False])


ENTRIES = [
    {"metric": "conversion", "p_value": 0.005, "estimate": 0.012},
    {"metric": "revenue", "p_value": 0.02, "estimate": 0.8},
    {"metric": "retention", "p_value": 0.03, "estimate": 0.03},
    {"metric": "latency", "p_value": 0.08, "estimate": -4.0},
]


class MultiMetricReportTests(unittest.TestCase):
    def test_report_carries_identity_order_and_method(self):
        report = summarize_metrics(ENTRIES, method="bh", alpha=0.05)
        self.assertEqual(report["report_type"], "multi_metric_correction")
        self.assertEqual(report["method"], "bh")
        self.assertEqual(report["alpha"], 0.05)
        self.assertEqual(report["input_order"], ["conversion", "revenue", "retention", "latency"])
        self.assertEqual(
            [row["metric"] for row in report["metrics"]], report["input_order"]
        )
        self.assertEqual([row["input_order"] for row in report["metrics"]], [0, 1, 2, 3])

    def test_duplicate_metric_names_rejected(self):
        dup = [{"metric": "conversion", "p_value": 0.01},
               {"metric": "conversion", "p_value": 0.02}]
        with self.assertRaises(ValueError):
            summarize_metrics(dup)

    def test_empty_metric_list_rejected(self):
        with self.assertRaises(ValueError):
            summarize_metrics([])

    def test_malformed_entry_rejected(self):
        with self.assertRaises(ValueError):
            summarize_metrics([{"metric": "x"}])
        with self.assertRaises(ValueError):
            summarize_metrics([{"p_value": 0.01}])

    def test_missing_and_illegal_p_values_survive_into_report(self):
        entries = [
            {"metric": "conversion", "p_value": 0.001},
            {"metric": "revenue", "p_value": None},
            {"metric": "retention", "p_value": float("nan")},
            {"metric": "latency", "p_value": 1.5},
        ]
        report = summarize_metrics(entries, method="bh")
        self.assertEqual(report["counts"], {"total": 4, "valid": 1, "invalid": 3, "rejected": 1})
        revenue = report["metrics"][1]
        self.assertFalse(revenue["valid"])
        self.assertEqual(revenue["corrected_decision"], "not significant")
        self.assertEqual(revenue["uncorrected_decision"], "unevaluable")
        self.assertEqual(revenue["conclusion_change"], "unevaluable")
        self.assertIsNone(revenue["adjusted_p_value"])
        # Strict JSON must round-trip despite the NaN input (no NaN leaks out).
        restored = json.loads(to_json(report))
        self.assertIsNone(restored["metrics"][2]["p_value"])

    def test_changed_metrics_are_traceable(self):
        # Under Holm only conversion survives; revenue/retention lose significance.
        report = summarize_metrics(ENTRIES, method="holm", alpha=0.05)
        changed = {item["metric"]: item for item in report["changed_metrics"]}
        self.assertEqual(set(changed), {"revenue", "retention"})
        self.assertEqual(changed["revenue"]["from"], "significant")
        self.assertEqual(changed["retention"]["to"], "not significant")
        rows = {row["metric"]: row for row in report["metrics"]}
        self.assertEqual(rows["conversion"]["conclusion_change"], "unchanged")

    def test_decision_rules_any_vs_all_key(self):
        # conversion + retention are key; under Holm only conversion passes.
        report = summarize_metrics(
            ENTRIES, method="holm", key_metrics=["conversion", "retention"]
        )
        rules = report["decision_rules"]
        self.assertTrue(rules["any"]["satisfied"])
        self.assertFalse(rules["all_key"]["satisfied"])
        self.assertIn("1/2", rules["all_key"]["basis"])
        self.assertEqual(report["decision_rule"], "any")
        self.assertTrue(report["decision"])

        strict = summarize_metrics(
            ENTRIES, method="holm",
            key_metrics=["conversion", "retention"], decision_rule="all_key",
        )
        self.assertFalse(strict["decision"])
        self.assertTrue(strict["decision_rules"]["any"]["satisfied"])

    def test_invalid_key_metric_fails_all_key_rule(self):
        entries = [
            {"metric": "conversion", "p_value": 0.001},
            {"metric": "revenue", "p_value": None},
        ]
        report = summarize_metrics(entries, method="bh", decision_rule="all_key")
        self.assertTrue(report["decision_rules"]["any"]["satisfied"])
        self.assertFalse(report["decision_rules"]["all_key"]["satisfied"])
        self.assertFalse(report["decision"])

    def test_unknown_key_metric_and_rule_rejected(self):
        with self.assertRaises(ValueError):
            summarize_metrics(ENTRIES, key_metrics=["nope"])
        with self.assertRaises(ValueError):
            summarize_metrics(ENTRIES, decision_rule="majority")

    def test_methods_yield_different_reports(self):
        bh_report = summarize_metrics(ENTRIES, method="bh")
        holm_report = summarize_metrics(ENTRIES, method="holm")
        self.assertEqual(bh_report["rejected_metrics"],
                         ["conversion", "revenue", "retention"])
        self.assertEqual(holm_report["rejected_metrics"], ["conversion"])
        # Under the strict all_key rule (key = the three BH-rejected metrics)
        # the methods yield opposite decisions.
        keys = ["conversion", "revenue", "retention"]
        bh_strict = summarize_metrics(ENTRIES, method="bh",
                                      key_metrics=keys, decision_rule="all_key")
        holm_strict = summarize_metrics(ENTRIES, method="holm",
                                        key_metrics=keys, decision_rule="all_key")
        self.assertTrue(bh_strict["decision"])
        self.assertFalse(holm_strict["decision"])

    def test_markdown_renders_from_same_dict_as_json(self):
        report = summarize_metrics(ENTRIES, method="holm",
                                   key_metrics=["conversion", "retention"],
                                   decision_rule="all_key")
        text = markdown(report)
        # Identity, method, alpha, per-metric evidence and both rules all appear.
        for token in ["HOLM", "alpha=0.05", "conversion", "revenue", "retention", "latency",
                      "lost_significance", "`any`", "`all_key`", "FAIL",
                      "Adjusted p", "Rank"]:
            self.assertIn(token, text)
        # Every adjusted p in Markdown matches the JSON dict exactly.
        for row in report["metrics"]:
            if row["adjusted_p_value"] is not None:
                self.assertIn(f"{row['adjusted_p_value']:.6g}", text)
        # Markdown rejections mirror the JSON rejection set.
        for row in report["metrics"]:
            self.assertEqual(row["corrected_significant"],
                             row["metric"] in report["rejected_metrics"])

    def test_tied_p_group_reported_in_json_and_markdown(self):
        entries = [{"metric": f"m{i}", "p_value": 0.01} for i in range(3)]
        report = summarize_metrics(entries, method="bh")
        self.assertEqual(report["tie_groups"], [[0, 1, 2]])
        text = markdown(report)
        self.assertIn("Tied p-value", text)
        self.assertIn("[0, 1, 2]", text)

    def test_json_is_strict_and_complete(self):
        report = summarize_metrics(ENTRIES, method="bh", alpha=0.1)
        restored = json.loads(to_json(report))
        self.assertEqual(restored["method"], "bh")
        self.assertEqual(restored["alpha"], 0.1)
        self.assertEqual(len(restored["metrics"]), 4)
        for row in restored["metrics"]:
            # No NaN/None-where-float-required surprises.
            self.assertIsInstance(row["valid"], bool)

    def test_single_metric_api_unchanged(self):
        result = summarize("metric", 1.0, 0.01)
        self.assertTrue(result["significant"])
        text = markdown(result)
        self.assertIn("### metric", text)


if __name__ == "__main__":
    unittest.main()
