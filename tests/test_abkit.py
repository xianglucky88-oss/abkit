import unittest

from abkit.assignment import bucket
from abkit.multiple import bh
from abkit.power import required_sample_size
from abkit.segment import group_means, interaction_effect
from abkit.variance import cuped


class AbkitTests(unittest.TestCase):
    def test_bucket_is_stable(self):
        self.assertEqual(bucket("user-1", "salt", 2), bucket("user-1", "salt", 2))

    def test_cuped_reduces_perfect_baseline(self):
        adjusted, theta = cuped([1, 2, 3], [1, 2, 3])
        self.assertGreater(theta, 0)
        self.assertAlmostEqual(adjusted[0], adjusted[1])

    def test_multiple_comparison(self):
        self.assertEqual(bh([0.001, 0.04, 0.9]), [True, False, False])

    def test_power_is_positive(self):
        self.assertGreater(required_sample_size(0.1), 0)


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            # android：每臂 3 个样本，真实效应 0.5
            {"platform": "android", "arm": "control", "revenue": 1.0},
            {"platform": "android", "arm": "control", "revenue": 2.0},
            {"platform": "android", "arm": "control", "revenue": 3.0},
            {"platform": "android", "arm": "treatment", "revenue": 1.5},
            {"platform": "android", "arm": "treatment", "revenue": 2.5},
            {"platform": "android", "arm": "treatment", "revenue": 3.5},
            # ios：单臂分层（只有对照）
            {"platform": "ios", "arm": "control", "revenue": 1.0},
            {"platform": "ios", "arm": "control", "revenue": 2.0},
            # web：样本不足（处理臂只有 1 个样本）
            {"platform": "web", "arm": "control", "revenue": 1.0},
            {"platform": "web", "arm": "control", "revenue": 2.0},
            {"platform": "web", "arm": "treatment", "revenue": 5.0},
        ]
        self.result = interaction_effect(self.rows, "platform", "arm", "revenue")
        self.by_segment = {s["segment"]: s for s in self.result["segments"]}

    def test_segment_statuses(self):
        self.assertEqual(self.by_segment["android"]["status"], "ok")
        self.assertEqual(self.by_segment["ios"]["status"], "not_estimable")
        self.assertEqual(self.by_segment["web"]["status"], "insufficient_n")

    def test_effect_only_for_valid_segments(self):
        effect = self.by_segment["android"]["contrasts"]["treatment"]["effect"]
        self.assertAlmostEqual(effect, 0.5)
        self.assertIsNone(self.by_segment["ios"]["contrasts"]["treatment"]["effect"])
        self.assertIsNone(self.by_segment["web"]["contrasts"]["treatment"]["effect"])

    def test_sample_sizes_are_retained(self):
        self.assertEqual(self.by_segment["ios"]["n"], {"control": 2, "treatment": 0})
        self.assertEqual(self.by_segment["web"]["n"]["treatment"], 1)
        self.assertNotIn("treatment", self.by_segment["ios"]["means"])

    def test_dirty_rows_are_skipped_and_counted(self):
        rows = [
            {"platform": "android", "arm": "control", "revenue": 1.0},
            {"platform": "android", "arm": "treatment", "revenue": 2.0},
            {"platform": "", "arm": "control", "revenue": 1.0},
            {"arm": "control", "revenue": 1.0},
            {"platform": "android", "arm": None, "revenue": 1.0},
            {"platform": "android", "arm": "control"},
            {"platform": "android", "arm": "control", "revenue": ""},
            {"platform": "android", "arm": "control", "revenue": "abc"},
            {"platform": "android", "arm": "control", "revenue": float("inf")},
            {"platform": "android", "arm": "control", "revenue": float("nan")},
        ]
        result = interaction_effect(rows, "platform", "arm", "revenue", min_n_per_arm=1)
        self.assertEqual(
            result["skipped_rows"],
            {"missing_segment": 2, "missing_treatment": 1,
             "missing_metric": 3, "non_finite_metric": 2},
        )
        segment = result["segments"][0]
        self.assertEqual(segment["status"], "ok")
        self.assertAlmostEqual(segment["contrasts"]["treatment"]["effect"], 1.0)

    def test_categories_sort_numerically_then_lexically(self):
        rows = [
            {"g": seg, "arm": arm, "m": value}
            for seg in ("10", "2", "1", "beta")
            for arm in ("a", "b")
            for value in (1.0, 2.0)
        ]
        result = interaction_effect(rows, "g", "arm", "m")
        self.assertEqual([s["segment"] for s in result["segments"]], ["1", "2", "10", "beta"])
        self.assertEqual(result["treatment_levels"], ["a", "b"])
        self.assertEqual(result["control"], "a")

    def test_multiple_treatment_levels_with_explicit_control(self):
        rows = [
            {"g": "x", "arm": arm, "m": base + offset}
            for arm, base in (("control", 1.0), ("a", 2.0), ("b", 4.0))
            for offset in (0.0, 1.0)
        ]
        result = interaction_effect(rows, "g", "arm", "m", control="control")
        contrasts = result["segments"][0]["contrasts"]
        self.assertEqual(result["treatment_levels"], ["a", "b", "control"])
        self.assertAlmostEqual(contrasts["a"]["effect"], 1.0)
        self.assertAlmostEqual(contrasts["b"]["effect"], 3.0)
        self.assertNotIn("control", contrasts)

    def test_degenerate_batches_are_rejected(self):
        with self.assertRaises(ValueError):  # 空批
            interaction_effect([], "g", "arm", "m")
        with self.assertRaises(ValueError):  # 清洗后无可用行
            interaction_effect([{"g": "", "arm": "a", "m": 1.0}], "g", "arm", "m")
        with self.assertRaises(ValueError):  # 处理水平不足两个
            interaction_effect([{"g": "x", "arm": "only", "m": 1.0}], "g", "arm", "m")
        with self.assertRaises(ValueError):  # 指定的 control 不存在
            interaction_effect(self.rows, "platform", "arm", "revenue", control="nope")
        with self.assertRaises(ValueError):  # min_n_per_arm 非法
            interaction_effect(self.rows, "platform", "arm", "revenue", min_n_per_arm=0)

    def test_group_means_keeps_sample_size_and_skips_dirty_rows(self):
        rows = [
            {"g": "x", "m": 1.0},
            {"g": "x", "m": 3.0},
            {"g": "y", "m": 10.0},
            {"g": "", "m": 5.0},
            {"g": "y", "m": float("nan")},
            {"g": "y"},
        ]
        result = group_means(rows, "g", "m")
        self.assertEqual(result["x"], {"n": 2, "mean": 2.0})
        self.assertEqual(result["y"], {"n": 1, "mean": 10.0})
        self.assertNotIn("", result)


if __name__ == "__main__":
    unittest.main()
