import math
import random
import unittest

from abkit.planning import plan_mean_metric, plan_ratio_metric
from abkit.power import (detectable_effect, detectable_effect_from_variances,
                         required_per_arm, required_sample_size)
from abkit.report import plan_markdown
from abkit.stats import normal_ppf
from abkit.variance import ratio_delta, ratio_estimate


def correlated_arm(n, shift, seed, rho=0.9, noise=0.25):
    rng = random.Random(seed)
    x = [rng.gauss(0.0, 1.0) for _ in range(n)]
    y = [shift + rho * v + rng.gauss(0.0, noise) for v in x]
    return y, x


def ratio_arm(n, ratio, seed, den_mean=5.0):
    """构造分子≈ratio·分母的比率数据，含跨期持久的用户级成分，返回 (num, den, num0, den0)。"""
    rng = random.Random(seed)
    den0 = [rng.gauss(den_mean, 1.0) for _ in range(n)]
    u = [rng.gauss(0.0, 0.5) for _ in range(n)]  # 用户级持久成分，前后期复现
    num0 = [ratio * d + ui + rng.gauss(0.0, 0.05) for d, ui in zip(den0, u)]
    den = [d + rng.gauss(0.0, 0.05) for d in den0]
    num = [ratio * d + ui + rng.gauss(0.0, 0.05) for d, ui in zip(den, u)]
    return num, den, num0, den0


class PowerConsistencyTests(unittest.TestCase):
    def test_generalized_matches_legacy(self):
        for effect, sd in [(0.1, 1.0), (0.25, 2.0), (1.3, 0.7)]:
            self.assertEqual(required_sample_size(effect, sd),
                             required_per_arm(effect, sd * sd, sd * sd))
            self.assertAlmostEqual(detectable_effect(100, sd),
                                   detectable_effect_from_variances(100, sd * sd, sd * sd))

    def test_rounding_boundary_hits_exact_integer(self):
        z = normal_ppf(0.975) + normal_ppf(0.8)
        n_target = 500
        effect = z * math.sqrt(2.0 / n_target)  # 使 n_exact 理论上恰为 500
        n = required_per_arm(effect, 1.0, 1.0)
        self.assertEqual(n, n_target)
        self.assertLessEqual(detectable_effect_from_variances(n, 1.0, 1.0), effect)
        self.assertGreater(detectable_effect_from_variances(n - 1, 1.0, 1.0), effect)

    def test_recommended_n_always_meets_power(self):
        for effect in (0.05, 0.1, 0.13, 0.337, 1.0):
            n = required_per_arm(effect, 1.2, 0.8)
            self.assertLessEqual(
                detectable_effect_from_variances(n, 1.2, 0.8), effect * (1 + 1e-12))
            self.assertTrue(n == 2 or detectable_effect_from_variances(n - 1, 1.2, 0.8) > effect)

    def test_required_per_arm_validates_inputs(self):
        with self.assertRaises(ValueError):
            required_per_arm(0.0, 1.0, 1.0)  # 非正/零效应
        with self.assertRaises(ValueError):
            required_per_arm(0.1, 0.0, 0.0)  # 方差和为零
        with self.assertRaises(ValueError):
            required_per_arm(0.1, -1.0, 2.0)
        with self.assertRaises(ValueError):
            required_per_arm(0.1, 1.0, 1.0, alpha=0.0)

    def test_negative_effect_uses_absolute_value(self):
        self.assertEqual(required_per_arm(-0.2, 1.0, 1.0), required_per_arm(0.2, 1.0, 1.0))


class RatioEstimateTests(unittest.TestCase):
    def test_zero_denominator_sum_is_explainable(self):
        est = ratio_estimate([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        self.assertFalse(est.ok)
        self.assertIsNone(est.ratio)
        self.assertIn("分母", est.reason)
        # 兼容包装保持旧行为
        self.assertEqual(ratio_delta([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]), (0.0, 0.0))

    def test_near_zero_mean_denominator_is_rejected(self):
        est = ratio_estimate([1.0, 2.0, 3.0], [1e-13, 2e-13, 3e-13])
        self.assertFalse(est.ok)
        self.assertIn("接近 0", est.reason)

    def test_zero_entries_in_denominator_are_allowed(self):
        est = ratio_estimate([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        self.assertTrue(est.ok)
        self.assertAlmostEqual(est.ratio, 1.0)

    def test_matches_legacy_ratio_delta(self):
        num, den = [1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]
        est = ratio_estimate(num, den)
        legacy_ratio, legacy_se = ratio_delta(num, den)
        self.assertAlmostEqual(est.ratio, legacy_ratio)
        self.assertAlmostEqual(est.se, legacy_se)


class MeanPlanningTests(unittest.TestCase):
    def setUp(self):
        self.y_c, self.x_c = correlated_arm(400, 0.0, seed=1)
        self.y_t, self.x_t = correlated_arm(400, 0.3, seed=2)

    def test_adjusted_plan_uses_adjusted_uncertainty(self):
        plan = plan_mean_metric(self.y_c, self.y_t, self.x_c, self.x_t)
        self.assertEqual(plan["status"], "ok")
        self.assertGreater(plan["variance_reduction"], 0.5)
        ss = plan["sample_size"]
        self.assertEqual(ss["basis"], "adjusted")
        self.assertLess(ss["recommended"], ss["per_arm_unadjusted"])
        # 推荐值与报告的不确定性口径一致
        self.assertEqual(ss["recommended"], required_per_arm(
            plan["effect"]["adjusted"],
            plan["variance"]["control_adjusted"],
            plan["variance"]["treatment_adjusted"]))
        target = abs(plan["effect"]["used_for_planning"])
        self.assertLessEqual(ss["detectable_effect_at_recommended_n"], target * (1 + 1e-9))
        md = plan_markdown(plan)
        self.assertIn("基于调整后不确定性", md)
        self.assertIn("方差缩减", md)

    def test_unadjusted_when_no_baseline(self):
        plan = plan_mean_metric(self.y_c, self.y_t)
        self.assertEqual(plan["status"], "ok")
        self.assertIsNone(plan["effect"]["adjusted"])
        ss = plan["sample_size"]
        self.assertEqual(ss["basis"], "unadjusted")
        self.assertEqual(ss["recommended"], ss["per_arm_unadjusted"])
        self.assertEqual(ss["recommended"], required_per_arm(
            plan["effect"]["raw"],
            plan["variance"]["control_raw"],
            plan["variance"]["treatment_raw"]))
        self.assertIn("基于未调整不确定性", plan_markdown(plan))

    def test_baseline_length_mismatch_falls_back(self):
        plan = plan_mean_metric(self.y_c, self.y_t, self.x_c[:-1], self.x_t)
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["sample_size"]["basis"], "unadjusted")
        self.assertTrue(any("不一致" in w for w in plan["warnings"]))
        self.assertEqual(plan["sample_size"]["recommended"],
                         plan["sample_size"]["per_arm_unadjusted"])

    def test_zero_variance_covariate_falls_back(self):
        plan = plan_mean_metric(self.y_c, self.y_t,
                                [1.0] * len(self.y_c), [1.0] * len(self.y_t))
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["variance_reduction"], 0.0)
        self.assertEqual(plan["sample_size"]["basis"], "unadjusted")
        self.assertTrue(any("无方差" in w for w in plan["warnings"]))

    def test_negative_effect_plans_on_absolute_value(self):
        control = [1.0, 2.0, 3.0, 4.0]
        treatment = [0.1, 0.2, 0.3, 0.4]
        plan = plan_mean_metric(control, treatment)
        self.assertEqual(plan["status"], "ok")
        self.assertLess(plan["effect"]["raw"], 0)
        self.assertTrue(any("负" in w for w in plan["warnings"]))
        self.assertEqual(plan["sample_size"]["recommended"],
                         required_per_arm(abs(plan["effect"]["raw"]),
                                          plan["variance"]["control_raw"],
                                          plan["variance"]["treatment_raw"]))

    def test_zero_effect_is_explainable(self):
        plan = plan_mean_metric([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(plan["status"], "zero_effect")
        self.assertIsNone(plan["sample_size"]["recommended"])
        self.assertIsNotNone(plan["mde_at_observed_n"])
        self.assertIn("效应为 0", plan["reason"])
        self.assertIn("无法给出样本量建议", plan_markdown(plan))

    def test_insufficient_data(self):
        plan = plan_mean_metric([1.0], [2.0])
        self.assertEqual(plan["status"], "insufficient_data")
        self.assertIsNone(plan["sample_size"]["recommended"])
        empty = plan_mean_metric([], [])
        self.assertEqual(empty["status"], "insufficient_data")
        self.assertIn("insufficient_data", plan_markdown(empty))

    def test_invalid_alpha_power_rejected(self):
        with self.assertRaises(ValueError):
            plan_mean_metric(self.y_c, self.y_t, alpha=1.5)
        with self.assertRaises(ValueError):
            plan_mean_metric(self.y_c, self.y_t, power=1.0)


class RatioPlanningTests(unittest.TestCase):
    def test_ratio_end_to_end_deterministic(self):
        c_num, c_den = [1.0, 2.0, 3.0, 4.0, 5.0], [2.0] * 5
        t_num, t_den = [2.0, 3.0, 4.0, 5.0, 6.0], [2.0] * 5
        plan = plan_ratio_metric(c_num, c_den, t_num, t_den)
        self.assertEqual(plan["status"], "ok")
        self.assertAlmostEqual(plan["ratio"]["control"], 1.5)
        self.assertAlmostEqual(plan["ratio"]["treatment"], 2.0)
        self.assertAlmostEqual(plan["effect"]["raw"], 0.5)
        # 线性化项方差 2.5，mean(d)=2 → 每观测方差 0.625
        self.assertAlmostEqual(plan["variance"]["control_raw"], 0.625)
        self.assertAlmostEqual(plan["standard_error"]["raw"], math.sqrt(0.25))
        self.assertEqual(plan["sample_size"]["basis"], "unadjusted")
        self.assertEqual(plan["sample_size"]["recommended"],
                         required_per_arm(0.5, 0.625, 0.625))

    def test_ratio_zero_denominator_is_explainable(self):
        plan = plan_ratio_metric([1.0, 2.0, 3.0], [0.0, 0.0, 0.0],
                                 [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        self.assertEqual(plan["status"], "invalid_metric")
        self.assertIsNone(plan["sample_size"]["recommended"])
        self.assertIn("分母", plan["reason"])
        md = plan_markdown(plan)
        self.assertIn("分母", md)
        self.assertIn("无法给出样本量建议", md)

    def test_ratio_near_zero_denominator_is_explainable(self):
        plan = plan_ratio_metric([1.0, 2.0, 3.0], [1e-13, 2e-13, 3e-13],
                                 [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        self.assertEqual(plan["status"], "invalid_metric")
        self.assertIn("接近 0", plan["reason"])

    def test_ratio_with_cuped_reduces_sample_size(self):
        c_num, c_den, c_num0, c_den0 = ratio_arm(500, 2.0, seed=3)
        t_num, t_den, t_num0, t_den0 = ratio_arm(500, 2.2, seed=4)
        plan = plan_ratio_metric(c_num, c_den, t_num, t_den,
                                 (c_num0, c_den0), (t_num0, t_den0))
        self.assertEqual(plan["status"], "ok")
        self.assertGreater(plan["variance_reduction"], 0.0)
        ss = plan["sample_size"]
        self.assertEqual(ss["basis"], "adjusted")
        self.assertLess(ss["recommended"], ss["per_arm_unadjusted"])
        self.assertEqual(ss["recommended"], required_per_arm(
            plan["effect"]["adjusted"],
            plan["variance"]["control_adjusted"],
            plan["variance"]["treatment_adjusted"]))

    def test_ratio_baseline_mismatch_falls_back(self):
        c_num, c_den, c_num0, c_den0 = ratio_arm(100, 2.0, seed=5)
        t_num, t_den, t_num0, t_den0 = ratio_arm(100, 2.2, seed=6)
        plan = plan_ratio_metric(c_num, c_den, t_num, t_den,
                                 (c_num0[:-1], c_den0), (t_num0, t_den0))
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["sample_size"]["basis"], "unadjusted")
        self.assertTrue(any("不一致" in w for w in plan["warnings"]))


if __name__ == "__main__":
    unittest.main()
