"""多臂分流与 SRM 检测的完整验收测试。"""

import math
import unittest

from abkit.assignment import (
    MIN_EXPECTED,
    arm_bucket,
    assign,
    assign_detail,
    bucket,
    normalize_ratios,
    run_srm_check,
    srm_p_value,
    srm_report,
)
from abkit.stats import chi2_sf


def _users(n: int) -> list[str]:
    return [f"user-{i}" for i in range(n)]


class ChiSquareTests(unittest.TestCase):
    def test_known_sf_values(self):
        # df=1: chi2_sf(z^2, 1) 应等于标准正态的双侧 p 值
        for z in (0.5, 1.0, 1.96, 2.58, 3.5):
            p_normal = 2 * (1 - 0.5 * (1 + math.erf(z / 2 ** 0.5)))
            self.assertAlmostEqual(chi2_sf(z * z, 1), p_normal, places=12)
        # 常用临界值：df=2, x=5.991 -> p≈0.05；df=4, x=9.488 -> p≈0.05
        self.assertAlmostEqual(chi2_sf(5.991, 2), 0.05, places=3)
        self.assertAlmostEqual(chi2_sf(9.488, 4), 0.05, places=3)
        self.assertEqual(chi2_sf(0.0, 3), 1.0)
        self.assertLess(chi2_sf(100.0, 2), 1e-20)

    def test_invalid_df(self):
        with self.assertRaises(ValueError):
            chi2_sf(1.0, 0)


class NormalizeRatiosTests(unittest.TestCase):
    def test_none_gives_equal(self):
        self.assertEqual(normalize_ratios(None, 3), [1 / 3, 1 / 3, 1 / 3])

    def test_integer_weights_and_sum_to_one(self):
        result = normalize_ratios([1, 2, 1], 3)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(sum(result), 1.0)
        self.assertAlmostEqual(result[1], 2 * result[0])

    def test_config_errors(self):
        with self.assertRaises(ValueError):
            normalize_ratios([1, 1], 3)          # 长度不匹配
        with self.assertRaises(ValueError):
            normalize_ratios([0, 0, 0], 3)       # 全零
        with self.assertRaises(ValueError):
            normalize_ratios([1, -1, 1], 3)      # 负数
        with self.assertRaises(ValueError):
            normalize_ratios([1, float("nan"), 1], 3)
        with self.assertRaises(ValueError):
            normalize_ratios([1, float("inf")], 2)

    def test_zero_probability_arm_allowed(self):
        result = normalize_ratios([1, 0, 1], 3)
        self.assertEqual(result[1], 0.0)
        self.assertAlmostEqual(result[0], 0.5)


class StableAssignmentTests(unittest.TestCase):
    def test_same_input_same_bucket(self):
        for buckets in (1, 2, 3, 7, 50):
            first = [bucket(u, "salt-x", buckets) for u in _users(200)]
            second = [bucket(u, "salt-x", buckets) for u in _users(200)]
            self.assertEqual(first, second)

    def test_bucket_in_range_and_all_arms_present(self):
        for buckets in (1, 2, 5, 13):
            counts = assign(_users(3000), "s", buckets)
            self.assertEqual(set(counts), set(range(buckets)))
            self.assertTrue(all(0 <= bucket(u, "s", buckets) < buckets
                                for u in _users(100)))

    def test_salt_changes_distribution_not_contract(self):
        a = assign(_users(2000), "salt-a", 4)
        b = assign(_users(2000), "salt-b", 4)
        self.assertEqual(sum(a.values()), sum(b.values()))
        self.assertNotEqual(a, b)  # 不同 salt 几乎必然产生不同切分

    def test_extreme_bucket_counts(self):
        self.assertEqual(bucket("u", "s", 1), 0)
        with self.assertRaises(ValueError):
            bucket("u", "s", 0)
        with self.assertRaises(ValueError):
            bucket("u", "s", 2 ** 32 + 1)
        with self.assertRaises(TypeError):
            bucket("u", "s", True)

    def test_unequal_ratio_stable_and_aligned(self):
        ratio = [0.1, 0.2, 0.3, 0.4]
        first = [arm_bucket(u, "salt", 4, ratio) for u in _users(500)]
        second = [arm_bucket(u, "salt", 4, ratio) for u in _users(500)]
        self.assertEqual(first, second)  # salt/桶数/配置不变 -> 稳定
        # 换 salt 或换比例配置后允许变化，但都落在合法臂内
        self.assertTrue(all(0 <= arm < 4 for arm in first))

    def test_equal_ratio_matches_legacy_bucket(self):
        for u in _users(50):
            self.assertEqual(arm_bucket(u, "s", 3, [1, 1, 1]), bucket(u, "s", 3))
            self.assertEqual(arm_bucket(u, "s", 3, None), bucket(u, "s", 3))

    def test_unequal_ratio_empirical_shares(self):
        ratio = [1, 2, 1]
        counts = assign(_users(8000), "salt", 3, ratio)
        total = sum(counts.values())
        shares = [counts[i] / total for i in range(3)]
        for observed, expected in zip(shares, (0.25, 0.5, 0.25)):
            self.assertAlmostEqual(observed, expected, delta=0.03)

    def test_duplicate_user_ids_counted_once(self):
        ids = _users(100) + _users(50)  # 后 50 个重复
        detail = assign_detail(ids, "s", 2, dedupe=True)
        self.assertEqual(detail["duplicates"], 50)
        self.assertEqual(sum(detail["counts"].values()), 100)
        self.assertEqual(len(detail["assignments"]), 100)
        # 不去重时保留全部记录
        raw = assign_detail(ids, "s", 2, dedupe=False)
        self.assertEqual(raw["duplicates"], 0)
        self.assertEqual(sum(raw["counts"].values()), 150)

    def test_zero_sample_bucket_still_in_counts(self):
        counts = assign(_users(5), "s", 4)
        self.assertEqual(set(counts), {0, 1, 2, 3})
        self.assertEqual(sum(counts.values()), 5)


class SrmPValueTests(unittest.TestCase):
    def test_two_group_matches_legacy_z_test(self):
        from abkit.stats import normal_cdf
        counts = {0: 462, 1: 538}
        n = 1000
        p0 = 0.5
        z = (462 / n - p0) / (p0 * (1 - p0) / n) ** 0.5
        p_z = 2 * (1 - normal_cdf(abs(z)))
        self.assertAlmostEqual(srm_p_value(counts), p_z, places=12)

    def test_balanced_multi_arm_not_significant(self):
        self.assertGreater(srm_p_value({0: 500, 1: 500, 2: 500}), 0.05)
        self.assertGreater(srm_p_value({0: 250, 1: 500, 2: 250}, [1, 2, 1]), 0.05)

    def test_multi_arm_detects_deviation_beyond_first_two(self):
        # 前两组完全平衡，第三组严重偏离 —— 旧实现必然漏报
        counts = {0: 400, 1: 400, 2: 600, 3: 400}
        self.assertLess(srm_p_value(counts), 0.001)

    def test_unequal_ratio_balanced_not_flagged(self):
        counts = {0: 103, 1: 198, 2: 297, 3: 402}
        self.assertGreater(srm_p_value(counts, [0.1, 0.2, 0.3, 0.4]), 0.1)

    def test_unequal_ratio_wrongly_equal_is_imbalanced(self):
        # 实际 1:1:1:1，但配置要求 1:2:3:4 -> 应显著
        counts = {i: 250 for i in range(4)}
        self.assertLess(srm_p_value(counts, [1, 2, 3, 4]), 0.001)

    def test_zero_probability_arm(self):
        self.assertEqual(srm_p_value({0: 500, 1: 0, 2: 500}, [1, 0, 1]), 1.0)
        self.assertEqual(srm_p_value({0: 500, 1: 5, 2: 495}, [1, 0, 1]), 0.0)

    def test_empty_and_single_bucket(self):
        self.assertEqual(srm_p_value({0: 0, 1: 0}), 1.0)
        self.assertEqual(srm_p_value({0: 10}), 1.0)

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            srm_p_value({})
        with self.assertRaises(ValueError):
            srm_p_value({0: -1, 1: 9})
        with self.assertRaises(ValueError):
            srm_p_value({0: 5, 1: 5}, [1, 1, 1])  # 比例长度不匹配

    def test_sparse_dict_aligned(self):
        # 缺失的桶编号按零样本处理而不是被静默忽略
        report = srm_report({0: 100, 2: 100}, [1, 1, 1])
        self.assertEqual(report["status"], "insufficient_sample")
        self.assertEqual(report["zero_sample_arms"], [1])


class SrmReportTests(unittest.TestCase):
    def test_imbalance_localizes_specific_arm(self):
        # 人为污染第 2 号臂，其余臂保持期望比例
        counts = {0: 2500, 1: 2500, 2: 3200, 3: 2500, 4: 2500}
        report = srm_report(counts)
        self.assertEqual(report["status"], "imbalance")
        self.assertLess(report["p_value"], report["alpha"])
        self.assertEqual(report["deviating_arms"], [2])
        self.assertTrue(report["per_arm"][2]["deviates"])
        self.assertFalse(report["per_arm"][0]["deviates"])
        self.assertIn("臂 2", report["alert"])

    def test_ok_status(self):
        report = srm_report({i: 2000 for i in range(4)})
        self.assertEqual(report["status"], "ok")
        self.assertGreaterEqual(report["p_value"], report["alpha"])
        self.assertEqual(report["deviating_arms"], [])

    def test_insufficient_sample_zero_arm(self):
        report = srm_report({0: 50, 1: 0, 2: 50})
        self.assertEqual(report["status"], "insufficient_sample")
        self.assertIsNone(report["p_value"])
        self.assertEqual(report["zero_sample_arms"], [1])
        self.assertIn("样本不足", report["alert"])

    def test_insufficient_sample_sparse_expected(self):
        # 100 个样本分到 0.01 概率的臂，期望只有 1 < 5
        report = srm_report({0: 99, 1: 1}, [0.99, 0.01], min_expected=5.0)
        self.assertEqual(report["status"], "insufficient_sample")

    def test_config_error_bad_ratio(self):
        report = srm_report({0: 10, 1: 10}, [1, 0])
        self.assertEqual(report["status"], "config_error")
        # 全零比例
        report2 = srm_report({0: 10, 1: 10}, [0, 0])
        self.assertEqual(report2["status"], "config_error")
        # 长度不匹配
        report3 = srm_report({0: 10, 1: 10}, [1, 1, 1])
        self.assertEqual(report3["status"], "config_error")

    def test_config_error_zero_probability_violation(self):
        report = srm_report({0: 100, 1: 3, 2: 97}, [0.5, 0.0, 0.5])
        self.assertEqual(report["status"], "config_error")
        self.assertEqual(report["deviating_arms"], [1])

    def test_explainability_fields_complete(self):
        report = srm_report({0: 230, 1: 270}, [1, 1])
        for field in ("status", "alert", "per_arm", "statistic", "p_value",
                      "expected_ratio", "deviating_arms", "total"):
            self.assertIn(field, report)
        entry = report["per_arm"][0]
        self.assertEqual(
            set(entry),
            {"arm", "observed", "expected", "expected_ratio",
             "observed_ratio", "adjusted_residual", "deviates", "zero_sample"},
        )
        self.assertAlmostEqual(entry["observed_ratio"] + report["per_arm"][1]["observed_ratio"], 1.0)


class EndToEndTests(unittest.TestCase):
    def test_healthy_multi_arm_pipeline(self):
        ids = _users(5000)
        report = run_srm_check(ids, "launch-42", 4, [1, 2, 3, 4])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["buckets"], 4)
        self.assertEqual(report["unique_users"], 5000)
        self.assertEqual(report["duplicates"], 0)
        self.assertEqual(sum(arm["observed"] for arm in report["per_arm"]), 5000)

    def test_stability_across_runs(self):
        ids = _users(1000)
        r1 = run_srm_check(ids, "stable-salt", 5, [3, 1, 2, 1, 3])
        r2 = run_srm_check(ids, "stable-salt", 5, [3, 1, 2, 1, 3])
        self.assertEqual(r1["assignments"], r2["assignments"])
        self.assertEqual(
            [a["observed"] for a in r1["per_arm"]],
            [a["observed"] for a in r2["per_arm"]],
        )

    def test_detect_tampered_arm_in_many_arm_experiment(self):
        # 8 臂实验：模拟分桶后把第 6 臂的 300 个用户错误地挪到第 3 臂
        salt = "multi-arm"
        ratio = [1] * 8
        assignments = {u: arm_bucket(u, salt, 8, ratio) for u in _users(16000)}
        counts = {i: 0 for i in range(8)}
        moved = 0
        for uid, arm in assignments.items():
            if arm == 6 and moved < 300:
                arm = 3
                moved += 1
            counts[arm] += 1
        report = srm_report(counts, ratio)
        self.assertEqual(report["status"], "imbalance")
        self.assertEqual(set(report["deviating_arms"]), {3, 6})
        # 残差符号可解释：3 号臂超员（正），6 号臂缺员（负）
        residuals = {e["arm"]: e["adjusted_residual"] for e in report["per_arm"]}
        self.assertGreater(residuals[3], 0)
        self.assertLess(residuals[6], 0)

    def test_end_to_end_with_duplicates(self):
        ids = _users(2000) + _users(200)
        report = run_srm_check(ids, "dup-salt", 3, [1, 1, 1])
        self.assertEqual(report["duplicates"], 200)
        self.assertEqual(report["unique_users"], 2000)
        self.assertIn("重复用户 ID", report["alert"])

    def test_config_error_does_not_raise(self):
        report = run_srm_check(_users(100), "s", 3, [1, 1])  # 比例长度错误
        self.assertEqual(report["status"], "config_error")
        report2 = run_srm_check(_users(100), "s", 0)          # 桶数非法
        self.assertEqual(report2["status"], "config_error")

    def test_extreme_many_buckets_small_sample(self):
        report = run_srm_check(_users(20), "s", 20)
        self.assertEqual(report["status"], "insufficient_sample")
        self.assertTrue(len(report["zero_sample_arms"]) > 0)

    def test_large_bucket_count_valid(self):
        report = run_srm_check(_users(20000), "s", 100)
        self.assertIn(report["status"], ("ok", "insufficient_sample", "imbalance"))
        self.assertEqual(report["buckets"], 100)

    def test_min_expected_threshold_respected(self):
        report = srm_report({0: 10000, 1: 1}, [0.9999, 0.0001])
        self.assertEqual(report["status"], "insufficient_sample")
        healthy = srm_report({0: 9999, 1: 100}, [0.99, 0.01])
        self.assertNotEqual(healthy["status"], "config_error")

    def test_all_zero_counts_is_insufficient_not_config_error(self):
        report = srm_report({0: 0, 1: 0, 2: 0})
        self.assertEqual(report["status"], "insufficient_sample")
        self.assertEqual(report["zero_sample_arms"], [0, 1, 2])
        end_to_end = run_srm_check([], "s", 3)
        self.assertEqual(end_to_end["status"], "insufficient_sample")

    def test_chi2_aggrees_with_brute_force_two_arm(self):
        # 任意两组计数：卡方 p 与直接 z 公式一致
        from abkit.stats import normal_cdf
        for a, b in ((10, 20), (480, 520), (990, 1010)):
            n = a + b
            z = (a / n - 0.5) / (0.25 / n) ** 0.5
            self.assertAlmostEqual(
                srm_p_value({0: a, 1: b}),
                2 * (1 - normal_cdf(abs(z))),
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
