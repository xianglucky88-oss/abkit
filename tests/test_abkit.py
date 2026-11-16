import unittest

from abkit.assignment import bucket
from abkit.multiple import bh
from abkit.power import required_sample_size
from abkit.sequential import (
    SPENDING_FAMILIES,
    GroupSequentialDesign,
    alpha_spending,
    monitor,
    z_test,
)
from abkit.simulator import batch_evaluate, calibration, run_scenario, simulate
from abkit.stats import normal_ppf, wilson_interval
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


class SequentialDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obf = GroupSequentialDesign(looks=5, alpha=0.05)
        cls.pocock = GroupSequentialDesign(looks=5, alpha=0.05, spending="pocock")
        cls.naive = GroupSequentialDesign(looks=5, alpha=0.05, spending="unadjusted")

    def test_spending_endpoints(self):
        for family in SPENDING_FAMILIES:
            self.assertAlmostEqual(alpha_spending(1.0, 0.05, family), 0.05)
            self.assertLess(alpha_spending(0.5, 0.05, family), 0.05)
            self.assertGreater(alpha_spending(0.5, 0.05, family), 0.0)

    def test_spending_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            alpha_spending(0.0)
        with self.assertRaises(ValueError):
            alpha_spending(1.2)
        with self.assertRaises(ValueError):
            alpha_spending(0.5, 0.05, "nope")
        with self.assertRaises(ValueError):
            GroupSequentialDesign(looks=0)
        with self.assertRaises(ValueError):
            GroupSequentialDesign(looks=3, info_fractions=[0.5, 0.4])

    def test_single_look_matches_z_test(self):
        design = GroupSequentialDesign(looks=1, alpha=0.05)
        self.assertAlmostEqual(design.critical_value(1), normal_ppf(0.975), places=6)
        self.assertEqual(design.spent, (0.05,))

    def test_obf_boundaries_match_reference(self):
        # Lan-DeMets OBF 型消耗、5 次等距查看的参考临界值
        reference = (4.3826, 3.0998, 2.5536, 2.2540, 2.0636)
        for got, want in zip(self.obf.boundaries_, reference):
            self.assertAlmostEqual(got, want, places=2)

    def test_pocock_boundaries_nearly_constant(self):
        spread = max(self.pocock.boundaries_) - min(self.pocock.boundaries_)
        self.assertLess(spread, 0.1)

    def test_design_spends_exact_alpha(self):
        self.assertAlmostEqual(self.obf.spent[-1], 0.05)
        self.assertAlmostEqual(self.pocock.spent[-1], 0.05)

    def test_unadjusted_design_inflates_type1(self):
        # 每次查看都用完整 alpha：真实累计一类错误约 0.142
        self.assertTrue(all(abs(b - normal_ppf(0.975)) < 1e-9 for b in self.naive.boundaries_))
        self.assertGreater(self.naive.spent[-1], 0.10)
        self.assertLess(self.naive.spent[-1], 0.20)

    def test_custom_info_fractions_complete_to_one(self):
        design = GroupSequentialDesign(looks=2, info_fractions=[0.5], alpha=0.05)
        self.assertEqual(design.info_fractions, (0.5, 1.0))
        self.assertAlmostEqual(design.spent[-1], 0.05)

    def test_z_test_single_look_semantics(self):
        result = z_test(2.0, 1.0)
        self.assertTrue(result["stop"])
        self.assertEqual(z_test(1.0, 0.0)["stop"], False)


class MonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obf = GroupSequentialDesign(looks=5, alpha=0.05)

    def test_efficacy_early_stop(self):
        control, treatment = simulate(400, 0.8, "test:efficacy")
        report = monitor(self.obf, control, treatment)
        self.assertTrue(report.significant)
        self.assertEqual(report.stop_reason, "efficacy")
        self.assertEqual(report.conclusion, "positive effect")
        self.assertLess(report.looks_used, 5)

    def test_null_runs_to_max_looks(self):
        control, treatment = simulate(400, 0.0, "test:null")
        report = monitor(self.obf, control, treatment)
        self.assertFalse(report.significant)
        self.assertEqual(report.stop_reason, "max_looks")
        self.assertEqual(report.looks_used, 5)
        self.assertEqual(report.conclusion, "inconclusive")

    def test_negative_effect_direction(self):
        control, treatment = simulate(400, -0.8, "test:negative")
        report = monitor(self.obf, control, treatment)
        self.assertTrue(report.significant)
        self.assertEqual(report.conclusion, "negative effect")

    def test_records_link_information_to_decision(self):
        control, treatment = simulate(100, 0.0, "test:records")
        report = monitor(self.obf, control, treatment)
        self.assertEqual(len(report.records), report.looks_used)
        for record, fraction in zip(report.records, self.obf.info_fractions):
            self.assertEqual(record.info_fraction, fraction)
            self.assertGreater(record.critical, 0)
            self.assertGreater(record.n_control, 0)
            self.assertIn(record.reason, ("efficacy", "continue", "max_looks"))
        for record in report.records[:-1]:
            self.assertEqual(record.reason, "continue")
            self.assertFalse(record.stop)
        self.assertTrue(report.records[-1].stop)

    def test_empty_samples_are_inconclusive(self):
        report = monitor(self.obf, [], [])
        self.assertFalse(report.significant)
        self.assertEqual(report.stop_reason, "max_looks")
        self.assertEqual(report.looks_used, 5)
        self.assertEqual(report.estimate, 0.0)


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obf = GroupSequentialDesign(looks=5, alpha=0.05)
        cls.naive = GroupSequentialDesign(looks=5, alpha=0.05, spending="unadjusted")

    def test_batch_is_deterministic(self):
        first = batch_evaluate(repetitions=50, n_per_arm=200, design=self.obf, seed="det")
        second = batch_evaluate(repetitions=50, n_per_arm=200, design=self.obf, seed="det")
        self.assertEqual(first, second)

    def test_seed_changes_results(self):
        first = batch_evaluate(repetitions=50, n_per_arm=200, design=self.obf, seed="a")
        second = batch_evaluate(repetitions=50, n_per_arm=200, design=self.obf, seed="b")
        self.assertNotEqual(first["positive"]["mean_estimate"],
                              second["positive"]["mean_estimate"])

    def test_zero_repetitions_is_safe(self):
        summary = run_scenario("null", 0.0, repetitions=0, design=self.obf)
        self.assertEqual(summary["repetitions"], 0)
        self.assertEqual(summary["rejection_rate"], 0.0)
        self.assertEqual(summary["rejection_ci"], (0.0, 1.0))
        self.assertEqual(summary["mean_looks_used"], 0.0)

    def test_empty_samples_never_reject(self):
        summary = run_scenario("null", 0.0, repetitions=5, n_per_arm=0,
                               design=self.obf, seed="empty")
        self.assertEqual(summary["rejection_rate"], 0.0)
        self.assertEqual(summary["stop_reasons"], {"max_looks": 5})

    def test_type1_controlled_and_naive_inflates(self):
        cal_obf = run_scenario("null", 0.0, 400, 400, self.obf, "t1")
        cal_naive = run_scenario("null", 0.0, 400, 400, self.naive, "t1")
        self.assertLessEqual(cal_obf["rejection_rate"], 0.10)
        self.assertGreaterEqual(cal_naive["rejection_rate"], 0.08)
        self.assertGreater(cal_naive["rejection_rate"], cal_obf["rejection_rate"])

    def test_power_and_direction(self):
        positive = run_scenario("positive", 0.2, 400, 400, self.obf, "power")
        negative = run_scenario("negative", -0.2, 400, 400, self.obf, "power")
        self.assertGreaterEqual(positive["rejection_rate"], 0.6)
        self.assertGreaterEqual(negative["rejection_rate"], 0.6)
        self.assertGreaterEqual(positive["direction_correct_rate"], 0.95)
        self.assertGreaterEqual(negative["direction_correct_rate"], 0.95)

    def test_calibration_compat(self):
        cal = calibration(10)
        self.assertGreaterEqual(cal["rejection_rate"], 0.0)
        self.assertLessEqual(cal["rejection_rate"], 1.0)
        self.assertIn("mean_estimate", cal)

    def test_wilson_interval_bounds(self):
        lo, hi = wilson_interval(5, 100)
        self.assertLess(lo, 0.05)
        self.assertGreater(hi, 0.05)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
