import unittest

from abkit.assignment import bucket
from abkit.multiple import bh
from abkit.power import required_sample_size
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


if __name__ == "__main__":
    unittest.main()
