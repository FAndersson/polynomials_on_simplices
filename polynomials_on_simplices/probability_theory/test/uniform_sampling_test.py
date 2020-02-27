import unittest

from polynomials_on_simplices.geometry.primitives.simplex import inside_simplex, unit
from polynomials_on_simplices.probability_theory.uniform_sampling import (
    closed_unit_interval_sample, left_closed_interval_sample, nsimplex_sampling, open_unit_interval_sample,
    right_closed_interval_sample)


class TestUnitIntervalSample(unittest.TestCase):
    def test_closed_unit_interval_sample(self):
        s = closed_unit_interval_sample()
        self.assertTrue(s >= 0.0)
        self.assertTrue(s <= 1.0)

    def test_left_closed_interval_sample(self):
        s = left_closed_interval_sample()
        self.assertTrue(s >= 0.0)
        self.assertTrue(s < 1.0)

    def test_right_closed_interval_sample(self):
        s = right_closed_interval_sample()
        self.assertTrue(s > 0.0)
        self.assertTrue(s <= 1.0)

    def test_open_unit_interval_sample(self):
        s = open_unit_interval_sample()
        self.assertTrue(s > 0.0)
        self.assertTrue(s < 1.0)


class TestNSimplexSampling(unittest.TestCase):
    def test_inside(self):
        points = nsimplex_sampling(3, 3)
        for i in range(3):
            self.assertTrue(inside_simplex(points[i], unit(3)))


if __name__ == '__main__':
    unittest.main()
