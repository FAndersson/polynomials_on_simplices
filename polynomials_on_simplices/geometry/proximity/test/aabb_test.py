import unittest

import numpy as np

from polynomials_on_simplices.geometry.proximity.aabb import (
    corner, create, diameter, empty, half_diagonal, intersection, is_empty, is_valid, midpoint, union, unit, volume)


class TestAABB(unittest.TestCase):
    def test_create(self):
        points = np.array([[-1, 4, -3], [2, -2, 6]])
        aabb = create(points)
        self.assertTrue(np.array_equal(aabb[0], np.array([-1, -2, -3])))
        self.assertTrue(np.array_equal(aabb[1], np.array([2, 4, 6])))

    def test_empty(self):
        self.assertTrue(is_empty(empty()))
        self.assertTrue(not is_valid(empty()))

    def test_valid(self):
        self.assertTrue(not is_valid(empty()))
        self.assertTrue(not is_valid(((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0))))
        self.assertTrue(is_valid(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))))

    def test_midpoint(self):
        self.assertTrue(midpoint(empty()) is None)
        self.assertTrue(np.array_equal(midpoint(unit()), np.array([0.5, 0.5, 0.5])))

    def test_half_diagonal(self):
        self.assertTrue(half_diagonal(empty()) is None)
        self.assertTrue(np.array_equal(half_diagonal(unit()), np.array([0.5, 0.5, 0.5])))
        self.assertTrue(np.array_equal(half_diagonal((unit()[0], 2 * unit()[1])), np.array([1.0, 1.0, 1.0])))

    def test_diameter(self):
        self.assertTrue(diameter(empty()) is None)
        self.assertTrue(diameter(unit()), np.sqrt(3))
        self.assertTrue(diameter((unit()[0], 2 * unit()[1])), np.sqrt(12))

    def test_volume(self):
        self.assertTrue(volume(empty()) is None)
        self.assertTrue(volume(unit()) == 1.0)
        self.assertTrue(volume((unit()[0], 2 * unit()[1])) == 8.0)
        self.assertTrue(volume((np.zeros(3), np.zeros(3))) == 0.0)

    @staticmethod
    def _generate_random_aabb():
        aabb = (20 * np.random.rand(3) - 10), (20 * np.random.rand(3) - 10)
        for i in range(3):
            if aabb[0][i] > aabb[1][i]:
                aabb[0][i], aabb[1][i] = aabb[1][i], aabb[0][i]
        return aabb

    def test_union(self):
        aabb1 = unit()
        aabb2 = ((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))
        aabb = union(aabb1, aabb2)
        self.assertTrue(np.array_equal(aabb[0], np.zeros(3)))
        self.assertTrue(np.array_equal(aabb[1], 3 * np.ones(3)))

        aabb1 = self._generate_random_aabb()
        aabb2 = empty()
        aabb = union(aabb1, aabb2)
        self.assertTrue(np.array_equal(aabb[0], aabb1[0]))
        self.assertTrue(np.array_equal(aabb[1], aabb1[1]))

    def test_intersection(self):
        aabb1 = unit()
        aabb2 = ((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))
        self.assertTrue(is_empty(intersection(aabb1, aabb2)))
        aabb2 = ((0.5, 0.5, 0.5), (3.0, 3.0, 3.0))
        aabb = intersection(aabb1, aabb2)
        self.assertTrue(np.array_equal(aabb[0], 0.5 * np.ones(3)))
        self.assertTrue(np.array_equal(aabb[1], np.ones(3)))

        aabb1 = self._generate_random_aabb()
        aabb2 = empty()
        aabb = intersection(aabb1, aabb2)
        self.assertTrue(is_empty(aabb))

    def test_corner(self):
        aabb = ((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))
        expected_corners = np.array([
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 2.0],
            [2.0, 3.0, 2.0],
            [3.0, 3.0, 2.0],
            [2.0, 2.0, 3.0],
            [3.0, 2.0, 3.0],
            [2.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ])

        for i in range(8):
            c = corner(aabb, i)
            self.assertTrue(np.array_equal(expected_corners[i], c))


if __name__ == "__main__":
    unittest.main()
