import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest

from polynomials_on_simplices.calculus.error_measures import relative_error
from polynomials_on_simplices.calculus.quadrature import (
    quadrature_interval_fixed, quadrature_tetrahedron, quadrature_tetrahedron_fixed,
    quadrature_tetrahedron_midpoint_rule, quadrature_triangle, quadrature_triangle_fixed,
    quadrature_triangle_midpoint_rule, quadrature_unit_interval_fixed, quadrature_unit_tetrahedron,
    quadrature_unit_tetrahedron_fixed, quadrature_unit_triangle, quadrature_unit_triangle_fixed)
from polynomials_on_simplices.geometry.primitives.simplex import unit, volume
from polynomials_on_simplices.polynomial.polynomials_simplex_bernstein_basis import bernstein_basis_simplex
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_bernstein_basis import bernstein_basis


class TestUnitInterval(unittest.TestCase):
    """Unit tests for quadrature over the unit interval"""

    def test_sin(self):
        """Verify quadrature of the sin function"""
        def fn(x):
            return math.sin(math.pi * x)
        res = math.pi * quadrature_unit_interval_fixed(fn, 5)
        self.assertTrue(relative_error(2.0, res) < 1e-3)

    def degree_r_bernstein_basis_eval(self, r):
        expected_results = [1 / 2, 1 / 3, 1 / 4, 1 / 5]
        for i in range(1, r + 1):
            basis_fns = bernstein_basis(i, 1)
            for basis_fn in basis_fns:
                res = quadrature_unit_interval_fixed(basis_fn, i)
                self.assertAlmostEqual(expected_results[i - 1], res)

    def test_quadrature_unit_interval(self):
        """Verify that the degree r quadrature is exact for Bernstein polynomials of degree <= r"""
        for r in range(1, 5):
            self.degree_r_bernstein_basis_eval(r)


class TestInterval(unittest.TestCase):
    """Unit tests for quadrature over arbitrary intervals"""

    def test_sin(self):
        """Verify quadrature of the sin function"""
        res = quadrature_interval_fixed(math.sin, 0, math.pi, 5)
        self.assertTrue(relative_error(2.0, res) < 1e-3)

    def test_degree_r_polynomials_fixed(self):
        """Verify quadrature of some polynomials"""
        def fn(x):
            return x
        res = quadrature_interval_fixed(fn, 1, 3, 1)
        self.assertAlmostEqual(res, 4)

        def fn(x):
            return x**2
        res = quadrature_interval_fixed(fn, -2, 2, 2)
        self.assertAlmostEqual(res, 16 / 3)

        def fn(x):
            return x**3
        res = quadrature_interval_fixed(fn, -3, 1, 3)
        self.assertAlmostEqual(res, -20)


class TestUnitTriangle(unittest.TestCase):
    """Unit tests for quadrature over the unit triangle"""

    def degree_r_bernstein_basis_eval(self, r):
        expected_results = [1 / 6, 1 / 12, 1 / 20, 1 / 30]
        for i in range(1, r + 1):
            basis_fns = bernstein_basis(i, 2)
            for basis_fn in basis_fns:
                res = quadrature_unit_triangle_fixed(lambda x1, x2: basis_fn([x1, x2]), i)
                self.assertAlmostEqual(expected_results[i - 1], res)

    def test_quadrature_unit_triangle_fixed(self):
        """Verify that the degree r quadrature is exact for Bernstein polynomials of degree <= r"""
        for r in range(1, 5):
            self.degree_r_bernstein_basis_eval(r)

    def test_quadrature_unit_triangle(self):
        """Verify that quadrature is accurate for Bernstein polynomials"""
        expected_results = [1 / 6, 1 / 12, 1 / 20, 1 / 30]
        for i in range(1, 5):
            basis_fns = bernstein_basis(i, 2)
            for basis_fn in basis_fns:
                res = quadrature_unit_triangle(lambda x1, x2: basis_fn([x1, x2]))
                self.assertAlmostEqual(expected_results[i - 1], res)


class TestTriangle(unittest.TestCase):
    """Unit tests for quadrature over arbitrary triangles"""

    def setUp(self):
        self.vertices = np.random.rand(3, 2)
        area = volume(self.vertices)
        area0 = volume(unit(2))
        j = area / area0
        self.expected_results = j * np.array([1 / 6, 1 / 12, 1 / 20, 1 / 30])

    def degree_r_bernstein_basis_eval(self, r):
        # Make sure that the degree r quadrature rule is correct for all Bernstein basis polynomials of degree <= r
        for i in range(1, r + 1):
            basis_fns = bernstein_basis_simplex(i, self.vertices)
            for basis_fn in basis_fns:
                def f(x, y):
                    return basis_fn((x, y))
                res = quadrature_triangle_fixed(f, self.vertices, r)
                self.assertAlmostEqual(self.expected_results[i - 1], res)

    def test_quadrature_triangle_fixed(self):
        """Verify that the degree r quadrature is exact for Bernstein polynomials of degree <= r"""
        for r in range(1, 5):
            self.degree_r_bernstein_basis_eval(r)

    def test_quadrature_triangle(self):
        """Verify that the degree r quadrature is accurate for Bernstein polynomials"""
        # Make sure that the quadrature rule is accurate for all Bernstein basis polynomials of degree <= 5
        for i in range(1, 5):
            basis_fns = bernstein_basis_simplex(i, self.vertices)
            for basis_fn in basis_fns:
                def f(x, y):
                    return basis_fn((x, y))
                res = quadrature_triangle(f, self.vertices)
                self.assertAlmostEqual(self.expected_results[i - 1], res)


class TestUnitTetrahedron(unittest.TestCase):
    """Unit tests for quadrature over the unit tetrahedron"""

    def degree_r_bernstein_basis_eval(self, r):
        # Make sure that the degree r quadrature rule is correct for all Bernstein basis polynomials of degree <= r
        expected_results = [1 / 24, 1 / 60, 1 / 120, 1 / 210]
        for i in range(1, r + 1):
            basis_fns = bernstein_basis(i, 3)
            for basis_fn in basis_fns:
                res = quadrature_unit_tetrahedron_fixed(lambda x1, x2, x3: basis_fn([x1, x2, x3]), i)
                self.assertAlmostEqual(expected_results[i - 1], res)

    def test_quadrature_unit_tetrahedron_fixed(self):
        """Verify that the degree r quadrature is exact for Bernstein polynomials of degree <= r"""
        for r in range(1, 5):
            self.degree_r_bernstein_basis_eval(r)

    def test_quadrature_unit_tetrahedron(self):
        """Verify that quadrature is accurate for Bernstein polynomials"""
        # Make sure that the quadrature rule is accurate for all Bernstein basis polynomials of degree <= 5
        expected_results = [1 / 24, 1 / 60, 1 / 120, 1 / 210]
        for i in range(1, 5):
            basis_fns = bernstein_basis(i, 3)
            for basis_fn in basis_fns:
                res = quadrature_unit_tetrahedron(lambda x1, x2, x3: basis_fn([x1, x2, x3]))
                self.assertAlmostEqual(expected_results[i - 1], res)


class TestTetrahedron(unittest.TestCase):
    """Unit tests for quadrature over arbitrary tetrahedra"""

    def setUp(self):
        self.vertices = np.random.rand(4, 3)
        v = volume(self.vertices)
        v0 = volume(unit(3))
        j = v / v0
        self.expected_results = j * np.array([1 / 24, 1 / 60, 1 / 120, 1 / 210])

    def degree_r_bernstein_basis_eval(self, r):
        # Make sure that the degree r quadrature rule is correct for all Bernstein basis polynomials of degree <= r
        for i in range(1, r + 1):
            basis_fns = bernstein_basis_simplex(i, self.vertices)
            for basis_fn in basis_fns:
                def f(x, y, z):
                    return basis_fn((x, y, z))
                res = quadrature_tetrahedron_fixed(f, self.vertices, r)
                self.assertAlmostEqual(self.expected_results[i - 1], res)

    def test_quadrature_tetrahedron_fixed(self):
        """Verify that the degree r quadrature is exact for Bernstein polynomials of degree <= r"""
        for r in range(1, 5):
            self.degree_r_bernstein_basis_eval(r)

    @pytest.mark.slow
    def test_quadrature_tetrahedron(self):
        """Verify that quadrature is accurate for Bernstein polynomials"""
        # Make sure that the quadrature rule is accurate for all Bernstein basis polynomials of degree <= 5
        for i in range(1, 5):
            basis_fns = bernstein_basis_simplex(i, self.vertices)
            for basis_fn in basis_fns:
                def f(x, y, z):
                    return basis_fn((x, y, z))
                res = quadrature_tetrahedron(f, self.vertices)
                self.assertAlmostEqual(self.expected_results[i - 1], res)


class TestTriangleMidpointRule(unittest.TestCase):
    @staticmethod
    def convergence_plot_interactive():
        basis_fns = bernstein_basis(5, 2)
        expected_value = 1 / 42
        fig = plt.figure()
        for basis_fn in basis_fns:
            n = range(1, 15)
            rel_errors = [0.0] * len(n)
            for i in range(len(n)):
                integral = quadrature_triangle_midpoint_rule(lambda x1, x2: basis_fn((x1, x2)), unit(2), n[i])
                rel_errors[i] = relative_error(expected_value, integral)
            plt.plot(n, rel_errors, figure=fig)
        plt.xlabel("Unit triangle grid resolution")
        plt.ylabel("Relative error")
        plt.show()


class TestTetrahedronMidpointRule(unittest.TestCase):
    @staticmethod
    def convergence_plot_interactive():
        basis_fns = bernstein_basis(5, 3)
        expected_value = 1 / 336
        fig = plt.figure()
        for basis_fn in basis_fns:
            n = range(1, 15)
            rel_errors = [0.0] * len(n)
            for i in range(len(n)):
                integral = quadrature_tetrahedron_midpoint_rule(lambda x1, x2, x3: basis_fn((x1, x2, x3)), unit(3),
                                                                n[i])
                rel_errors[i] = relative_error(expected_value, integral)
            plt.plot(n, rel_errors, figure=fig)
        plt.xlabel("Unit triangle grid resolution")
        plt.ylabel("Relative error")
        plt.show()


if __name__ == "__main__":
    unittest.main()
