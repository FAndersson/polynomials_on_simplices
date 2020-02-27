import sys

import numpy as np
import pytest

from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_bernstein_basis_calculus import (
    integrate_bernstein_basis_fn_simplex, integrate_bernstein_basis_fn_unit_simplex)
from polynomials_on_simplices.calculus.quadrature import (
    quadrature_interval_fixed, quadrature_tetrahedron_fixed, quadrature_triangle_fixed)
from polynomials_on_simplices.polynomial.polynomials_simplex_bernstein_basis import bernstein_basis_fn_simplex


def test_integrate_unit_simplex():
    # 1D
    expected_results = [1, 1 / 2, 1 / 3, 1 / 4]
    for r in range(4):
        result = integrate_bernstein_basis_fn_unit_simplex(r, 1)
        assert abs(result - expected_results[r]) < 1e-10

    # 2D
    expected_results = [1 / 2, 1 / 6, 1 / 12, 1 / 20]
    for r in range(4):
        result = integrate_bernstein_basis_fn_unit_simplex(r, 2)
        assert abs(result - expected_results[r]) < 1e-10

    # 3D
    expected_results = [1 / 6, 1 / 24, 1 / 60, 1 / 120]
    for r in range(4):
        result = integrate_bernstein_basis_fn_unit_simplex(r, 3)
        assert abs(result - expected_results[r]) < 1e-10


def test_integrate_simplex():
    # 1D
    vertices = np.array([[1], [3]])
    r = 2
    result = integrate_bernstein_basis_fn_simplex(r, vertices)
    p = bernstein_basis_fn_simplex(0, r, vertices)
    expected_result = quadrature_interval_fixed(p, 1, 3, r)
    assert abs(result - expected_result) < 1e-10

    # 2D
    vertices = np.array([[2, 3], [4, 3], [2, 5]])
    r = 2
    result = integrate_bernstein_basis_fn_simplex(r, vertices)
    p = bernstein_basis_fn_simplex((0, 0), r, vertices)

    def f(x, y):
        return p([x, y])
    expected_result = quadrature_triangle_fixed(f, vertices, r)
    assert abs(result - expected_result) < 1e-10

    # 3D
    vertices = np.array([[1, 1, 1], [3, 2, 0], [2, 3, 0], [3, 3, 3]])
    r = 2
    result = integrate_bernstein_basis_fn_simplex(r, vertices)
    p = bernstein_basis_fn_simplex((0, 0, 0), r, vertices)

    def f(x, y, z):
        return p([x, y, z])
    expected_result = quadrature_tetrahedron_fixed(f, vertices, r)
    assert abs(result - expected_result) < 1e-10


if __name__ == '__main__':
    pytest.main(sys.argv)
