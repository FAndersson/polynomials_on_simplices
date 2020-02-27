import sys

import numpy as np
import pytest

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_lagrange_basis_calculus import (
    integrate_lagrange_basis_fn_simplex, integrate_lagrange_basis_fn_unit_simplex)
from polynomials_on_simplices.calculus.quadrature import (
    quadrature_interval_fixed, quadrature_tetrahedron_fixed, quadrature_triangle_fixed)
from polynomials_on_simplices.polynomial.polynomials_base import get_dimension
from polynomials_on_simplices.polynomial.polynomials_simplex_lagrange_basis import lagrange_basis_fn_simplex


def test_integrate_unit_simplex():
    # 1D
    expected_results = [[1], [1 / 2, 1 / 2], [1 / 6, 2 / 3, 1 / 6], [1 / 8, 3 / 8, 3 / 8, 1 / 8]]
    for r in range(4):
        for i in range(get_dimension(r, 1)):
            result = integrate_lagrange_basis_fn_unit_simplex(i, r)
            assert abs(result - expected_results[r][i]) < 1e-10

    # 2D
    expected_results = [
        [1 / 2],
        [1 / 6, 1 / 6, 1 / 6],
        [0, 1 / 6, 0, 1 / 6, 1 / 6, 0],
        [1 / 60, 3 / 80, 3 / 80, 1 / 60, 3 / 80, 9 / 40, 3 / 80, 3 / 80, 3 / 80, 1 / 60]
    ]
    for r in range(4):
        i = 0
        for nu in multiindex.generate_all(2, r):
            result = integrate_lagrange_basis_fn_unit_simplex(nu, r)
            assert abs(result - expected_results[r][i]) < 1e-10
            i += 1

    # 3D
    expected_results = [
        [1 / 6],
        [1 / 24, 1 / 24, 1 / 24, 1 / 24],
        [-1 / 120, 1 / 30, -1 / 120, 1 / 30, 1 / 30, -1 / 120, 1 / 30, 1 / 30, 1 / 30, -1 / 120],
        [1 / 240, 0, 0, 1 / 240, 0, 3 / 80, 0, 0, 0, 1 / 240, 0, 3 / 80, 0, 3 / 80, 3 / 80, 0, 0, 0, 0, 1 / 240]
    ]
    for r in range(4):
        i = 0
        for nu in multiindex.generate_all(3, r):
            result = integrate_lagrange_basis_fn_unit_simplex(nu, r)
            assert abs(result - expected_results[r][i]) < 1e-10
            i += 1


def test_integrate_simplex():
    # 1D
    vertices = np.array([[1], [3]])
    r = 2
    for i in range(3):
        result = integrate_lagrange_basis_fn_simplex(i, r, vertices)
        p = lagrange_basis_fn_simplex(i, r, vertices)
        expected_result = quadrature_interval_fixed(p, 1, 3, r)
        assert abs(result - expected_result) < 1e-10

    # 2D
    vertices = np.array([[2, 3], [4, 3], [2, 5]])
    r = 2
    for nu in multiindex.generate_all(2, r):
        result = integrate_lagrange_basis_fn_simplex(nu, r, vertices)
        p = lagrange_basis_fn_simplex(nu, r, vertices)

        def f(x, y):
            return p([x, y])
        expected_result = quadrature_triangle_fixed(f, vertices, r)
        assert abs(result - expected_result) < 1e-10

    # 3D
    vertices = np.array([[1, 1, 1], [3, 2, 0], [2, 3, 0], [3, 3, 3]])
    r = 2
    for nu in multiindex.generate_all(3, r):
        result = integrate_lagrange_basis_fn_simplex(nu, r, vertices)
        p = lagrange_basis_fn_simplex(nu, r, vertices)

        def f(x, y, z):
            return p([x, y, z])
        expected_result = quadrature_tetrahedron_fixed(f, vertices, r)
        assert abs(result - expected_result) < 1e-10


if __name__ == '__main__':
    pytest.main(sys.argv)
