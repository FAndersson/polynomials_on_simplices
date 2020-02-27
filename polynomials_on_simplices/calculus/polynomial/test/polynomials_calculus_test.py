import sys

import numpy as np
import pytest

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.finite_difference import (
    central_difference, central_difference_jacobian, second_central_difference)
from polynomials_on_simplices.calculus.polynomial.polynomials_calculus import (
    derivative, gradient, hessian, jacobian, partial_derivatives_array)
from polynomials_on_simplices.polynomial.polynomials_base import get_dimension, polynomials_equal
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import Polynomial, zero_polynomial
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_bernstein_basis import PolynomialBernstein
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import PolynomialLagrange


@pytest.mark.slow
def test_higher_derivatives_bernstein_basis():
    # Test first derivatives
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            for r in [0, 1, 2, 3]:
                dim = get_dimension(r, m)
                if n == 1:
                    a = np.random.random_sample(dim)
                else:
                    a = np.random.random_sample((dim, n))

                p = PolynomialBernstein(a, r, m)

                for i in range(m):
                    if m == 1:
                        def dp_dxi_fd(x):
                            return central_difference(p, x)
                    else:
                        def dp_dxi_fd(x):
                            return central_difference_jacobian(p, n, x)[:, i]
                    alpha = multiindex.unit_multiindex(m, i)
                    assert polynomials_equal(derivative(p, alpha), dp_dxi_fd, r, m, rel_tol=1e-5, abs_tol=1e-6)

    # Test second derivatives
    n = 1
    for m in [1, 2, 3]:
        for r in [0, 1, 2, 3]:
            dim = get_dimension(r, m)
            a = np.random.random_sample(dim)
            p = PolynomialBernstein(a, r, m)

            for i in range(m):
                for j in range(m):
                    if m == 1:
                        def dp_dxi_dxj_fd(x):
                            return second_central_difference(p, x)
                    else:
                        def dp_dxi_dxj_fd(x):
                            return second_central_difference(p, x)[i, j]
                    alpha = multiindex.zero_multiindex(m)
                    alpha += multiindex.unit_multiindex(m, i)
                    alpha += multiindex.unit_multiindex(m, j)
                    assert polynomials_equal(derivative(p, alpha), dp_dxi_dxj_fd, r, m, rel_tol=1e-4, abs_tol=1e-4)


@pytest.mark.slow
def test_higher_derivatives_lagrange_basis():
    # Test first derivatives
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            for r in [0, 1, 2, 3]:
                dim = get_dimension(r, m)
                if n == 1:
                    a = np.random.random_sample(dim)
                else:
                    a = np.random.random_sample((dim, n))

                p = PolynomialLagrange(a, r, m)

                for i in range(m):
                    if m == 1:
                        def dp_dxi_fd(x):
                            return central_difference(p, x)
                    else:
                        def dp_dxi_fd(x):
                            return central_difference_jacobian(p, n, x)[:, i]
                    alpha = multiindex.unit_multiindex(m, i)
                    assert polynomials_equal(derivative(p, alpha), dp_dxi_fd, r, m, rel_tol=1e-5, abs_tol=1e-6)

    # Test second derivatives
    n = 1
    for m in [1, 2, 3]:
        for r in [0, 1, 2, 3]:
            dim = get_dimension(r, m)
            a = np.random.random_sample(dim)
            p = PolynomialLagrange(a, r, m)

            for i in range(m):
                for j in range(m):
                    if m == 1:
                        def dp_dxi_dxj_fd(x):
                            return second_central_difference(p, x)
                    else:
                        def dp_dxi_dxj_fd(x):
                            return second_central_difference(p, x)[i, j]
                    alpha = multiindex.zero_multiindex(m)
                    alpha += multiindex.unit_multiindex(m, i)
                    alpha += multiindex.unit_multiindex(m, j)
                    assert polynomials_equal(derivative(p, alpha), dp_dxi_dxj_fd, r, m, rel_tol=1e-4, abs_tol=1e-4)


def test_higher_derivatives_monomial_basis():
    # Verify the derivative function on polynomials in the monomial basis using finite difference

    # Test first derivatives
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            for r in [0, 1, 2, 3]:
                dim = get_dimension(r, m)
                if n == 1:
                    a = np.random.random_sample(dim)
                else:
                    a = np.random.random_sample((dim, n))

                p = Polynomial(a, r, m)

                for i in range(m):
                    if m == 1:
                        def dp_dxi_fd(x):
                            return central_difference(p, x)
                    else:
                        def dp_dxi_fd(x):
                            return central_difference_jacobian(p, n, x)[:, i]
                    alpha = multiindex.unit_multiindex(m, i)
                    assert polynomials_equal(derivative(p, alpha), dp_dxi_fd, r, m, rel_tol=1e-5)

    # Test second derivatives
    n = 1
    for m in [1, 2, 3]:
        for r in [0, 1, 2, 3]:
            dim = get_dimension(r, m)
            a = np.random.random_sample(dim)
            p = Polynomial(a, r, m)

            for i in range(m):
                for j in range(m):
                    if m == 1:
                        def dp_dxi_dxj_fd(x):
                            return second_central_difference(p, x)
                    else:
                        def dp_dxi_dxj_fd(x):
                            return second_central_difference(p, x)[i, j]
                    alpha = multiindex.zero_multiindex(m)
                    alpha += multiindex.unit_multiindex(m, i)
                    alpha += multiindex.unit_multiindex(m, j)
                    assert polynomials_equal(derivative(p, alpha), dp_dxi_dxj_fd, r, m, rel_tol=1e-4, abs_tol=1e-4)


def test_gradient_monomial_basis():
    # Verify the gradient function on polynomials in the monomial basis using finite difference

    for m in [1, 2, 3]:
        for r in [0, 1, 2, 3]:
            dim = get_dimension(r, m)
            a = np.random.random_sample(dim)

            p = Polynomial(a, r, m)

            g = gradient(p)
            if m == 1:
                x = np.random.rand()
            else:
                x = np.random.rand(m)
            g_fd = central_difference(p, x)
            if m == 1:
                assert abs(g[0](x) - g_fd) < 1e-6
            else:
                for i in range(m):
                    assert abs(g[i](x) - g_fd[i]) < 1e-6


def test_jacobian_monomial_basis():
    for m in [1, 2, 3]:
        for n in [2, 3]:
            for r in [1, 2, 3]:
                dim = get_dimension(r, m)
                a = np.random.random_sample((dim, n))

                p = Polynomial(a, r, m)

                j = jacobian(p)
                assert len(j) == n
                assert len(j[0]) == m
                if m == 1:
                    x = np.random.rand()
                else:
                    x = np.random.rand(m)
                j_fd = central_difference_jacobian(p, n, x)
                for k in range(n):
                    for l in range(m):
                        assert abs(j[k][l](x) - j_fd[k][l]) < 1e-6


def test_hessian():
    # Test Hessian of a cubic polynomial in the monomial basis on a real domain
    # p(x) = 1 + 2x + 3x^2 + 4x^3
    # d^2 p = 6 + 24x
    a = [1, 2, 3, 4]
    p = Polynomial(a, 3, 1)
    h = hessian(p)
    ae = [6, 24]
    pe = Polynomial(ae, 1, 1)
    h_expected = [[pe]]
    assert polynomials_equal(h_expected[0][0], h[0][0], 1, 1)

    # Test Hessian of a cubic polynomial in the monomial basis on a 2d domain
    # p(x) = 1 + 2 x_1 + 3 x_1^2 + 4 x_1^3 + 5x_2 + 6 x_1 x_2 + 7 x_1^2 x_2 + 8 x_2^2 + 9 x_1 x_2^2 + 10 x_2^3
    # d^2 p / dx_1^2 = 6 + 24 x_1 + 14 x_2
    # d^2 p / dx_1 dx_2 = 6 + 14 x_1 + 18 x_2 = d^2 p / dx_2 dx_1
    # d^2 p / dx_2^2 = 16 + 18 x_1 + 60 x_2
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p = Polynomial(a, 3, 2)
    h = hessian(p)
    h_expected = np.empty((2, 2), dtype=object)
    h_expected[0][0] = Polynomial([6, 24, 14], 1, 2)
    h_expected[0][1] = Polynomial([6, 14, 18], 1, 2)
    h_expected[1][0] = Polynomial([6, 14, 18], 1, 2)
    h_expected[1][1] = Polynomial([16, 18, 60], 1, 2)
    for i in range(2):
        for j in range(2):
            assert polynomials_equal(h_expected[i][j], h[i][j], 1, 2)


def test_partial_derivatives_array():
    # Test that it agrees with the gradient for r = 1
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p = Polynomial(a, 3, 2)
    g = gradient(p)
    d = partial_derivatives_array(p, 1)
    for i in range(len(g)):
        assert polynomials_equal(g[i], d[i], 2, 2)

    # Test that it agrees with the Hessian for r = 2
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p = Polynomial(a, 3, 2)
    h = hessian(p)
    d = partial_derivatives_array(p, 2)
    for i in range(len(h)):
        for j in range(len(h[0])):
            assert polynomials_equal(h[i][j], d[i][j], 1, 2)

    # Test array of third derivatives of a cubic polynomial in the monomial basis on a 2d domain
    # p(x) = 1 + 2 x_1 + 3 x_1^2 + 4 x_1^3 + 5x_2 + 6 x_1 x_2 + 7 x_1^2 x_2 + 8 x_2^2 + 9 x_1 x_2^2 + 10 x_2^3
    # d^3 p / dx_1^3 = 24
    # d^3 p / dx_1^2 dx_2 = 14
    # d^3 p / dx_1 dx_2^2 = 18
    # d^3 p / dx_2^3 = 60
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p = Polynomial(a, 3, 2)
    d = partial_derivatives_array(p, 3)
    d_expected = np.empty((2, 2, 2), dtype=object)
    d_expected[0][0][0] = Polynomial([24], 0, 2)
    d_expected[0][0][1] = Polynomial([14], 0, 2)
    d_expected[0][1][0] = Polynomial([14], 0, 2)
    d_expected[1][0][0] = Polynomial([14], 0, 2)
    d_expected[0][1][1] = Polynomial([18], 0, 2)
    d_expected[1][0][1] = Polynomial([18], 0, 2)
    d_expected[1][1][0] = Polynomial([18], 0, 2)
    d_expected[1][1][1] = Polynomial([60], 0, 2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert polynomials_equal(d_expected[i][j][k], d[i][j][k], 0, 2)

    # Test array of fourth derivatives of a quintic polynomial in the monomial basis on a 2d domain
    # Using the polynomial p(x) = x_1^5 for trivial derivatives
    a = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p = Polynomial(a, 5, 2)
    d = partial_derivatives_array(p, 4)
    d_expected = np.empty((2, 2, 2, 2), dtype=object)
    d_expected[0][0][0][0] = Polynomial([0, 120, 0], 1, 2)
    d_expected[0][0][1][0] = zero_polynomial(2, 1)
    d_expected[0][1][0][0] = zero_polynomial(2, 1)
    d_expected[1][0][0][0] = zero_polynomial(2, 1)
    d_expected[0][1][1][0] = zero_polynomial(2, 1)
    d_expected[1][0][1][0] = zero_polynomial(2, 1)
    d_expected[1][1][0][0] = zero_polynomial(2, 1)
    d_expected[1][1][1][0] = zero_polynomial(2, 1)
    d_expected[0][0][0][1] = zero_polynomial(2, 1)
    d_expected[0][0][1][1] = zero_polynomial(2, 1)
    d_expected[0][1][0][1] = zero_polynomial(2, 1)
    d_expected[1][0][0][1] = zero_polynomial(2, 1)
    d_expected[0][1][1][1] = zero_polynomial(2, 1)
    d_expected[1][0][1][1] = zero_polynomial(2, 1)
    d_expected[1][1][0][1] = zero_polynomial(2, 1)
    d_expected[1][1][1][1] = zero_polynomial(2, 1)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    assert polynomials_equal(d_expected[i][j][k][l], d[i][j][k][l], 1, 2)


if __name__ == '__main__':
    pytest.main(sys.argv)
