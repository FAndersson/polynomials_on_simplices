import sys

import numpy as np
import pytest

from polynomials_on_simplices.polynomial.polynomials_base import get_dimension, polynomials_equal
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import Polynomial, unique_identifier_monomial_basis
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_bases import convert_polynomial_to_basis
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_bernstein_basis import (
    PolynomialBernstein, unique_identifier_bernstein_basis)
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import (
    PolynomialLagrange, unique_identifier_lagrange_basis)


@pytest.mark.slow
def test_monomial_to_bernstein():
    r = 3
    for m in [1, 2, 3]:
        dim = get_dimension(r, m)

        # n = 1
        a = np.random.rand(dim)
        p = Polynomial(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_bernstein_basis())
        assert q.basis() == unique_identifier_bernstein_basis()
        assert polynomials_equal(p, q, r, m)

        # n = 2
        a = np.random.random_sample((dim, 2))
        p = Polynomial(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_bernstein_basis())
        assert q.basis() == unique_identifier_bernstein_basis()
        assert polynomials_equal(p, q, r, m)


def test_bernstein_to_monomial():
    r = 3
    for m in [1, 2, 3]:
        dim = get_dimension(r, m)

        # n = 1
        a = np.random.rand(dim)
        p = PolynomialBernstein(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_monomial_basis())
        assert q.basis() == unique_identifier_monomial_basis()
        assert polynomials_equal(p, q, r, m)

        # n = 2
        a = np.random.random_sample((dim, 2))
        p = PolynomialBernstein(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_monomial_basis())
        assert q.basis() == unique_identifier_monomial_basis()
        assert polynomials_equal(p, q, r, m)


def test_monomial_to_lagrange():
    r = 3
    for m in [1, 2, 3]:
        dim = get_dimension(r, m)

        # n = 1
        a = np.random.rand(dim)
        p = Polynomial(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_lagrange_basis())
        assert q.basis() == unique_identifier_lagrange_basis()
        assert polynomials_equal(p, q, r, m)

        # n = 2
        a = np.random.random_sample((dim, 2))
        p = Polynomial(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_lagrange_basis())
        assert q.basis() == unique_identifier_lagrange_basis()
        assert polynomials_equal(p, q, r, m)


def test_lagrange_to_monomial():
    r = 3
    for m in [1, 2, 3]:
        dim = get_dimension(r, m)

        # n = 1
        a = np.random.rand(dim)
        p = PolynomialLagrange(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_monomial_basis())
        assert q.basis() == unique_identifier_monomial_basis()
        assert polynomials_equal(p, q, r, m)

        # n = 2
        a = np.random.random_sample((dim, 2))
        p = PolynomialLagrange(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_monomial_basis())
        assert q.basis() == unique_identifier_monomial_basis()
        assert polynomials_equal(p, q, r, m)


def test_bernstein_to_lagrange():
    r = 3
    for m in [1, 2, 3]:
        dim = get_dimension(r, m)

        # n = 1
        a = np.random.rand(dim)
        p = PolynomialBernstein(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_lagrange_basis())
        assert q.basis() == unique_identifier_lagrange_basis()
        assert polynomials_equal(p, q, r, m)

        # n = 2
        a = np.random.random_sample((dim, 2))
        p = PolynomialBernstein(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_lagrange_basis())
        assert q.basis() == unique_identifier_lagrange_basis()
        assert polynomials_equal(p, q, r, m)


@pytest.mark.slow
def test_lagrange_to_bernstein():
    r = 3
    for m in [1, 2, 3]:
        dim = get_dimension(r, m)

        # n = 1
        a = np.random.rand(dim)
        p = PolynomialLagrange(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_bernstein_basis())
        assert q.basis() == unique_identifier_bernstein_basis()
        assert polynomials_equal(p, q, r, m)

        # n = 2
        a = np.random.random_sample((dim, 2))
        p = PolynomialLagrange(a, r, m)
        q = convert_polynomial_to_basis(p, unique_identifier_bernstein_basis())
        assert q.basis() == unique_identifier_bernstein_basis()
        assert polynomials_equal(p, q, r, m)


if __name__ == '__main__':
    pytest.main(sys.argv)
