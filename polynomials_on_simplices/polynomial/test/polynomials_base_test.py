import sys

import pytest

from polynomials_on_simplices.polynomial.polynomials_base import get_dimension, polynomials_equal
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import Polynomial


def test_dimension_1d():
    # Verify the dimension of the space of polynomials on a 1-dimensional domain
    for r in range(4):
        d = get_dimension(r, 1)
        assert d == r + 1


def test_dimension_2d():
    # Verify the dimension of the space of polynomials on a 2-dimensional domain
    expected_dimension = [1, 3, 6, 10]
    for r in range(4):
        d = get_dimension(r, 2)
        assert d == expected_dimension[r]


def test_dimension_3d():
    # Verify the dimension of the space of polynomials on a 3-dimensional domain
    expected_dimension = [1, 4, 10, 20]
    for r in range(4):
        d = get_dimension(r, 3)
        assert d == expected_dimension[r]


def test_polynomials_equal():
    # A relative error of 10^{-15} should be considered equal
    value = 1e6
    value_approx = 1e6 - 1e-9
    m = 1
    r = 1
    p1 = Polynomial([value, value], r, m)
    p2 = Polynomial([value_approx, value_approx], r, m)
    assert polynomials_equal(p1, p2, r, m)


if __name__ == '__main__':
    pytest.main(sys.argv)
