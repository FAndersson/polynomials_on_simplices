import sys

import pytest

from polynomials_on_simplices.algebra.algebraic_operations import composition


def test_composition_11():
    # g univariate scalar valued function

    def f(x):
        return x

    def g(x):
        return 2 * x

    def fog_expected(x):
        return 2 * x

    fog = composition(f, g)

    assert fog(0) == fog_expected(0)
    assert fog(1) == fog_expected(1)
    assert fog(2) == fog_expected(2)


def test_composition_12():
    # g univariate vector valued function

    def f(x, y):
        return 2 * x * y**2

    def g(x):
        return 3 * x, 2 * x

    def fog_expected(x):
        return 24 * x**3

    fog = composition(f, g)

    assert fog(0) == fog_expected(0)
    assert fog(1) == fog_expected(1)
    assert fog(2) == fog_expected(2)


def test_composition_21():
    # g bivariate scalar valued function

    def f(x):
        return 2 * x

    def g(x, y):
        return 3 * x**2 * y

    def fog_expected(x, y):
        return 6 * x**2 * y

    fog = composition(f, g)

    assert fog(0, 1) == fog_expected(0, 1)
    assert fog(1, 2) == fog_expected(1, 2)
    assert fog(2, 3) == fog_expected(2, 3)


def test_composition_22():
    # g bivariate vector valued function

    def f(x, y):
        return 2 * x, x * y

    def g(x, y):
        return 3 * y, 2 * x

    def fog_expected(x, y):
        return 6 * y, 6 * x * y

    fog = composition(f, g)

    assert fog(0, 1) == fog_expected(0, 1)
    assert fog(1, 2) == fog_expected(1, 2)
    assert fog(2, 3) == fog_expected(2, 3)


if __name__ == "__main__":
    pytest.main(sys.argv)
