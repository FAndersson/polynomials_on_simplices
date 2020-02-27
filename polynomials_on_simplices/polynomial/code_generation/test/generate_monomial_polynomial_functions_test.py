import sys

import numpy as np
import pytest

from polynomials_on_simplices.polynomial.code_generation.generate_monomial_polynomial_functions import (
    generate_function_general, generate_function_specific)


class TestGenerateFunctionGeneralUnivariate:
    @staticmethod
    def test_scalar_valued():
        code = generate_function_general(1, 2)
        expected_code = """def monomial_polynomial_12(a, x):
    r\"\"\"
    Evaluate a univariate degree 2 monomial polynomial.

    .. math:: p(x) = a_0 + a_1 x + a_2 x^2.

    :param a: Coefficient in front of each monomial base polynomial, :math:`\\text{a}[i] = a_i`.
    :param x: Point where the polynomial should be evaluated.
    :return: Value of the polynomial.
    \"\"\"
    return a[0] + x * (a[1] + x * a[2])
"""
        assert code == expected_code


class TestGenerateFunctionGeneralBivariate:
    @staticmethod
    def test_scalar_valued():
        code = generate_function_general(2, 2)
        expected_code = """def monomial_polynomial_22(a, x):
    r\"\"\"
    Evaluate a bivariate degree 2 monomial polynomial.

    .. math:: p(x) = a_0 + a_1 x_1 + a_2 x_1^2 + a_3 x_2 + a_4 x_1 x_2 + a_5 x_2^2.

    :param a: Coefficient in front of each monomial base polynomial, :math:`\\text{a}[i] = a_{\\nu_i}`, where
        :math:`\\nu_i` is the i:th multi-index in the sequence of all multi-indices of dimension 2 with norm
        :math:`\\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
    :param x: Point where the polynomial should be evaluated.
    :return: Value of the polynomial.
    \"\"\"
    return a[0] + a[1] * x[0] + a[2] * x[0]**2 + a[3] * x[1] + a[4] * x[0] * x[1] + a[5] * x[1]**2
"""
        assert code == expected_code


class TestGenerateFunctionSpecificUnivariate:
    @staticmethod
    def test_scalar_valued():
        a = [2, 3]
        code, fn_name = generate_function_specific(1, 1, a)
        expected_code = "def " + fn_name + "(x):\n    return 2 + x * 3\n"
        assert code == expected_code

        a = [1, 0, 0]
        code, fn_name = generate_function_specific(1, 2, a)
        expected_code = "def " + fn_name + "(x):\n    return 1\n"
        assert code == expected_code

        a = [0, 1, 0]
        code, fn_name = generate_function_specific(1, 2, a)
        expected_code = "def " + fn_name + "(x):\n    return x\n"
        assert code == expected_code

        a = [0, 0, 1]
        code, fn_name = generate_function_specific(1, 2, a)
        expected_code = "def " + fn_name + "(x):\n    return x * x\n"
        assert code == expected_code

        a = [1, 2, -3]
        code, fn_name = generate_function_specific(1, 2, a)
        expected_code = "def " + fn_name + "(x):\n    return 1 + x * (2 - x * 3)\n"
        assert code == expected_code

    @staticmethod
    def test_vector_valued():
        a = np.array([[2, 3], [4, 5]])
        code, fn_name = generate_function_specific(1, 1, a)
        expected_code = "def " + fn_name + "(x):\n    return np.array([2 + x * 4, 3 + x * 5])\n"
        assert code == expected_code

        a = np.array([[1, 0], [0, 1], [1, 1]])
        code, fn_name = generate_function_specific(1, 2, a)
        expected_code = "def " + fn_name + "(x):\n    return np.array([1 + x * x, x * (1 + x)])\n"
        assert code == expected_code


class TestGenerateFunctionSpecificBivariate:
    @staticmethod
    def test_scalar_valued():
        a = [2, 3, 4]
        code, fn_name = generate_function_specific(2, 1, a)
        expected_code = "def " + fn_name + "(x):\n    return 2 + 3 * x[0] + 4 * x[1]\n"
        assert code == expected_code

        a = [1, 0, 0, 0, 0, 0]
        code, fn_name = generate_function_specific(2, 2, a)
        expected_code = "def " + fn_name + "(x):\n    return 1\n"
        assert code == expected_code

        a = [0, 0, 1, 0, 0, 0]
        code, fn_name = generate_function_specific(2, 2, a)
        expected_code = "def " + fn_name + "(x):\n    return x[0]**2\n"
        assert code == expected_code

        a = [0, 0, 0, 0, 1, 0]
        code, fn_name = generate_function_specific(2, 2, a)
        expected_code = "def " + fn_name + "(x):\n    return x[0] * x[1]\n"
        assert code == expected_code

    @staticmethod
    def test_vector_valued():
        a = np.array([[2, 3], [4, 5], [6, 7]])
        code, fn_name = generate_function_specific(2, 1, a)
        expected_code = "def " + fn_name + "(x):\n    return np.array([2 + 4 * x[0] + 6 * x[1]," \
                                           " 3 + 5 * x[0] + 7 * x[1]])\n"
        assert code == expected_code

        a = np.array([[1, 0], [0, 1], [1, 1]])
        code, fn_name = generate_function_specific(2, 1, a)
        expected_code = "def " + fn_name + "(x):\n    return np.array([1 + x[1], x[0] + x[1]])\n"
        assert code == expected_code


def generate_monomial_polynomial_functions():
    for m in [1, 2, 3]:
        for r in range(1, 7):
            print(generate_function_general(m, r))
            print()
            print()


if __name__ == '__main__':
    pytest.main(sys.argv)
