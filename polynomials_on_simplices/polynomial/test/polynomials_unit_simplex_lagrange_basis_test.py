import sys

import numpy as np
import pytest

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.finite_difference import central_difference, central_difference_jacobian
from polynomials_on_simplices.geometry.mesh.simplicial_complex import (
    simplex_dimension, simplex_sub_simplices_fixed_dimension, simplex_vertices)
from polynomials_on_simplices.geometry.primitives.simplex import cartesian_to_barycentric_unit, dimension, unit
from polynomials_on_simplices.polynomial.polynomials_base import polynomials_equal
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import unique_identifier_monomial_basis
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_bernstein_basis import bernstein_basis_fn
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import (
    PolynomialLagrange, dual_lagrange_basis_fn, dual_vector_valued_lagrange_basis,
    generate_lagrange_basis_fn_coefficients, generate_lagrange_point, generate_lagrange_points,
    get_associated_sub_simplex, get_dimension, lagrange_basis_fn, unique_identifier_lagrange_basis, unit_polynomial,
    vector_valued_lagrange_basis, zero_polynomial)
from polynomials_on_simplices.probability_theory.uniform_sampling import nsimplex_sampling


def test_lagrange_point_r0():
    r = 0
    expected_x = [
        [0.0],
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    for n in [1, 2, 3]:
        nu = multiindex.generate(n, r, 0)
        x = generate_lagrange_point(n, r, nu)
        assert (x == expected_x[n - 1]).all()


def test_lagrange_points_r0():
    r = 0
    expected_x = [
        [0.0],
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    for n in [1, 2, 3]:
        x = generate_lagrange_points(n, r)[0]
        assert (x == expected_x[n - 1]).all()


def test_generate_lagrange_basis_fn_coefficients():
    coeffs = generate_lagrange_basis_fn_coefficients(1, 1)
    expected_coeffs = np.array([0.0, 1.0])
    assert np.array_equal(expected_coeffs, coeffs)


def test_call():
    # Test calling a scalar valued univariate polynomial
    p = PolynomialLagrange([1, 1, 1], 2, 1)
    value = p(0.5)
    expected_value = 1
    assert value == expected_value

    # Test calling a vector valued univariate polynomial
    p = PolynomialLagrange([[1, 1], [1, 1], [1, 1]], 2, 1)
    value = p(0.5)
    expected_value = np.array([1, 1])
    assert np.linalg.norm(value - expected_value) < 1e-10

    # Test calling a scalar valued bivariate polynomial
    p = PolynomialLagrange([1, 1, 1, 1, 1, 1], 2, 2)
    value = p([1 / 3, 1 / 3])
    expected_value = 1
    assert abs(value - expected_value) < 1e-12

    # Test calling a vector valued bivariate polynomial
    p = PolynomialLagrange([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], 2, 2)
    value = p([1 / 3, 1 / 3])
    expected_value = np.array([1, 1])
    assert np.linalg.norm(value - expected_value) < 1e-10


def test_getitem():
    # Test getting the components of a polynomial in P(R^m, R^n)
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            r = 3
            dim = get_dimension(r, m)
            if n == 1:
                a = np.random.random_sample(dim)
            else:
                a = np.random.random_sample((dim, n))

            p = PolynomialLagrange(a, r, m)

            for i in range(n):
                if n == 1:
                    def pi_expected(x):
                        return p(x)
                else:
                    def pi_expected(x):
                        return p(x)[i]
                assert polynomials_equal(p[i], pi_expected, r, m)


def test_add():
    # Test adding two polynomials in P(R^m, R^n)
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            r = 3
            dim = get_dimension(r, m)
            if n == 1:
                a = np.random.random_sample(dim)
                b = np.random.random_sample(dim)
            else:
                a = np.random.random_sample((dim, n))
                b = np.random.random_sample((dim, n))

            p1 = PolynomialLagrange(a, r, m)
            p2 = PolynomialLagrange(b, r, m)

            def p_expected(x):
                return p1(x) + p2(x)
            assert polynomials_equal(p1 + p2, p_expected, r, m)

    # Test adding two polynomials in P(R^m, R^n) of different degree
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            r1 = 2
            r2 = 3
            dim1 = get_dimension(r1, m)
            dim2 = get_dimension(r2, m)
            if n == 1:
                a = np.random.random_sample(dim1)
                b = np.random.random_sample(dim2)
            else:
                a = np.random.random_sample((dim1, n))
                b = np.random.random_sample((dim2, n))

            p1 = PolynomialLagrange(a, r1, m)
            p2 = PolynomialLagrange(b, r2, m)

            def p_expected(x):
                return p1(x) + p2(x)
            assert polynomials_equal(p1 + p2, p_expected, r, m)


def test_sub():
    # Test subtracting two polynomials in P(R^m, R^n)
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            r = 3
            dim = get_dimension(r, m)
            if n == 1:
                a = np.random.random_sample(dim)
                b = np.random.random_sample(dim)
            else:
                a = np.random.random_sample((dim, n))
                b = np.random.random_sample((dim, n))

            p1 = PolynomialLagrange(a, r, m)
            p2 = PolynomialLagrange(b, r, m)

            def p_expected(x):
                return p1(x) - p2(x)
            assert polynomials_equal(p1 - p2, p_expected, r, m)

    # Test subtracting two polynomials in P(R^m, R^n) of different degree
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            r1 = 2
            r2 = 3
            dim1 = get_dimension(r1, m)
            dim2 = get_dimension(r2, m)
            if n == 1:
                a = np.random.random_sample(dim1)
                b = np.random.random_sample(dim2)
            else:
                a = np.random.random_sample((dim1, n))
                b = np.random.random_sample((dim2, n))

            p1 = PolynomialLagrange(a, r1, m)
            p2 = PolynomialLagrange(b, r2, m)

            def p_expected(x):
                return p1(x) - p2(x)
            assert polynomials_equal(p1 - p2, p_expected, r, m)


@pytest.mark.slow
def test_mul():
    # Test multiplying a polynomial in P(R^m, R^n) with a scalar
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            r = 3
            dim = get_dimension(r, m)
            if n == 1:
                a = np.random.random_sample(dim)
            else:
                a = np.random.random_sample((dim, n))
            s = np.random.rand()

            p = PolynomialLagrange(a, r, m)

            def p_expected(x):
                return s * p(x)
            assert polynomials_equal(s * p, p_expected, r, m)
            assert polynomials_equal(p * s, p_expected, r, m)

    # Test multiplying a polynomial in P(R^m) = P(R^m, R^1) with a vector
    for m in [1, 2, 3]:
        r = 3
        dim = get_dimension(r, m)
        a = np.random.random_sample(dim)
        v = np.random.rand(2)

        p = PolynomialLagrange(a, r, m)

        def p_expected(x):
            return v * p(x)
        # Can't do this, because this will be handled by the Numpy ndarray __mul__ method and result in Numpy array
        # of polynomials
        # assert polynomials_equal(v * p, p_expected, r, m)
        assert polynomials_equal(p * v, p_expected, r, m)

    # Test multiplying two polynomials in P(R^m) = P(R^m, R^1)
    for m in [1, 2, 3]:
        r1 = 3
        r2 = 2
        dim1 = get_dimension(r1, m)
        a = np.random.random_sample(dim1)
        dim2 = get_dimension(r2, m)
        b = np.random.random_sample(dim2)

        p1 = PolynomialLagrange(a, r1, m)
        p2 = PolynomialLagrange(b, r2, m)

        def p_expected(x):
            return p1(x) * p2(x)
        assert polynomials_equal(p1 * p2, p_expected, r1 + r2, m)


def test_div():
    # Test dividing a polynomial in P(R^m, R^n) with a scalar
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            r = 3
            dim = get_dimension(r, m)
            if n == 1:
                a = np.random.random_sample(dim)
            else:
                a = np.random.random_sample((dim, n))
            s = np.random.rand()

            p = PolynomialLagrange(a, r, m)

            def p_expected(x):
                return p(x) / s
            assert polynomials_equal(p / s, p_expected, r, m)


def test_pow():
    # Test taking the power of a polynomial in P(R^m, R^n)
    r = 3
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            if n == 1:
                exponents = range(r + 1)
            else:
                exponents = multiindex.generate_all(n, r)
            for exponent in exponents:
                r_base = 1
                dim = get_dimension(r_base, m)
                if n == 1:
                    a = np.random.random_sample(dim)
                else:
                    a = np.random.random_sample((dim, n))

                p = PolynomialLagrange(a, r_base, m)

                if n == 1:
                    def p_expected(x):
                        return p(x)**exponent
                else:
                    def p_expected(x):
                        return multiindex.power(p(x), exponent)
                assert polynomials_equal(p**exponent, p_expected, r, m)


@pytest.mark.slow
def test_partial_derivative():
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
                    if n == 1:
                        if m == 1:
                            def dp_dxi_fd(x):
                                return central_difference(p, x)
                        else:
                            def dp_dxi_fd(x):
                                return central_difference(p, x)[i]
                    else:
                        def dp_dxi_fd(x):
                            return central_difference_jacobian(p, n, x)[:, i]
                    assert polynomials_equal(p.partial_derivative(i), dp_dxi_fd, r, m, rel_tol=1e-4)


@pytest.mark.slow
def test_degree_elevate():
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            for r in [0, 1, 2, 3]:
                dim = get_dimension(r, m)
                if n == 1:
                    a = np.random.random_sample(dim)
                else:
                    a = np.random.random_sample((dim, n))

                p = PolynomialLagrange(a, r, m)

                for s in range(r, r + 3):
                    q = p.degree_elevate(s)

                    assert polynomials_equal(p, q, s, m)


@pytest.mark.slow
def test_to_monomial_basis():
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            for r in [0, 1, 2, 3]:
                dim = get_dimension(r, m)
                if n == 1:
                    a = np.random.random_sample(dim)
                else:
                    a = np.random.random_sample((dim, n))

                p = PolynomialLagrange(a, r, m)
                q = p.to_monomial_basis()
                assert q.basis() == unique_identifier_monomial_basis()
                assert polynomials_equal(p, q, r, m)


def test_latex_str():
    # Test univariate polynomial
    p = PolynomialLagrange([1, -2, 3], 2, 1)
    assert p.latex_str() == "l_{0, 2}(x) - 2 l_{1, 2}(x) + 3 l_{2, 2}(x)"

    # Test bivariate polynomial
    p = PolynomialLagrange([1, -2, 3], 1, 2)
    assert p.latex_str() == "l_{(0, 0), 1}(x) - 2 l_{(1, 0), 1}(x) + 3 l_{(0, 1), 1}(x)"
    p = PolynomialLagrange([0, 1, 0], 1, 2)
    assert p.latex_str() == "l_{(1, 0), 1}(x)"
    p = PolynomialLagrange([0, 0, -1], 1, 2)
    assert p.latex_str() == "-l_{(0, 1), 1}(x)"
    p = PolynomialLagrange([1, -2, 3, 0.5, 1, 2], 2, 2)
    assert p.latex_str() == r"l_{(0, 0), 2}(x) - 2 l_{(1, 0), 2}(x) + 3 l_{(2, 0), 2}(x) + " \
                            r"\frac{1}{2} l_{(0, 1), 2}(x) + l_{(1, 1), 2}(x) + 2 l_{(0, 2), 2}(x)"

    # Test vector valued polynomial
    p = PolynomialLagrange([[1, 1], [-2, -3], [3, 2]], 2, 1)
    assert p.latex_str() == r"\begin{pmatrix}1 \\ 1\end{pmatrix} l_{0, 2}(x)" \
                            r" + \begin{pmatrix}-2 \\ -3\end{pmatrix} l_{1, 2}(x)" \
                            r" + \begin{pmatrix}3 \\ 2\end{pmatrix} l_{2, 2}(x)"

    # Test bivariate vector valued polynomial
    p = PolynomialLagrange([[1, 1], [-2, -3], [3, 2]], 1, 2)
    assert p.latex_str() == r"\begin{pmatrix}1 \\ 1\end{pmatrix} l_{(0, 0), 1}(x)" \
                            r" + \begin{pmatrix}-2 \\ -3\end{pmatrix} l_{(1, 0), 1}(x)" \
                            r" + \begin{pmatrix}3 \\ 2\end{pmatrix} l_{(0, 1), 1}(x)"


def test_latex_str_expanded():
    # Test univariate polynomial
    p = PolynomialLagrange([1, -2, 3], 2, 1)
    assert p.latex_str_expanded() == r"(1 - 3 x + 2 x^2) - 2 (4 x - 4 x^2) + 3 (-x + 2 x^2)"

    # Test bivariate polynomial
    p = PolynomialLagrange([1, -2, 3], 1, 2)
    assert p.latex_str_expanded() == "(1 - x_1 - x_2) - 2 x_1 + 3 x_2"
    p = PolynomialLagrange([0, 1, 0], 1, 2)
    assert p.latex_str_expanded() == "x_1"
    p = PolynomialLagrange([0, 0, -1], 1, 2)
    assert p.latex_str_expanded() == "-x_2"
    p = PolynomialLagrange([1, -2, 3, 0.5, 1, 2], 2, 2)
    assert p.latex_str_expanded() == r"(1 - 3 x_1 + 2 x_1^2 - 3 x_2 + 4 x_1 x_2 + 2 x_2^2)" \
                                     r" - 2 (4 x_1 - 4 x_1^2 - 4 x_1 x_2)" \
                                     r" + 3 (-x_1 + 2 x_1^2) + " \
                                     r"\frac{1}{2} (4 x_2 - 4 x_1 x_2 - 4 x_2^2)" \
                                     r" + (4 x_1 x_2)" \
                                     r" + 2 (-x_2 + 2 x_2^2)"

    # Test vector valued polynomial
    p = PolynomialLagrange([[1, 1], [-2, -3], [3, 2]], 2, 1)
    assert p.latex_str_expanded() == r"\begin{pmatrix}1 \\ 1\end{pmatrix} (1 - 3 x + 2 x^2)" \
                                     r" + \begin{pmatrix}-2 \\ -3\end{pmatrix} (4 x - 4 x^2)" \
                                     r" + \begin{pmatrix}3 \\ 2\end{pmatrix} (-x + 2 x^2)"

    # Test bivariate vector valued polynomial
    p = PolynomialLagrange([[1, 1], [-2, -3], [3, 2]], 1, 2)
    assert p.latex_str_expanded() == r"\begin{pmatrix}1 \\ 1\end{pmatrix} (1 - x_1 - x_2)" \
                                     r" + \begin{pmatrix}-2 \\ -3\end{pmatrix} x_1" \
                                     r" + \begin{pmatrix}3 \\ 2\end{pmatrix} x_2"


def test_code_str():
    # Test univariate polynomial
    p = PolynomialLagrange([1, -2, 3], 2, 1)
    fn_name = "test_code_str"
    code = p.code_str(fn_name)
    compiled_code = compile(code, fn_name, 'exec')
    exec(compiled_code, globals(), locals())
    test_fn = locals()[fn_name]
    x = np.random.rand()
    assert p(x) == test_fn(x)


class TestUnivariateLagrangeBasis:
    @staticmethod
    def test_basis_fn():
        m = 1
        for r in range(4):
            x = generate_lagrange_points(m, r)
            for nu in multiindex.generate_all(m, r):
                l = lagrange_basis_fn(nu, r)
                for i in range(len(x)):
                    if i == multiindex.get_index(nu, r):
                        assert abs(l(x[i]) - 1.0) < 1e-10
                    else:
                        assert abs(l(x[i])) < 1e-10

    @staticmethod
    def test_vector_valued_basis_fn():
        m = 1
        n = 2
        r = 2
        basis = vector_valued_lagrange_basis(r, m, n, ordering="sequential")
        basis_expected = [
            PolynomialLagrange([[1, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [1, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [1, 0]], r, m),
            PolynomialLagrange([[0, 1], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 1], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 1]], r, m)
        ]
        for i in range(len(basis)):
            assert polynomials_equal(basis[i][0], basis_expected[i][0], r, m)
            assert polynomials_equal(basis[i][1], basis_expected[i][1], r, m)

        basis = vector_valued_lagrange_basis(r, m, n, ordering="interleaved")
        basis_expected = [
            PolynomialLagrange([[1, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 1], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [1, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 1], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [1, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 1]], r, m)
        ]
        for i in range(len(basis)):
            assert polynomials_equal(basis[i][0], basis_expected[i][0], r, m)
            assert polynomials_equal(basis[i][1], basis_expected[i][1], r, m)

    @staticmethod
    def test_dual_basis_fn():
        for r in range(4):
            for mu in multiindex.generate_all(1, r):
                q = dual_lagrange_basis_fn(mu, r)
                for nu in multiindex.generate_all(1, r):
                    p = lagrange_basis_fn(nu, r)
                    if mu == nu:
                        assert abs(q(p) - 1.0) < 1e-10
                    else:
                        assert abs(q(p)) < 1e-10

    @staticmethod
    def test_dual_vector_valued_basis_fn():
        m = 1
        n = 2
        for r in range(4):
            basis = vector_valued_lagrange_basis(r, m, n, ordering="sequential")
            dual_basis = dual_vector_valued_lagrange_basis(r, m, n, ordering="sequential")
            for i in range(len(basis)):
                p = basis[i]
                for j in range(len(basis)):
                    q = dual_basis[j]
                    if i == j:
                        assert abs(q(p) - 1.0) < 1e-10
                    else:
                        assert abs(q(p)) < 1e-10

        for r in range(4):
            basis = vector_valued_lagrange_basis(r, m, n, ordering="interleaved")
            dual_basis = dual_vector_valued_lagrange_basis(r, m, n, ordering="interleaved")
            for i in range(len(basis)):
                p = basis[i]
                for j in range(len(basis)):
                    q = dual_basis[j]
                    if i == j:
                        assert abs(q(p) - 1.0) < 1e-10
                    else:
                        assert abs(q(p)) < 1e-10


class TestBivariateLagrangeBasis:
    @staticmethod
    def test_basis_fn():
        m = 2
        for r in range(4):
            x = generate_lagrange_points(m, r)
            for nu in multiindex.generate_all(m, r):
                l = lagrange_basis_fn(nu, r)
                for i in range(len(x)):
                    if i == multiindex.get_index(nu, r):
                        assert abs(l(x[i]) - 1.0) < 1e-10
                    else:
                        assert abs(l(x[i])) < 1e-10

    @staticmethod
    def test_basis_fn_vector_valued():
        m = 2
        n = 2
        r = 2
        basis = vector_valued_lagrange_basis(r, m, n, ordering="sequential")
        basis_expected = [
            PolynomialLagrange([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], r, m),
            PolynomialLagrange([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]], r, m)
        ]
        for i in range(len(basis)):
            assert polynomials_equal(basis[i][0], basis_expected[i][0], r, m)
            assert polynomials_equal(basis[i][1], basis_expected[i][1], r, m)

        basis = vector_valued_lagrange_basis(r, m, n, ordering="interleaved")
        basis_expected = [
            PolynomialLagrange([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], r, m),
            PolynomialLagrange([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]], r, m)
        ]
        for i in range(len(basis)):
            assert polynomials_equal(basis[i][0], basis_expected[i][0], r, m)
            assert polynomials_equal(basis[i][1], basis_expected[i][1], r, m)

    @staticmethod
    def test_dual_basis_fn():
        m = 2
        r = 3
        for mu in multiindex.generate_all(m, r):
            q = dual_lagrange_basis_fn(mu, r)
            for nu in multiindex.generate_all(m, r):
                p = lagrange_basis_fn(nu, r)
                if mu == nu:
                    assert abs(q(p) - 1.0) < 1e-10
                else:
                    assert abs(q(p)) < 1e-10

    @staticmethod
    @pytest.mark.slow
    def test_dual_vector_valued_basis_fn():
        m = 2
        n = 2
        for r in range(4):
            basis = vector_valued_lagrange_basis(r, m, n)
            dual_basis = dual_vector_valued_lagrange_basis(r, m, n)
            for i in range(len(basis)):
                p = basis[i]
                for j in range(len(basis)):
                    q = dual_basis[j]
                    if i == j:
                        assert abs(q(p) - 1.0) < 1e-10
                    else:
                        assert abs(q(p)) < 1e-10


def test_zero_polynomial():
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        p = zero_polynomial(0, m)
        assert p.basis() == unique_identifier_lagrange_basis()
        assert abs(p(x)) < 1e-12
        for n in [2, 3]:
            p = zero_polynomial(0, m, n)
            assert p.basis() == unique_identifier_lagrange_basis()
            assert np.linalg.norm(p(x) - np.zeros(n)) < 1e-12

    # Test zero polynomial expressed in the degree 2 polynomial basis
    r = 2
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        p = zero_polynomial(r, m, 1)
        assert p.basis() == unique_identifier_lagrange_basis()
        assert p.degree() == r
        assert abs(p(x)) < 1e-12
        for n in [2, 3]:
            p = zero_polynomial(r, m, n)
            assert p.basis() == unique_identifier_lagrange_basis()
            assert p.degree() == r
            assert np.linalg.norm(p(x) - np.zeros(n)) < 1e-12


def test_unit_polynomial():
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        p = unit_polynomial(0, m)
        assert p.basis() == unique_identifier_lagrange_basis()
        assert abs(p(x) - 1) < 1e-12
        for n in [2, 3]:
            p = unit_polynomial(0, m, n)
            assert p.basis() == unique_identifier_lagrange_basis()
            assert np.linalg.norm(p(x) - np.ones(n)) < 1e-12

    # Test unit polynomial expressed in the degree 2 polynomial basis
    r = 2
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        p = unit_polynomial(r, m, 1)
        assert p.basis() == unique_identifier_lagrange_basis()
        assert p.degree() == r
        assert abs(p(x) - 1) < 1e-12
        for n in [2, 3]:
            p = unit_polynomial(r, m, n)
            assert p.basis() == unique_identifier_lagrange_basis()
            assert p.degree() == r
            assert np.linalg.norm(p(x) - np.ones(n)) < 1e-12


def _polynomial_vanish_on_simplex(p, vertices):
    n = dimension(vertices)
    if n == 0:
        x = vertices[0]
        if len(x) == 1:
            x = x[0]
        return abs(p(x)) < 1e-14
    r = p.degree()
    dim = get_dimension(r, n)
    points = nsimplex_sampling(n, dim)
    for point in points:
        b = cartesian_to_barycentric_unit(point)
        x = sum(b[i] * vertices[i] for i in range(len(b)))
        if abs(p(x)) > 1e-14:
            return False
    return True


def test_geometric_decomposition_lagrange_1d():
    n = 1
    simplex = unit(n)
    indexed_simplex = [0, 1]
    r = 0
    f, mu = get_associated_sub_simplex((0,), r)
    assert f == indexed_simplex
    for r in [1, 2, 3]:
        for nu in multiindex.generate_all(n, r):
            f, mu = get_associated_sub_simplex(nu, r)
            p = lagrange_basis_fn(nu, r)
            # Validate that the nu-basis polynomial vanishes on all k-dimensional
            # sub simplices g, g != f, where k = dim(f)
            for g in simplex_sub_simplices_fixed_dimension(indexed_simplex, simplex_dimension(f)):
                if tuple(sorted(f)) == g:
                    continue
                vertices = simplex_vertices(g, simplex)
                assert _polynomial_vanish_on_simplex(p, vertices)


def test_geometric_decomposition_lagrange_2d():
    n = 2
    simplex = unit(n)
    indexed_simplex = [0, 1, 2]
    r = 0
    f, mu = get_associated_sub_simplex((0, 0), r)
    assert f == indexed_simplex
    for r in [1, 2, 3]:
        for nu in multiindex.generate_all(n, r):
            f, mu = get_associated_sub_simplex(nu, r)
            p = lagrange_basis_fn(nu, r)
            # Validate that the nu-basis polynomial vanishes on all k-dimensional
            # sub simplices g, g != f, where k = dim(f)
            for g in simplex_sub_simplices_fixed_dimension(indexed_simplex, simplex_dimension(f)):
                if tuple(sorted(f)) == g:
                    continue
                vertices = simplex_vertices(g, simplex)
                assert _polynomial_vanish_on_simplex(p, vertices)


def test_geometric_decomposition_lagrange_3d():
    n = 3
    simplex = unit(n)
    indexed_simplex = [0, 1, 2, 3]
    r = 0
    f, mu = get_associated_sub_simplex((0, 0, 0), r)
    assert f == indexed_simplex
    for r in [1, 2, 3]:
        for nu in multiindex.generate_all(n, r):
            f, mu = get_associated_sub_simplex(nu, r)
            p = lagrange_basis_fn(nu, r)
            # Validate that the nu-basis polynomial vanishes on all k-dimensional
            # sub simplices g, g != f, where k = dim(f)
            for g in simplex_sub_simplices_fixed_dimension(indexed_simplex, simplex_dimension(f)):
                if tuple(sorted(f)) == g:
                    continue
                vertices = simplex_vertices(g, simplex)
                assert _polynomial_vanish_on_simplex(p, vertices)


# The geometric decomposition of Lagrange polynomials also holds for Bernstein polynomials
def test_geometric_decomposition_bernstein_1d():
    n = 1
    simplex = unit(n)
    indexed_simplex = [0, 1]
    r = 0
    f, mu = get_associated_sub_simplex((0,), r)
    assert f == indexed_simplex
    for r in [1, 2, 3]:
        for nu in multiindex.generate_all(n, r):
            f, mu = get_associated_sub_simplex(nu, r)
            p = bernstein_basis_fn(nu, r)
            # Validate that the nu-basis polynomial vanishes on all k-dimensional
            # sub simplices g, g != f, where k = dim(f)
            for g in simplex_sub_simplices_fixed_dimension(indexed_simplex, simplex_dimension(f)):
                if tuple(sorted(f)) == g:
                    continue
                vertices = simplex_vertices(g, simplex)
                assert _polynomial_vanish_on_simplex(p, vertices)


def test_geometric_decomposition_bernstein_2d():
    n = 2
    simplex = unit(n)
    indexed_simplex = [0, 1, 2]
    r = 0
    f, mu = get_associated_sub_simplex((0, 0), r)
    assert f == indexed_simplex
    for r in [1, 2, 3]:
        for nu in multiindex.generate_all(n, r):
            f, mu = get_associated_sub_simplex(nu, r)
            p = bernstein_basis_fn(nu, r)
            # Validate that the nu-basis polynomial vanishes on all k-dimensional
            # sub simplices g, g != f, where k = dim(f)
            for g in simplex_sub_simplices_fixed_dimension(indexed_simplex, simplex_dimension(f)):
                if tuple(sorted(f)) == g:
                    continue
                vertices = simplex_vertices(g, simplex)
                assert _polynomial_vanish_on_simplex(p, vertices)


def test_geometric_decomposition_bernstein_3d():
    n = 3
    simplex = unit(n)
    indexed_simplex = [0, 1, 2, 3]
    r = 0
    f, mu = get_associated_sub_simplex((0, 0, 0), r)
    assert f == indexed_simplex
    for r in [1, 2, 3]:
        for nu in multiindex.generate_all(n, r):
            f, mu = get_associated_sub_simplex(nu, r)
            p = bernstein_basis_fn(nu, r)
            # Validate that the nu-basis polynomial vanishes on all k-dimensional
            # sub simplices g, g != f, where k = dim(f)
            for g in simplex_sub_simplices_fixed_dimension(indexed_simplex, simplex_dimension(f)):
                if tuple(sorted(f)) == g:
                    continue
                vertices = simplex_vertices(g, simplex)
                assert _polynomial_vanish_on_simplex(p, vertices)


if __name__ == '__main__':
    pytest.main(sys.argv)
