import sys

import numpy as np
import pytest

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.finite_difference import central_difference, central_difference_jacobian
from polynomials_on_simplices.geometry.primitives.simplex import affine_map_from_unit
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import unique_identifier_monomial_basis
from polynomials_on_simplices.polynomial.polynomials_simplex_base import polynomials_equal_on_simplex
from polynomials_on_simplices.polynomial.polynomials_simplex_bernstein_basis import (
    PolynomialBernsteinSimplex, bernstein_basis_fn_simplex, dual_bernstein_basis_fn_simplex,
    dual_vector_valued_bernstein_basis_simplex, get_dimension, unique_identifier_bernstein_basis_simplex,
    unit_polynomial_simplex, vector_valued_bernstein_basis_simplex, zero_polynomial_simplex)
from polynomials_on_simplices.probability_theory.uniform_sampling import nsimplex_sampling


def test_call():
    vertices = np.random.random_sample((2, 1))
    # Test calling a scalar valued univariate polynomial
    p = PolynomialBernsteinSimplex([1, 1, 1], vertices, 2)
    value = p(0.5)
    expected_value = 1
    assert abs(value - expected_value) < 1e-12

    # Test calling a vector valued univariate polynomial
    p = PolynomialBernsteinSimplex([[1, 1], [1, 1], [1, 1]], vertices, 2)
    value = p(0.5)
    expected_value = np.array([1, 1])
    assert np.linalg.norm(value - expected_value) < 1e-10

    vertices = np.random.random_sample((3, 2))
    # Test calling a scalar valued bivariate polynomial
    p = PolynomialBernsteinSimplex([1, 1, 1, 1, 1, 1], vertices, 2)
    value = p([1 / 3, 1 / 3])
    expected_value = 1
    assert abs(value - expected_value) < 1e-12

    # Test calling a vector valued bivariate polynomial
    p = PolynomialBernsteinSimplex([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], vertices, 2)
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

            vertices = np.random.random_sample((m + 1, m))
            p = PolynomialBernsteinSimplex(a, vertices, r)

            for i in range(n):
                if n == 1:
                    def pi_expected(x):
                        return p(x)
                else:
                    def pi_expected(x):
                        return p(x)[i]
                assert polynomials_equal_on_simplex(p[i], pi_expected, r, vertices)


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

            vertices = np.random.random_sample((m + 1, m))
            p1 = PolynomialBernsteinSimplex(a, vertices, r)
            p2 = PolynomialBernsteinSimplex(b, vertices, r)

            def p_expected(x):
                return p1(x) + p2(x)
            assert polynomials_equal_on_simplex(p1 + p2, p_expected, r, vertices)

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

            vertices = np.random.random_sample((m + 1, m))
            p1 = PolynomialBernsteinSimplex(a, vertices, r1)
            p2 = PolynomialBernsteinSimplex(b, vertices, r2)

            def p_expected(x):
                return p1(x) + p2(x)
            assert polynomials_equal_on_simplex(p1 + p2, p_expected, max(r1, r2), vertices)


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

            vertices = np.random.random_sample((m + 1, m))
            p1 = PolynomialBernsteinSimplex(a, vertices, r)
            p2 = PolynomialBernsteinSimplex(b, vertices, r)

            def p_expected(x):
                return p1(x) - p2(x)
            assert polynomials_equal_on_simplex(p1 - p2, p_expected, r, vertices)

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

            vertices = np.random.random_sample((m + 1, m))
            p1 = PolynomialBernsteinSimplex(a, vertices, r1)
            p2 = PolynomialBernsteinSimplex(b, vertices, r2)

            def p_expected(x):
                return p1(x) - p2(x)
            assert polynomials_equal_on_simplex(p1 - p2, p_expected, r, vertices)


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

            vertices = np.random.random_sample((m + 1, m))
            p = PolynomialBernsteinSimplex(a, vertices, r)

            def p_expected(x):
                return s * p(x)
            assert polynomials_equal_on_simplex(s * p, p_expected, r, vertices)
            assert polynomials_equal_on_simplex(p * s, p_expected, r, vertices)

    # Test multiplying a polynomial in P(R^m) = P(R^m, R^1) with a vector
    for m in [1, 2, 3]:
        r = 3
        dim = get_dimension(r, m)
        a = np.random.random_sample(dim)
        v = np.random.rand(2)

        vertices = np.random.random_sample((m + 1, m))
        p = PolynomialBernsteinSimplex(a, vertices, r)

        def p_expected(x):
            return v * p(x)
        # Can't do this, because this will be handled by the Numpy ndarray __mul__ method and result in Numpy array
        # of polynomials
        # assert polynomials_equal(v * p, p_expected, r, m)
        assert polynomials_equal_on_simplex(p * v, p_expected, r, vertices)

    # Test multiplying two polynomials in P(R^m) = P(R^m, R^1)
    for m in [1, 2, 3]:
        r1 = 3
        r2 = 2
        dim1 = get_dimension(r1, m)
        a = np.random.random_sample(dim1)
        dim2 = get_dimension(r2, m)
        b = np.random.random_sample(dim2)

        vertices = np.random.random_sample((m + 1, m))
        p1 = PolynomialBernsteinSimplex(a, vertices, r1)
        p2 = PolynomialBernsteinSimplex(b, vertices, r2)

        def p_expected(x):
            return p1(x) * p2(x)
        assert polynomials_equal_on_simplex(p1 * p2, p_expected, r1 + r2, vertices)


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

            vertices = np.random.random_sample((m + 1, m))
            p = PolynomialBernsteinSimplex(a, vertices, r)

            def p_expected(x):
                return p(x) / s
            assert polynomials_equal_on_simplex(p / s, p_expected, r, vertices)


@pytest.mark.slow
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

                vertices = np.random.random_sample((m + 1, m))
                p = PolynomialBernsteinSimplex(a, vertices, r_base)

                if n == 1:
                    def p_expected(x):
                        return p(x)**exponent
                else:
                    def p_expected(x):
                        return multiindex.power(p(x), exponent)
                assert polynomials_equal_on_simplex(p**exponent, p_expected, r, vertices)


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

                vertices = np.random.random_sample((m + 1, m))
                p = PolynomialBernsteinSimplex(a, vertices, r)

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
                    assert polynomials_equal_on_simplex(p.partial_derivative(i), dp_dxi_fd, r, vertices, rel_tol=1e-4)


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

                vertices = np.random.random_sample((m + 1, m))
                p = PolynomialBernsteinSimplex(a, vertices, r)

                for s in range(r, r + 3):
                    q = p.degree_elevate(s)

                    assert polynomials_equal_on_simplex(p, q, s, vertices)


@pytest.mark.slow
def test_to_monomial_basis():
    for m in [1, 2, 3]:
        for n in [2, 3]:
            for r in [0, 1, 2, 3]:
                dim = get_dimension(r, m)
                if n == 1:
                    a = np.random.random_sample(dim)
                else:
                    a = np.random.random_sample((dim, n))

                vertices = np.random.random_sample((m + 1, m))
                p = PolynomialBernsteinSimplex(a, vertices, r)
                q = p.to_monomial_basis()
                assert q.basis() == unique_identifier_monomial_basis()
                assert polynomials_equal_on_simplex(p, q, r, vertices)


def test_latex_str():
    # Test univariate polynomial
    vertices = np.array([[1], [3]])
    p = PolynomialBernsteinSimplex([1, -2, 3], vertices, 2)
    assert p.latex_str() == "b_{0, 2}(x) - 2 b_{1, 2}(x) + 3 b_{2, 2}(x)"

    # Test bivariate polynomial
    vertices = np.array([[2, 3], [4, 3], [2, 5]])
    p = PolynomialBernsteinSimplex([1, -2, 3], vertices, 1)
    assert p.latex_str() == "b_{(0, 0), 1}(x) - 2 b_{(1, 0), 1}(x) + 3 b_{(0, 1), 1}(x)"
    p = PolynomialBernsteinSimplex([0, 1, 0], vertices, 1)
    assert p.latex_str() == "b_{(1, 0), 1}(x)"
    p = PolynomialBernsteinSimplex([0, 0, -1], vertices, 1)
    assert p.latex_str() == "-b_{(0, 1), 1}(x)"
    p = PolynomialBernsteinSimplex([1, -2, 3, 0.5, 1, 2], vertices, 2)
    assert p.latex_str() == r"b_{(0, 0), 2}(x) - 2 b_{(1, 0), 2}(x) + 3 b_{(2, 0), 2}(x) + " \
                            r"\frac{1}{2} b_{(0, 1), 2}(x) + b_{(1, 1), 2}(x) + 2 b_{(0, 2), 2}(x)"

    # Test vector valued polynomial
    vertices = np.array([[1], [3]])
    p = PolynomialBernsteinSimplex([[1, 1], [-2, -3], [3, 2]], vertices, 2)
    assert p.latex_str() == r"\begin{pmatrix}1 \\ 1\end{pmatrix} b_{0, 2}(x)" \
                            r" + \begin{pmatrix}-2 \\ -3\end{pmatrix} b_{1, 2}(x)" \
                            r" + \begin{pmatrix}3 \\ 2\end{pmatrix} b_{2, 2}(x)"

    # Test bivariate vector valued polynomial
    vertices = np.array([[2, 3], [4, 3], [2, 5]])
    p = PolynomialBernsteinSimplex([[1, 1], [-2, -3], [3, 2]], vertices, 1)
    assert p.latex_str() == r"\begin{pmatrix}1 \\ 1\end{pmatrix} b_{(0, 0), 1}(x)" \
                            r" + \begin{pmatrix}-2 \\ -3\end{pmatrix} b_{(1, 0), 1}(x)" \
                            r" + \begin{pmatrix}3 \\ 2\end{pmatrix} b_{(0, 1), 1}(x)"


def test_latex_str_expanded():
    # Test univariate polynomial
    vertices = np.array([[1], [3]])
    p = PolynomialBernsteinSimplex([1, -2], vertices, 1)
    assert p.latex_str_expanded() == r"(\frac{3}{2} - \frac{1}{2} x) - 2 (-\frac{1}{2} + \frac{1}{2} x)"

    # Test bivariate polynomial
    vertices = np.array([[2, 3], [4, 3], [2, 5]])
    p = PolynomialBernsteinSimplex([1, -2, 3], vertices, 1)
    assert p.latex_str_expanded() == r"(\frac{7}{2} - \frac{1}{2} x_1 - \frac{1}{2} x_2)"\
                                     r" - 2 (-1 + \frac{1}{2} x_1)"\
                                     r" + 3 (-\frac{3}{2} + \frac{1}{2} x_2)"

    # Test vector valued polynomial
    vertices = np.array([[1], [3]])
    p = PolynomialBernsteinSimplex([[1, 1], [-2, -3]], vertices, 1)
    assert p.latex_str_expanded() == r"\begin{pmatrix}1 \\ 1\end{pmatrix} (\frac{3}{2} - \frac{1}{2} x)" \
                                     r" + \begin{pmatrix}-2 \\ -3\end{pmatrix} (-\frac{1}{2} + \frac{1}{2} x)"

    # Test bivariate vector valued polynomial
    vertices = np.array([[2, 3], [4, 3], [2, 5]])
    p = PolynomialBernsteinSimplex([[1, 1], [-2, -3], [3, 2]], vertices, 1)
    assert p.latex_str_expanded() == (
        r"\begin{pmatrix}1 \\ 1\end{pmatrix} (\frac{7}{2} - \frac{1}{2} x_1 - \frac{1}{2} x_2)"
        + r" + \begin{pmatrix}-2 \\ -3\end{pmatrix} (-1 + \frac{1}{2} x_1)"
        + r" + \begin{pmatrix}3 \\ 2\end{pmatrix} (-\frac{3}{2} + \frac{1}{2} x_2)")


def test_code_str():
    # Test univariate polynomial
    vertices = np.array([[1], [3]])
    p = PolynomialBernsteinSimplex([1, -2, 3], vertices, 2)
    fn_name = "test_code_str"
    code = p.code_str(fn_name)
    compiled_code = compile(code, fn_name, 'exec')
    exec(compiled_code, globals(), locals())
    test_fn = locals()[fn_name]
    x = np.random.rand()
    assert p(x) == test_fn(x)

    # Test vector valued univariate polynomial
    vertices = np.array([[1], [3]])
    p = PolynomialBernsteinSimplex([[1, 1], [-2, -2], [3, 3]], vertices, 2)
    fn_name = "test_code_str"
    code = p.code_str(fn_name)
    compiled_code = compile(code, fn_name, 'exec')
    exec(compiled_code, globals(), locals())
    test_fn = locals()[fn_name]
    x = np.random.rand()
    assert np.linalg.norm(p(x) - test_fn(x)) < 1e-14

    # Test bivariate polynomial
    vertices = np.array([[2, 3], [4, 3], [2, 5]])
    p = PolynomialBernsteinSimplex([1, -2, 3], vertices, 1)
    fn_name = "test_code_str"
    code = p.code_str(fn_name)
    compiled_code = compile(code, fn_name, 'exec')
    exec(compiled_code, globals(), locals())
    test_fn = locals()[fn_name]
    x = np.random.rand(2)
    assert p(x) == test_fn(x)

    # Test vector valued bivariate polynomial
    vertices = np.array([[2, 3], [4, 3], [2, 5]])
    p = PolynomialBernsteinSimplex([[1, 1], [-2, -2], [3, 3]], vertices, 1)
    fn_name = "test_code_str"
    code = p.code_str(fn_name)
    compiled_code = compile(code, fn_name, 'exec')
    exec(compiled_code, globals(), locals())
    test_fn = locals()[fn_name]
    x = np.random.rand(2)
    assert np.linalg.norm(p(x) - test_fn(x)) < 1e-14


class TestUnivariateBernsteinBasis:
    @staticmethod
    def get_expected_basis_fns():
        return [
            [
                lambda x: 1
            ],
            [
                lambda x: -0.5 * x + 1.5,
                lambda x: 0.5 * x - 0.5
            ],
            [
                lambda x: (0.5 * x - 1.5)**2,
                lambda x: -0.5 * x**2 + 2.0 * x - 1.5,
                lambda x: 0.25 * (x - 1)**2
            ],
            [
                lambda x: -(0.5 * x - 1.5)**3,
                lambda x: 1.5 * (0.5 * x - 1.5)**2 * (x - 1),
                lambda x: (-0.375 * x + 1.125) * (x - 1)**2,
                lambda x: 0.125 * (x - 1)**3
            ]
        ]

    @staticmethod
    def test_basis_fn():
        vertices = np.array([[1], [3]])
        for r in range(4):
            for i in range(r + 1):
                p = bernstein_basis_fn_simplex(i, r, vertices)
                p_expected = TestUnivariateBernsteinBasis.get_expected_basis_fns()[r][i]
                assert polynomials_equal_on_simplex(p, p_expected, r, vertices)

    @staticmethod
    def test_vector_valued_basis_fn():
        vertices = np.array([[1], [3]])
        n = 2
        r = 2
        basis = vector_valued_bernstein_basis_simplex(r, vertices, n, ordering="sequential")
        basis_expected = [
            PolynomialBernsteinSimplex([[1, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [1, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [1, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 1], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 1], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 1]], vertices, r)
        ]
        for i in range(len(basis)):
            assert polynomials_equal_on_simplex(basis[i][0], basis_expected[i][0], r, vertices)
            assert polynomials_equal_on_simplex(basis[i][1], basis_expected[i][1], r, vertices)

        basis = vector_valued_bernstein_basis_simplex(r, vertices, n, ordering="interleaved")
        basis_expected = [
            PolynomialBernsteinSimplex([[1, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 1], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [1, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 1], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [1, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 1]], vertices, r)
        ]
        for i in range(len(basis)):
            assert polynomials_equal_on_simplex(basis[i][0], basis_expected[i][0], r, vertices)
            assert polynomials_equal_on_simplex(basis[i][1], basis_expected[i][1], r, vertices)

    @staticmethod
    def test_dual_basis_fn():
        vertices = np.array([[1], [3]])
        for r in range(1, 4):
            for mu in multiindex.generate_all(1, r):
                q = dual_bernstein_basis_fn_simplex(mu, r, vertices)
                for nu in multiindex.generate_all(1, r):
                    p = bernstein_basis_fn_simplex(nu, r, vertices)
                    if mu == nu:
                        assert abs(q(p) - 1.0) < 1e-10
                    else:
                        assert abs(q(p)) < 1e-10

    @staticmethod
    def test_dual_vector_valued_basis_fn():
        vertices = np.array([[1], [3]])
        n = 2

        for r in range(4):
            basis = vector_valued_bernstein_basis_simplex(r, vertices, n, ordering="sequential")
            dual_basis = dual_vector_valued_bernstein_basis_simplex(r, vertices, n, ordering="sequential")
            for i in range(len(basis)):
                p = basis[i]
                for j in range(len(basis)):
                    q = dual_basis[j]
                    if i == j:
                        assert abs(q(p) - 1.0) < 1e-10
                    else:
                        assert abs(q(p)) < 1e-10

        for r in range(4):
            basis = vector_valued_bernstein_basis_simplex(r, vertices, n, ordering="interleaved")
            dual_basis = dual_vector_valued_bernstein_basis_simplex(r, vertices, n, ordering="interleaved")
            for i in range(len(basis)):
                p = basis[i]
                for j in range(len(basis)):
                    q = dual_basis[j]
                    if i == j:
                        assert abs(q(p) - 1.0) < 1e-10
                    else:
                        assert abs(q(p)) < 1e-10


class TestBivariateBernsteinBasis:
    @staticmethod
    def get_expected_basis_fns():
        return [
            {
                (0, 0): lambda x: 1
            },
            {
                (0, 0): lambda x: -0.5 * x[0] - 0.5 * x[1] + 3.5,
                (1, 0): lambda x: 0.5 * x[0] - 1.0,
                (0, 1): lambda x: 0.5 * x[1] - 1.5
            },
            {
                (0, 0): lambda x: (0.5 * x[0] + 0.5 * x[1] - 3.5)**2,
                (1, 0): lambda x: -2 * (0.5 * x[0] - 1.0) * (0.5 * x[0] + 0.5 * x[1] - 3.5),
                (2, 0): lambda x: (0.5 * x[0] - 1.0)**2,
                (0, 1): lambda x: -2 * (0.5 * x[1] - 1.5) * (0.5 * x[0] + 0.5 * x[1] - 3.5),
                (1, 1): lambda x: 2 * (0.5 * x[0] - 1.0) * (0.5 * x[1] - 1.5),
                (0, 2): lambda x: (0.5 * x[1] - 1.5)**2
            },
            {
                (0, 0): lambda x: -(0.5 * x[0] + 0.5 * x[1] - 3.5)**3,
                (1, 0): lambda x: (1.5 * x[0] - 3.0) * (0.5 * x[0] + 0.5 * x[1] - 3.5)**2,
                (2, 0): lambda x: (0.5 * x[0] - 1.0)**2 * (-1.5 * x[0] - 1.5 * x[1] + 10.5),
                (3, 0): lambda x: (0.5 * x[0] - 1.0)**3,
                (0, 1): lambda x: (1.5 * x[1] - 4.5) * (0.5 * x[0] + 0.5 * x[1] - 3.5)**2,
                (1, 1): lambda x: -6 * (0.5 * x[0] - 1.0) * (0.5 * x[1] - 1.5) * (0.5 * x[0] + 0.5 * x[1] - 3.5),
                (2, 1): lambda x: (0.5 * x[0] - 1.0)**2 * (1.5 * x[1] - 4.5),
                (0, 2): lambda x: (0.5 * x[1] - 1.5)**2 * (-1.5 * x[0] - 1.5 * x[1] + 10.5),
                (1, 2): lambda x: (1.5 * x[0] - 3.0) * (0.5 * x[1] - 1.5)**2,
                (0, 3): lambda x: (0.5 * x[1] - 1.5)**3
            },
        ]

    @staticmethod
    def test_basis_fn():
        vertices = np.array([[2, 3], [4, 3], [2, 5]])
        m = 2
        for r in range(4):
            for nu in multiindex.generate_all(m, r):
                p = bernstein_basis_fn_simplex(nu, r, vertices)
                p_expected = TestBivariateBernsteinBasis.get_expected_basis_fns()[r][nu]
                assert polynomials_equal_on_simplex(p, p_expected, r, vertices)

    @staticmethod
    def test_basis_fn_vector_valued():
        vertices = np.array([[2, 3], [4, 3], [2, 5]])
        n = 2
        r = 2
        basis = vector_valued_bernstein_basis_simplex(r, vertices, n, ordering="sequential")
        basis_expected = [
            PolynomialBernsteinSimplex([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]], vertices, r)
        ]
        for i in range(len(basis)):
            assert polynomials_equal_on_simplex(basis[i][0], basis_expected[i][0], r, vertices)
            assert polynomials_equal_on_simplex(basis[i][1], basis_expected[i][1], r, vertices)

        basis = vector_valued_bernstein_basis_simplex(r, vertices, n, ordering="interleaved")
        basis_expected = [
            PolynomialBernsteinSimplex([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], vertices, r),
            PolynomialBernsteinSimplex([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]], vertices, r)
        ]
        for i in range(len(basis)):
            assert polynomials_equal_on_simplex(basis[i][0], basis_expected[i][0], r, vertices)
            assert polynomials_equal_on_simplex(basis[i][1], basis_expected[i][1], r, vertices)

    @staticmethod
    @pytest.mark.slow
    def test_dual_basis_fn():
        vertices = np.array([[2, 3], [4, 3], [2, 5]])
        m = 2
        r = 3
        for mu in multiindex.generate_all(m, r):
            q = dual_bernstein_basis_fn_simplex(mu, r, vertices)
            for nu in multiindex.generate_all(m, r):
                p = bernstein_basis_fn_simplex(nu, r, vertices)
                if mu == nu:
                    assert abs(q(p) - 1.0) < 1e-10
                else:
                    assert abs(q(p)) < 1e-10

    @staticmethod
    @pytest.mark.slow
    def test_dual_vector_valued_basis_fn():
        vertices = np.array([[2, 3], [4, 3], [2, 5]])
        n = 2
        for r in range(4):
            basis = vector_valued_bernstein_basis_simplex(r, vertices, n)
            dual_basis = dual_vector_valued_bernstein_basis_simplex(r, vertices, n)
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
        vertices = np.random.random_sample((m + 1, m))
        p = zero_polynomial_simplex(vertices, 1)
        assert p.basis() == unique_identifier_bernstein_basis_simplex(vertices)
        assert abs(p(x)) < 1e-12
        for n in [2, 3]:
            p = zero_polynomial_simplex(vertices, n)
            assert p.basis() == unique_identifier_bernstein_basis_simplex(vertices)
            assert np.linalg.norm(p(x)) < 1e-12

    # Test zero polynomial expressed in the degree 2 polynomial basis
    r = 2
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        vertices = np.random.random_sample((m + 1, m))
        p = zero_polynomial_simplex(vertices, r, 1)
        assert p.basis() == unique_identifier_bernstein_basis_simplex(vertices)
        assert p.degree() == r
        assert abs(p(x)) < 1e-12
        for n in [2, 3]:
            p = zero_polynomial_simplex(vertices, r, n)
            assert p.basis() == unique_identifier_bernstein_basis_simplex(vertices)
            assert p.degree() == r
            assert np.linalg.norm(p(x)) < 1e-12


def test_unit_polynomial():
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        vertices = np.random.random_sample((m + 1, m))
        phi = affine_map_from_unit(vertices)
        x = phi(x)
        p = unit_polynomial_simplex(vertices, 0)
        assert p.basis() == unique_identifier_bernstein_basis_simplex(vertices)
        assert abs(p(x) - 1) < 1e-12
        for n in [2, 3]:
            p = unit_polynomial_simplex(vertices, 0, n)
            assert p.basis() == unique_identifier_bernstein_basis_simplex(vertices)
            assert np.linalg.norm(p(x) - np.ones(n)) < 1e-12

    # Test unit polynomial expressed in the degree 2 polynomial basis
    r = 2
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        vertices = np.random.random_sample((m + 1, m))
        phi = affine_map_from_unit(vertices)
        x = phi(x)
        p = unit_polynomial_simplex(vertices, r, 1)
        assert p.basis() == unique_identifier_bernstein_basis_simplex(vertices)
        assert p.degree() == r
        assert abs(p(x) - 1) < 1e-12
        for n in [2, 3]:
            p = unit_polynomial_simplex(vertices, r, n)
            assert p.basis() == unique_identifier_bernstein_basis_simplex(vertices)
            assert p.degree() == r
            assert np.linalg.norm(p(x) - np.ones(n)) < 1e-12


if __name__ == '__main__':
    pytest.main(sys.argv)
