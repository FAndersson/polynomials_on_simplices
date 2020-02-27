import sys

import numpy as np
import pytest

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.finite_difference import central_difference, central_difference_jacobian
from polynomials_on_simplices.polynomial.polynomials_base import polynomials_equal
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import (
    Polynomial, dual_monomial_basis_fn, dual_vector_valued_monomial_basis, get_dimension, monomial_basis_fn,
    unique_identifier_monomial_basis, unit_polynomial, vector_valued_monomial_basis, zero_polynomial)
from polynomials_on_simplices.probability_theory.uniform_sampling import nsimplex_sampling


def test_iter_components():
    # Iterate over the components of a vector valued polynomial
    r = 1
    m = 1
    p = Polynomial([[1, 2], [3, 4]], r, m)
    expected_components = [
        Polynomial([1, 3], r, m),
        Polynomial([2, 4], r, m)
    ]
    assert polynomials_equal(p[0], expected_components[0], r, m)
    assert polynomials_equal(p[1], expected_components[1], r, m)

    # Verify that we can iterate over the components of the polynomial
    i = 0
    for pi in p:
        assert polynomials_equal(pi, expected_components[i], r, m)
        i += 1

    # Verify that we can convert the polynomial into a list of its components
    pl = list(p)
    assert len(pl) == len(expected_components)
    for i in range(len(pl)):
        assert polynomials_equal(pl[i], expected_components[i], r, m)

    # Verify that we can dot-multiply a vector valued polynomial with a Numpy array
    a = np.array([2, 3])
    pa = np.dot(p, a)
    assert polynomials_equal(pa, 2 * expected_components[0] + 3 * expected_components[1], r, m)


def test_call():
    # Test calling a scalar valued univariate polynomial
    p = Polynomial([1, 2, 3], 2, 1)
    value = p(0.5)
    expected_value = 2.75
    assert value == expected_value

    p = Polynomial([1, 2, -3], 2, 1)
    value = p(0.5)
    expected_value = 1.25
    assert value == expected_value

    # Test calling a vector valued univariate polynomial
    p = Polynomial([[1, 4], [2, 5], [3, 6]], 2, 1)
    value = p(0.5)
    expected_value = np.array([2.75, 8])
    assert np.linalg.norm(value - expected_value) < 1e-10

    # Test calling a scalar valued bivariate polynomial
    p = Polynomial([1, 1, 1, 1, 1, 1], 2, 2)
    value = p([1 / 3, 1 / 3])
    expected_value = 2
    assert value == expected_value

    # Test calling a vector valued bivariate polynomial
    p = Polynomial([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], 2, 2)
    value = p([1 / 3, 1 / 3])
    expected_value = np.array([2, 2])
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

            p = Polynomial(a, r, m)

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

            p1 = Polynomial(a, r, m)
            p2 = Polynomial(b, r, m)

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

            p1 = Polynomial(a, r1, m)
            p2 = Polynomial(b, r2, m)

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

            p1 = Polynomial(a, r, m)
            p2 = Polynomial(b, r, m)

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

            p1 = Polynomial(a, r1, m)
            p2 = Polynomial(b, r2, m)

            def p_expected(x):
                return p1(x) - p2(x)
            assert polynomials_equal(p1 - p2, p_expected, r, m)


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

            p = Polynomial(a, r, m)

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

        p = Polynomial(a, r, m)

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

        p1 = Polynomial(a, r1, m)
        p2 = Polynomial(b, r2, m)

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

            p = Polynomial(a, r, m)

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

                p = Polynomial(a, r_base, m)

                if n == 1:
                    def p_expected(x):
                        return p(x)**exponent
                else:
                    def p_expected(x):
                        return multiindex.power(p(x), exponent)
                assert polynomials_equal(p**exponent, p_expected, r, m)


def test_partial_derivative():
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
                    assert polynomials_equal(p.partial_derivative(i), dp_dxi_fd, r, m, rel_tol=1e-5)


def test_degree_elevate():
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            for r in [0, 1, 2, 3]:
                dim = get_dimension(r, m)
                if n == 1:
                    a = np.random.random_sample(dim)
                else:
                    a = np.random.random_sample((dim, n))

                p = Polynomial(a, r, m)

                for s in range(r, r + 3):
                    q = p.degree_elevate(s)

                    assert polynomials_equal(p, q, s, m)


def test_latex_str():
    # Test univariate polynomial
    p = Polynomial([1, -2, 3], 2, 1)
    assert p.latex_str() == "1 - 2 x + 3 x^2"

    # Test bivariate polynomial
    p = Polynomial([1, -2, 3], 1, 2)
    assert p.latex_str() == "1 - 2 x_1 + 3 x_2"
    p = Polynomial([0, 1, 0], 1, 2)
    assert p.latex_str() == "x_1"
    p = Polynomial([0, 0, -1], 1, 2)
    assert p.latex_str() == "-x_2"
    p = Polynomial([1, -2, 3, 0.5, 1, 2], 2, 2)
    assert p.latex_str() == r"1 - 2 x_1 + 3 x_1^2 + \frac{1}{2} x_2 + x_1 x_2 + 2 x_2^2"

    # Test vector valued polynomial
    p = Polynomial([[1, 1], [-2, -3], [3, 2]], 2, 1)
    assert p.latex_str() == r"\begin{pmatrix}1 \\ 1\end{pmatrix} + \begin{pmatrix}-2 \\ -3\end{pmatrix} x" \
                            r" + \begin{pmatrix}3 \\ 2\end{pmatrix} x^2"

    # Test bivariate vector valued polynomial
    p = Polynomial([[1, 1], [-2, -3], [3, 2]], 1, 2)
    assert p.latex_str() == r"\begin{pmatrix}1 \\ 1\end{pmatrix} + \begin{pmatrix}-2 \\ -3\end{pmatrix} x_1" \
                            r" + \begin{pmatrix}3 \\ 2\end{pmatrix} x_2"


def test_code_str():
    # Test univariate polynomial
    p = Polynomial([1, -2, 3], 2, 1)
    fn_name = "test_code_str"
    code = p.code_str(fn_name)
    compiled_code = compile(code, fn_name, 'exec')
    exec(compiled_code, globals(), locals())
    test_fn = locals()[fn_name]
    x = np.random.rand()
    assert p(x) == test_fn(x)


class TestUnivariateMonomialBasis:
    @staticmethod
    def get_expected_basis_fns():
        return [
            lambda x: 1,
            lambda x: x**1,
            lambda x: x**2,
            lambda x: x**3,
        ]

    @staticmethod
    def test_basis_fn():
        m = 1
        for r in range(4):
            p = monomial_basis_fn(r)
            p_expected = TestUnivariateMonomialBasis.get_expected_basis_fns()[r]
            assert polynomials_equal(p, p_expected, r, m)

    @staticmethod
    def test_vector_valued_basis_fn():
        m = 1
        n = 2
        r = 2
        basis = vector_valued_monomial_basis(r, m, n, ordering="sequential")
        basis_expected = [
            Polynomial([[1, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [1, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [1, 0]], r, m),
            Polynomial([[0, 1], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 1], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 1]], r, m)
        ]
        for i in range(len(basis)):
            assert polynomials_equal(basis[i][0], basis_expected[i][0], r, m)
            assert polynomials_equal(basis[i][1], basis_expected[i][1], r, m)

        basis = vector_valued_monomial_basis(r, m, n, ordering="interleaved")
        basis_expected = [
            Polynomial([[1, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 1], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [1, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 1], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [1, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 1]], r, m)
        ]
        for i in range(len(basis)):
            assert polynomials_equal(basis[i][0], basis_expected[i][0], r, m)
            assert polynomials_equal(basis[i][1], basis_expected[i][1], r, m)

    @staticmethod
    def test_dual_basis_fn():
        for i in range(4):
            qi = dual_monomial_basis_fn(i)
            for j in range(4):
                pj = monomial_basis_fn(j)
                if i == j:
                    assert qi(pj) == 1.0
                else:
                    assert qi(pj) == 0.0

    @staticmethod
    def test_dual_vector_valued_basis_fn():
        m = 1
        n = 2
        for r in range(4):
            basis = vector_valued_monomial_basis(r, m, n, ordering="sequential")
            dual_basis = dual_vector_valued_monomial_basis(r, m, n, ordering="sequential")
            for i in range(len(basis)):
                p = basis[i]
                for j in range(len(basis)):
                    q = dual_basis[j]
                    if i == j:
                        assert abs(q(p) - 1.0) < 1e-10
                    else:
                        assert abs(q(p)) < 1e-10

        for r in range(4):
            basis = vector_valued_monomial_basis(r, m, n, ordering="interleaved")
            dual_basis = dual_vector_valued_monomial_basis(r, m, n, ordering="interleaved")
            for i in range(len(basis)):
                p = basis[i]
                for j in range(len(basis)):
                    q = dual_basis[j]
                    if i == j:
                        assert abs(q(p) - 1.0) < 1e-10
                    else:
                        assert abs(q(p)) < 1e-10


class TestBivariateMonomialBasis:
    @staticmethod
    def get_expected_basis_fns():
        return {
            (0, 0): lambda x: 1,
            (1, 0): lambda x: x[0],
            (2, 0): lambda x: x[0]**2,
            (3, 0): lambda x: x[0]**3,
            (0, 1): lambda x: x[1],
            (1, 1): lambda x: x[0] * x[1],
            (2, 1): lambda x: x[0]**2 * x[1],
            (0, 2): lambda x: x[1]**2,
            (1, 2): lambda x: x[0] * x[1]**2,
            (0, 3): lambda x: x[1]**3
        }

    @staticmethod
    def test_basis_fn():
        m = 2
        for r in range(4):
            for nu in multiindex.generate_all(m, r):
                p = monomial_basis_fn(nu)
                p_expected = TestBivariateMonomialBasis.get_expected_basis_fns()[nu]
                assert polynomials_equal(p, p_expected, r, m)

    @staticmethod
    def test_basis_fn_vector_valued():
        m = 2
        n = 2
        r = 2
        basis = vector_valued_monomial_basis(r, m, n, ordering="sequential")
        basis_expected = [
            Polynomial([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], r, m),
            Polynomial([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]], r, m)
        ]
        for i in range(len(basis)):
            assert polynomials_equal(basis[i][0], basis_expected[i][0], r, m)
            assert polynomials_equal(basis[i][1], basis_expected[i][1], r, m)

        basis = vector_valued_monomial_basis(r, m, n, ordering="interleaved")
        basis_expected = [
            Polynomial([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], r, m),
            Polynomial([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]], r, m)
        ]
        for i in range(len(basis)):
            assert polynomials_equal(basis[i][0], basis_expected[i][0], r, m)
            assert polynomials_equal(basis[i][1], basis_expected[i][1], r, m)

    @staticmethod
    def test_dual_basis_fn():
        m = 2
        r = 3
        for nu in multiindex.generate_all(m, r):
            q = dual_monomial_basis_fn(nu)
            for mu in multiindex.generate_all(m, r):
                p = monomial_basis_fn(mu)
                if nu == mu:
                    assert q(p) == 1.0
                else:
                    assert q(p) == 0.0

    @staticmethod
    @pytest.mark.slow
    def test_dual_vector_valued_basis_fn():
        m = 2
        n = 2
        for r in range(4):
            basis = vector_valued_monomial_basis(r, m, n)
            dual_basis = dual_vector_valued_monomial_basis(r, m, n)
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
        p = zero_polynomial(m, 1)
        assert p.basis() == unique_identifier_monomial_basis()
        assert abs(p(x)) < 1e-12
        for n in [2, 3]:
            p = zero_polynomial(m, n)
            assert p.basis() == unique_identifier_monomial_basis()
            assert np.linalg.norm(p(x) - np.zeros(n)) < 1e-12

    # Test zero polynomial expressed in the degree 2 polynomial basis
    r = 2
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        p = zero_polynomial(r, m, 1)
        assert p.basis() == unique_identifier_monomial_basis()
        assert p.degree() == r
        assert abs(p(x)) < 1e-12
        for n in [2, 3]:
            p = zero_polynomial(r, m, n)
            assert p.basis() == unique_identifier_monomial_basis()
            assert p.degree() == r
            assert np.linalg.norm(p(x) - np.zeros(n)) < 1e-12


def test_unit_polynomial():
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        p = unit_polynomial(m, 1)
        assert p.basis() == unique_identifier_monomial_basis()
        assert abs(p(x) - 1) < 1e-12
        for n in [2, 3]:
            p = unit_polynomial(m, n)
            assert p.basis() == unique_identifier_monomial_basis()
            assert np.linalg.norm(p(x) - np.ones(n)) < 1e-12

    # Test unit polynomial expressed in the degree 2 polynomial basis
    r = 2
    for m in [1, 2, 3]:
        x = nsimplex_sampling(m, 1)[0]
        p = unit_polynomial(r, m, 1)
        assert p.basis() == unique_identifier_monomial_basis()
        assert p.degree() == r
        assert abs(p(x) - 1) < 1e-12
        for n in [2, 3]:
            p = unit_polynomial(r, m, n)
            assert p.basis() == unique_identifier_monomial_basis()
            assert p.degree() == r
            assert np.linalg.norm(p(x) - np.ones(n)) < 1e-12


if __name__ == '__main__':
    pytest.main(sys.argv)
