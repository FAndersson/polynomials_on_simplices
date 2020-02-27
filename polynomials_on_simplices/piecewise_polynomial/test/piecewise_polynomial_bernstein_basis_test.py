import sys

import numpy as np
import pytest

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.geometry.mesh.basic_meshes.tet_meshes import (
    rectangular_box_triangulation, unit_cube_vertices)
from polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes import (
    rectangle_triangulation, rectangle_vertices, unit_square_vertices)
from polynomials_on_simplices.geometry.mesh.simplicial_complex import simplex_vertices
from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial import piecewise_polynomials_equal
from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial_bernstein_basis import (
    PiecewisePolynomialBernstein, dual_piecewise_polynomial_bernstein_basis, generate_inverse_local_to_global_map,
    generate_local_to_global_map, piecewise_polynomial_bernstein_basis, unit_piecewise_polynomial_bernstein,
    zero_piecewise_polynomial_bernstein)
from polynomials_on_simplices.polynomial.polynomials_base import get_dimension
from polynomials_on_simplices.polynomial.polynomials_simplex_base import polynomials_equal_on_simplex


def line_strip_primitives(resolution):
    """
    Create the indexed primitives of a line strip.

    :param resolution: Number of vertices in the line strip.
    :return: resolution - 1 by 2 list of primitives.
    """
    primitives = np.empty((resolution - 1, 2), dtype=int)
    for i in range(resolution - 1):
        primitives[i][0] = i
        primitives[i][1] = i + 1
    return primitives


def create_scalar_valued_univariate_piecewise_polynomial():
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    r = 2
    coeff = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    return PiecewisePolynomialBernstein(coeff, lines, vertices, r)


def create_random_scalar_valued_univariate_piecewise_polynomial(r=2):
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    dim = (n - 1) * get_dimension(r, 1)
    coeff = np.random.rand(dim)
    return PiecewisePolynomialBernstein(coeff, lines, vertices, r)


def create_vector_valued_univariate_piecewise_polynomial():
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    r = 2
    coeff = [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]]
    return PiecewisePolynomialBernstein(coeff, lines, vertices, r)


def create_random_vector_valued_univariate_piecewise_polynomial(r=2):
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    dim = (n - 1) * get_dimension(r, 1)
    coeff = np.random.random_sample((dim, 2))
    return PiecewisePolynomialBernstein(coeff, lines, vertices, r)


def create_scalar_valued_bivariate_piecewise_polynomial():
    triangles = rectangle_triangulation()
    vertices = rectangle_vertices(1, 1)[:, 0:2]
    r = 2
    coeff = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
    return PiecewisePolynomialBernstein(coeff, triangles, vertices, r)


def create_random_scalar_valued_bivariate_piecewise_polynomial(r=2):
    triangles = rectangle_triangulation()
    vertices = rectangle_vertices(1, 1)[:, 0:2]
    dim = 2 * get_dimension(r, 2)
    coeff = np.random.rand(dim)
    return PiecewisePolynomialBernstein(coeff, triangles, vertices, r)


def create_vector_valued_bivariate_piecewise_polynomial():
    triangles = rectangle_triangulation()
    vertices = rectangle_vertices(1, 1)[:, 0:2]
    r = 2
    coeff = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
    return PiecewisePolynomialBernstein(coeff, triangles, vertices, r)


def create_random_vector_valued_bivariate_piecewise_polynomial(r=2):
    triangles = rectangle_triangulation()
    vertices = rectangle_vertices(1, 1)[:, 0:2]
    dim = 2 * get_dimension(r, 2)
    coeff = np.random.random_sample((dim, 2))
    return PiecewisePolynomialBernstein(coeff, triangles, vertices, r)


def test_call():
    # Test calling a scalar valued univariate piecewise polynomial
    p = create_scalar_valued_univariate_piecewise_polynomial()
    expected_value = 1.5
    assert abs(p(0.0625) - expected_value) < 1e-12
    assert abs(p(0.3125) - expected_value) < 1e-12
    assert abs(p(0.5625) - expected_value) < 1e-12
    assert abs(p(0.8125) - expected_value) < 1e-12

    # Test calling a vector valued univariate piecewise polynomial
    p = create_vector_valued_univariate_piecewise_polynomial()
    expected_value = np.array([1.5, 1.5])
    assert np.linalg.norm(p(0.0625) - expected_value) < 1e-12
    assert np.linalg.norm(p(0.3125) - expected_value) < 1e-12
    assert np.linalg.norm(p(0.5625) - expected_value) < 1e-12
    assert np.linalg.norm(p(0.8125) - expected_value) < 1e-12

    # Test calling a scalar valued bivariate piecewise polynomial
    p = create_scalar_valued_bivariate_piecewise_polynomial()
    expected_value = 3.555555555555556
    assert abs(p([-1 / 6, -1 / 6]) - expected_value) < 1e-12
    assert abs(p([1 / 6, 1 / 6]) - expected_value) < 1e-12

    # Test calling a vector valued bivariate piecewise polynomial
    p = create_vector_valued_bivariate_piecewise_polynomial()
    expected_value = np.array([3.555555555555556, 3.555555555555556])
    assert np.linalg.norm(p([-1 / 6, -1 / 6]) - expected_value) < 1e-12
    assert np.linalg.norm(p([1 / 6, 1 / 6]) - expected_value) < 1e-12


def test_getitem():
    # Test getting the components of a scalar valued univariate piecewise polynomial
    p = create_scalar_valued_univariate_piecewise_polynomial()

    def p0_expected(x):
        return p(x)
    assert piecewise_polynomials_equal(p[0], p0_expected, 2)

    # Test getting the components of a vector valued univariate piecewise polynomial
    p = create_vector_valued_univariate_piecewise_polynomial()
    for i in range(2):
        def pi_expected(x):
            return p(x)[i]

        assert piecewise_polynomials_equal(p[i], pi_expected, 2)

    # Test getting the components of a scalar valued bivariate piecewise polynomial
    p = create_scalar_valued_bivariate_piecewise_polynomial()

    def p0_expected(x):
        return p(x)
    assert piecewise_polynomials_equal(p[0], p0_expected, 2)

    # Test getting the components of a vector valued bivariate piecewise polynomial
    p = create_vector_valued_bivariate_piecewise_polynomial()
    for i in range(2):
        def pi_expected(x):
            return p(x)[i]

        assert piecewise_polynomials_equal(p[i], pi_expected, 2)


def test_add():
    # Test adding two scalar valued univariate piecewise polynomials
    p1 = create_random_scalar_valued_univariate_piecewise_polynomial()
    p2 = create_random_scalar_valued_univariate_piecewise_polynomial()

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two vector valued univariate piecewise polynomials
    p1 = create_random_vector_valued_univariate_piecewise_polynomial()
    p2 = create_random_vector_valued_univariate_piecewise_polynomial()

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two scalar valued bivariate piecewise polynomials
    p1 = create_random_scalar_valued_bivariate_piecewise_polynomial()
    p2 = create_random_scalar_valued_bivariate_piecewise_polynomial()

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two vector valued bivariate piecewise polynomials
    p1 = create_random_vector_valued_bivariate_piecewise_polynomial()
    p2 = create_random_vector_valued_bivariate_piecewise_polynomial()

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two scalar valued univariate piecewise polynomials of different degree
    p1 = create_random_scalar_valued_univariate_piecewise_polynomial(2)
    p2 = create_random_scalar_valued_univariate_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two vector valued univariate piecewise polynomials of different degree
    p1 = create_random_vector_valued_univariate_piecewise_polynomial(2)
    p2 = create_random_vector_valued_univariate_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two scalar valued bivariate piecewise polynomials of different degree
    p1 = create_random_scalar_valued_bivariate_piecewise_polynomial(3)
    p2 = create_random_scalar_valued_bivariate_piecewise_polynomial(2)

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two vector valued bivariate piecewise polynomials of different degree
    p1 = create_random_vector_valued_bivariate_piecewise_polynomial(2)
    p2 = create_random_vector_valued_bivariate_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)


def test_sub():
    # Test subtracting two scalar valued univariate piecewise polynomials
    p1 = create_random_scalar_valued_univariate_piecewise_polynomial()
    p2 = create_random_scalar_valued_univariate_piecewise_polynomial()

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two vector valued univariate piecewise polynomials
    p1 = create_random_vector_valued_univariate_piecewise_polynomial()
    p2 = create_random_vector_valued_univariate_piecewise_polynomial()

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two scalar valued bivariate piecewise polynomials
    p1 = create_random_scalar_valued_bivariate_piecewise_polynomial()
    p2 = create_random_scalar_valued_bivariate_piecewise_polynomial()

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two vector valued bivariate piecewise polynomials
    p1 = create_random_vector_valued_bivariate_piecewise_polynomial()
    p2 = create_random_vector_valued_bivariate_piecewise_polynomial()

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two scalar valued univariate piecewise polynomials of different degree
    p1 = create_random_scalar_valued_univariate_piecewise_polynomial(2)
    p2 = create_random_scalar_valued_univariate_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two vector valued univariate piecewise polynomials of different degree
    p1 = create_random_vector_valued_univariate_piecewise_polynomial(2)
    p2 = create_random_vector_valued_univariate_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two scalar valued bivariate piecewise polynomials of different degree
    p1 = create_random_scalar_valued_bivariate_piecewise_polynomial(3)
    p2 = create_random_scalar_valued_bivariate_piecewise_polynomial(2)

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two vector valued bivariate piecewise polynomials of different degree
    p1 = create_random_vector_valued_bivariate_piecewise_polynomial(2)
    p2 = create_random_vector_valued_bivariate_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)


def test_mul():
    # Test multiplying a scalar valued univariate piecewise polynomial with a scalar
    p = create_random_scalar_valued_univariate_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return s * p(x)

    assert piecewise_polynomials_equal(s * p, p_expected)

    # Test multiplying a vector valued univariate piecewise polynomial with a scalar
    p = create_random_vector_valued_univariate_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return s * p(x)

    assert piecewise_polynomials_equal(s * p, p_expected)

    # Test multiplying a scalar valued bivariate piecewise polynomial with a scalar
    p = create_random_scalar_valued_bivariate_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return s * p(x)

    assert piecewise_polynomials_equal(s * p, p_expected)

    # Test multiplying a vector valued bivariate piecewise polynomial with a scalar
    p = create_random_vector_valued_bivariate_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return s * p(x)

    assert piecewise_polynomials_equal(s * p, p_expected)

    # Test multiplying a scalar valued univariate piecewise polynomial with a vector
    p = create_random_scalar_valued_univariate_piecewise_polynomial()
    v = np.random.rand(2)

    def p_expected(x):
        return v * p(x)

    assert piecewise_polynomials_equal(p * v, p_expected)

    # Test multiplying a scalar valued bivariate piecewise polynomial with a vector
    p = create_random_scalar_valued_bivariate_piecewise_polynomial()
    v = np.random.rand(2)

    def p_expected(x):
        return v * p(x)

    assert piecewise_polynomials_equal(p * v, p_expected)

    # Test multiplying two scalar valued univariate piecewise polynomials
    p1 = create_random_scalar_valued_univariate_piecewise_polynomial(2)
    p2 = create_random_scalar_valued_univariate_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) * p2(x)

    assert piecewise_polynomials_equal(p1 * p2, p_expected)

    # Test multiplying two scalar valued bivariate piecewise polynomials
    p1 = create_random_scalar_valued_bivariate_piecewise_polynomial(3)
    p2 = create_random_scalar_valued_bivariate_piecewise_polynomial(2)

    def p_expected(x):
        return p1(x) * p2(x)

    assert piecewise_polynomials_equal(p1 * p2, p_expected)


def test_div():
    # Test dividing a scalar valued univariate piecewise polynomial with a scalar
    p = create_random_scalar_valued_univariate_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return p(x) / s

    assert piecewise_polynomials_equal(p / s, p_expected)

    # Test dividing a vector valued univariate piecewise polynomial with a scalar
    p = create_random_vector_valued_univariate_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return p(x) / s

    assert piecewise_polynomials_equal(p / s, p_expected)

    # Test dividing a scalar valued bivariate piecewise polynomial with a scalar
    p = create_random_scalar_valued_bivariate_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return p(x) / s

    assert piecewise_polynomials_equal(p / s, p_expected)

    # Test dividing a vector valued bivariate piecewise polynomial with a scalar
    p = create_random_vector_valued_bivariate_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return p(x) / s

    assert piecewise_polynomials_equal(p / s, p_expected)


@pytest.mark.slow
def test_pow():
    # Test taking the power of a scalar valued univariate piecewise polynomial
    p = create_random_scalar_valued_univariate_piecewise_polynomial()
    r = 3
    for exponent in range(r):
        def p_expected(x):
            return p(x)**exponent

        assert piecewise_polynomials_equal(p**exponent, p_expected)

    # Test taking the power of a vector valued univariate piecewise polynomial
    p = create_random_vector_valued_univariate_piecewise_polynomial()
    r = 3
    for exponent in multiindex.generate_all(2, r):
        def p_expected(x):
            return multiindex.power(p(x), exponent)

        assert piecewise_polynomials_equal(p**exponent, p_expected)

    # Test taking the power of a scalar valued bivariate piecewise polynomial
    p = create_random_scalar_valued_bivariate_piecewise_polynomial()
    r = 3
    for exponent in range(r):
        def p_expected(x):
            return p(x)**exponent

        assert piecewise_polynomials_equal(p**exponent, p_expected)

    # Test taking the power of a vector valued bivariate piecewise polynomial
    p = create_random_vector_valued_bivariate_piecewise_polynomial()
    r = 3
    for exponent in multiindex.generate_all(2, r):
        def p_expected(x):
            return multiindex.power(p(x), exponent)

        assert piecewise_polynomials_equal(p**exponent, p_expected)


def test_degree_elevate():
    # Test degree elevating a scalar valued univariate piecewise polynomial
    p = create_random_scalar_valued_univariate_piecewise_polynomial()
    r = p.degree()
    for s in range(r, r + 3):
        q = p.degree_elevate(s)

        assert piecewise_polynomials_equal(p, q)

    # Test degree elevating a vector valued univariate piecewise polynomial
    p = create_random_vector_valued_univariate_piecewise_polynomial()
    r = p.degree()
    for s in range(r, r + 3):
        q = p.degree_elevate(s)

        assert piecewise_polynomials_equal(p, q)

    # Test degree elevating a scalar valued bivariate piecewise polynomial
    p = create_random_scalar_valued_bivariate_piecewise_polynomial()
    r = p.degree()
    for s in range(r, r + 3):
        q = p.degree_elevate(s)

        assert piecewise_polynomials_equal(p, q)

    # Test degree elevating a vector valued bivariate piecewise polynomial
    p = create_random_vector_valued_bivariate_piecewise_polynomial()
    r = p.degree()
    for s in range(r, r + 3):
        q = p.degree_elevate(s)

        assert piecewise_polynomials_equal(p, q)


def test_restrict_to_simplex():
    # Test restricting a scalar valued univariate piecewise polynomial to individual edges in the triangulation
    p = create_random_scalar_valued_univariate_piecewise_polynomial()
    for i in range(len(p.triangles)):
        p_rest = p.restrict_to_simplex(i)
        vertices = simplex_vertices(p.triangles[i], p.vertices)

        assert polynomials_equal_on_simplex(p_rest, p, p.degree(), vertices)

    # Test restricting a vector valued univariate piecewise polynomial to individual edges in the triangulation
    p = create_random_vector_valued_univariate_piecewise_polynomial()
    for i in range(len(p.triangles)):
        p_rest = p.restrict_to_simplex(i)
        vertices = simplex_vertices(p.triangles[i], p.vertices)

        assert polynomials_equal_on_simplex(p_rest, p, p.degree(), vertices)

    # Test restricting a scalar valued bivariate piecewise polynomial to individual triangles in the triangulation
    p = create_random_scalar_valued_bivariate_piecewise_polynomial()
    for i in range(len(p.triangles)):
        p_rest = p.restrict_to_simplex(i)
        vertices = simplex_vertices(p.triangles[i], p.vertices)

        assert polynomials_equal_on_simplex(p_rest, p, p.degree(), vertices)

    # Test restricting a vector valued bivariate piecewise polynomial to individual triangles in the triangulation
    p = create_random_vector_valued_bivariate_piecewise_polynomial()
    for i in range(len(p.triangles)):
        p_rest = p.restrict_to_simplex(i)
        vertices = simplex_vertices(p.triangles[i], p.vertices)

        assert polynomials_equal_on_simplex(p_rest, p, p.degree(), vertices)


def delta(i, j):
    if i == j:
        return 1.0
    else:
        return 0.0


class TestUnivariateBernsteinBasis:
    @staticmethod
    @pytest.mark.slow
    def test_basis():
        m = 1
        res = 5
        lines = line_strip_primitives(res)
        vertices = np.linspace(0, 1, res)
        for r in range(4):
            tau, num_dofs = generate_local_to_global_map(lines, r)
            tau_inv = generate_inverse_local_to_global_map(tau, len(lines), num_dofs, r, m)
            basis = piecewise_polynomial_bernstein_basis(lines, vertices, r, tau)
            dual_basis = dual_piecewise_polynomial_bernstein_basis(lines, vertices, r, tau, num_dofs, tau_inv)
            for i in range(num_dofs):
                for j in range(num_dofs):
                    assert abs(dual_basis[j](basis[i]) - delta(i, j)) < 1e-14


class TestBivariateBernsteinBasis:
    @staticmethod
    @pytest.mark.slow
    def test_basis():
        m = 2
        res = 3
        triangles = rectangle_triangulation(res, res)
        vertices = unit_square_vertices(res, res)[:, 0:2]
        for r in range(4):
            tau, num_dofs = generate_local_to_global_map(triangles, r)
            tau_inv = generate_inverse_local_to_global_map(tau, len(triangles), num_dofs, r, m)
            basis = piecewise_polynomial_bernstein_basis(triangles, vertices, r, tau)
            dual_basis = dual_piecewise_polynomial_bernstein_basis(triangles, vertices, r, tau, num_dofs, tau_inv)
            for i in range(num_dofs):
                for j in range(num_dofs):
                    assert abs(dual_basis[j](basis[i]) - delta(i, j)) < 1e-14


def test_zero_piecewise_polynomial():
    # Test zero piecewise polynomial on a 1D mesh
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    p = zero_piecewise_polynomial_bernstein(lines, vertices)
    assert p.basis() == "Bernstein"
    x = np.random.rand()
    assert abs(p(x)) < 1e-15
    for n in [2, 3]:
        p = zero_piecewise_polynomial_bernstein(lines, vertices, n)
        assert p.basis() == "Bernstein"
        assert np.linalg.norm(p(x)) < 1e-15

    # Test zero piecewise polynomial on a 2D mesh
    n = 5
    triangles = rectangle_triangulation(n, n)
    vertices = unit_square_vertices(n, n)[:, 0:2]
    p = zero_piecewise_polynomial_bernstein(triangles, vertices)
    assert p.basis() == "Bernstein"
    x = np.random.rand(2)
    assert abs(p(x)) < 1e-15
    for n in [2, 3]:
        p = zero_piecewise_polynomial_bernstein(triangles, vertices, n)
        assert p.basis() == "Bernstein"
        assert np.linalg.norm(p(x)) < 1e-15

    # Test zero piecewise polynomial on a 3D mesh
    n = 5
    triangles = rectangular_box_triangulation(n, n, n)
    vertices = unit_cube_vertices(n, n, n)
    p = zero_piecewise_polynomial_bernstein(triangles, vertices)
    assert p.basis() == "Bernstein"
    x = np.random.rand(3)
    assert abs(p(x)) < 1e-15
    for n in [2, 3]:
        p = zero_piecewise_polynomial_bernstein(triangles, vertices, n)
        assert p.basis() == "Bernstein"
        assert np.linalg.norm(p(x)) < 1e-15


@pytest.mark.slow
def test_unit_piecewise_polynomial():
    # Test unit piecewise polynomial on a 1D mesh
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    p = unit_piecewise_polynomial_bernstein(lines, vertices)
    assert p.basis() == "Bernstein"
    x = np.random.rand()
    assert abs(p(x) - 1) < 1e-15
    for n in [2, 3]:
        p = unit_piecewise_polynomial_bernstein(lines, vertices, n)
        assert p.basis() == "Bernstein"
        assert np.linalg.norm(p(x) - np.ones(n)) < 1e-15

    # Test unit piecewise polynomial on a 2D mesh
    n = 5
    triangles = rectangle_triangulation(n, n)
    vertices = unit_square_vertices(n, n)[:, 0:2]
    p = unit_piecewise_polynomial_bernstein(triangles, vertices)
    assert p.basis() == "Bernstein"
    x = np.random.rand(2)
    assert abs(p(x) - 1) < 1e-15
    for n in [2, 3]:
        p = unit_piecewise_polynomial_bernstein(triangles, vertices, n)
        assert p.basis() == "Bernstein"
        assert np.linalg.norm(p(x) - np.ones(n)) < 1e-15

    # Test unit piecewise polynomial on a 3D mesh
    n = 5
    triangles = rectangular_box_triangulation(n, n, n)
    vertices = unit_cube_vertices(n, n, n)
    p = unit_piecewise_polynomial_bernstein(triangles, vertices)
    assert p.basis() == "Bernstein"
    x = np.random.rand(3)
    assert abs(p(x) - 1) < 1e-15
    for n in [2, 3]:
        p = unit_piecewise_polynomial_bernstein(triangles, vertices, n)
        assert p.basis() == "Bernstein"
        assert np.linalg.norm(p(x) - np.ones(n)) < 1e-15


if __name__ == '__main__':
    pytest.main(sys.argv)
