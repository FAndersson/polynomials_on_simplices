import sys

import numpy as np
import pytest

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.finite_difference import central_difference, central_difference_jacobian
from polynomials_on_simplices.geometry.mesh.basic_meshes.tet_meshes import (
    rectangular_box_triangulation, unit_cube_vertices)
from polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes import (
    rectangle_triangulation, rectangle_vertices, unit_square_vertices)
from polynomials_on_simplices.geometry.mesh.simplicial_complex import simplex_vertices
from polynomials_on_simplices.piecewise_polynomial.continuous_piecewise_polynomial_lagrange_basis import (
    ContinuousPiecewisePolynomialLagrange, continuous_piecewise_polynomial_lagrange_basis,
    dual_continuous_piecewise_polynomial_lagrange_basis, generate_local_to_global_map,
    generate_local_to_global_preimage_map, unit_continuous_piecewise_polynomial_lagrange,
    zero_continuous_piecewise_polynomial_lagrange)
from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial import piecewise_polynomials_equal
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


def create_scalar_valued_univariate_continuous_piecewise_polynomial():
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    r = 2
    coeff = [1, 2, 3, 2, 1, 2, 3, 2, 1]
    return ContinuousPiecewisePolynomialLagrange(coeff, lines, vertices, r)


def create_random_scalar_valued_univariate_continuous_piecewise_polynomial(r=2):
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    dim = (n - 1) * get_dimension(r, 1) - (n - 2)
    coeff = np.random.rand(dim)
    return ContinuousPiecewisePolynomialLagrange(coeff, lines, vertices, r)


def create_vector_valued_univariate_continuous_piecewise_polynomial():
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    r = 2
    coeff = [[1, 1], [2, 2], [3, 3], [2, 2], [1, 1], [2, 2], [3, 3], [2, 2], [1, 1]]
    return ContinuousPiecewisePolynomialLagrange(coeff, lines, vertices, r)


def create_random_vector_valued_univariate_continuous_piecewise_polynomial(r=2):
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    dim = (n - 1) * get_dimension(r, 1) - (n - 2)
    coeff = np.random.random_sample((dim, 2))
    return ContinuousPiecewisePolynomialLagrange(coeff, lines, vertices, r)


def create_scalar_valued_bivariate_continuous_piecewise_polynomial():
    triangles = rectangle_triangulation()
    vertices = rectangle_vertices(1, 1)[:, 0:2]
    r = 2
    coeff = [1, 2, 3, 4, 5, 6, 2, 1, 5]
    return ContinuousPiecewisePolynomialLagrange(coeff, triangles, vertices, r)


def create_random_scalar_valued_bivariate_continuous_piecewise_polynomial(r=2):
    triangles = rectangle_triangulation()
    vertices = rectangle_vertices(1, 1)[:, 0:2]
    dim = 2 * get_dimension(r, 2) - get_dimension(r, 1)
    coeff = np.random.rand(dim)
    return ContinuousPiecewisePolynomialLagrange(coeff, triangles, vertices, r)


def create_vector_valued_bivariate_continuous_piecewise_polynomial():
    triangles = rectangle_triangulation()
    vertices = rectangle_vertices(1, 1)[:, 0:2]
    r = 2
    coeff = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [2, 2], [1, 1], [5, 5]]
    return ContinuousPiecewisePolynomialLagrange(coeff, triangles, vertices, r)


def create_random_vector_valued_bivariate_continuous_piecewise_polynomial(r=2):
    triangles = rectangle_triangulation()
    vertices = rectangle_vertices(1, 1)[:, 0:2]
    dim = 2 * get_dimension(r, 2) - get_dimension(r, 1)
    coeff = np.random.random_sample((dim, 2))
    return ContinuousPiecewisePolynomialLagrange(coeff, triangles, vertices, r)


def test_call():
    # Test calling a scalar valued univariate continuous piecewise polynomial
    p = create_scalar_valued_univariate_continuous_piecewise_polynomial()
    expected_value = 1.5
    assert abs(p(0.0625) - expected_value) < 1e-12
    assert abs(p(0.4375) - expected_value) < 1e-12
    assert abs(p(0.5625) - expected_value) < 1e-12
    assert abs(p(0.9375) - expected_value) < 1e-12

    # Test calling a vector valued univariate continuous piecewise polynomial
    p = create_vector_valued_univariate_continuous_piecewise_polynomial()
    expected_value = np.array([1.5, 1.5])
    assert np.linalg.norm(p(0.0625) - expected_value) < 1e-12
    assert np.linalg.norm(p(0.4375) - expected_value) < 1e-12
    assert np.linalg.norm(p(0.5625) - expected_value) < 1e-12
    assert np.linalg.norm(p(0.9375) - expected_value) < 1e-12

    # Test calling a scalar valued bivariate continuous piecewise polynomial
    p = create_scalar_valued_bivariate_continuous_piecewise_polynomial()
    expected_value = 3.7777777777777772
    expected_value_2 = 4.222222222222222
    assert abs(p([-1 / 6, -1 / 6]) - expected_value) < 1e-12
    assert abs(p([1 / 6, 1 / 6]) - expected_value_2) < 1e-12

    # Test calling a vector valued bivariate continuous piecewise polynomial
    p = create_vector_valued_bivariate_continuous_piecewise_polynomial()
    expected_value = np.array([3.7777777777777772, 3.7777777777777772])
    expected_value_2 = np.array([4.222222222222222, 4.222222222222222])
    assert np.linalg.norm(p([-1 / 6, -1 / 6]) - expected_value) < 1e-12
    assert np.linalg.norm(p([1 / 6, 1 / 6]) - expected_value_2) < 1e-12


def test_call_on_boundary():
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    r = 2
    # Test calling a scalar valued univariate continuous piecewise polynomial that is zero on the boundary
    coeff = [2, 3, 2, 1, 2, 3, 2, 1]
    p = ContinuousPiecewisePolynomialLagrange(coeff, lines, vertices, r, boundary_simplices=[[0]])
    assert p(0) == 0
    # Test calling a scalar valued univariate continuous piecewise polynomial with prescribed value on the boundary
    coeff = [2, 3, 2, 1, 2, 3, 2, 1, 1]
    p = ContinuousPiecewisePolynomialLagrange(coeff, lines, vertices, r, boundary_simplices=[[0]],
                                              keep_boundary_dofs_last=True)
    assert p(0) == 1

    triangles = rectangle_triangulation()
    vertices = rectangle_vertices(1, 1)[:, 0:2]
    r = 2
    # Test calling a scalar valued bivariate continuous piecewise polynomial that is zero on the boundary
    coeff = [4, 5, 6, 2, 1, 5]
    p = ContinuousPiecewisePolynomialLagrange(coeff, triangles, vertices, r, boundary_simplices=[[0, 1]])
    assert p((-0.5, -0.5)) == 0
    assert p((0.0, -0.5)) == 0
    assert p((0.5, -0.5)) == 0
    # Test calling a scalar valued bivariate continuous piecewise polynomial that with prescribed value on the boundary
    coeff = [1, 2, 3, 4, 5, 2, -1, -1, -1]
    p = ContinuousPiecewisePolynomialLagrange(coeff, triangles, vertices, r, boundary_simplices=[[2, 3]],
                                              keep_boundary_dofs_last=True)
    assert p((-0.5, 0.5)) == -1
    assert p((0.0, 0.5)) == -1
    assert p((0.5, 0.5)) == -1


def test_call_failing_spatial_partition():
    r = 1
    n = 3
    triangles = rectangle_triangulation(n, n)
    vertices = rectangle_vertices(1.0, 1.0, n, n)[:, 0:2]
    boundary_simplices = [[0, 1], [1, 2], [8]]

    tau, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                    keep_boundary_dofs_last=True)
    basis = continuous_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau, num_dofs, boundary_simplices,
                                                           True)
    p = basis[0]

    expected_value = 0.6
    assert abs(p([-0.4, 0.1]) - expected_value) < 1e-12


def test_getitem():
    # Test getting the components of a scalar valued univariate piecewise polynomial
    p = create_scalar_valued_univariate_continuous_piecewise_polynomial()

    def p0_expected(x):
        return p(x)
    assert piecewise_polynomials_equal(p[0], p0_expected, 2)

    # Test getting the components of a vector valued univariate piecewise polynomial
    p = create_vector_valued_univariate_continuous_piecewise_polynomial()
    for i in range(2):
        def pi_expected(x):
            return p(x)[i]

        assert piecewise_polynomials_equal(p[i], pi_expected, 2)

    # Test getting the components of a scalar valued bivariate piecewise polynomial
    p = create_scalar_valued_bivariate_continuous_piecewise_polynomial()

    def p0_expected(x):
        return p(x)
    assert piecewise_polynomials_equal(p[0], p0_expected, 2)

    # Test getting the components of a vector valued bivariate piecewise polynomial
    p = create_vector_valued_bivariate_continuous_piecewise_polynomial()
    for i in range(2):
        def pi_expected(x):
            return p(x)[i]

        assert piecewise_polynomials_equal(p[i], pi_expected, 2)


def test_add():
    # Test adding two scalar valued univariate piecewise polynomials
    p1 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()
    p2 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two vector valued univariate piecewise polynomials
    p1 = create_random_vector_valued_univariate_continuous_piecewise_polynomial()
    p2 = create_random_vector_valued_univariate_continuous_piecewise_polynomial()

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two scalar valued bivariate piecewise polynomials
    p1 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()
    p2 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two vector valued bivariate piecewise polynomials
    p1 = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()
    p2 = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two scalar valued univariate piecewise polynomials of different degree
    p1 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial(2)
    p2 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two vector valued univariate piecewise polynomials of different degree
    p1 = create_random_vector_valued_univariate_continuous_piecewise_polynomial(2)
    p2 = create_random_vector_valued_univariate_continuous_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two scalar valued bivariate piecewise polynomials of different degree
    p1 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial(3)
    p2 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial(2)

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)

    # Test adding two vector valued bivariate piecewise polynomials of different degree
    p1 = create_random_vector_valued_bivariate_continuous_piecewise_polynomial(2)
    p2 = create_random_vector_valued_bivariate_continuous_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) + p2(x)

    assert piecewise_polynomials_equal(p1 + p2, p_expected)


def test_sub():
    # Test subtracting two scalar valued univariate piecewise polynomials
    p1 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()
    p2 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two vector valued univariate piecewise polynomials
    p1 = create_random_vector_valued_univariate_continuous_piecewise_polynomial()
    p2 = create_random_vector_valued_univariate_continuous_piecewise_polynomial()

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two scalar valued bivariate piecewise polynomials
    p1 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()
    p2 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two vector valued bivariate piecewise polynomials
    p1 = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()
    p2 = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two scalar valued univariate piecewise polynomials of different degree
    p1 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial(2)
    p2 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two vector valued univariate piecewise polynomials of different degree
    p1 = create_random_vector_valued_univariate_continuous_piecewise_polynomial(2)
    p2 = create_random_vector_valued_univariate_continuous_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two scalar valued bivariate piecewise polynomials of different degree
    p1 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial(3)
    p2 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial(2)

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)

    # Test subtracting two vector valued bivariate piecewise polynomials of different degree
    p1 = create_random_vector_valued_bivariate_continuous_piecewise_polynomial(2)
    p2 = create_random_vector_valued_bivariate_continuous_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) - p2(x)

    assert piecewise_polynomials_equal(p1 - p2, p_expected)


def test_mul():
    # Test multiplying a scalar valued univariate piecewise polynomial with a scalar
    p = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return s * p(x)

    assert piecewise_polynomials_equal(s * p, p_expected)

    # Test multiplying a vector valued univariate piecewise polynomial with a scalar
    p = create_random_vector_valued_univariate_continuous_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return s * p(x)

    assert piecewise_polynomials_equal(s * p, p_expected)

    # Test multiplying a scalar valued bivariate piecewise polynomial with a scalar
    p = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return s * p(x)

    assert piecewise_polynomials_equal(s * p, p_expected)

    # Test multiplying a vector valued bivariate piecewise polynomial with a scalar
    p = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return s * p(x)

    assert piecewise_polynomials_equal(s * p, p_expected)

    # Test multiplying a scalar valued univariate piecewise polynomial with a vector
    p = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()
    v = np.random.rand(2)

    def p_expected(x):
        return v * p(x)

    assert piecewise_polynomials_equal(p * v, p_expected)

    # Test multiplying a scalar valued bivariate piecewise polynomial with a vector
    p = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()
    v = np.random.rand(2)

    def p_expected(x):
        return v * p(x)

    assert piecewise_polynomials_equal(p * v, p_expected)

    # Test multiplying two scalar valued univariate piecewise polynomials
    p1 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial(2)
    p2 = create_random_scalar_valued_univariate_continuous_piecewise_polynomial(3)

    def p_expected(x):
        return p1(x) * p2(x)

    assert piecewise_polynomials_equal(p1 * p2, p_expected)

    # Test multiplying two scalar valued bivariate piecewise polynomials
    p1 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial(3)
    p2 = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial(2)

    def p_expected(x):
        return p1(x) * p2(x)

    assert piecewise_polynomials_equal(p1 * p2, p_expected)


def test_div():
    # Test dividing a scalar valued univariate piecewise polynomial with a scalar
    p = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return p(x) / s

    assert piecewise_polynomials_equal(p / s, p_expected)

    # Test dividing a vector valued univariate piecewise polynomial with a scalar
    p = create_random_vector_valued_univariate_continuous_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return p(x) / s

    assert piecewise_polynomials_equal(p / s, p_expected)

    # Test dividing a scalar valued bivariate piecewise polynomial with a scalar
    p = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return p(x) / s

    assert piecewise_polynomials_equal(p / s, p_expected)

    # Test dividing a vector valued bivariate piecewise polynomial with a scalar
    p = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()
    s = np.random.rand()

    def p_expected(x):
        return p(x) / s

    assert piecewise_polynomials_equal(p / s, p_expected)


@pytest.mark.slow
def test_pow():
    # Test taking the power of a scalar valued univariate piecewise polynomial
    p = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()
    r = 3
    for exponent in range(r):
        def p_expected(x):
            return p(x)**exponent

        assert piecewise_polynomials_equal(p**exponent, p_expected)

    # Test taking the power of a vector valued univariate piecewise polynomial
    p = create_random_vector_valued_univariate_continuous_piecewise_polynomial()
    r = 3
    for exponent in multiindex.generate_all(2, r):
        def p_expected(x):
            return multiindex.power(p(x), exponent)

        assert piecewise_polynomials_equal(p**exponent, p_expected)

    # Test taking the power of a scalar valued bivariate piecewise polynomial
    p = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()
    r = 3
    for exponent in range(r):
        def p_expected(x):
            return p(x)**exponent

        assert piecewise_polynomials_equal(p**exponent, p_expected)

    # Test taking the power of a vector valued bivariate piecewise polynomial
    p = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()
    r = 3
    for exponent in multiindex.generate_all(2, r):
        def p_expected(x):
            return multiindex.power(p(x), exponent)

        assert piecewise_polynomials_equal(p**exponent, p_expected)


def test_degree_elevate():
    # Test degree elevating a scalar valued univariate piecewise polynomial
    p = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()
    r = p.degree()
    for s in range(r, r + 3):
        q = p.degree_elevate(s)

        assert piecewise_polynomials_equal(p, q)

    # Test degree elevating a vector valued univariate piecewise polynomial
    p = create_random_vector_valued_univariate_continuous_piecewise_polynomial()
    r = p.degree()
    for s in range(r, r + 3):
        q = p.degree_elevate(s)

        assert piecewise_polynomials_equal(p, q)

    # Test degree elevating a scalar valued bivariate piecewise polynomial
    p = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()
    r = p.degree()
    for s in range(r, r + 3):
        q = p.degree_elevate(s)

        assert piecewise_polynomials_equal(p, q)

    # Test degree elevating a vector valued bivariate piecewise polynomial
    p = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()
    r = p.degree()
    for s in range(r, r + 3):
        q = p.degree_elevate(s)

        assert piecewise_polynomials_equal(p, q)


def test_restrict_to_simplex():
    # Test restricting a scalar valued univariate piecewise polynomial to individual edges in the triangulation
    p = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()
    for i in range(len(p.triangles)):
        p_rest = p.restrict_to_simplex(i)
        vertices = simplex_vertices(p.triangles[i], p.vertices)

        assert polynomials_equal_on_simplex(p_rest, p, p.degree(), vertices)

    # Test restricting a vector valued univariate piecewise polynomial to individual edges in the triangulation
    p = create_random_vector_valued_univariate_continuous_piecewise_polynomial()
    for i in range(len(p.triangles)):
        p_rest = p.restrict_to_simplex(i)
        vertices = simplex_vertices(p.triangles[i], p.vertices)

        assert polynomials_equal_on_simplex(p_rest, p, p.degree(), vertices)

    # Test restricting a scalar valued bivariate piecewise polynomial to individual triangles in the triangulation
    p = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()
    for i in range(len(p.triangles)):
        p_rest = p.restrict_to_simplex(i)
        vertices = simplex_vertices(p.triangles[i], p.vertices)

        assert polynomials_equal_on_simplex(p_rest, p, p.degree(), vertices)

    # Test restricting a vector valued bivariate piecewise polynomial to individual triangles in the triangulation
    p = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()
    for i in range(len(p.triangles)):
        p_rest = p.restrict_to_simplex(i)
        vertices = simplex_vertices(p.triangles[i], p.vertices)

        assert polynomials_equal_on_simplex(p_rest, p, p.degree(), vertices)


@pytest.mark.slow
def test_partial_derivative():
    # Test partial derivative of a scalar valued univariate piecewise polynomial
    p = create_random_scalar_valued_univariate_continuous_piecewise_polynomial()
    q = p.weak_partial_derivative()

    def dp_dxi_fd(x):
        return central_difference(p, x)

    assert piecewise_polynomials_equal(q, dp_dxi_fd, rel_tol=1e-4)

    # Test partial derivative of a vector valued univariate piecewise polynomial
    p = create_random_vector_valued_univariate_continuous_piecewise_polynomial()
    q = p.weak_partial_derivative()

    def dp_dxi_fd(x):
        return central_difference_jacobian(p, 2, x)

    assert piecewise_polynomials_equal(q, dp_dxi_fd, rel_tol=1e-4)

    # Test partial derivative of a scalar valued bivariate piecewise polynomial
    p = create_random_scalar_valued_bivariate_continuous_piecewise_polynomial()
    for i in range(2):
        q = p.weak_partial_derivative(i)

        def dp_dxi_fd(x):
            return central_difference(p, x)[i]

        assert piecewise_polynomials_equal(q, dp_dxi_fd, rel_tol=1e-4)

    # Test partial derivative of a vector valued bivariate piecewise polynomial
    p = create_random_vector_valued_bivariate_continuous_piecewise_polynomial()
    for i in range(2):
        q = p.weak_partial_derivative(i)

        def dp_dxi_fd(x):
            return central_difference_jacobian(p, 2, x)[:, i]

        assert piecewise_polynomials_equal(q, dp_dxi_fd, rel_tol=1e-4)


def delta(i, j):
    if i == j:
        return 1.0
    else:
        return 0.0


class TestUnivariateLagrangeBasis:
    @staticmethod
    def test_basis():
        m = 1
        res = 5
        lines = line_strip_primitives(res)
        vertices = np.linspace(0, 1, res)
        for r in range(1, 4):
            tau, num_dofs = generate_local_to_global_map(lines, r)
            tau_preim = generate_local_to_global_preimage_map(tau, len(lines), num_dofs, r, m)
            basis = continuous_piecewise_polynomial_lagrange_basis(lines, vertices, r, tau)
            dual_basis = dual_continuous_piecewise_polynomial_lagrange_basis(lines, vertices, r, tau, num_dofs,
                                                                             tau_preim)
            for i in range(num_dofs):
                for j in range(num_dofs):
                    assert abs(dual_basis[j](basis[i]) - delta(i, j)) < 1e-15


class TestBivariateLagrangeBasis:
    @staticmethod
    @pytest.mark.slow
    def test_basis_unit_square():
        m = 2
        res = 5
        triangles = rectangle_triangulation(res, res)
        vertices = unit_square_vertices(res, res)[:, 0:2]
        for r in range(1, 4):
            tau, num_dofs = generate_local_to_global_map(triangles, r)
            tau_preim = generate_local_to_global_preimage_map(tau, len(triangles), num_dofs, r, m)
            basis = continuous_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau)
            dual_basis = dual_continuous_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau, num_dofs,
                                                                             tau_preim)
            for i in range(num_dofs):
                for j in range(num_dofs):
                    assert abs(dual_basis[j](basis[i]) - delta(i, j)) < 1e-14

    @staticmethod
    @pytest.mark.slow
    def test_basis_square():
        m = 2
        res = 5
        triangles = rectangle_triangulation(res, res)
        vertices = rectangle_vertices(1, 1, res, res)[:, 0:2]
        for r in range(1, 4):
            tau, num_dofs = generate_local_to_global_map(triangles, r)
            tau_preim = generate_local_to_global_preimage_map(tau, len(triangles), num_dofs, r, m)
            basis = continuous_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau)
            dual_basis = dual_continuous_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau, num_dofs,
                                                                             tau_preim)
            for i in range(num_dofs):
                for j in range(num_dofs):
                    assert abs(dual_basis[j](basis[i]) - delta(i, j)) < 1e-14

    @staticmethod
    @pytest.mark.slow
    def test_basis_unit_square_with_boundary():
        m = 2
        res = 5
        triangles = rectangle_triangulation(res, res)
        vertices = unit_square_vertices(res, res)[:, 0:2]
        boundary_simplices = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 9], [9, 14], [14, 19], [19, 24]]
        for r in range(1, 4):
            tau, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, boundary_simplices, True)
            tau_preim = generate_local_to_global_preimage_map(tau, len(triangles), num_dofs, r, m)
            basis = continuous_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau, num_dofs,
                                                                   boundary_simplices, True)
            dual_basis = dual_continuous_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau, num_dofs,
                                                                             tau_preim, boundary_simplices, True)
            for i in range(num_dofs):
                for j in range(num_dofs):
                    assert abs(dual_basis[j](basis[i]) - delta(i, j)) < 1e-14


def test_zero_continuous_piecewise_polynomial():
    # Test zero continuous piecewise polynomial on a 1D mesh
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    p = zero_continuous_piecewise_polynomial_lagrange(lines, vertices)
    assert p.basis() == "Lagrange"
    x = np.random.rand()
    assert abs(p(x)) < 1e-15
    for n in [2, 3]:
        p = zero_continuous_piecewise_polynomial_lagrange(lines, vertices, n)
        assert p.basis() == "Lagrange"
        assert np.linalg.norm(p(x)) < 1e-15

    # Test zero continuous piecewise polynomial on a 2D mesh
    n = 5
    triangles = rectangle_triangulation(n, n)
    vertices = unit_square_vertices(n, n)[:, 0:2]
    p = zero_continuous_piecewise_polynomial_lagrange(triangles, vertices)
    assert p.basis() == "Lagrange"
    x = np.random.rand(2)
    assert abs(p(x)) < 1e-15
    for n in [2, 3]:
        p = zero_continuous_piecewise_polynomial_lagrange(triangles, vertices, n)
        assert p.basis() == "Lagrange"
        assert np.linalg.norm(p(x)) < 1e-15

    # Test zero continuous piecewise polynomial on a 3D mesh
    n = 5
    triangles = rectangular_box_triangulation(n, n, n)
    vertices = unit_cube_vertices(n, n, n)
    p = zero_continuous_piecewise_polynomial_lagrange(triangles, vertices)
    assert p.basis() == "Lagrange"
    x = np.random.rand(3)
    assert abs(p(x)) < 1e-15
    for n in [2, 3]:
        p = zero_continuous_piecewise_polynomial_lagrange(triangles, vertices, n)
        assert p.basis() == "Lagrange"
        assert np.linalg.norm(p(x)) < 1e-15


@pytest.mark.slow
def test_unit_continuous_piecewise_polynomial():
    # Test unit continuous piecewise polynomial on a 1D mesh
    n = 5
    lines = line_strip_primitives(n)
    vertices = np.linspace(0, 1, n)
    p = unit_continuous_piecewise_polynomial_lagrange(lines, vertices)
    assert p.basis() == "Lagrange"
    x = np.random.rand()
    assert abs(p(x) - 1) < 1e-15
    for n in [2, 3]:
        p = unit_continuous_piecewise_polynomial_lagrange(lines, vertices, n)
        assert p.basis() == "Lagrange"
        assert np.linalg.norm(p(x) - np.ones(n)) < 1e-15

    # Test unit continuous piecewise polynomial on a 2D mesh
    n = 5
    triangles = rectangle_triangulation(n, n)
    vertices = unit_square_vertices(n, n)[:, 0:2]
    p = unit_continuous_piecewise_polynomial_lagrange(triangles, vertices)
    assert p.basis() == "Lagrange"
    x = np.random.rand(2)
    assert abs(p(x) - 1) < 1e-15
    for n in [2, 3]:
        p = unit_continuous_piecewise_polynomial_lagrange(triangles, vertices, n)
        assert p.basis() == "Lagrange"
        assert np.linalg.norm(p(x) - np.ones(n)) < 1e-15

    # Test unit continuous piecewise polynomial on a 3D mesh
    n = 5
    triangles = rectangular_box_triangulation(n, n, n)
    vertices = unit_cube_vertices(n, n, n)
    p = unit_continuous_piecewise_polynomial_lagrange(triangles, vertices)
    assert p.basis() == "Lagrange"
    x = np.random.rand(3)
    assert abs(p(x) - 1) < 1e-15
    for n in [2, 3]:
        p = unit_continuous_piecewise_polynomial_lagrange(triangles, vertices, n)
        assert p.basis() == "Lagrange"
        assert np.linalg.norm(p(x) - np.ones(n)) < 1e-15


if __name__ == '__main__':
    pytest.main(sys.argv)
