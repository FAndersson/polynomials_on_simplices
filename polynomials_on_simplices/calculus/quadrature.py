"""Numerical evaluation of integrals over triangles (or simplices).

For triangle quadrature rules, see D.A. Dunavant, High degree efficient symmetrical Gaussian
quadrature rules for the triangle.
For tetrahedron quadrature rules, see Yu Jinyun
Symmetric Gaussian Quadrature Formulae for Tetrahedronal Regions,
and the PHG FEM software: http://lsec.cc.ac.cn/phg/index_en.htm.
"""

import numpy as np
from scipy.integrate import dblquad, quadrature, tplquad
from sympy.integrals.quadrature import gauss_legendre

from polynomials_on_simplices.algebra.algebraic_operations import composition
from polynomials_on_simplices.calculus.affine_map import create_affine_map
from polynomials_on_simplices.geometry.mesh.basic_meshes.tet_meshes import (
    general_tetrahedron_vertices, tetrahedron_triangulation)
from polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes import (
    general_triangle_vertices, triangle_triangulation)
from polynomials_on_simplices.geometry.mesh.simplicial_complex import simplex_vertices
from polynomials_on_simplices.geometry.mesh.triangle_mesh import vertices as triangle_vertices
from polynomials_on_simplices.geometry.primitives.simplex import affine_transformation_from_unit, centroid, volume
from polynomials_on_simplices.geometry.primitives.triangle import area


def quadrature_unit_interval(f):
    r"""
    Numerically compute the integral :math:`\int_0^1 f(x) \, dx`.

    :param f: Function to integrate.
    :type f: Callable f(x)
    :return: Approximate value of the integral.
    """
    return quadrature(f, 0, 1, vec_func=False)[0]


def gauss_legendre_unit_interval(n, n_digits=10):
    r"""
    Compute quadrature points and weights for Gauss-Legendre quadrature modified from the interval [-1, 1] to the
    unit interval.

    .. math::
        \int_0^1 f(x)\,dx \approx \sum_{i=0}^{n-1} w_i f(x_i).

    :param n: The order of quadrature (number of points and weights).
    :param n_digits: Number of significant digits of the points and weights to return.
    :return: Tuple (x, w) of lists containing the points x and the weights w.
    """
    x, w = gauss_legendre(n, n_digits)
    x_mod = np.empty(len(x))
    w_mod = np.empty(len(w))
    # \int_0^1 f(x) dx = 1/2 \int_{-1}^1 f((y + 1) / 2) dy
    # so that
    # x_mod = (x + 1) / 2
    # w_mod = 0.5 * w
    for i in range(len(x)):
        x_mod[i] = float((x[i] + 1) / 2)
        w_mod[i] = float(0.5 * w[i])
    return x_mod, w_mod


def quadrature_unit_interval_fixed(f, r):
    r"""
    Numerically compute the integral :math:`\int_0^1 f(x) \, dx`.

    :param f: Function to integrate.
    :type f: Callable f(x)
    :param r: Quadrature degree. The result is guaranteed to be exact for all polynomials of degree <= r.
    :return: Approximate value of the integral.
    """
    # Compute number of Gauss points necessary
    n = np.ceil((r + 1) / 2)
    xg, wg = gauss_legendre_unit_interval(n)
    integral = 0
    for i in range(len(xg)):
        integral += wg[i] * f(xg[i])
    return integral


def quadrature_interval(f, a, b):
    r"""
    Numerically compute the integral :math:`\int_a^b f(x) \, dx`.

    :param f: Function to integrate.
    :type f: Callable f(x)
    :param a: Start point of the interval over which we should integrate f.
    :param b: End point of the interval over which we should integrate f.
    :return: Approximate value of the integral.
    """
    return quadrature(f, a, b, vec_func=False)[0]


def quadrature_interval_fixed(f, a, b, n):
    r"""
    Numerically compute the integral :math:`\int_a^b f(x) \, dx`.

    :param f: Function to integrate.
    :type f: Callable f(x)
    :param a: Start point of the interval over which we should integrate f.
    :param b: End point of the interval over which we should integrate f.
    :param n: Quadrature degree. The result is guaranteed to be exact for all polynomials of degree <= n.
    :return: Approximate value of the integral.
    """
    # Pullback the function to the unit interval
    ta, tb = affine_transformation_from_unit(np.array([[a], [b]]))
    phi = create_affine_map(ta, tb)
    pb_f = composition(f, phi)
    # Integrate the pulled-back function over the unit interval
    d = np.abs(ta)
    return d * quadrature_unit_interval_fixed(pb_f, n)


def quadrature_unit_triangle(f):
    r"""
    Numerically compute the integral :math:`\int_{\Delta^2_c} f(x) \, dx`.

    :param f: Function to integrate.
    :type f: Callable f(x, y)
    :return: Approximate value of the integral.
    """
    def gfun(_):
        return 0

    def hfun(x):
        return 1 - x

    return dblquad(f, 0, 1, gfun, hfun)[0]


def quadrature_unit_triangle_fixed(f, n):
    r"""
    Numerically compute the integral :math:`\int_{\Delta^2_c} f(x) \, dx`.

    :param f: Function to integrate.
    :type f: Callable f(x, y)
    :param n: Quadrature degree. The result is guaranteed to be exact for all polynomials of degree <= n.
    :return: Approximate value of the integral.
    """
    if n == 1:
        return 1 / 2 * f(1 / 3, 1 / 3)
    if n == 2:
        return 1 / 6 * (f(1 / 2, 1 / 2)
                        + f(0, 1 / 2)
                        + f(1 / 2, 0))
    if n == 3:
        return 1 / 96 * (-27 * f(1 / 3, 1 / 3)
                         + 25 * f(0.2, 0.6)
                         + 25 * f(0.6, 0.2)
                         + 25 * f(0.2, 0.2))
    if n == 4:
        return 0.5 * (0.223381589678011 * f(0.108103018168070, 0.445948490915965)
                      + 0.223381589678011 * f(0.445948490915965, 0.108103018168070)
                      + 0.223381589678011 * f(0.445948490915965, 0.445948490915965)
                      + 0.109951743655322 * f(0.816847572980459, 0.091576213509771)
                      + 0.109951743655322 * f(0.091576213509771, 0.816847572980459)
                      + 0.109951743655322 * f(0.091576213509771, 0.091576213509771))
    raise ValueError("Quadrature not implemented for n > 4")


def quadrature_triangle(f, vertices):
    r"""
    Numerically compute the integral :math:`\int_{T} f(x) \, dx`, for a given triangle T.

    :param f: Function to integrate.
    :type f: Callable f(x, y)
    :param vertices: Vertices of the triangle (3 by 2 array).
    :return: Approximate value of the integral.
    """
    # Pullback the function to the unit triangle
    ta, tb = affine_transformation_from_unit(vertices)
    phi = create_affine_map(ta, tb, multiple_arguments=True)
    pb_f = composition(f, phi)
    # Integrate the pulled-back function over the unit triangle
    d = np.abs(np.linalg.det(ta))
    return d * quadrature_unit_triangle(pb_f)


def quadrature_triangle_fixed(f, vertices, n):
    r"""
    Numerically compute the integral :math:`\int_{T} f(x) \, dx`, for a given triangle T.

    :param f: Function to integrate.
    :type f: Callable f(x, y)
    :param vertices: Vertices of the triangle (3 by 2 array).
    :param n: Quadrature degree. The result is guaranteed to be exact for all polynomials of degree <= n.
    :return: Approximate value of the integral.
    """
    # Pullback the function to the unit triangle
    ta, tb = affine_transformation_from_unit(vertices)
    phi = create_affine_map(ta, tb, multiple_arguments=True)
    pb_f = composition(f, phi)
    # Integrate the pulled-back function over the unit triangle
    d = np.abs(np.linalg.det(ta))
    return d * quadrature_unit_triangle_fixed(pb_f, n)


def quadrature_unit_tetrahedron(f):
    r"""
    Numerically compute the integral :math:`\int_{\Delta^3_c} f(x) \, dx`.

    :param f: Function to integrate.
    :type f: Callable f(x, y, z)
    :return: Approximate value of the integral.
    """
    def gfun(_):
        return 0

    def hfun(x):
        return 1 - x

    def qfun(_, __):
        return 0

    def rfun(x, y):
        return 1 - x - y

    return tplquad(f, 0, 1, gfun, hfun, qfun, rfun)[0]


def quadrature_unit_tetrahedron_fixed(f, n):
    r"""
    Numerically compute the integral :math:`\int_{\Delta^3_c} f(x) \, dx`.

    :param f: Function to integrate.
    :type f: Callable f(x, y, z)
    :param n: Quadrature degree. The result is guaranteed to be exact for all polynomials of degree <= n.
    :return: Approximate value of the integral.
    """
    if n == 1:
        return 1 / 6 * f(1 / 4, 1 / 4, 1 / 4)
    if n == 2:
        # pt1 = (5.0 + 3 * sqrt(5.0)) / 20, pt2, 3, 4 = (5.0 - sqrt(5.0)) / 20
        return 1 / 24 * (f(0.58541019662496852, 0.1381966011250105, 0.1381966011250105)
                         + f(0.1381966011250105, 0.58541019662496852, 0.1381966011250105)
                         + f(0.1381966011250105, 0.1381966011250105, 0.58541019662496852)
                         + f(0.1381966011250105, 0.1381966011250105, 0.1381966011250105))
    if n == 3:
        return 1 / 6 * (-0.8 * f(1 / 4, 1 / 4, 1 / 4)
                        + 0.45 * f(0.5, 1 / 6, 1 / 6)
                        + 0.45 * f(1 / 6, 0.5, 1 / 6)
                        + 0.45 * f(1 / 6, 1 / 6, 0.5)
                        + 0.45 * f(1 / 6, 1 / 6, 1 / 6))
    if n == 4:
        return 1 / 6 * (0.07349304311636194934358694586367885 * f(0.7217942490673264,
                                                                  0.09273525031089122628655892066032137,
                                                                  0.09273525031089122628655892066032137)
                        + 0.07349304311636194934358694586367885 * f(0.09273525031089122628655892066032137,
                                                                    0.7217942490673264,
                                                                    0.09273525031089122628655892066032137)
                        + 0.07349304311636194934358694586367885 * f(0.09273525031089122628655892066032137,
                                                                    0.09273525031089122628655892066032137,
                                                                    0.7217942490673264)
                        + 0.07349304311636194934358694586367885 * f(0.09273525031089122628655892066032137,
                                                                    0.09273525031089122628655892066032137,
                                                                    0.09273525031089122628655892066032137)
                        + 0.11268792571801585036501492847638892 * f(0.06734224221009821,
                                                                    0.31088591926330060975814749494040332,
                                                                    0.31088591926330060975814749494040332)
                        + 0.11268792571801585036501492847638892 * f(0.31088591926330060975814749494040332,
                                                                    0.06734224221009821,
                                                                    0.31088591926330060975814749494040332)
                        + 0.11268792571801585036501492847638892 * f(0.31088591926330060975814749494040332,
                                                                    0.31088591926330060975814749494040332,
                                                                    0.06734224221009821)
                        + 0.11268792571801585036501492847638892 * f(0.31088591926330060975814749494040332,
                                                                    0.31088591926330060975814749494040332,
                                                                    0.31088591926330060975814749494040332)
                        + 0.04254602077708146686093208377328816 * f(0.04550370412564965000000000000000000,
                                                                    0.04550370412564965000000000000000000,
                                                                    0.45449629587435036)
                        + 0.04254602077708146686093208377328816 * f(0.04550370412564965000000000000000000,
                                                                    0.45449629587435036,
                                                                    0.04550370412564965000000000000000000)
                        + 0.04254602077708146686093208377328816 * f(0.04550370412564965000000000000000000,
                                                                    0.45449629587435036,
                                                                    0.45449629587435036)
                        + 0.04254602077708146686093208377328816 * f(0.45449629587435036,
                                                                    0.04550370412564965000000000000000000,
                                                                    0.04550370412564965000000000000000000)
                        + 0.04254602077708146686093208377328816 * f(0.45449629587435036,
                                                                    0.04550370412564965000000000000000000,
                                                                    0.45449629587435036)
                        + 0.04254602077708146686093208377328816 * f(0.45449629587435036,
                                                                    0.45449629587435036,
                                                                    0.04550370412564965000000000000000000))
    raise ValueError("Quadrature not implemented for n > 4")


def quadrature_tetrahedron(f, vertices):
    r"""
    Numerically compute the integral :math:`\int_{T} f(x) \, dx`, for a given tetrahedron T.

    :param f: Function to integrate.
    :type f: Callable f(x, y, z)
    :param vertices: Vertices of the tetrahedron (4 by 3 array).
    :return: Approximate value of the integral.
    """
    # Pullback the function to the unit tetrahedron
    ta, tb = affine_transformation_from_unit(vertices)
    phi = create_affine_map(ta, tb, multiple_arguments=True)
    pb_f = composition(f, phi)
    # Integrate the pulled-back function over the unit tetrahedron
    d = np.abs(np.linalg.det(ta))
    return d * quadrature_unit_tetrahedron(pb_f)


def quadrature_tetrahedron_fixed(f, vertices, n):
    r"""
    Numerically compute the integral :math:`\int_{T} f(x) \, dx`, for a given tetrahedron T.

    :param f: Function to integrate.
    :type f: Callable f(x, y, z)
    :param vertices: Vertices of the tetrahedron (4 by 3 array).
    :param n: Quadrature degree. The result is guaranteed to be exact for all polynomials of degree <= n.
    :return: Approximate value of the integral.
    """
    # Pullback the function to the unit tetrahedron
    ta, tb = affine_transformation_from_unit(vertices)
    phi = create_affine_map(ta, tb, multiple_arguments=True)
    pb_f = composition(f, phi)
    # Integrate the pulled-back function over the unit tetrahedron
    d = np.abs(np.linalg.det(ta))
    return d * quadrature_unit_tetrahedron_fixed(pb_f, n)


def quadrature_triangle_midpoint_rule(f, vertices, n):
    r"""
    Numerically compute the integral :math:`\int_T f \, dx` using the midpoint rule.

    .. note::

        Only useful for testing, since :func:`quadrature_triangle` and :func:`quadrature_triangle_fixed` are much
        more efficient.

    :param f: Function to integrate.
    :type f: Callable f(x, y)
    :param vertices: Vertices of the triangle to integrate over (3 by 2 array).
    :param n: Number of smaller triangles along each edge of the full triangle,
        over which the function is approximated with a constant value.
    :return: Approximate value of the integral.
    """
    sub_triangles = triangle_triangulation(n + 1)
    sub_vertices = general_triangle_vertices(vertices, n + 1)
    integral = 0
    # All sub triangles have the same area
    sub_triangle_area = area(vertices) / len(sub_triangles)
    for i in range(len(sub_triangles)):
        tri_vertices = triangle_vertices(sub_triangles[i], sub_vertices)
        midpoint = centroid(tri_vertices)
        integral += sub_triangle_area * f(midpoint[0], midpoint[1])
    return integral


def quadrature_tetrahedron_midpoint_rule(f, vertices, n):
    r"""
    Numerically compute the integral :math:`\int_T f \, dx` using the midpoint rule.

    .. note::

        Only useful for testing, since :func:`quadrature_tetrahedron` and :func:`quadrature_tetrahedron_fixed` are
        much more efficient.

    :param f: Function to integrate.
    :type f: Callable f(x, y, z)
    :param vertices: Vertices of the tetrahedron to integrate over (4 by 3 array).
    :param n: Number of smaller tetrahedrons along each edge of the full tetrahedron,
        over which the function is approximated with a constant value.
    :return: Approximate value of the integral.
    """
    sub_tetrahedrons = tetrahedron_triangulation(n + 1)
    sub_vertices = general_tetrahedron_vertices(vertices, n + 1)
    integral = 0
    # All sub tetrahedrons have the same volume
    sub_tetrahedron_volume = volume(vertices) / len(sub_tetrahedrons)
    for i in range(len(sub_tetrahedrons)):
        tet_vertices = simplex_vertices(sub_tetrahedrons[i], sub_vertices)
        midpoint = centroid(tet_vertices)
        integral += sub_tetrahedron_volume * f(midpoint[0], midpoint[1], midpoint[2])
    return integral
