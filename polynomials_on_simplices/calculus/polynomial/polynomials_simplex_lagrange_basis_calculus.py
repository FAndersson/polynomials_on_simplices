"""
Calculus functionality (differentiation and integration) for polynomials expressed in the Lagrange basis (see
:mod:`~polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis`).
"""

import numbers

import numpy as np

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_monomial_basis_calculus import (
    integrate_polynomial_unit_simplex)
from polynomials_on_simplices.geometry.primitives.simplex import affine_transformation_from_unit, dimension
from polynomials_on_simplices.polynomial.polynomials_base import get_dimension
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import (
    get_lagrange_basis_fn_coefficients)


def integrate_lagrange_basis_fn_unit_simplex(nu, r):
    r"""
    Integrate the Lagrange base polynomial as specified by nu and r over the n-dimensional unit simplex.

    .. math::

        \int_{\Delta_c^n} l_{\nu, r}(x) \, dx,

    where n is equal to the length of nu.

    :param nu: Multi-index indicating which Lagrange basis polynomial should be integrated.
        The polynomial will have the value 1 at the point associated with the multi-index,
        and value 0 at all other points.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of the basis polynomial.
    :return: Evaluation of the integral of the basis polynomial over the unit simplex.
    :rtype: float

    .. rubric:: Examples

    :math:`\int_0^1 l_{0, 1}(x) \, dx = \int_0^1 1 - x \, dx = 1 - \frac{1}{2} = \frac{1}{2}`.

    >>> integrate_lagrange_basis_fn_unit_simplex(0, 1) == 1 / 2
    True

    :math:`\int_0^1 l_{0, 2}(x) \, dx = \int_0^1 2x^2 - 3x + 1 \, dx = \frac{2}{3} - \frac{3}{2} + 1
    = \frac{4}{6} - \frac{9}{6} + \frac{6}{6} = \frac{1}{6}`.

    >>> abs(integrate_lagrange_basis_fn_unit_simplex(0, 2) - 1 / 6) < 1e-10
    True

    :math:`\int_{\Delta_c^2} l_{(0, 1)), 2}(x) \, dx = \int_0^1 \int_0^{1 - x_1} -4 x_1 x_2 - 4 x_2^2 + 4 x_2
    = \int_0^1 -2 x_1 (1 - x_1)^2 - \frac{4}{3} (1 - x_1)^3 + 2 (1 - x_1)^2
    = -\frac{2}{3} \frac{1}{4} - \frac{4}{3} \frac{1}{4} + 2 \frac{1}{3}
    = - \frac{1}{6} - \frac{2}{6} + \frac{4}{6} = \frac{1}{6}`.

    >>> abs(integrate_lagrange_basis_fn_unit_simplex((0, 1), 2) - 1 / 6) < 1e-10
    True
    """
    try:
        n = len(nu)
    except TypeError:
        n = 1
    coeff = get_lagrange_basis_fn_coefficients(nu, r)
    return integrate_polynomial_unit_simplex(r, n, coeff)


def integrate_lagrange_polynomial_unit_simplex(r, a, n):
    r"""
    Integrate a degree r Lagrange polynomial over the n-dimensional unit simplex.

    .. math::

        \int_{\Delta_c^n} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1} a_{\nu_i} l_{\nu_i, r}(x) \, dx,

    where :math:`\nu_i` is the i:th n-dimensional multi-index of norm <= r (see
    :func:`polynomials_on_simplices.algebra.multiindex.generate` and
    :func:`polynomials_on_simplices.algebra.multiindex.generate_all` for the sequence of and ordering of multi-indices).

    :param int r: Degree of the polynomial.
    :param int n: Dimension of the unit simplex.
    :param a: Coefficients :math:`a_{\nu_i}` in the polynomial (array where element i is the coefficient
        :math:`a_{\nu_i}`)
    :return: Evaluation of the integral of the polynomial over the unit simplex.

    .. rubric:: Examples

    :math:`\int_0^1 2(1 - x) + x \, dx = 1 + \frac{1}{2} = \frac{3}{2}`.

    >>> integrate_lagrange_polynomial_unit_simplex(1, [2, 1], 1) == 3 / 2
    True
    """
    res = 0
    i = 0
    assert len(a) == get_dimension(r, n)
    for nu in multiindex.generate_all(n, r):
        res += a[i] * integrate_lagrange_basis_fn_unit_simplex(nu, r)
        i += 1
    return res


def integrate_lagrange_basis_fn_simplex(nu, r, vertices):
    r"""
    Integrate the Lagrange base polynomial as specified by nu and r over an n-dimensional simplex T.

    .. math::

        \int_{T} l_{\nu, r}(x) \, dx
        = \int_{\Phi(\Delta_c^n)} \bar{l}_{\nu, r} \circ \Phi^{-1}(x)
        = |\det D\Phi| \int_{\Delta_c^n} \bar{l}_{\nu, r}(x),

    where n is equal to the length of nu.

    :param nu: Multi-index indicating which Lagrange basis polynomial should be integrated.
        The polynomial will have the value 1 at the point associated with the multi-index,
        and value 0 at all other points.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of the basis polynomial.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: Evaluation of the integral of the basis polynomial over the simplex T.
    :rtype: float
    """
    iu = integrate_lagrange_basis_fn_unit_simplex(nu, r)
    a, b = affine_transformation_from_unit(vertices)
    if isinstance(a, numbers.Number):
        det = a
    else:
        det = np.linalg.det(a)
    assert det > 0
    return det * iu


def integrate_lagrange_polynomial_simplex(r, a, vertices):
    r"""
    Integrate a degree r Lagrange polynomial over an n-dimensional simplex T.

    .. math::

        \int_{T} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1} a_{\nu_i} l_{\nu_i, r}(x) \, dx
        = \int_{\Phi(\Delta_c^n)} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1}
            a_{\nu_i} (bar{l}_{\nu_i, r} \circ \Phi^{-1})(x) \, dx
        = |\det D\Phi| \int_{\Delta_c^n} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1}
            a_{\nu_i} bar{l}_{\nu_i, r}(x) \, dx,

    where :math:`\nu_i` is the i:th n-dimensional multi-index of norm <= r (see
    :func:`polynomials_on_simplices.algebra.multiindex.generate` and
    :func:`polynomials_on_simplices.algebra.multiindex.generate_all` for the sequence of and ordering of multi-indices).

    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :param int r: Degree of the polynomial.
    :param a: Coefficients :math:`a_{\nu_i}` in the polynomial (array where element i is the coefficient
        :math:`a_{\nu_i}`)
    :return: Evaluation of the integral of the polynomial over the simplex T.
    """
    res = 0
    i = 0
    n = dimension(vertices)
    assert len(a) == get_dimension(r, n)
    for nu in multiindex.generate_all(n, r):
        res += a[i] * integrate_lagrange_basis_fn_simplex(nu, r, vertices)
        i += 1
    return res
