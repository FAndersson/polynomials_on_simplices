"""
Calculus functionality (differentiation and integration) for polynomials expressed in the Bernstein basis (see
:mod:`~polynomials_on_simplices.polynomial.polynomials_unit_simplex_bernstein_basis`).
"""

import numbers

import numpy as np

from polynomials_on_simplices.geometry.primitives.simplex import affine_transformation_from_unit, dimension
from polynomials_on_simplices.polynomial.polynomials_base import get_dimension


def integrate_bernstein_basis_fn_unit_simplex(r, n):
    r"""
    Integrate a degree r Bernstein base polynomial over the n-dimensional unit simplex.

    .. math::

        \int_{\Delta_c^n} b_{\nu, r}(x) \, dx = \frac{1}{\prod_{i = 1}^n (r + i)}.

    Note that this is independent of nu, i.e. each basis polynomial has the same integral over the unit simplex.

    :param int r: Degree of the basis polynomial.
    :param int n: Dimension of the unit simplex.
    :return: Evaluation of the integral of the basis polynomial over the unit simplex.
    :rtype: float

    .. rubric:: Examples

    :math:`\int_{\Delta_c^2} b_{\nu, 2}(x) \, dx = \frac{1}{\prod_{i = 1}^2 (2 + i)} = \frac{1}{3 \cdot 4}`.

    >>> integrate_bernstein_basis_fn_unit_simplex(2, 2) == 1 / 12
    True
    """
    denom = 1
    for i in range(1, n + 1):
        denom *= (r + i)
    return 1 / denom


def integrate_bernstein_polynomial_unit_simplex(r, a, n):
    r"""
    Integrate a degree r Bernstein polynomial over the n-dimensional unit simplex.

    .. math::

        \int_{\Delta_c^n} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1} a_{\nu_i} b_{\nu_i, r}(x) \, dx
        = \frac{1}{\prod_{i = 1}^n (r + i)} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1} a_{\nu_i},

    where :math:`\nu_i` is the i:th n-dimensional multi-index of norm <= r (see
    :func:`polynomials_on_simplices.algebra.multiindex.generate` and
    :func:`polynomials_on_simplices.algebra.multiindex.generate_all` for the sequence of and ordering of multi-indices).

    :param int r: Degree of the polynomial.
    :param a: Coefficients :math:`a_{\nu_i}` in the polynomial (array where element i is the coefficient
        :math:`a_{\nu_i}`)
    :param int n: Dimension of the unit simplex.
    :return: Evaluation of the integral of the polynomial over the unit simplex.

    .. rubric:: Examples

    :math:`\int_0^1 2(1 - x) + x \, dx = 1 + \frac{1}{2} = \frac{3}{2}`.

    >>> integrate_bernstein_polynomial_unit_simplex(1, [2, 1], 1) == 3 / 2
    True
    """
    assert len(a) == get_dimension(r, n)
    c = integrate_bernstein_basis_fn_unit_simplex(r, n)
    return c * sum(a)


def integrate_bernstein_basis_fn_simplex(r, vertices):
    r"""
    Integrate a degree r Bernstein base polynomial over an n-dimensional simplex T.

    .. math::

        \int_{T} b_{\nu, r}(x) \, dx
        = \int_{\Phi(\Delta_c^n)} \bar{b}_{\nu, r} \circ \Phi^{-1}(x)
        = |\det D\Phi| \int_{\Delta_c^n} \bar{b}_{\nu, r}(x)
        = |\det D\Phi| \frac{1}{\prod_{i = 1}^n (r + i)}.

    Note that this is independent of nu, i.e. each basis polynomial has the same integral over the simplex T.

    :param int r: Degree of the basis polynomial.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: Evaluation of the integral of the basis polynomial over the simplex T.
    :rtype: float
    """
    n = dimension(vertices)
    iu = integrate_bernstein_basis_fn_unit_simplex(r, n)
    a, b = affine_transformation_from_unit(vertices)
    if isinstance(a, numbers.Number):
        det = a
    else:
        det = np.linalg.det(a)
    assert det > 0
    return det * iu


def integrate_bernstein_polynomial_simplex(r, a, vertices):
    r"""
    Integrate a degree r Bernstein polynomial over an n-dimensional simplex T.

    .. math::

        \int_{T} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1} a_{\nu_i} b_{\nu_i, r}(x) \, dx
        = \int_{\Phi(\Delta_c^n)} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1}
            a_{\nu_i} (bar{b}_{\nu_i, r} \circ \Phi^{-1})(x) \, dx
        = |\det D\Phi| \int_{\Delta_c^n} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1}
            a_{\nu_i} bar{b}_{\nu_i, r}(x) \, dx
        = |\det D\Phi| \frac{1}{\prod_{i = 1}^n (r + i)} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1} a_{\nu_i},

    where :math:`\nu_i` is the i:th n-dimensional multi-index of norm <= r (see
    :func:`polynomials_on_simplices.algebra.multiindex.generate` and
    :func:`polynomials_on_simplices.algebra.multiindex.generate_all` for the sequence of and ordering of multi-indices).

    :param int r: Degree of the polynomial.
    :param a: Coefficients :math:`a_{\nu_i}` in the polynomial (array where element i is the coefficient
        :math:`a_{\nu_i}`)
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: Evaluation of the integral of the polynomial over the simplex T.
    """
    assert len(a) == get_dimension(r, dimension(vertices))
    c = integrate_bernstein_basis_fn_simplex(r, vertices)
    return c * sum(a)
