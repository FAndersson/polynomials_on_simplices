"""
Calculus functionality (differentiation and integration) for polynomials expressed in the monomial basis (see
:mod:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis`).
For integration of a polynomial over a simplex see [Baldoni_2008]_.

.. rubric:: References

.. [Baldoni_2008] Velleda Baldoni, Nicole Berline Jesus A. De Loera, Matthias Koeppe, and Michele Vergne,
    *How to integrate a polynomial over a simplex*, Mathematics of Computation, 80, 09, 2008,
    doi:10.1090/S0025-5718-2010-02378-6
    URL http://www.ams.org/journals/mcom/2011-80-273/S0025-5718-2010-02378-6/home.html.
"""

import math

import numpy as np
from scipy.special import binom

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.polynomial.monomial_integrals_unit_simplex_cache import (
    monomial_integrals_unit_simplex_all_cache)
from polynomials_on_simplices.geometry.primitives.simplex import volume
import polynomials_on_simplices.polynomial.polynomials_base as polynomials_base
from polynomials_on_simplices.set_theory import nfold_cartesian_product


def integrate_homogeneous_polynomial_unit_simplex(nu):
    r"""
    Integrate a homogeneous polynomial over the n-dimensional simplex unit simplex :math:`\Delta_c^n`.

    .. math::

        \int_{\Delta_c^n} x^{\nu} \, dx,

    where n is the length of nu.

    :param nu: Exponent for the homogeneous polynomial.
    :type: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :return: Evaluation of the integral.
    """
    try:
        n = len(nu)
    except TypeError:
        n = 1
        nu = (nu,)

    # We have b_{\nu, |\nu|}(x) = \binom{|\nu|}{\nu} x^{\nu}.
    # Using this we can integrate the homogeneous polynomial by taking advantage of the easy integration formula
    # for Bernstein basis polynomials
    from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_bernstein_basis_calculus import \
        integrate_bernstein_basis_fn_unit_simplex
    r = multiindex.norm(nu)
    return 1 / multiindex.multinom(nu) * integrate_bernstein_basis_fn_unit_simplex(r, n)


def integrate_homogeneous_polynomial_simplex(nu, vertices):
    r"""
    Integrate a homogeneous polynomial over an n-dimensional simplex.

    .. math::

        \int_{\Delta} x^{\nu} \, dx.

    :param nu: Exponent for the homogeneous polynomial.
    :type: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: Evaluation of the integral.
    """
    # If nu is not a multi-index we convert it to a multi-index
    # (to be able to handle both integers and multi-indices)
    if not isinstance(nu, multiindex.MultiIndex):
        nu = multiindex.MultiIndex(nu)
    # Vertices data structure need to support basis arithmetic (multiplication of
    # rows (vertices) with a scalar)
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)

    v = volume(vertices)
    q = multiindex.norm(nu)
    if q == 0:
        return v
    # This is an implementation of formula (18), page 18 in [Baldoni_2008]_.
    n = len(vertices) - 1
    res = 0
    for i in multiindex.generate_all_non_decreasing(q, n):
        if q == 1:
            for eps in {-1, 1}:
                x = eps * vertices[i[0]]
                res += eps * multiindex.power(x, nu)
        else:
            for eps in nfold_cartesian_product({-1, 1}, q):
                w = 1
                for eps_i in eps:
                    w *= eps_i
                x = 0
                for k in range(len(eps)):
                    x += eps[k] * vertices[i[k]]
                # w * f(x)
                res += w * multiindex.power(x, nu)
    res *= v / (2**q * math.factorial(q) * binom(q + n, q))
    return res


def integrate_polynomial_unit_simplex(r, n, a):
    r"""
    Integrate a general degree r polynomial over the n-dimensional unit simplex :math:`\Delta_c^n`.

    .. math::

        \int_{\Delta_c^n} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1} a_{\nu_i} x^{\nu_i} \, dx,

    where :math:`\nu_i` is the i:th n-dimensional multi-index of norm <= r (see
    :func:`polynomials_on_simplices.algebra.multiindex.generate` and
    :func:`polynomials_on_simplices.algebra.multiindex.generate_all` for the sequence of and ordering of multi-indices).

    :param int r: Degree of the polynomial.
    :param int n: Dimension of the unit simplex.
    :param a: Coefficients :math:`a_{\nu_i}` in the polynomial (array where element i is the coefficient
        :math:`a_{\nu_i}`)
    :type a: List
    :return: Evaluation of the integral.

    .. rubric:: Examples

    :math:`\int_0^1 x + 3x^2 \, dx = 1/2 + 3 \cdot 1/3 = 1.5`.

    >>> integrate_polynomial_unit_simplex(2, 1, [0, 1, 3])
    1.5
    """
    assert len(a) == polynomials_base.get_dimension(r, n)

    if (n, r) in monomial_integrals_unit_simplex_all_cache:
        return sum(ai * mon_i for (ai, mon_i) in zip(a, monomial_integrals_unit_simplex_all_cache[(n, r)]))

    res = 0
    i = 0
    for nu in multiindex.generate_all(n, r):
        res += a[i] * integrate_homogeneous_polynomial_unit_simplex(nu)
        i += 1
    return res


def integrate_polynomial_simplex(r, a, vertices):
    r"""
    Integrate a general degree r polynomial over an n-dimensional simplex.

    .. math::

        \int_{\Delta} \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^n)) - 1} a_{\nu_i} x^{\nu_i} \, dx,

    where :math:`\nu_i` is the i:th n-dimensional multi-index of norm <= r (see
    :func:`polynomials_on_simplices.algebra.multiindex.generate` and
    :func:`polynomials_on_simplices.algebra.multiindex.generate_all` for the sequence of and ordering of multi-indices).

    :param int r: Degree of the polynomial.
    :param a: Coefficients :math:`a_{\nu_i}` in the polynomial (array where element i is the coefficient
        :math:`a_{\nu_i}`)
    :type a: List
    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: Evaluation of the integral.

    .. rubric:: Examples

    :math:`\int_0^1 x + 3x^2 \, dx = 1/2 + 3 \cdot 1/3 = 1.5`.

    >>> integrate_polynomial_simplex(2, [0, 1, 3], [[0.0], [1.0]])
    1.5
    """
    res = 0
    n = len(vertices) - 1
    i = 0
    for nu in multiindex.generate_all(n, r):
        res += a[i] * integrate_homogeneous_polynomial_simplex(nu, vertices)
        i += 1
    return res


def l2_inner_product_unit(p1, p2):
    r"""
    Compute the :math:`L^2` inner product of two polynomials.

    .. math::

        \langle p_1, p_2 \rangle = \int_{\Delta_c^n} p_1(x) p_2(x) \, dx.

    :param p1: First polynomial expressed in the monomial basis.
    :type p1: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`
    :param p2: Second polynomial expressed in the monomial basis.
    :type p2: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`
    :return: :math:`L^2` inner product of the two polynomials.
    :rtype: float
    """
    p = p1 * p2
    m = p.domain_dimension()
    return integrate_polynomial_unit_simplex(p.degree(), m, p.coeff)


def l2_norm2_unit(p):
    r"""
    Compute the squared :math:`L^2` norm of a polynomial :math:`p \in \mathcal{P}((\Delta_c^n)`.

    .. math::

        \| p \|^2 = \langle p, p \rangle =  \int_{\Delta_c^n} p(x)^2 \, dx.

    :param p: Polynomial expressed in the monomial basis.
    :type p: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`
    :return: Squared :math:`L^2` norm of the polynomial.
    :rtype: float
    """
    return l2_inner_product_unit(p, p)


def l2_inner_product_simplex(p1, p2, vertices):
    r"""
    Compute the :math:`L^2` inner product of two polynomials.

    .. math::

        \langle p_1, p_2 \rangle = \int_{T} p_1(x) p_2(x) \, dx,

    where T is an n-dimensional simplex.

    :param p1: First polynomial expressed in the monomial basis.
    :type p1: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`
    :param p2: Second polynomial expressed in the monomial basis.
    :type p2: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: :math:`L^2` inner product of the two polynomials.
    :rtype: float
    """
    p = p1 * p2
    return integrate_polynomial_simplex(p.degree(), p.coeff, vertices)


def l2_norm2_simplex(p, vertices):
    r"""
    Compute the squared :math:`L^2` norm of a polynomial :math:`p \in \mathcal{P}((T)`.

    .. math::

        \| p \|^2 = \langle p, p \rangle =  \int_{T} p(x)^2 \, dx,

    where T is an n-dimensional simplex.

    :param p: Polynomial expressed in the monomial basis.
    :type p: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: Squared :math:`L^2` norm of the polynomial.
    :rtype: float
    """
    return l2_inner_product_simplex(p, p, vertices)
