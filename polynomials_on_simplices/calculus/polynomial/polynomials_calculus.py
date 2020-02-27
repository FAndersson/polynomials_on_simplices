"""
Calculus functionality (differentiation and integration) for polynomials (objects implementing the
:class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase` interface).
"""

import copy

import numpy as np

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_bernstein_basis_calculus import (
    integrate_bernstein_polynomial_simplex, integrate_bernstein_polynomial_unit_simplex)
from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_lagrange_basis_calculus import (
    integrate_lagrange_polynomial_simplex, integrate_lagrange_polynomial_unit_simplex)
from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_monomial_basis_calculus import (
    integrate_polynomial_unit_simplex)
from polynomials_on_simplices.polynomial.polynomials_base import PolynomialBase
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import unique_identifier_monomial_basis
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_bernstein_basis import (
    unique_identifier_bernstein_basis)
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import unique_identifier_lagrange_basis


def gradient(p):
    r"""
    Compute the gradient of a polynomial,

    .. math:: (\nabla p)_i = \frac{\partial p}{\partial x^i}.

    :param p: Polynomial whose gradient we want to compute.
    :type p: Implementation of the :class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase`
        interface
    :return: Gradient of the polynomial as a list of polynomials (where the i:th entry is the i:th partial derivative
        of p).
    """
    assert isinstance(p, PolynomialBase)

    m = p.domain_dimension()
    n = p.target_dimension()
    if n > 1:
        return jacobian(p)
    return [p.partial_derivative(i) for i in range(m)]


def jacobian(p):
    r"""
    Compute the Jacobian of a polynomial,

    .. math:: (J_p)^i_j = \frac{\partial p^i}{\partial x^j}.

    :param p: Polynomial whose Jacobian we want to compute.
    :type p: Implementation of the :class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase`
        interface
    :return: Jacobian of the polynomial as a list of list of polynomials (where the i:th entry is the gradient of the
        i:th entry of p).
    """
    assert isinstance(p, PolynomialBase)

    n = p.target_dimension()
    return [gradient(p[i]) for i in range(n)]


def derivative(p, alpha):
    r"""
    Compute the higher order partial derivative :math:`\partial^{\alpha}` of the given polynomial.

    .. math:: \partial^{\alpha} : \mathcal{P} \to \mathcal{P},

    .. math::

        (\partial^{\alpha} p)(x)
        = \frac{\partial^{|\alpha|} p(x)}{\partial (x^1)^{\alpha_1} \ldots \partial (x^n)^{\alpha_n}}.

    :param p: Polynomial we want to take the derivative of.
    :type p: Implementation of the :class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase`
        interface
    :param alpha: Multi-index defining the higher order partial derivative.
    :type alpha: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :return: Higher order partial derivative of the given polynomial.
    """
    assert isinstance(p, PolynomialBase)

    dp = copy.deepcopy(p)
    try:
        m = len(alpha)
        assert m == p.domain_dimension()
        # Multivariate polynomial
        for i in range(m):
            for j in range(alpha[i]):
                dp = dp.partial_derivative(i)
        return dp
    except TypeError:
        m = 1
        assert m == p.domain_dimension()
        # univariate polynomial
        for i in range(alpha):
            dp = dp.partial_derivative()
        return dp


def hessian(p):
    r"""
    Compute the Hessian matrix of a polynomial,

    .. math:: (H_p)_{ij} = \frac{\partial^2 p}{\partial x^i \partial x^j}.

    :param p: Polynomial whose Hessian we want to compute.
    :type p: Implementation of the :class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase`
        interface
    :return: Hessian of the polynomial as a size (n, n) 2d-array of polynomials (where element (i, j) of the array
        is equal to :math:`\frac{\partial^2 p}{\partial x^i \partial x^j}` and n is the dimension of the polynomial
        domain).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    assert isinstance(p, PolynomialBase)

    m = p.domain_dimension()
    n = p.target_dimension()
    # Can only compute the Hessian of a scalar valued polynomial
    assert n == 1
    h = np.empty((m, m), dtype=object)
    for i in range(m):
        for j in range(i, m):
            alpha = multiindex.zero_multiindex(m)
            alpha[i] += 1
            alpha[j] += 1
            h[i][j] = derivative(p, alpha)
            if i != j:
                h[j][i] = h[i][j]
    return h


def partial_derivatives_array(p, r):
    r"""
    Compute the multi-array consisting of all the degree r partial derivatives of a polynomial,

    .. math::

        (D_p)_{i_1, i_2, \ldots, i_r} = \frac{\partial^r p}{\partial x^{i_1} \partial x^{i_2} \cdots \partial x^{i_r}}.

    For r = 1 this gives the gradient of p and for r = 2 it gives the Hessian matrix.

    :param p: Polynomial whose derivative multi-array we want to compute.
    :type p: Implementation of the :class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase`
        interface
    :param int r: Degree of the partial derivatives we want to compute.
    :return: Degree r derivatives of the polynomial as an r-dimensional array where each array is indexed from 0 to
        n-1 (where element :math:`(i_1, i_2, \ldots, i_r)` of the array is equal to
        :math:`\frac{\partial^r p}{\partial x^{i_1} \partial x^{i_2} \cdots \partial x^{i_r}}` and n is the
        dimension of the polynomial domain).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    assert isinstance(p, PolynomialBase)

    m = p.domain_dimension()
    n = p.target_dimension()
    # Can only compute the derivative multi-array of a scalar valued polynomial
    assert n == 1
    # Partial derivatives can't have negative degree
    assert r >= 0

    # Handle the special case of zero:th order partial derivatives
    if r == 0:
        return p

    d = np.empty((m,) * r, dtype=object)
    for i in multiindex.generate_all_non_decreasing(r, m - 1):
        alpha = multiindex.zero_multiindex(m)
        for r in range(len(i)):
            alpha[i[r]] += 1
        pd = derivative(p, alpha)
        for ip in _unique_permutations(i.to_tuple()):
            d[ip] = pd
    return d


def integrate_unit_simplex(p):
    r"""
    Integrate a general degree r polynomial over the n-dimensional unit simplex :math:`\Delta_c^n`.

    .. math::

        \int_{\Delta_c^n} p(x) \, dx.

    :param p: Polynomial that should be integrated over the unit simplex.
    :type p: Implementation of the :class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase`
        interface
    :return: Evaluation of the integral.
    """
    assert isinstance(p, PolynomialBase)

    if p.basis() == unique_identifier_monomial_basis():
        return integrate_polynomial_unit_simplex(p.degree(), p.domain_dimension(), p.coeff)
    if p.basis() == unique_identifier_lagrange_basis():
        return integrate_lagrange_polynomial_unit_simplex(p.degree(), p.coeff, p.domain_dimension())
    if p.basis() == unique_identifier_bernstein_basis():
        return integrate_bernstein_polynomial_unit_simplex(p.degree(), p.coeff, p.domain_dimension())
    raise TypeError("Cannot integrate polynomial: Unknown polynomial basis")


def integrate_simplex(p, vertices):
    r"""
    Integrate a general degree r polynomial over an n-dimensional simplex T.

    .. math::

        \int_{T} p(x) \, dx.

    :param p: Polynomial that should be integrated over the simplex.
    :type p: Implementation of the :class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase`
        interface
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: Evaluation of the integral.
    """
    assert isinstance(p, PolynomialBase)

    if p.basis().startswith("Lagrange"):
        return integrate_lagrange_polynomial_simplex(p.degree(), p.coeff, vertices)
    if p.basis().startswith("Bernstein"):
        return integrate_bernstein_polynomial_simplex(p.degree(), p.coeff, vertices)
    raise TypeError("Cannot integrate polynomial: Unknown polynomial basis")


def _unique_permutations(t):
    """
    Compute all unique permutations of the elements of a tuple.

    :param t: Tuple of elements which we want to find all permutations of.
    :type t: tuple
    :return: List of all permutations of the input tuple.
    :rtype: List[tuple].

    .. rubric:: Examples

    >>> t = (0, 0, 1)
    >>> p = set(_unique_permutations(t))
    >>> p_expected = {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
    >>> p == p_expected
    True

    >>> t = (0, 0, 1, 2)
    >>> p = set(_unique_permutations(t))
    >>> p_expected_1 = {(0, 0, 1, 2), (0, 0, 2, 1), (0, 1, 0, 2), (0, 1, 2, 0), (0, 2, 0, 1), (0, 2, 1, 0)}
    >>> p_expected_2 = {(1, 0, 0, 2), (1, 0, 2, 0), (1, 2, 0, 0), (2, 0, 0, 1), (2, 0, 1, 0), (2, 1, 0, 0)}
    >>> p_expected = p_expected_1.union(p_expected_2)
    >>> p == p_expected
    True
    """
    permutations = set()
    from polynomials_on_simplices.algebra.permutations import permutations as generate_permutations
    from polynomials_on_simplices.algebra.permutations import permute_positions
    for permutation in generate_permutations(len(t)):
        permutations.add(tuple(permute_positions(permutation, list(t))))
    return list(permutations)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
