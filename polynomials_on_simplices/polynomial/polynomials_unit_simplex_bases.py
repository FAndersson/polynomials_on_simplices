"""Functionality for working with different bases for polynomials on the unit simplex, and for converting between
these bases.
"""

import copy

import numpy as np

from polynomials_on_simplices.polynomial.polynomials_monomial_basis import (
    dual_monomial_basis, dual_monomial_basis_fn, dual_vector_valued_monomial_basis,
    dual_vector_valued_monomial_basis_fn, monomial_basis, monomial_basis_fn, monomial_basis_fn_latex,
    monomial_basis_fn_latex_compact, monomial_basis_latex, monomial_basis_latex_compact,
    unique_identifier_monomial_basis, vector_valued_monomial_basis, vector_valued_monomial_basis_fn)
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_bernstein_basis import (
    PolynomialBernstein, bernstein_basis, bernstein_basis_fn, bernstein_basis_fn_latex,
    bernstein_basis_fn_latex_compact, bernstein_basis_latex, bernstein_basis_latex_compact, dual_bernstein_basis,
    dual_bernstein_basis_fn, dual_vector_valued_bernstein_basis, dual_vector_valued_bernstein_basis_fn,
    unique_identifier_bernstein_basis, vector_valued_bernstein_basis, vector_valued_bernstein_basis_fn)
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import (
    PolynomialLagrange, dual_lagrange_basis, dual_lagrange_basis_fn, dual_vector_valued_lagrange_basis,
    dual_vector_valued_lagrange_basis_fn, lagrange_basis, lagrange_basis_fn, lagrange_basis_fn_latex,
    lagrange_basis_fn_latex_compact, lagrange_basis_latex, lagrange_basis_latex_compact,
    unique_identifier_lagrange_basis, vector_valued_lagrange_basis, vector_valued_lagrange_basis_fn)


def convert_polynomial_to_basis(p, target_basis):
    r"""
    Convert a polynomial in :math:`\mathcal{P}_r (\Delta_c^m)` to the given basis.

    :param p: Polynomial expanded in some basis.
    :param str target_basis: Unique identifier for the basis we want to expand the polynomial in.
    :return: Polynomial expanded in the given basis.
    """
    if p.basis() == target_basis:
        return copy.deepcopy(p)
    if target_basis == unique_identifier_monomial_basis():
        return p.to_monomial_basis()
    m = p.domain_dimension()
    n = p.target_dimension()
    r = p.degree()
    coeff = np.empty(p.coeff.shape)
    dual_basis = dual_polynomial_basis(r, m, target_basis)
    if n == 1:
        for i in range(len(coeff)):
            coeff[i] = dual_basis[i](p)
    else:
        # Handle each component of p separately
        for i in range(len(coeff)):
            for j in range(n):
                coeff[i][j] = dual_basis[i](p[j])
    if target_basis == unique_identifier_lagrange_basis():
        return PolynomialLagrange(coeff, r, m)
    else:
        if target_basis == unique_identifier_bernstein_basis():
            return PolynomialBernstein(coeff, r, m)
        else:
            raise ValueError("Unknown polynomial basis")


def polynomial_basis_fn(nu, r, basis):
    r"""
    Generate a basis polynomial in the space :math:`\mathcal{P}_r(\Delta_c^n)` (where n is equal to the length
    of nu) in the given basis.

    :param nu: Multi-index indicating which basis polynomial should be generated.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param str basis: Unique identifier for the basis we should generate a base polynomial for.
    :return: The base polynomial as specified by nu, r and basis.
    :rtype: Implementation of :class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase`.
    """
    if basis == unique_identifier_monomial_basis():
        return monomial_basis_fn(nu)
    if basis == unique_identifier_lagrange_basis():
        return lagrange_basis_fn(nu, r)
    if basis == unique_identifier_bernstein_basis():
        return bernstein_basis_fn(nu, r)
    raise ValueError("Unknown polynomial basis")


def polynomial_basis(r, n, basis):
    r"""
    Generate all base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)` in the given basis.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the unit simplex.
    :param str basis: Unique identifier for the basis we should generate base polynomials for.
    :return: List of base polynomials in the specified basis.
    """
    if basis == unique_identifier_monomial_basis():
        return monomial_basis(r, n)
    if basis == unique_identifier_lagrange_basis():
        return lagrange_basis(r, n)
    if basis == unique_identifier_bernstein_basis():
        return bernstein_basis(r, n)
    raise ValueError("Unknown polynomial basis")


def vector_valued_polynomial_basis_fn(nu, r, i, n, basis):
    r"""
    Generate a basis polynomial for the space :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)` (where m is equal to
    the length of nu) in the given basis.

    :param nu: Multi-index indicating which basis polynomial should be generated.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param int i: Index of the vector component that is non-zero.
    :param int n: Dimension of the target.
    :param str basis: Unique identifier for the basis we should generate a base polynomial for.
    :return: The base polynomial as specified by nu, r and basis.
    :rtype: Implementation of :class:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase`.
    """
    if basis == unique_identifier_monomial_basis():
        return vector_valued_monomial_basis_fn(nu, i, n)
    if basis == unique_identifier_lagrange_basis():
        return vector_valued_lagrange_basis_fn(nu, r, i, n)
    if basis == unique_identifier_bernstein_basis():
        return vector_valued_bernstein_basis_fn(nu, r, i, n)
    raise ValueError("Unknown polynomial basis")


def vector_valued_polynomial_basis(r, m, n, basis, ordering="interleaved"):
    r"""
    Generate all base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)` in the given basis.

    :param int r: Degree of the polynomial space.
    :param int m: Dimension of the domain.
    :param int n: Dimension of the target.
    :param str basis: Unique identifier for the basis we should generate base polynomials for.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: List of base polynomials in the specified basis.
    """
    if n == 1:
        return polynomial_basis(m, r, basis)
    if basis == unique_identifier_monomial_basis():
        return vector_valued_monomial_basis(r, m, n, ordering)
    if basis == unique_identifier_lagrange_basis():
        return vector_valued_lagrange_basis(r, m, n, ordering)
    if basis == unique_identifier_bernstein_basis():
        return vector_valued_bernstein_basis(r, m, n, ordering)
    raise ValueError("Unknown polynomial basis")


def dual_polynomial_basis_fn(mu, r, basis):
    r"""
    Generate a dual basis function to a polynomial basis, i.e. the linear map
    :math:`q_{\mu, r} : \mathcal{P}_r(\Delta_c^n) \to \mathbb{R}` such that

    .. math::

        q_{\mu, r}(p_{\nu, r}) = \delta_{\mu, \nu},

    where :math:`p_{\nu, r}` is the degree r basis polynomial indexed by the multi-index :math:`\nu` in the given
    basis and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial space.
    :param str basis: Unique identifier for the basis we should generate a dual base function for.
    :return: The dual basis function as specified by mu, r and basis.
    :rtype: Callable :math:`q_{\mu, r}(p)`.
    """
    if basis == unique_identifier_monomial_basis():
        return dual_monomial_basis_fn(mu)
    if basis == unique_identifier_lagrange_basis():
        return dual_lagrange_basis_fn(mu, r)
    if basis == unique_identifier_bernstein_basis():
        return dual_bernstein_basis_fn(mu, r)
    raise ValueError("Unknown polynomial basis")


def dual_polynomial_basis(r, n, basis):
    r"""
    Generate all dual base functions for the space :math:`\mathcal{P}_r(\Delta_c^n)` in the given basis (i.e. a basis
    for :math:`\mathcal{P}_r(\Delta_c^n)^*`).

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the domain.
    :param str basis: Unique identifier for the basis we should generate dual base functions for.
    :return: List of dual base functions.
    :rtype: List[callable `q(p)`].
    """
    if basis == unique_identifier_monomial_basis():
        return dual_monomial_basis(r, n)
    if basis == unique_identifier_lagrange_basis():
        return dual_lagrange_basis(r, n)
    if basis == unique_identifier_bernstein_basis():
        return dual_bernstein_basis(r, n)
    raise ValueError("Unknown polynomial basis")


def dual_vector_valued_polynomial_basis_fn(mu, r, i, n, basis):
    r"""
    Generate a dual basis function to a vector valued polynomial basis, i.e. the linear map
    :math:`q_{\mu, i} : \mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n) \to \mathbb{R}` that satisfies

    .. math::

        q_{\mu, i}(p_{\nu, j}) = \delta_{\mu, \nu} \delta_{i, j},

    where :math:`p_{\nu, j}` is the degree :math:`|\nu|` vector valued basis polynomial indexed by the
    multi-index :math:`\nu` with a non-zero i:th component in the given basis (see
    :func:`vector_valued_polynomial_basis_fn`) and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...].
    :param int r: Degree of polynomial space.
    :param int i: Integer indicating which dual basis function should be generated.
    :param int n: Dimension of the target.
    :param str basis: Unique identifier for the basis we should generate a dual base function for.
    :return: The dual basis function as specified by mu, r and i.
    :rtype: Callable :math:`q_{\mu, i}(p)`.
    """
    if basis == unique_identifier_monomial_basis():
        return dual_vector_valued_monomial_basis_fn(mu, i, n)
    if basis == unique_identifier_lagrange_basis():
        return dual_vector_valued_lagrange_basis_fn(mu, r, i, n)
    if basis == unique_identifier_bernstein_basis():
        return dual_vector_valued_bernstein_basis_fn(mu, r, i, n)
    raise ValueError("Unknown polynomial basis")


def dual_vector_valued_polynomial_basis(r, m, n, basis, ordering="interleaved"):
    r"""
    Generate all dual base functions for the space :math:`\mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n)` in the
    given basis (i.e. the basis for :math:`\mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n)^*`).

    See :func:`dual_vector_valued_polynomial_basis_fn`.

    :param int r: Degree of the polynomial space.
    :param int m: Dimension of the domain.
    :param int n: Dimension of the target.
    :param str basis: Unique identifier for the basis we should generate dual base functions for.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: List of dual base functions.
    :rtype: List[callable `q(p)`].
    """
    if basis == unique_identifier_monomial_basis():
        return dual_vector_valued_monomial_basis(r, m, n, ordering)
    if basis == unique_identifier_lagrange_basis():
        return dual_vector_valued_lagrange_basis(r, m, n, ordering)
    if basis == unique_identifier_bernstein_basis():
        return dual_vector_valued_bernstein_basis(r, m, n, ordering)
    raise ValueError("Unknown polynomial basis")


def polynomial_basis_fn_latex(nu, r, basis):
    r"""
    Generate Latex string for a basis polynomial for the space :math:`\mathcal{P}_r(\mathbb{R}^n)` (where n is equal
    to the length of nu) in the given basis.

    :param nu: Multi-index indicating which basis polynomial should be generated.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param str basis: Unique identifier for the basis we should generate a basis polynomial Latex string for.
    :return: Latex string for the base polynomial as specified by nu, r and basis.
    :rtype: str

    .. rubric:: Examples

    >>> polynomial_basis_fn_latex(3, 3, unique_identifier_monomial_basis())
    'x^3'
    >>> polynomial_basis_fn_latex((1, 1, 1), 3, unique_identifier_bernstein_basis())
    '6 x_1 x_2 x_3'
    """
    if basis == unique_identifier_monomial_basis():
        return monomial_basis_fn_latex(nu)
    if basis == unique_identifier_lagrange_basis():
        return lagrange_basis_fn_latex(nu, r)
    if basis == unique_identifier_bernstein_basis():
        return bernstein_basis_fn_latex(nu, r)
    raise ValueError("Unknown polynomial basis")


def polynomial_basis_fn_latex_compact(nu, r, basis):
    r"""
    Generate compact Latex string for a basis polynomial for the space :math:`\mathcal{P}_r(\mathbb{R}^n)` (where n
    is equal to the length of nu) in the given basis, using the common shorthand notation for the given basis.

    :param nu: Multi-index indicating which basis polynomial should be generated.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param str basis: Unique identifier for the basis we should generate a basis polynomial Latex string for.
    :return: Latex string for the base polynomial as specified by nu, r and basis.
    :rtype: str

    .. rubric:: Examples

    >>> polynomial_basis_fn_latex_compact(3, 3, unique_identifier_monomial_basis())
    'x^3'
    >>> polynomial_basis_fn_latex_compact((1, 1), 3, unique_identifier_monomial_basis())
    'x^{(1, 1)}'
    >>> polynomial_basis_fn_latex_compact((1, 1, 1), 3, unique_identifier_bernstein_basis())
    'b_{(1, 1, 1), 3}(x)'
    """
    if basis == unique_identifier_monomial_basis():
        return monomial_basis_fn_latex_compact(nu)
    if basis == unique_identifier_lagrange_basis():
        return lagrange_basis_fn_latex_compact(nu, r)
    if basis == unique_identifier_bernstein_basis():
        return bernstein_basis_fn_latex_compact(nu, r)
    raise ValueError("Unknown polynomial basis")


def polynomial_basis_latex(r, n, basis):
    r"""
    Generate Latex strings for all base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)` in the given
    basis.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the unit simplex.
    :param str basis: Unique identifier for the basis we should generate base polynomial Latex strings for.
    :return: List of Latex strings for each base polynomials in the specified basis.
    :rtype: List[str]

    .. rubric:: Examples

    >>> polynomial_basis_latex(2,1,unique_identifier_monomial_basis())
    ['1', 'x', 'x^2']
    >>> polynomial_basis_latex(2,2,unique_identifier_bernstein_basis())
    ['(1 - x_1 - x_2)^2', '2 x_1 (1 - x_1 - x_2)', 'x_1^2', '2 x_2 (1 - x_1 - x_2)', '2 x_1 x_2', 'x_2^2']
    """
    if basis == unique_identifier_monomial_basis():
        return monomial_basis_latex(r, n)
    if basis == unique_identifier_lagrange_basis():
        return lagrange_basis_latex(r, n)
    if basis == unique_identifier_bernstein_basis():
        return bernstein_basis_latex(r, n)
    raise ValueError("Unknown polynomial basis")


def polynomial_basis_latex_compact(r, n, basis):
    r"""
    Generate compact Latex strings for all base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)` in the
    given basis.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the unit simplex.
    :param str basis: Unique identifier for the basis we should generate base polynomial Latex strings for.
    :return: List of Latex strings for each base polynomials in the specified basis.
    :rtype: List[str]

    .. rubric:: Examples

    >>> polynomial_basis_latex_compact(2,1,unique_identifier_monomial_basis())
    ['1', 'x', 'x^2']
    >>> polynomial_basis_latex_compact(1,2,unique_identifier_bernstein_basis())
    ['b_{(0, 0), 1}(x)', 'b_{(1, 0), 1}(x)', 'b_{(0, 1), 1}(x)']
    """
    if basis == unique_identifier_monomial_basis():
        return monomial_basis_latex_compact(r, n)
    if basis == unique_identifier_lagrange_basis():
        return lagrange_basis_latex_compact(r, n)
    if basis == unique_identifier_bernstein_basis():
        return bernstein_basis_latex_compact(r, n)
    raise ValueError("Unknown polynomial basis")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
