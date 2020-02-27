"""Abstract base class and basic functionality for polynomials defined on a simplex."""

import abc
import math
import numbers

import numpy as np

from polynomials_on_simplices.geometry.primitives.simplex import affine_map_from_unit, dimension
from polynomials_on_simplices.polynomial.polynomials_base import PolynomialBase, get_dimension
from polynomials_on_simplices.probability_theory.uniform_sampling import nsimplex_sampling


def polynomials_equal_on_simplex(p1, p2, r, vertices, rel_tol=1e-9, abs_tol=1e-7):
    r"""
    Check if two polynomials p1 and p2 are approximately equal by comparing their values on a given n-dimensional
    simplex T.

    For scalar valued polynomials, the two polynomials are considered equal if

    .. code-block:: python

        math.isclose(p1(xi), p2(xi), rel_tol=rel_tol, abs_tol=abs_tol)

    is true for a set of random points :math:`\{ x_i \}_{i = 0}^{d - 1}` in the given n-dimensional simplex T,
    where :math:`d` is the dimension of the polynomial space p1 and p2 belongs to (as given by the
    :func:`~polynomials_on_simplices.polynomial.polynomials_base.get_dimension` function).

    For vector valued polynomials the same check is done component wise.

    :param p1: First polynomial.
    :type p1: Callable p1(x)
    :param p2: Second polynomial.
    :type p2: Callable p2(x)
    :param int r: Degree of the polynomials.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :param float rel_tol: Tolerance for the relative error. See :func:`math.isclose <python:math.isclose>` for details.
    :param float abs_tol: Tolerance for the absolute error. See :func:`math.isclose <python:math.isclose>` for details.
    :return: Whether or not the two polynomials are approximately equal.
    :rtype: bool
    """
    # Note: This function takes callables as input instead of instances of the PolynomialBase abstract base class.
    # The reason for this is that the former is more general. It allows us to check for equality for callables
    # that supposedly are polynomials but doesn't implement the PolynomialBase interface.

    n = dimension(vertices)
    phi = affine_map_from_unit(vertices)

    # Generate random domain points where we should check for approximate equality
    dim = get_dimension(r, n)
    if n == 1:
        x_values = [phi(x[0]) for x in nsimplex_sampling(n, dim)]
    else:
        x_values = [phi(x) for x in nsimplex_sampling(n, dim)]

    # Check for approximate equality for the polynomial values at the random domain points
    for x in x_values:
        p = p1(x)
        q = p2(x)
        try:
            len(p)
            # Vector valued polynomials, check for component wise equality
            for i in range(len(p)):
                if not math.isclose(p[i], q[i], rel_tol=rel_tol, abs_tol=abs_tol):
                    return False
        except TypeError:
            # Scalar valued polynomials
            if not math.isclose(p, q, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
    return True


class PolynomialSimplexBase(PolynomialBase, abc.ABC):
    """
    Abstract base class for a polynomial defined on an m-dimensional simplex T.
    """

    def __init__(self, coeff, vertices, r=None):
        r"""
        :param coeff: Coefficients for the polynomial in the chosen basis for :math:`\mathcal{P}_r (T)`
            (see :meth:`~polynomials_on_simplices.polynomial.polynomials_base.PolynomialBase.basis`). If p is
            expressed in the chosen basis :math:`\{ b_{\nu, r} \}` as
            :math:`p(x) = \sum_{\nu} a_{\nu} b_{\nu, r}(x)` then :math:`\text{coeff}[i] = a_{\nu(i)}`, where
            :math:`\nu(i)` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
            :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
            Array of scalars for a scalar valued polynomial (n = 1) and array of n-dimensional vectors for a vector
            valued polynomial (:math:`n \geq 2`).
        :param vertices: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
            simplex).
        :param int r: Degree of the polynomial space. Optional, will be inferred from the number of polynomial
            coefficients if not specified.
        """
        m = len(vertices[0])
        PolynomialBase.__init__(self, coeff, r, m)
        self.vertices = vertices

    def __getitem__(self, i):
        """
        Get the i:th component of the polynomial (for a vector valued polynomial).

        :param int i: Component to get.
        :return: The i:th component of the vector valued polynomial (real valued polynomial).
        :rtype: Instance of self.__class__
        """
        assert i >= 0
        assert i < self.target_dimension()
        if self.target_dimension() == 1:
            return self.__class__(self.coeff, self.vertices, self.r)
        else:
            return self.__class__(self.coeff[:, i], self.vertices, self.r)

    def __add__(self, other):
        """
        Addition of this polynomial with another polynomial, self + other.

        :param other: Other polynomial.
        :return: Sum of the two polynomials.
        :rtype: Instance of self.__class__
        """
        # Added polynomials need to have the same domain and target dimension
        assert self.domain_dimension() == other.domain_dimension()
        assert self.target_dimension() == other.target_dimension()
        # For now require that both polynomials are expressed in the same basis.
        # If not we would need to transform them to some common basis, and what basis
        # this is would need to be specified by the user.
        assert self.basis() == other.basis()
        if self.degree() == other.degree():
            return self.__class__(self.coeff + other.coeff, self.vertices, self.r)
        if self.degree() > other.degree():
            return self + other.degree_elevate(self.degree())
        else:
            return self.degree_elevate(other.degree()) + other

    def __sub__(self, other):
        """
        Subtraction of this polynomial with another polynomial, self - other.

        :param other: Other polynomial.
        :return: Difference of the two polynomials.
        :rtype: Instance of self.__class__
        """
        # Subtracted polynomials need to have the same domain and target dimension
        assert self.domain_dimension() == other.domain_dimension()
        assert self.target_dimension() == other.target_dimension()
        # For now require that both polynomials are expressed in the same basis.
        # If not we would need to transform them to some common basis, and what basis
        # this is would need to be specified by the user.
        assert self.basis() == other.basis()
        if self.degree() == other.degree():
            return self.__class__(self.coeff - other.coeff, self.vertices, self.r)
        if self.degree() > other.degree():
            return self - other.degree_elevate(self.degree())
        else:
            return self.degree_elevate(other.degree()) - other

    def multiply_with_constant(self, c):
        """
        Multiplication of this polynomial with a constant scalar or a vector (only for a scalar valued polynomial),
        self * c.

        :param c: Scalar or vector we should multiply this polynomial with.
        :type c: Union[float, :class:`Numpy array <numpy.ndarray>`]
        :return: Product of this polynomial with the constant.
        :rtype: Instance of self.__class__
        """
        if isinstance(c, numbers.Number):
            # Multiplication of the polynomial with a scalar
            return self.__class__(self.coeff * c, self.vertices, self.r)
        if isinstance(c, np.ndarray):
            # Multiplication of the polynomial with a vector
            # Can only multiply a scalar valued polynomials with a vector, to produce a vector valued polynomial
            assert self.n == 1
            return self.__class__(np.outer(self.coeff, c), self.vertices, self.r)
        assert False  # Unknown type for the constant c

    def __truediv__(self, s):
        """
        Division of this polynomial with a scalar, self / s.

        :param float s: Scalar to divide with.
        :return: Division of this polynomial with s.
        :rtype: Instance of self.__class__
        """
        # Only division by a scalar is implemented
        assert isinstance(s, numbers.Number)
        return self.__class__(self.coeff / s, self.vertices, self.r)

    def simplex_vertices(self):
        """
        Get the vertices of the simplex T on which this polynomial is defined.

        :return: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
            simplex).
        """
        return self.vertices
