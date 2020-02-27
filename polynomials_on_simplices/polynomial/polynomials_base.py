"""Abstract base class and basic functionality for polynomials."""

import abc
import math
import numbers

import numpy as np
from scipy.special import binom


def get_dimension(r, n):
    """
    Get the dimension of the space of polynomials of degree <= r on an n-dimensional domain.

    :param int r: Maximum polynomial degree.
    :param int n: Dimension of the domain.
    :return: Dimension of the space of polynomials (number of basis functions needed to span the space).
    :rtype: int
    """
    return int(binom(r + n, n))


def get_degree_from_dimension(dim, n):
    """
    Get the maximum polynomial degree r for a polynomial in the space of all polynomials on an n-dimensional domain
    with given dimension. In a sense this is the inverse of the :func:`get_dimension` function.

    :param int dim: Dimension of the polynomial space.
    :param int n: Dimension of the domain.
    :return: Maximum polynomial degree.
    :rtype: int
    """
    # FIXME: is there a non brute force way of computing this number?
    for r in range(20):
        if get_dimension(r, n) == dim:
            return r
    assert False


def polynomials_equal(p1, p2, r, m, rel_tol=1e-9, abs_tol=1e-7):
    r"""
    Check if two polynomials p1 and p2 are approximately equal.

    For scalar valued polynomials, the two polynomials are considered equal if

    .. code-block:: python

        math.isclose(p1(xi), p2(xi), rel_tol=rel_tol, abs_tol=abs_tol)

    is true for a set of random points :math:`\{ x_i \}_{i = 0}^{d - 1}` from the m-dimensional unit cube,
    where :math:`d` is the dimension of the polynomial space p1 and p2 belongs to (as given by the
    :func:`get_dimension` function).

    For vector valued polynomials the same check is done component wise.

    :param p1: First polynomial.
    :type p1: Callable p1(x)
    :param p2: Second polynomial.
    :type p2: Callable p2(x)
    :param int r: Degree of the polynomials.
    :param int m: Dimension of the domain of the polynomials.
    :param float rel_tol: Tolerance for the relative error. See :func:`math.isclose <python:math.isclose>` for details.
    :param float abs_tol: Tolerance for the absolute error. See :func:`math.isclose <python:math.isclose>` for details.
    :return: Whether or not the two polynomials are approximately equal.
    :rtype: bool
    """
    # Note: This function takes callables as input instead of instances of the PolynomialBase abstract base class.
    # The reason for this is that the former is more general. It allows us to check for equality for callables
    # that supposedly are polynomials but doesn't implement the PolynomialBase interface.

    # Generate random domain points where we should check for approximate equality
    dim = get_dimension(r, m)
    if m == 1:
        x_values = np.random.random_sample(dim)
    else:
        x_values = np.random.random_sample((dim, m))

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


class PolynomialComponentsIterator:
    """
    Iterator for iterating over the components of a vector valued polynomial.
    """

    def __init__(self, p):
        """
        :param p: Vector valued polynomial.
        :type p: Instance of PolynomialBase
        """
        assert p.target_dimension() > 1
        self._p = p
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._p):
            raise StopIteration
        pc = self._p[self._i]
        self._i += 1
        return pc


class PolynomialBase(abc.ABC):
    r"""
    Abstract base class for a polynomial. The space of polynomials :math:`\mathcal{P}
    = \mathcal{P}(\mathbb{R}^m, \mathbb{R}^n)` is defined as

    .. math::

        \mathcal{P} = \{ p : \mathbb{R}^m \to \mathbb{R}^n | p(x) = \sum_{\nu} a_{\nu} x^{\nu},
        \nu \in \mathbb{N}_0^m, a_{\nu} \in \mathbb{R}^n \}.

    The domain dimension m and the target dimension n of the polynomial is given by the :meth:`domain_dimension`
    and :meth:`target_dimension` functions respectively.

    For a computable polynomial we must have :math:`a_{\nu} = 0` for all but finitely many :math:`\nu`, and then the
    degree of :math:`p` is defined as :math:`r = \deg{p} = \max_{\nu : a_{\nu} \neq 0} |\nu|`. The degree is given
    by the :meth:`degree` method.

    There are many common bases for the space of polynomials. The basis used for a specific polynomial is given by
    the :meth:`basis` method.

    This class also defines the basic algebraic and differentiable structures of the space of polynomials.

    **Ring structure:**

    Addition: :math:`+ : \mathcal{P} \times \mathcal{P} \to \mathcal{P}, (p_1 + p_2)(x) = p_1(x) + p_2(x)`.

    Multiplication: :math:`\cdot : \mathcal{P} \times \mathcal{P} \to \mathcal{P},
    (p_1 \cdot p_2)(x) = p_1(x) \cdot p_2(x)`.

    **Vector space structure:**

    Scalar multiplication: :math:`\cdot : \mathbb{R} \times \mathcal{P} \to \mathcal{P}, (s \cdot p)(x) = s \cdot p(x)`.

    **Differentiable structure:**

    `i`:th partial derivative: :math:`\partial_i : \mathcal{P} \to \mathcal{P},
    (\partial_i p)(x) = \frac{\partial p(x)}{\partial x^i}`.
    """

    def __init__(self, coeff, r=None, m=1):
        r"""
        :param coeff: Coefficients for the polynomial in the chosen basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`
            (see :meth:`basis`). If p is expressed in the chosen basis :math:`\{ b_{\nu, r} \}` as
            :math:`p(x) = \sum_{\nu} a_{\nu} b_{\nu, r}(x)` then :math:`\text{coeff}[i] = a_{\nu(i)}`, where
            :math:`\nu(i)` is the i:th multi-index in the sequence of all multi-indices of dimension m with norm
            :math:`\leq r` (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
            Array of scalars for a scalar valued polynomial (n = 1) and array of n-dimensional vectors for a vector
            valued polynomial (:math:`n \geq 2`).
        :param int r: Degree of the polynomial space. Optional, will be inferred from the number of polynomial
            coefficients if not specified.
        :param int m: Dimension of the domain of the polynomial.
        """
        assert len(coeff) > 0
        assert isinstance(m, int)
        assert m >= 0
        self.coeff = _to_numpy_array(coeff)
        self.m = m
        try:
            self.n = len(self.coeff[0])
        except TypeError:
            self.n = 1
        if r is not None:
            # Check consistency
            if m > 0:
                assert r == get_degree_from_dimension(len(self.coeff), m)
            else:
                assert len(self.coeff) == 1
            self.r = r
        else:
            self.r = get_degree_from_dimension(len(self.coeff), m)

    def __str__(self):
        return self.latex_str()

    def domain_dimension(self):
        """
        Get dimension of the polynomial domain.

        :return: Dimension of the domain of the polynomial.
        :rtype: int
        """
        return self.m

    def target_dimension(self):
        """
        Get dimension of the polynomial target.

        :return: Dimension of the target of the polynomial.
        :rtype: int
        """
        return self.n

    def degree(self):
        """
        Get degree of the polynomial.

        :return: Polynomial degree.
        :rtype: int
        """
        return self.r

    @abc.abstractmethod
    def basis(self):
        r"""
        Get basis for the space :math:`\mathcal{P}_r (\mathbb{R}^m)` used to express this polynomial.

        :return: Unique identifier for the basis used.
        :rtype: str
        """
        pass

    @abc.abstractmethod
    def __call__(self, x):
        r"""
        Evaluate the polynomial at a point :math:`x \in \mathbb{R}^m`.

        :param x: Point where the polynomial should be evaluated.
        :type x: float or length m :class:`Numpy array <numpy.ndarray>`
        :return: Value of the polynomial.
        :rtype: float or length n :class:`Numpy array <numpy.ndarray>`.
        """
        pass

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
            return self.__class__(self.coeff, self.r, self.m)
        else:
            return self.__class__(self.coeff[:, i], self.r, self.m)

    def __len__(self):
        """
        Get the number of components of the polynomial. Only applicable for a vector valued polynomial.

        :return: The number of components of the vector valued polynomial.
        :rtype: int
        """
        if self.target_dimension() == 1:
            raise TypeError("Scalar valued polynomials doesn't have a length")
        return self.target_dimension()

    def __iter__(self):
        """
        Iterate over the components of a vector valued polynomial.
        """
        return PolynomialComponentsIterator(self)

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
            return self.__class__(self.coeff + other.coeff, self.r, self.m)
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
            return self.__class__(self.coeff - other.coeff, self.r, self.m)
        if self.degree() > other.degree():
            return self - other.degree_elevate(self.degree())
        else:
            return self.degree_elevate(other.degree()) - other

    @abc.abstractmethod
    def __mul__(self, other):
        """
        Multiplication of this polynomial with another polynomial, a scalar, or a vector (for a scalar valued
        polynomial), self * other.

        :param other: Polynomial, scalar or vector we should multiply this polynomial with.
        :return: Product of this polynomial with other.
        """
        pass

    def __rmul__(self, other):
        """
        Multiplication of this polynomial with another polynomial or a scalar, other * self.

        :param other: Other polynomial or scalar.
        :return: Product of this polynomial with other.
        """
        return self * other

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
            return self.__class__(self.coeff * c, self.r, self.m)
        if isinstance(c, np.ndarray):
            # Multiplication of the polynomial with a vector
            # Can only multiply a scalar valued polynomials with a vector, to produce a vector valued polynomial
            assert self.n == 1
            return self.__class__(np.outer(self.coeff, c), self.r, self.m)
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
        return self.__class__(self.coeff / s, self.r, self.m)

    @abc.abstractmethod
    def __pow__(self, exp):
        r"""
        Raise the polynomial to a power.

        .. math::

            (p^{\mu})(x) = p(x)^{\mu} =  p_1(x)^{\mu_1} p_2(x)^{\mu_2} \ldots p_n(x)^{\mu_n}.

        :param exp: Power we want the raise the polynomial to (natural number or multi-index depending on the dimension
            of the target of the polynomial).
        :type exp: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
        :return: This polynomial raised to the given power.
        """
        pass

    @abc.abstractmethod
    def partial_derivative(self, i=0):
        """
        Compute the i:th partial derivative of the polynomial.

        :param int i: Index of partial derivative.
        :return: i:th partial derivative of this polynomial.
        """
        pass

    @abc.abstractmethod
    def degree_elevate(self, s):
        r"""
        Express the polynomial using a higher degree basis.

        Let :math:`p(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} b_{\nu, r}(x)` be this
        polynomial, where :math:`\{ b_{\nu, r} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}` is the chosen
        basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`. Let :math:`\{ b_{\nu, s} \}_{\substack{\nu \in \mathbb{N}_0^m
        \\ |\nu| \leq s}}, s \geq r` be the corresponding basis for :math:`\mathcal{P}_s (\mathbb{R}^m)`. Then this
        function returns a polynomial :math:`q(x)`

        .. math:: q(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq s}} \tilde{a}_{\nu} b_{\nu, s}(x),

        such that :math:`p(x) = q(x) \, \forall x \in \mathbb{R}^m`.

        :param int s: New degree for the polynomial basis the polynomial should be expressed in.
        :return: Elevation of this polynomial to the higher degree basis.
        """
        pass

    @abc.abstractmethod
    def to_monomial_basis(self):
        """
        Compute the monomial representation of this polynomial.

        :return: This polynomial expressed in the monomial basis.
        """
        pass

    @abc.abstractmethod
    def latex_str(self):
        r"""
        Generate a Latex string for this polynomial.

        :return: Latex string for this polynomial.
        :rtype: str
        """
        pass

    @abc.abstractmethod
    def code_str(self, fn_name):
        r"""
        Generate a function code string for evaluating this polynomial.

        :param str fn_name: Name for the function in the generated code.
        :return: Code string for evaluating this polynomial.
        :rtype: str
        """
        pass


def _to_numpy_array(arr):
    """
    Help function for converting an iterable to a Numpy array.

    :param arr: Array we want to convert.
    :type arr: Iterable[float]
    :return: Input array converted to a Numpy array.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    >>> _to_numpy_array([1.0, 2.0, 3.0])
    array([1., 2., 3.])
    """
    if isinstance(arr, np.ndarray):
        if len(arr.shape) == 2 and arr.shape[1] == 1:
            return arr.flatten()
        return np.copy(arr)
    return np.array(arr)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
