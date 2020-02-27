r"""Polynomials on the m-dimensional unit simplex with values in :math:`\mathbb{R}^n`, expressed using the
Lagrange basis.

.. math:: l(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} l_{\nu, r}(x),

where :math:`a_{\nu} \in \mathbb{R}^n`.

Basis polynomials in the Lagrange basis are uniquely determined by selecting a sequence of
:math:`\dim \mathcal{P}_r (\Delta_c^m)` unique points (Lagrange points) in the unit simplex and demanding that the
i:th basis function has the value one at the i:th of these points and zero at all the other points.

Here we have used evenly spaced Lagrange points, so that for :math:`\dim \mathcal{P}_r (\Delta_c^m)` we have the
Lagrange points

.. math:: \{ x_{\nu} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}, x_{\nu} = \frac{\nu}{r}.

The basis polynomials :math:`l_{\nu, r}(x), \nu \in \mathbb{N}_0^n, |\nu| \leq r` are thus uniquely determined by the
conditions

.. math:: l_{\nu, r}(x_{\mu}) = \begin{cases} 1 & \nu = \mu \\ 0 & \text{else} \end{cases}.

The set :math:`\{ l_{\nu, r} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}` is a basis for the space
of all polynomials of degree less than or equal to r on the unit simplex, :math:`\mathcal{P}_r (\Delta_c^m)`.
"""

import numbers

import numpy as np

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.generic_tools.str_utils import (
    convert_float_to_fraction, str_dot_product, str_number, str_number_array)
from polynomials_on_simplices.polynomial.code_generation.generate_lagrange_polynomial_functions_simplex import (
    generate_function_specific)
from polynomials_on_simplices.polynomial.polynomials_base import PolynomialBase, get_dimension
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import Polynomial
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import zero_polynomial as zero_polynomial_monomial
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis_cache import (
    lagrange_basis_coefficients_cache)


def unique_identifier_lagrange_basis():
    """
    Get unique identifier for the Lagrange polynomial basis on the unit simplex.

    :return: Unique identifier.
    :rtype: str
    """
    return "Lagrange"


def generate_lagrange_point(n, r, nu):
    r"""
    Generate a Lagrange point indexed by a multi-index from the set of evenly spaced Lagrange points on the
    n-dimensional unit simplex (:math:`\Delta_c^n`)
    (Lagrange basis points are constructed so that each basis function has the value 1 at one of the points,
    and 0 at all the other points).

    .. math:: x_{\nu} = \frac{\nu}{r}.

    :param int n: Dimension of the simplex.
    :param int r: Degree of the polynomial.
    :param nu: Multi-index :math:`\nu` indexing the Lagrange point, where :math:`\frac{\nu_i}{r}` gives
        the i:th coordinate of the Lagrange point.
    :return: Point in the n-dimensional unit simplex.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> generate_lagrange_point(1, 2, (1,))
    array([0.5])
    >>> generate_lagrange_point(2, 2, (1, 0))
    array([0.5, 0. ])
    >>> generate_lagrange_point(2, 2, (0, 1))
    array([0. , 0.5])
    """
    # Handle special case
    if r == 0:
        return np.zeros(n)
    point = np.empty(n)
    for i in range(len(nu)):
        point[i] = nu[i]
    point /= r
    return point


def generate_lagrange_points(n, r):
    r"""
    Generate evenly spaced Lagrange points on the n-dimensional unit simplex (:math:`\Delta_c^n`)
    (Lagrange basis points are constructed so that each basis function has the value 1 at one of the points,
    and 0 at all the other points).

    .. math:: \{ x_{\nu} \}_{\substack{\nu \in \mathbb{N}_0^n \\ |\nu| \leq r}}, x_{\nu} = \frac{\nu}{r}.

    :param int n: Dimension of the simplex.
    :param int r: Degree of the polynomial.
    :return: List of points in the n-dimensional unit simplex.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> generate_lagrange_points(1, 2)
    array([0. , 0.5, 1. ])
    >>> generate_lagrange_points(2, 2)
    array([[0. , 0. ],
           [0.5, 0. ],
           [1. , 0. ],
           [0. , 0.5],
           [0.5, 0.5],
           [0. , 1. ]])
    """
    if n == 1:
        return np.linspace(0.0, 1.0, r + 1)
    points = np.zeros((get_dimension(r, n), n))
    # Handle special case
    if r == 0:
        return points
    i = 0
    for mi in multiindex.MultiIndexIterator(n, r):
        points[i] = generate_lagrange_point(n, r, mi)
        i += 1
    return points


def get_lagrange_basis_fn_coefficients(nu, r):
    r"""
    Get monomial coefficients for a Lagrange basis polynomial in the basis for the space
    :math:`\mathcal{P}_r(\Delta_c^n)`, with evenly spaced Lagrange points (see :func:`generate_lagrange_points`).

    :param nu: Multi-index indicating which Lagrange basis polynomial we should get monomial coefficients for.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: Array containing the coefficients for the basis polynomial.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    try:
        n = len(nu)
    except TypeError:
        n = 1

    # First look for the coefficients in the cache (no need to generate cached coefficients)
    if n in range(1, len(lagrange_basis_coefficients_cache) + 1):
        if r in range(1, len(lagrange_basis_coefficients_cache[n - 1]) + 1):
            if isinstance(nu, multiindex.MultiIndex):
                return lagrange_basis_coefficients_cache[n - 1][r - 1][nu.to_tuple()]
            else:
                if not isinstance(nu, tuple):
                    return lagrange_basis_coefficients_cache[n - 1][r - 1][(nu,)]
                else:
                    return lagrange_basis_coefficients_cache[n - 1][r - 1][nu]

    # Coefficients need to be generated
    return generate_lagrange_basis_fn_coefficients(nu, r)


def generate_lagrange_basis_fn_coefficients(nu, r):
    r"""
    Generate monomial coefficients for a Lagrange basis polynomial in the basis for the space
    :math:`\mathcal{P}_r(\Delta_c^n)`, with evenly spaced Lagrange points (see :func:`generate_lagrange_points`).

    :param nu: Multi-index indicating which Lagrange basis polynomial we should generate monomial coefficients for.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: Array containing the coefficients for the basis polynomial.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    try:
        n = len(nu)
    except TypeError:
        n = 1

    if not isinstance(nu, multiindex.MultiIndex):
        nu = multiindex.MultiIndex(nu)
    i = multiindex.get_index(nu, r)
    return generate_lagrange_base_coefficients(r, n)[:, i]


def get_lagrange_base_coefficients(r, n):
    r"""
    Get monomial coefficients for all the Lagrange base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`,
    with evenly spaced Lagrange points (see :func:`generate_lagrange_points`).

    :param int n: Dimension of the unit simplex.
    :param int r: Degree of the polynomial space.
    :return: Matrix containing the coefficients for each base polynomial as column vectors.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    # First look for the coefficients in the cache (no need to generate cached coefficients)
    if n in range(1, len(lagrange_basis_coefficients_cache) + 1):
        if r in range(1, len(lagrange_basis_coefficients_cache[n - 1]) + 1):
            coeffs = np.array([c for mi, c in lagrange_basis_coefficients_cache[n - 1][r - 1].items()]).T
            return coeffs

    # Coefficients need to be generated
    return generate_lagrange_base_coefficients(r, n)


def generate_lagrange_base_coefficients(r, n):
    r"""
    Generate monomial coefficients for all the Lagrange base polynomials
    for the space :math:`\mathcal{P}_r(\Delta_c^n)`, with evenly spaced Lagrange points (see
    :func:`generate_lagrange_points`).

    :param int n: Dimension of the unit simplex.
    :param int r: Degree of the polynomial space.
    :return: Matrix containing the coefficients for each base polynomial as column vectors.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    points = generate_lagrange_points(n, r)
    poly_dim = len(points)
    a = np.empty((poly_dim, poly_dim))
    if n == 1:
        for i in range(poly_dim):
            for j in range(poly_dim):
                a[i][j] = points[i]**j
    else:
        mis = multiindex.generate_all(n, r)
        for i in range(poly_dim):
            for j in range(poly_dim):
                a[i][j] = multiindex.power(points[i], mis[j])
    b = np.identity(poly_dim)
    return np.linalg.solve(a, b)


class PolynomialLagrange(PolynomialBase):
    r"""
    Implementation of the abstract polynomial base class for a polynomial on the m-dimensional unit simplex,
    expressed in the Lagrange basis.

    .. math:: l(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} l_{\nu_i, r}(x).
    """

    def __init__(self, coeff, r=None, m=1):
        r"""
        :param coeff: Coefficients for the polynomial in the Lagrange basis for :math:`\mathcal{P}_r (\mathbb{R}^m,
            \mathbb{R}^n). \text{coeff}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence
            of all multi-indices of dimension m with norm :math:`\leq r`
            (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
            Array of scalars for a scalar valued polynomial (n = 1) and array of n-dimensional vectors for a vector
            valued polynomial (:math:`n \geq 2`).
        :param int m: Dimension of the domain of the polynomial.
        :param int r: Degree of the polynomial space. Optional, will be inferred from the number of polynomial
            coefficients if not specified.
        """
        PolynomialBase.__init__(self, coeff, r, m)
        r = self.degree()
        if not (m, r) in PolynomialLagrange._basis_polynomials_monomial_form:
            PolynomialLagrange._basis_polynomials_monomial_form[(m, r)] = lagrange_basis_monomial(r, m)

        # Compile function for evaluating the polynomial
        self._eval_code, self._eval_fn_name = generate_function_specific(m, self.r, self.coeff)
        compiled_code = compile(self._eval_code, '<auto generated monomial polynomial function, '
                                + str(self.coeff) + '>', 'exec')
        exec(compiled_code, globals(), locals())
        self._eval_fn = locals()[self._eval_fn_name]

    def __repr__(self):
        return "polynomials_on_simplices.algebra.polynomial.polynomials_unit_simplex_lagrange_basis.PolynomialLagrange("\
               + str(self.coeff) + ", " + str(self.domain_dimension()) + ", " + str(self.degree()) + ")"

    def basis(self):
        r"""
        Get basis for the space :math:`\mathcal{P}_r (\mathbb{R}^m)` used to express this polynomial.

        :return: Unique identifier for the basis used.
        :rtype: str
        """
        return unique_identifier_lagrange_basis()

    def __call__(self, x):
        r"""
        Evaluate the polynomial at a point :math:`x \in \mathbb{R}^m`.

        :param x: Point where the polynomial should be evaluated.
        :type x: float or length m :class:`Numpy array <numpy.ndarray>`
        :return: Value of the polynomial.
        :rtype: float or length n :class:`Numpy array <numpy.ndarray>`.
        """
        return self._eval_fn(x)

    def __mul__(self, other):
        """
        Multiplication of this polynomial with another polynomial, a scalar, or a vector (for a scalar valued
        polynomial), self * other.

        :param other: Polynomial, scalar or vector we should multiply this polynomial with.
        :type: PolynomialLagrange, scalar or vector
        :return: Product of this polynomial with other.
        :rtype: :class:`PolynomialLagrange`.
        """
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            return self.multiply_with_constant(other)
        # Multiplication of two polynomials
        # Multiplied polynomials need to have the same domain dimension
        assert self.domain_dimension() == other.domain_dimension()
        # Cannot multiply two vector valued polynomials
        assert self.target_dimension() == 1
        assert other.target_dimension() == 1
        m = self.domain_dimension()
        r = self.degree() + other.degree()
        dim = get_dimension(r, m)
        coeff = np.empty(dim)
        x = generate_lagrange_points(m, r)
        for i in range(len(x)):
            coeff[i] = self(x[i]) * other(x[i])
        return PolynomialLagrange(coeff, r, m)

    def __pow__(self, exp):
        r"""
        Raise the polynomial to a power.

        .. math::

            (l^{\mu})(x) = l(x)^{\mu} =  l_1(x)^{\mu_1} l_2(x)^{\mu_2} \ldots l_n(x)^{\mu_n}.

        :param exp: Power we want the raise the polynomial to (natural number or multi-index depending on the dimension
            of the target of the polynomial).
        :type exp: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
        :return: This polynomial raised to the given power.
        :rtype: :class:`PolynomialLagrange`.
        """
        if isinstance(exp, numbers.Integral):
            assert exp >= 0
            assert self.target_dimension() == 1
            if exp == 0:
                return unit_polynomial(0, self.m)
            if exp == 1:
                return PolynomialLagrange(self.coeff, self.r, self.m)
            return self * self**(exp - 1)
        else:
            assert len(exp) == self.target_dimension()
            assert [entry >= 0 for entry in exp]
            m = self.domain_dimension()
            r = self.degree() * multiindex.norm(exp)
            dim = get_dimension(r, m)
            coeff = np.empty(dim)
            # Get the coefficients by applying the dual basis (evaluate at
            # Lagrange points) to the exponentiated polynomial
            x = generate_lagrange_points(m, r)
            for i in range(len(x)):
                coeff[i] = multiindex.power(self(x[i]), exp)
            return PolynomialLagrange(coeff, r, m)

    def partial_derivative(self, i=0):
        """
        Compute the i:th partial derivative of the polynomial.

        :param int i: Index of partial derivative.
        :return: i:th partial derivative of this polynomial.
        :rtype: :class:`PolynomialLagrange`.
        """
        assert isinstance(i, numbers.Integral)
        assert i >= 0
        m = self.domain_dimension()
        n = self.target_dimension()
        assert i < m
        r = self.degree()
        if r == 0:
            return zero_polynomial(0, m, n)

        dim = get_dimension(r - 1, m)
        if n == 1:
            coeff = np.zeros(dim)
        else:
            coeff = np.zeros((dim, n))
        # Express polynomial in monomial basis
        p = self.to_monomial_basis()
        # Compute derivative for the polynomial in the monomial basis
        dp = p.partial_derivative(i)
        # Convert the derivative to the Lagrange basis
        x = generate_lagrange_points(m, r - 1)
        for j in range(len(x)):
            coeff[j] = dp(x[j])
        return PolynomialLagrange(coeff, r - 1, m)

    def degree_elevate(self, s):
        r"""
        Express the polynomial using a higher degree basis.

        Let :math:`p(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} l_{\nu, r}(x)` be this
        polynomial, where :math:`\{ l_{\nu, r} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}` is the Lagrange
        basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`. Let :math:`\{ l_{\nu, s} \}_{\substack{\nu \in \mathbb{N}_0^m
        \\ |\nu| \leq s}}, s \geq r` be the Lagrange basis for :math:`\mathcal{P}_s (\mathbb{R}^m)`. Then this function
        returns a polynomial :math:`q(x)`

        .. math:: q(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq s}} \tilde{a}_{\nu} l_{\nu, s}(x),

        such that :math:`p(x) = q(x) \, \forall x \in \Delta_c^m`.

        :param int s: New degree for the polynomial basis the polynomial should be expressed in.
        :return: Elevation of this polynomial to the higher degree basis.
        :rtype: :class:`PolynomialLagrange`.
        """
        assert s >= self.degree()
        m = self.domain_dimension()
        n = self.target_dimension()
        r = self.degree()
        if s == self.degree():
            return PolynomialLagrange(self.coeff, r, m)
        dim = get_dimension(s, m)
        if n == 1:
            coeff = np.zeros(dim)
        else:
            coeff = np.zeros((dim, n))
        dual_basis = dual_lagrange_basis(s, m)
        for i in range(len(dual_basis)):
            coeff[i] = dual_basis[i](self)
        return PolynomialLagrange(coeff, s, m)

    def to_monomial_basis(self):
        """
        Compute the monomial representation of this polynomial.

        :return: This polynomial expressed in the monomial basis.
        :rtype: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`.
        """
        return sum([b * a for (a, b) in zip(self.coeff, self._basis_polynomials_monomial_form[self.m, self.r])],
                   zero_polynomial_monomial(0, self.m, self.n))

    _basis_polynomials_monomial_form = {}

    def latex_str(self):
        r"""
        Generate a Latex string for this polynomial.

        :return: Latex string for this polynomial.
        :rtype: str
        """
        try:
            len(self.coeff[0])
            coeff_strs = [str_number_array(c, latex=True) for c in self.coeff]
            basis_strs = lagrange_basis_latex_compact(self.r, self.m)
            return str_dot_product(coeff_strs, basis_strs)
        except TypeError:
            coeff_strs = [str_number(c, latex_fraction=True) for c in self.coeff]
            basis_strs = lagrange_basis_latex_compact(self.r, self.m)
            return str_dot_product(coeff_strs, basis_strs)

    def latex_str_expanded(self):
        r"""
        Generate a Latex string for this polynomial, where each basis function has been expanded in the monomial
        basis.

        :return: Latex string for this polynomial.
        :rtype: str
        """
        try:
            len(self.coeff[0])
            coeff_strs = [str_number_array(c, latex=True) for c in self.coeff]
            basis_strs = lagrange_basis_latex(self.r, self.m)
            for i in range(len(basis_strs)):
                if len(basis_strs[i]) > 3:
                    basis_strs[i] = "(" + basis_strs[i] + ")"
            return str_dot_product(coeff_strs, basis_strs)
        except TypeError:
            coeff_strs = [str_number(c, latex_fraction=True) for c in self.coeff]
            basis_strs = lagrange_basis_latex(self.r, self.m)
            for i in range(len(basis_strs)):
                if len(basis_strs[i]) > 3:
                    basis_strs[i] = "(" + basis_strs[i] + ")"
            return str_dot_product(coeff_strs, basis_strs)

    def code_str(self, fn_name):
        r"""
        Generate a function code string for evaluating this polynomial.

        :param str fn_name: Name for the function in the generated code.
        :return: Code string for evaluating this polynomial.
        :rtype: str
        """
        return self._eval_code.replace(self._eval_fn_name, fn_name)


def lagrange_basis_fn(nu, r):
    r"""
    Generate a Lagrange basis polynomial on the unit simplex (:math:`\Delta_c^n`),
    where n is equal to the length of nu.

    :param nu: Multi-index indicating which Lagrange basis polynomial should be generated.
        The polynomial will have the value 1 at the point associated with the multi-index,
        and value 0 at all other points.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: The Lagrange base polynomial as specified by nu and r.
    :rtype: :class:`PolynomialLagrange`.

    .. rubric:: Examples

    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> lagrange_basis_fn(1, 1)(x1) - x1
    0
    >>> sp.simplify(lagrange_basis_fn(2, 2)(x1) - (2*x1**2 - x1))
    0
    >>> sp.simplify(lagrange_basis_fn((1, 1), 2)((x1, x2)) - 4*x1*x2)
    0
    """
    try:
        m = len(nu)
    except TypeError:
        m = 1
        nu = (nu,)
    dim = get_dimension(r, m)
    coeff = np.zeros(dim, dtype=int)
    i = multiindex.get_index(nu, r)
    coeff[i] = 1
    return PolynomialLagrange(coeff, r, m)


def lagrange_basis(r, n):
    r"""
    Generate all Lagrange base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`.

    :param int n: Dimension of the space.
    :param int r: Degree of the polynomial space.
    :return: List of base polynomials.
    :rtype: List[:class:`PolynomialLagrange`].
    """
    basis = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis.append(lagrange_basis_fn(mi, r))
    return basis


def vector_valued_lagrange_basis_fn(nu, r, i, n):
    r"""
    Generate a vector valued Lagrange basis polynomial on the m-dimensional unit simplex,
    :math:`l_{\nu, r, i} : \Delta_c^m \to \mathbb{R}^n`.

    The vector valued basis polynomial is generated by specifying a scalar valued basis polynomial and the component
    of the vector valued basis polynomial that should be equal to the scalar valued basis polynomial. All other
    components of the vector valued basis polynomial will be zero, i.e.

    .. math:: l_{\nu, r, i}^j (x) = \begin{cases} l_{\nu, r} (x), & i = j \\ 0, & \text{else} \end{cases},

    where m is equal to the length of nu.

    :param nu: Multi-index indicating which scalar valued Lagrange basis polynomial should be generated for the
        non-zero component.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param int i: Index of the vector component that is non-zero.
    :param int n: Dimension of the target.
    :return: The Lagrange base polynomial as specified by nu, r, i and n.
    :rtype: :class:`PolynomialLagrange`.

    .. rubric:: Examples

    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> vector_valued_lagrange_basis_fn(0, 1, 0, 2)(x1)
    array([-x1 + 1, 0], dtype=object)
    >>> vector_valued_lagrange_basis_fn(1, 1, 1, 2)(x1)
    array([0, x1], dtype=object)
    >>> vector_valued_lagrange_basis_fn((1, 0), 2, 0, 2)((x1, x2))
    array([-4*x1**2 - 4*x1*x2 + 4*x1, 0], dtype=object)
    >>> vector_valued_lagrange_basis_fn((1, 1), 3, 1, 3)((x1, x2))
    array([0, -27*x1**2*x2 - 27*x1*x2**2 + 27*x1*x2, 0], dtype=object)
    """
    if n == 1:
        assert i == 0
        return lagrange_basis_fn(nu, r)
    assert i >= 0
    assert i < n
    try:
        m = len(nu)
    except TypeError:
        m = 1
        nu = (nu,)
    dim = get_dimension(r, m)
    coeff = np.zeros((dim, n), dtype=int)
    j = multiindex.get_index(nu, r)
    coeff[j][i] = 1
    return PolynomialLagrange(coeff, r, m)


def vector_valued_lagrange_basis(r, m, n, ordering="interleaved"):
    r"""
    Generate all Lagrange base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)`.

    :param int m: Dimension of the unit simplex.
    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the target.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: List of base polynomials.
    :rtype: List[:class:`PolynomialLagrange`].
    """
    basis = []
    if ordering == "interleaved":
        for mi in multiindex.MultiIndexIterator(m, r):
            for i in range(n):
                basis.append(vector_valued_lagrange_basis_fn(mi, r, i, n))
    else:
        for i in range(n):
            for mi in multiindex.MultiIndexIterator(m, r):
                basis.append(vector_valued_lagrange_basis_fn(mi, r, i, n))
    return basis


def lagrange_basis_fn_monomial(nu, r):
    r"""
    Generate a Lagrange basis polynomial on the unit simplex (:math:`\Delta_c^n`),
    where n is equal to the length of nu, expanded in the monomial basis.

    This is the same polynomial as given by the :func:`lagrange_basis_fn` function, but expressed in the monomial
    basis.

    :param nu: Multi-index indicating which Lagrange basis polynomial should be generated
        The polynomial will have the value 1 at the point associated with the multi-index,
        and value 0 at all other points.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: The Lagrange base polynomial as specified by nu and r.
    :rtype: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`.

    .. rubric:: Examples

    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> lagrange_basis_fn(1, 1)(x1) - x1
    0
    >>> sp.simplify(lagrange_basis_fn(2, 2)(x1) - (2*x1**2 - x1))
    0
    >>> sp.simplify(lagrange_basis_fn((1, 1), 2)((x1, x2)) - 4*x1*x2)
    0
    """
    try:
        m = len(nu)
    except TypeError:
        m = 1

    coeff = get_lagrange_basis_fn_coefficients(nu, r)
    return Polynomial(coeff, r, m)


def lagrange_basis_monomial(r, n):
    r"""
    Generate all Lagrange base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`, expanded in the monomial
    basis.

    This is the same set of polynomials as given by the :func:`lagrange_basis` function, but expressed in the
    monomial basis.

    :param int n: Dimension of the space.
    :param int r: Degree of the polynomial space.
    :return: List of base polynomials.
    :rtype: List[:class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`].
    """
    basis = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis.append(lagrange_basis_fn_monomial(mi, r))
    return basis


def dual_lagrange_basis_fn(mu, r):
    r"""
    Generate a dual basis function to the Lagrange polynomial basis, i.e. the linear map
    :math:`q_{\mu, r} : \mathcal{P}_r(\Delta_c^n) \to \mathbb{R}` that satisfies

    .. math::

        q_{\mu, r}(l_{\nu, r}) = \delta_{\mu, \nu},

    where :math:`l_{\nu, r}` is the degree r Lagrange basis polynomial indexed by the multi-index :math:`\nu`
    (see :func:`lagrange_basis_fn`) and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual Lagrange basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...].
    :param int r: Degree of polynomial space.
    :return: The dual Lagrange basis function as specified by mu and r.
    :rtype: Callable :math:`q_{\mu, r}(l)`.
    """
    try:
        m = len(mu)
    except TypeError:
        m = 1
        mu = multiindex.MultiIndex(mu)
    x_nu = generate_lagrange_point(m, r, mu)
    if m == 1:
        x_nu = x_nu[0]

    def q(p):
        return p(x_nu)
    return q


def dual_lagrange_basis(r, n):
    r"""
    Generate all dual Lagrange base functions for the space :math:`\mathcal{P}_r(\Delta_c^n)` (i.e. the Lagrange basis
    for :math:`\mathcal{P}_r(\Delta_c^n)^*`).

    See :func:`dual_lagrange_basis_fn`.

    :param int n: Dimension of the space.
    :param int r: Degree of the polynomial space.
    :return: List of dual base functions.
    :rtype: List[callable `q(l)`].
    """
    dual_basis = []
    for mi in multiindex.MultiIndexIterator(n, r):
        dual_basis.append(dual_lagrange_basis_fn(mi, r))
    return dual_basis


def dual_vector_valued_lagrange_basis_fn(mu, r, i, n):
    r"""
    Generate a dual basis function to the vector valued Lagrange polynomial basis, i.e. the linear map
    :math:`q_{\mu, r, i} : \mathcal{P}_r(\Delta_c^m, \mathbb{R}^n) \to \mathbb{R}` that satisfies

    .. math::

        q_{\mu, r, i}(l_{\nu, r, j}) = \delta_{\mu, \nu} \delta_{i, j},

    where :math:`l_{\nu, r, j}` is the degree r vector valued Lagrange basis polynomial indexed by the
    multi-index :math:`\nu` with a non-zero i:th component (see :func:`vector_valued_lagrange_basis_fn`) and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual Lagrange basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...].
    :param int r: Degree of polynomial space.
    :param int i: Integer indicating which dual Lagrange basis function should be generated.
    :param int n: Dimension of the target.
    :return: The dual Lagrange basis function as specified by mu, r and i.
    :rtype: Callable :math:`q_{\mu, r, i}(l)`.
    """
    if n == 1:
        assert i == 0
        return dual_lagrange_basis_fn(mu, r)
    assert i >= 0
    assert i < n

    qs = dual_lagrange_basis_fn(mu, r)

    def q(p):
        assert p.target_dimension() == n
        return qs(p[i])

    return q


def dual_vector_valued_lagrange_basis(r, m, n, ordering="interleaved"):
    r"""
    Generate all dual Lagrange base functions for the space :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)` (i.e. the
    Lagrange basis for :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)^*`).

    See :func:`dual_vector_valued_lagrange_basis_fn`.

    :param int m: Dimension of the space.
    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the target.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: List of dual base functions.
    :rtype: List[callable `q(l)`].
    """
    dual_basis = []
    if ordering == "interleaved":
        for mi in multiindex.MultiIndexIterator(m, r):
            for i in range(n):
                dual_basis.append(dual_vector_valued_lagrange_basis_fn(mi, r, i, n))
    else:
        for i in range(n):
            for mi in multiindex.MultiIndexIterator(m, r):
                dual_basis.append(dual_vector_valued_lagrange_basis_fn(mi, r, i, n))
    return dual_basis


def lagrange_basis_fn_latex(nu, r):
    r"""
    Generate Latex string for a Lagrange basis polynomial on the unit simplex (:math:`\Delta_c^n`),
    where n is equal to the length of nu.

    :param nu: Multi-index indicating which Lagrange basis polynomial we should generate Latex string for.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: Latex string for the Lagrange base polynomial as specified by nu and r.
    :rtype: str

    .. rubric:: Examples

    >>> lagrange_basis_fn_latex(2, 3)
    '-9 / 2 x + 18 x^2 - 27 / 2 x^3'
    >>> lagrange_basis_fn_latex((1, 1), 3)
    '27 x_1 x_2 - 27 x_1^2 x_2 - 27 x_1 x_2^2'
    """
    from polynomials_on_simplices.polynomial.polynomials_monomial_basis import monomial_basis_fn_latex
    l_monomial = lagrange_basis_fn_monomial(nu, r)
    latex_str = ""
    for i in range(len(l_monomial.coeff)):
        c = l_monomial.coeff[i]
        if c != 0:
            mi = multiindex.generate(l_monomial.domain_dimension(), r, i)
            if latex_str == "":
                if c == 1:
                    latex_str += monomial_basis_fn_latex(mi)
                else:
                    if c == -1:
                        latex_str += "-" + monomial_basis_fn_latex(mi)
                    else:
                        latex_str += convert_float_to_fraction(str(c)) + " " + monomial_basis_fn_latex(mi)
            else:
                if c == 1:
                    latex_str += " + " + monomial_basis_fn_latex(mi)
                else:
                    if c == -1:
                        latex_str += " - " + monomial_basis_fn_latex(mi)
                    else:
                        if c < 0:
                            latex_str += " - " + convert_float_to_fraction(str(-c)) + " " + monomial_basis_fn_latex(mi)
                        else:
                            latex_str += " + " + convert_float_to_fraction(str(c)) + " " + monomial_basis_fn_latex(mi)
    if latex_str == "":
        return "1"
    else:
        return latex_str


def lagrange_basis_fn_latex_compact(nu, r):
    r"""
    Generate compact Latex string for a Lagrange basis polynomial on the unit simplex (:math:`\Delta_c^n`),
    where n is equal to the length of nu.

    :param nu: Multi-index indicating which Lagrange basis polynomial we should generate Latex string for.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: Latex string for the Lagrange base polynomial as specified by nu and r.
    :rtype: str

    .. rubric:: Examples

    >>> lagrange_basis_fn_latex_compact(2, 3)
    'l_{2, 3}(x)'
    >>> lagrange_basis_fn_latex_compact((1, 1), 3)
    'l_{(1, 1), 3}(x)'
    """
    try:
        n = len(nu)
    except TypeError:
        n = 1
        nu = multiindex.MultiIndex(nu)

    if n == 1:
        return "l_{" + str(nu[0]) + ", " + str(r) + "}(x)"
    else:
        return "l_{" + str(nu) + ", " + str(r) + "}(x)"


def lagrange_basis_latex(r, n):
    r"""
    Generate Latex strings for all Lagrange base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`.

    :param int n: Dimension of the unit simplex.
    :param int r: Degree of the polynomial space.
    :return: List of Latex strings for each Lagrange base polynomial.
    :rtype: List[str]

    .. rubric:: Examples

    >>> lagrange_basis_latex(2, 1)
    ['1 - 3 x + 2 x^2', '4 x - 4 x^2', '-x + 2 x^2']
    >>> basis_strings = lagrange_basis_latex(2, 2)
    >>> expected_basis_strings = list()
    >>> expected_basis_strings.append('1 - 3 x_1 + 2 x_1^2 - 3 x_2 + 4 x_1 x_2 + 2 x_2^2')
    >>> expected_basis_strings.append('4 x_1 - 4 x_1^2 - 4 x_1 x_2')
    >>> expected_basis_strings.append('-x_1 + 2 x_1^2')
    >>> expected_basis_strings.append('4 x_2 - 4 x_1 x_2 - 4 x_2^2')
    >>> expected_basis_strings.append('4 x_1 x_2')
    >>> expected_basis_strings.append('-x_2 + 2 x_2^2')
    >>> basis_strings == expected_basis_strings
    True
    """
    basis_latex_strings = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis_latex_strings.append(lagrange_basis_fn_latex(mi, r))
    return basis_latex_strings


def lagrange_basis_latex_compact(r, n):
    r"""
    Generate compact Latex strings for all Lagrange base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`.

    :param int n: Dimension of the unit simplex.
    :param int r: Degree of the polynomial space.
    :return: List of Latex strings for each Lagrange base polynomial.
    :rtype: List[str]

    .. rubric:: Examples

    >>> lagrange_basis_latex_compact(2, 1)
    ['l_{0, 2}(x)', 'l_{1, 2}(x)', 'l_{2, 2}(x)']
    >>> lagrange_basis_latex_compact(1, 2)
    ['l_{(0, 0), 1}(x)', 'l_{(1, 0), 1}(x)', 'l_{(0, 1), 1}(x)']
    """
    basis_latex_strings = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis_latex_strings.append(lagrange_basis_fn_latex_compact(mi, r))
    return basis_latex_strings


def zero_polynomial(r=0, m=1, n=1):
    r"""
    Get the Lagrange polynomial :math:`l \in \mathcal{P}(\Delta_c^m, \mathbb{R}^n)` which is identically zero.

    :param int m: Dimension of the polynomial domain.
    :param int n: Dimension of the polynomial target.
    :param int r: The zero polynomial will be expressed in the Lagrange basis for
        :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)`.
    :return: The zero polynomial.
    :rtype: :class:`PolynomialLagrange`.
    """
    dim = get_dimension(r, m)
    if n == 1:
        coeff = np.zeros(dim)
    else:
        coeff = np.zeros((dim, n))
    return PolynomialLagrange(coeff, r, m)


def unit_polynomial(r=0, m=1, n=1):
    r"""
    Get the Lagrange polynomial :math:`l \in \mathcal{P}(\Delta_c^m, \mathbb{R}^n)` which is identically one.

    :param int m: Dimension of the polynomial domain.
    :param int n: Dimension of the polynomial target.
    :param int r: The unit polynomial will be expressed in the Lagrange basis for
        :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)`.
    :return: The unit polynomial.
    :rtype: :class:`PolynomialLagrange`.
    """
    dim = get_dimension(r, m)
    if n == 1:
        coeff = np.ones(dim)
    else:
        coeff = np.ones((dim, n))
    return PolynomialLagrange(coeff, r, m)


def get_associated_sub_simplex(nu, r, simplex=None):
    r"""
    Get the sub simplex associated with a Lagrange basis polynomial.

    For a Lagrange basis polynomial p on a simplex T there exist a unique sub simplex f of T such that

    - :math:`p|_f \neq 0`,

    - :math:`p|_g = 0` for all :math:`g \in \Delta_k(T), g \neq f`,

    - :math:`\dim f < \dim h`, where :math:`h` is any other sub simplex of T for which the above two conditions hold,

    where :math:`\Delta_k(T)` is the set of all k-dimensional sub simplices of T and :math:`k = \dim f`.

    :param nu: Multi-index indicating for which Lagrange basis polynomial we should get the associated sub simplex.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param simplex: Vertex indices for the vertices of the simplex. [0, 1, ..., n] assumed if not specified.
    :type simplex: Optional[List[int]]
    :return: Tuple containing the associated sub simplex, and the sub multi-index associated with the sub simplex
        (where all non-zero entries has been removed).

    .. rubric:: Examples

    >>> get_associated_sub_simplex((0,), 1)
    ([0], (1,))
    >>> get_associated_sub_simplex((1,), 1)
    ([1], (1,))
    >>> get_associated_sub_simplex((1, 0, 1), 2, [1, 2, 3, 4])
    ([2, 4], (1, 1))
    """
    try:
        n = len(nu)
    except TypeError:
        n = 1
    if simplex is None:
        simplex = tuple(range(n + 1))
    if r == 0:
        return list(simplex), multiindex.zero_multiindex(n)
    mi = multiindex.general_to_exact_norm(nu, r)
    sub_simplex = []
    sub_multi_index = []
    for i in range(len(mi)):
        if mi[i] != 0:
            sub_simplex.append(simplex[i])
            sub_multi_index.append(mi[i])
    return sub_simplex, tuple(sub_multi_index)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
