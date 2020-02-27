r"""Polynomials on :math:`\mathbb{R}^m` with values in :math:`\mathbb{R}^n`, expressed using the monomial basis.

An monomial in n variables of degree r is a product

.. math:: m = x_1^{d_1} x_2^{d_2} \ldots x_n^{d_n},

where :math:`d_i \mathbb{N}_0, i = 1, 2, \ldots, n` and :math:`\sum_{i = 1}^n d_i = r`.
I.e. :math:`d = (d_1, d_2, \ldots, d_n) \in \mathbb{N}_0^n` and in multi-index notation we simply have

.. math:: m = x^d, x = (x_1, x_2, \ldots, x_n).

The set :math:`\{ x^{\nu} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}` is a basis for the space
of all polynomials of degree less than or equal to r on :math:`\mathbb{R}^m, \mathcal{P}_r (\mathbb{R}^m)`.

.. math:: p(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} x^{\nu},

where :math:`a_{\nu} \in \mathbb{R}^n`.
"""

import copy
import numbers

import numpy as np

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.generic_tools.str_utils import str_dot_product, str_number, str_number_array
from polynomials_on_simplices.polynomial.code_generation.generate_monomial_polynomial_functions import (
    generate_function_specific)
from polynomials_on_simplices.polynomial.polynomials_base import PolynomialBase, get_dimension


def unique_identifier_monomial_basis():
    """
    Get unique identifier for the monomial polynomial basis.

    :return: Unique identifier.
    :rtype: str
    """
    return "monomial"


class Polynomial(PolynomialBase):
    r"""
    Implementation of the abstract polynomial base class for a polynomial using the monomial basis.

    .. math:: p(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} x^{\nu_i}.
    """

    def __init__(self, coeff, r=None, m=1):
        r"""
        :param coeff: Coefficients for the polynomial in the monomial basis for :math:`\mathcal{P}_r (\mathbb{R}^m,
            \mathbb{R}^n). \text{coeff}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence
            of all multi-indices of dimension m with norm :math:`\leq r`
            (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
            Array of scalars for a scalar valued polynomial (n = 1) and array of n-dimensional vectors for a vector
            valued polynomial (:math:`n \geq 2`).
        :param int r: Degree of the polynomial space. Optional, will be inferred from the number of polynomial
            coefficients if not specified.
        :param int m: Dimension of the domain of the polynomial.
        """
        PolynomialBase.__init__(self, coeff, r, m)
        # Compile function for evaluating the polynomial
        self._eval_code, self._eval_fn_name = generate_function_specific(m, self.r, self.coeff)
        compiled_code = compile(self._eval_code, '<auto generated monomial polynomial function, '
                                + str(self.coeff) + '>', 'exec')
        exec(compiled_code, globals(), locals())
        self._eval_fn = locals()[self._eval_fn_name]

    def __repr__(self):
        return "polynomials_on_simplices.algebra.polynomial.polynomials_monomial_basis.Polynomial(" + str(self.coeff) + ", "\
               + str(self.domain_dimension()) + ", " + str(self.degree()) + ")"

    def basis(self):
        r"""
        Get basis for the space :math:`\mathcal{P}_r (\mathbb{R}^m)` used to express this polynomial.

        :return: Unique identifier for the basis used.
        :rtype: str
        """
        return unique_identifier_monomial_basis()

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
        :type: Polynomial, scalar or vector
        :return: Product of this polynomial with other.
        :rtype: :class:`Polynomial`.
        """
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            return self.multiply_with_constant(other)
        # Multiplication of two polynomials
        # Multiplied polynomials need to have the same domain dimension
        assert self.domain_dimension() == other.domain_dimension()
        # Cannot multiply two vector valued polynomials
        assert self.target_dimension() == 1
        assert other.target_dimension() == 1
        # Formula below is for multiplying two monomial polynomials
        # The general case can be handled by converting the other polynomial to the monomial basis
        assert isinstance(other, Polynomial)
        m = self.domain_dimension()
        if m == 1:
            # Univariate case
            # We get a degree k monomial if we multiply the i:th term in the first polynomial
            # by the (k-i):th term in the second polynomial. So we get the coefficients of the
            # product polynomial by summing all of these terms, for all values of k, which is
            # exactly what the Numpy convolve function does.
            return Polynomial(np.convolve(self.coeff, other.coeff), self.r + other.r, self.m)
        else:
            # Multivariate case
            r1 = self.degree()
            r2 = other.degree()
            r = r1 + r2
            coeff = np.zeros(get_dimension(r, m))
            mus = multiindex.generate_all(m, r1)
            nus = multiindex.generate_all(m, r2)
            for i in range(len(self.coeff)):
                if self.coeff[i] == 0:
                    continue
                mu = mus[i]
                for j in range(len(other.coeff)):
                    if other.coeff[j] == 0:
                        continue
                    nu = nus[j]
                    k = multiindex.get_index(mu + nu, r)
                    coeff[k] += self.coeff[i] * other.coeff[j]
            return Polynomial(coeff, r, m)

    def __pow__(self, exp):
        r"""
        Raise the polynomial to a power.

        .. math::

            (p^{\mu})(x) = p(x)^{\mu} =  p_1(x)^{\mu_1} p_2(x)^{\mu_2} \ldots p_n(x)^{\mu_n}.

        :param exp: Power we want the raise the polynomial to (natural number or multi-index depending on the dimension
            of the target of the polynomial).
        :type exp: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
        :return: This polynomial raised to the given power.
        :rtype: :class:`Polynomial`.
        """
        if isinstance(exp, numbers.Integral):
            assert exp >= 0
            assert self.target_dimension() == 1
            if exp == 0:
                return unit_polynomial(0, self.m)
            if exp == 1:
                return Polynomial(self.coeff, self.r, self.m)
            return self * self**(exp - 1)
        else:
            assert len(exp) == self.target_dimension()
            assert [entry >= 0 for entry in exp]
            p = monomial_basis_fn(multiindex.zero_multiindex(self.m))
            for i in range(len(exp)):
                p *= self[i]**exp[i]
            return p

    def partial_derivative(self, i=0):
        """
        Compute the i:th partial derivative of the polynomial.

        :param int i: Index of partial derivative.
        :return: i:th partial derivative of this polynomial.
        :rtype: :class:`Polynomial`.
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
        j = 0
        for tau in multiindex.MultiIndexIterator(m, r - 1):
            tp1 = tau + multiindex.unit_multiindex(m, i)
            coeff[j] = (tau[i] + 1) * self.coeff[multiindex.get_index(tp1, r)]
            j += 1
        return Polynomial(coeff, r - 1, m)

    def degree_elevate(self, s):
        r"""
        Express the polynomial using a higher degree basis.

        Let :math:`p(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} x^{\nu}` be this
        polynomial expressed in the monomial basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`.
        Then this function returns a polynomial :math:`q(x)`

        .. math:: q(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq s}} \tilde{a}_{\nu} x^{\nu},

        such that :math:`p(x) = q(x) \, \forall x \in \mathbb{R}^m`.

        :param int s: New degree for the polynomial basis the polynomial should be expressed in.
        :return: Elevation of this polynomial to the higher degree basis.
        :rtype: :class:`Polynomial`.
        """
        assert s >= self.degree()
        m = self.domain_dimension()
        n = self.target_dimension()
        r = self.degree()
        if s == self.degree():
            return Polynomial(self.coeff, r, m)
        dim = get_dimension(s, m)
        if n == 1:
            coeff = np.zeros(dim)
        else:
            coeff = np.zeros((dim, n))
        curr_idx = 0
        curr_mi = multiindex.generate(m, r, curr_idx)
        i = 0
        for mi in multiindex.generate_all(m, s):
            if mi == curr_mi:
                coeff[i] = self.coeff[curr_idx]
                curr_idx += 1
                if curr_idx < len(self.coeff):
                    curr_mi = multiindex.generate(m, r, curr_idx)
            i += 1
        return Polynomial(coeff, s, m)

    def to_monomial_basis(self):
        """
        Compute the monomial representation of this polynomial.

        :return: This polynomial expressed in the monomial basis.
        :rtype: :class:`Polynomial`.
        """
        return copy.deepcopy(self)

    def latex_str(self):
        r"""
        Generate a Latex string for this polynomial.

        :return: Latex string for this polynomial.
        :rtype: str
        """
        try:
            len(self.coeff[0])
            # Vector valued polynomial
            coeff_strs = [str_number_array(c, latex=True) for c in self.coeff]
            basis_strs = monomial_basis_latex(self.r, self.m)
            return str_dot_product(coeff_strs, basis_strs)
        except TypeError:
            # Scalar valued polynomial
            coeff_strs = [str_number(c, latex_fraction=True) for c in self.coeff]
            basis_strs = monomial_basis_latex(self.r, self.m)
            return str_dot_product(coeff_strs, basis_strs)

    def code_str(self, fn_name):
        r"""
        Generate a function code string for evaluating this polynomial.

        :param str fn_name: Name for the function in the generated code.
        :return: Code string for evaluating this polynomial.
        :rtype: str
        """
        return self._eval_code.replace(self._eval_fn_name, fn_name)


def monomial_basis_fn(nu):
    r"""
    Generate a monomial basis polynomial :math:`p_{\nu}` in the space :math:`\mathcal{P}_r(\mathbb{R}^n)`,
    where :math:`r = |\nu|` and n is equal to the length of nu.

    .. math:: p_{\nu}(x) = x^{\nu}.

    :param nu: Multi-index indicating which monomial basis polynomial should be generated. Gives the exponent for
        each x_i term.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :return: The monomial base polynomial as specified by nu.
    :rtype: :class:`Polynomial`.

    .. rubric:: Examples

    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> monomial_basis_fn(1)(x1)
    x1
    >>> monomial_basis_fn(2)(x1)
    x1**2
    >>> monomial_basis_fn((1, 1))((x1, x2))
    x1*x2
    """
    try:
        m = len(nu)
    except TypeError:
        m = 1
    if not isinstance(nu, multiindex.MultiIndex):
        nu = multiindex.MultiIndex(nu)
    r = multiindex.norm(nu)
    dim = get_dimension(r, m)
    coeff = np.zeros(dim, dtype=int)
    i = multiindex.get_index(nu, r)
    coeff[i] = 1
    return Polynomial(coeff, r, m)


def monomial_basis(r, n):
    r"""
    Generate all monomial base polynomials for the space :math:`\mathcal{P}_r(\mathbb{R}^n)`.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the domain.
    :return: List of base polynomials.
    :rtype: List[:class:`Polynomial`].
    """
    basis = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis.append(monomial_basis_fn(mi))
    return basis


def vector_valued_monomial_basis_fn(nu, i, n):
    r"""
    Generate a vector valued monomial basis polynomial :math:`p_{\nu, i}` in the space
    :math:`\mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n)`, where :math:`r = |\nu|` and m is equal to the length of nu.

    The vector valued basis polynomial is generated by specifying a scalar valued basis polynomial and the component
    of the vector valued basis polynomial that should be equal to the scalar valued basis polynomial. All other
    components of the vector valued basis polynomial will be zero, i.e.

    .. math:: p_{\nu, i}^j (x) = \begin{cases} p_{\nu} (x), & i = j \\ 0, & \text{else} \end{cases}.

    :param nu: Multi-index indicating which scalar valued monomial basis polynomial should be generated for the
        non-zero component.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int i: Index of the vector component that is non-zero.
    :param int n: Dimension of the target.
    :return: The monomial base polynomial as specified by nu, r, i and n.
    :rtype: :class:`Polynomial`.

    .. rubric:: Examples

    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> vector_valued_monomial_basis_fn(0, 0, 2)(x1)
    array([1, 0])
    >>> vector_valued_monomial_basis_fn(1, 1, 2)(x1)
    array([0, x1], dtype=object)
    >>> vector_valued_monomial_basis_fn((1, 0), 0, 2)((x1, x2))
    array([x1, 0], dtype=object)
    >>> vector_valued_monomial_basis_fn((1, 1), 1, 3)((x1, x2))
    array([0, x1*x2, 0], dtype=object)
    """
    if n == 1:
        assert i == 0
        return monomial_basis_fn(nu)
    assert i >= 0
    assert i < n
    try:
        m = len(nu)
    except TypeError:
        m = 1
    if not isinstance(nu, multiindex.MultiIndex):
        nu = multiindex.MultiIndex(nu)
    r = multiindex.norm(nu)
    dim = get_dimension(r, m)
    coeff = np.zeros((dim, n), dtype=int)
    j = multiindex.get_index(nu, r)
    coeff[j][i] = 1
    return Polynomial(coeff, r, m)


def vector_valued_monomial_basis(r, m, n, ordering="interleaved"):
    r"""
    Generate all monomial base polynomials for the space :math:`\mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n)`.

    :param int r: Degree of the polynomial space.
    :param int m: Dimension of the domain.
    :param int n: Dimension of the target.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: List of base polynomials.
    :rtype: List[:class:`Polynomial`].
    """
    basis = []
    if ordering == "interleaved":
        for mi in multiindex.MultiIndexIterator(m, r):
            for i in range(n):
                basis.append(vector_valued_monomial_basis_fn(mi, i, n))
    else:
        for i in range(n):
            for mi in multiindex.MultiIndexIterator(m, r):
                basis.append(vector_valued_monomial_basis_fn(mi, i, n))
    return basis


def dual_monomial_basis_fn(mu):
    r"""
    Generate a dual basis function to the monomial polynomial basis, i.e. the linear map
    :math:`q_{\mu} : \mathcal{P}_r(\mathbb{R}^n) \to \mathbb{R}` that satisfies

    .. math::

        q_{\mu}(p_{\nu}) = \delta_{\mu, \nu},

    where :math:`p_{\nu}` is the monomial basis polynomial indexed by the multi-index :math:`\nu`, :math:`n = |\nu|`
    (see :func:`monomial_basis_fn`) and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual monomial basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :return: The dual monomial basis function as specified by mu.
    :rtype: Callable :math:`q_{\mu}(p)`.
    """
    try:
        m = len(mu)
    except TypeError:
        m = 1
        mu = multiindex.MultiIndex(mu)
    if m == 1:
        zero = 0
    else:
        zero = np.zeros(m)

    def q(p):
        from polynomials_on_simplices.calculus.polynomial.polynomials_calculus import derivative
        return derivative(p, mu)(zero) / multiindex.factorial(mu)
    return q


def dual_monomial_basis(r, n):
    r"""
    Generate all dual monomial base functions for the space :math:`\mathcal{P}_r(\mathbb{R}^n)` (i.e. the monomial basis
    for :math:`\mathcal{P}_r(\mathbb{R}^n)^*`).

    See :func:`dual_monomial_basis_fn`.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the domain.
    :return: List of dual base functions.
    :rtype: List[callable `q(p)`].
    """
    dual_basis = []
    for mi in multiindex.MultiIndexIterator(n, r):
        dual_basis.append(dual_monomial_basis_fn(mi))
    return dual_basis


def dual_vector_valued_monomial_basis_fn(mu, i, n):
    r"""
    Generate a dual basis function to the vector valued monomial polynomial basis, i.e. the linear map
    :math:`q_{\mu, i} : \mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n) \to \mathbb{R}` that satisfies

    .. math::

        q_{\mu, i}(p_{\nu, j}) = \delta_{\mu, \nu} \delta_{i, j},

    where :math:`p_{\nu, j}` is the degree :math:`|\nu|` vector valued monomial basis polynomial indexed by the
    multi-index :math:`\nu` with a non-zero i:th component (see :func:`vector_valued_monomial_basis_fn`) and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual monomial basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...].
    :param int i: Integer indicating which dual monomial basis function should be generated.
    :param int n: Dimension of the target.
    :return: The dual monomial basis function as specified by mu, r and i.
    :rtype: Callable :math:`q_{\mu, i}(p)`.
    """
    if n == 1:
        assert i == 0
        return dual_monomial_basis_fn(mu)
    assert i >= 0
    assert i < n

    qs = dual_monomial_basis_fn(mu)

    def q(p):
        assert p.target_dimension() == n
        return qs(p[i])

    return q


def dual_vector_valued_monomial_basis(r, m, n, ordering="interleaved"):
    r"""
    Generate all dual monomial base functions for the space :math:`\mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n)` (i.e.
    the monomial basis for :math:`\mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n)^*`).

    See :func:`dual_vector_valued_monomial_basis_fn`.

    :param int r: Degree of the polynomial space.
    :param int m: Dimension of the domain.
    :param int n: Dimension of the target.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: List of dual base functions.
    :rtype: List[callable `q(p)`].
    """
    dual_basis = []
    if ordering == "interleaved":
        for mi in multiindex.MultiIndexIterator(m, r):
            for i in range(n):
                dual_basis.append(dual_vector_valued_monomial_basis_fn(mi, i, n))
    else:
        for i in range(n):
            for mi in multiindex.MultiIndexIterator(m, r):
                dual_basis.append(dual_vector_valued_monomial_basis_fn(mi, i, n))
    return dual_basis


def monomial_basis_fn_latex(nu):
    r"""
    Generate Latex string for a monomial basis polynomial in the space :math:`\mathcal{P}_r(\mathbb{R}^n)`,
    where n is equal to the length of nu.

    :param nu: Multi-index indicating which monomial basis polynomial we should generate Latex string for. Gives the
        exponent for each x_i term.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :return: Latex string for the monomial base polynomial as specified by nu.
    :rtype: str

    .. rubric:: Examples

    >>> monomial_basis_fn_latex(3)
    'x^3'
    >>> monomial_basis_fn_latex((1, 1, 0))
    'x_1 x_2'
    """
    try:
        n = len(nu)
    except TypeError:
        n = 1
        nu = multiindex.MultiIndex(nu)

    if n == 1:
        variables = ["x"]
    else:
        variables = ["x_" + str(i) for i in range(1, n + 1)]

    latex_str_parts = []
    for i in range(len(nu)):
        if nu[i] == 0:
            continue
        if nu[i] == 1:
            latex_str_parts.append(variables[i])
        else:
            latex_str_parts.append(variables[i] + "^" + str(nu[i]))
    if len(latex_str_parts) == 0:
        return "1"
    return " ".join(latex_str_parts)


def monomial_basis_fn_latex_compact(nu):
    r"""
    Generate compact Latex string for a monomial basis polynomial in the space :math:`\mathcal{P}_r(\mathbb{R}^n)`,
    where n is equal to the length of nu.

    :param nu: Multi-index indicating which monomial basis polynomial we should generate Latex string for. Gives the
        exponent for each x_i term.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :return: Latex string for the monomial base polynomial as specified by nu.
    :rtype: str

    .. rubric:: Examples

    >>> monomial_basis_fn_latex_compact(3)
    'x^3'
    >>> monomial_basis_fn_latex_compact((1, 1, 0))
    'x^{(1, 1, 0)}'
    """
    try:
        n = len(nu)
    except TypeError:
        n = 1
        nu = multiindex.MultiIndex(nu)

    if n == 1:
        if nu[0] == 0:
            return "1"
        if nu[0] == 1:
            return "x"
        return "x^" + str(nu[0])
    else:
        return "x^{" + str(nu) + "}"


def monomial_basis_latex(r, n):
    r"""
    Generate Latex strings for all monomial base polynomials for the space :math:`\mathcal{P}_r(\mathbb{R}^n)`.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the domain.
    :return: List of Latex strings for each monomial base polynomial.
    :rtype: List[str]

    .. rubric:: Examples

    >>> monomial_basis_latex(2, 1)
    ['1', 'x', 'x^2']
    >>> monomial_basis_latex(2, 2)
    ['1', 'x_1', 'x_1^2', 'x_2', 'x_1 x_2', 'x_2^2']
    """
    basis_latex_strings = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis_latex_strings.append(monomial_basis_fn_latex(mi))
    return basis_latex_strings


def monomial_basis_latex_compact(r, n):
    r"""
    Generate compact Latex strings for all monomial base polynomials for the space :math:`\mathcal{P}_r(\mathbb{R}^n)`.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the space.
    :return: List of Latex strings for each monomial base polynomial.
    :rtype: List[str]

    .. rubric:: Examples

    >>> monomial_basis_latex_compact(2, 1)
    ['1', 'x', 'x^2']
    >>> monomial_basis_latex_compact(2, 2)
    ['x^{(0, 0)}', 'x^{(1, 0)}', 'x^{(2, 0)}', 'x^{(0, 1)}', 'x^{(1, 1)}', 'x^{(0, 2)}']
    """
    basis_latex_strings = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis_latex_strings.append(monomial_basis_fn_latex_compact(mi))
    return basis_latex_strings


def zero_polynomial(r=0, m=1, n=1):
    r"""
    Get the monomial polynomial :math:`p \in \mathcal{P}(\mathbb{R}^m, \mathbb{R}^n)` which is identically zero.

    :param int m: Dimension of the polynomial domain.
    :param int n: Dimension of the polynomial target.
    :param int r: The zero polynomial will be expressed in the monomial basis for
        :math:`\mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n)`.
    :return: The zero polynomial.
    :rtype: :class:`Polynomial`.
    """
    dim = get_dimension(r, m)
    if n == 1:
        coeff = np.zeros(dim)
    else:
        coeff = np.zeros((dim, n))
    return Polynomial(coeff, r, m)


def unit_polynomial(r=0, m=1, n=1):
    r"""
    Get the monomial polynomial :math:`p \in \mathcal{P}(\mathbb{R}^m, \mathbb{R}^n)` which is identically one.

    :param int m: Dimension of the polynomial domain.
    :param int n: Dimension of the polynomial target.
    :param int r: The unit polynomial will be expressed in the monomial basis for
        :math:`\mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n)`.
    :return: The unit polynomial.
    :rtype: :class:`Polynomial`.
    """
    dim = get_dimension(r, m)
    if n == 1:
        coeff = np.zeros(dim)
        coeff[0] = 1
    else:
        coeff = np.zeros((dim, n))
        coeff[0] = np.ones(n)
    return Polynomial(coeff, r, m)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
