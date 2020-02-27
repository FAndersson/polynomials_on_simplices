r"""Polynomials on the m-dimensional unit simplex with values in :math:`\mathbb{R}^n`, expressed using the Bernstein
basis.

.. math:: b(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} b_{\nu, r}(x),

where :math:`a_{\nu} \in \mathbb{R}^n` and

.. math:: b_{\nu, r}(x) = \binom{r}{\nu} x^{\nu} (1 - |x|)^{r - |\nu|}.

The set :math:`\{ b_{\nu, r} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}` is a basis for the space
of all polynomials of degree less than or equal to r on the unit simplex, :math:`\mathcal{P}_r (\Delta_c^m)`.

The operations on Bernstein polynomials on a simplex used here is a generalization of the corresponding operations
on Bernstein polynomials on the unit interval, as given in [Farouki_and_Rajan_1988]_.

.. rubric:: References

.. [Farouki_and_Rajan_1988] R.T. Farouki and V.T. Rajan. *Algorithms for polynomials in bernstein form*,
    Computer Aided Geometric Design, 5 (1):1 – 26, 1988. ISSN 0167-8396.
    doi:10.1016/0167-8396(88)90016-7. URL http://www.sciencedirect.com/science/article/pii/0167839688900167.

"""

import math
import numbers

import numpy as np
from scipy.special import binom

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_monomial_basis_calculus import (
    integrate_polynomial_unit_simplex)
from polynomials_on_simplices.generic_tools.str_utils import (
    str_dot_product, str_number, str_number_array, str_product, str_sum)
from polynomials_on_simplices.geometry.primitives.simplex import volume_unit
from polynomials_on_simplices.polynomial.code_generation.generate_barycentric_polynomial_functions_simplex import (
    generate_function_specific)
from polynomials_on_simplices.polynomial.polynomials_base import PolynomialBase, get_dimension
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import Polynomial
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import zero_polynomial as zero_polynomial_monomial


def unique_identifier_bernstein_basis():
    """
    Get unique identifier for the Bernstein polynomial basis on the unit simplex.

    :return: Unique identifier.
    :rtype: str
    """
    return "Bernstein"


class PolynomialBernstein(PolynomialBase):
    r"""
    Implementation of the abstract polynomial base class for a polynomial on the m-dimensional unit simplex,
    expressed in the Bernstein basis.

    .. math:: b(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} b_{\nu_i, r}(x).
    """

    def __init__(self, coeff, r=None, m=1):
        r"""
        :param coeff: Coefficients for the polynomial in the Bernstein basis for :math:`\mathcal{P}_r (\mathbb{R}^m,
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

        r = self.r

        # Cache products of polynomial coefficients and binomial coefficients to avoid unnecessary re-computations
        self._a = np.empty(self.coeff.shape, dtype=self.coeff.dtype)
        if m == 1:
            for i in range(0, r + 1):
                self._a[i] = self.coeff[i] * binom(r, i)
        else:
            mis = multiindex.generate_all(m, r)
            for i in range(len(mis)):
                nu = mis[i]
                self._a[i] = self.coeff[i] * multiindex.multinom_general(r, nu)

        # Compile function for evaluating the polynomial
        self._eval_code, self._eval_fn_name = generate_function_specific(m, self.r, self._a)
        compiled_code = compile(self._eval_code, '<auto generated barycentric polynomial function, '
                                + str(self._a) + '>', 'exec')
        exec(compiled_code, globals(), locals())
        self._eval_fn = locals()[self._eval_fn_name]

    def __repr__(self):
        return "polynomials_on_simplices.algebra.polynomial.polynomials_unit_simplex_bernstein_basis.PolynomialBernstein("\
               + str(self.coeff) + ", " + str(self.domain_dimension()) + ", " + str(self.degree()) + ")"

    def basis(self):
        r"""
        Get basis for the space :math:`\mathcal{P}_r (\mathbb{R}^m)` used to express this polynomial.

        :return: Unique identifier for the basis used.
        :rtype: str
        """
        return unique_identifier_bernstein_basis()

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
        :type: PolynomialBernstein, scalar or vector
        :return: Product of this polynomial with other.
        :rtype: :class:`PolynomialBernstein`.
        """
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            return self.multiply_with_constant(other)
        # Multiplication of two polynomials
        # Multiplied polynomials need to have the same domain dimension
        assert self.domain_dimension() == other.domain_dimension()
        # Cannot multiply two vector valued polynomials
        assert self.target_dimension() == 1
        assert other.target_dimension() == 1
        # Formula below is for multiplying two Bernstein polynomials
        # The general case can be handled by converting both polynomials to the monomial basis, do the multiplication
        # there and then convert the result to the Bernstein basis
        assert isinstance(other, PolynomialBernstein)
        m = self.domain_dimension()
        r = self.degree()
        s = other.degree()
        dim = get_dimension(r + s, m)
        coeff = np.zeros(dim)
        if m == 1:
            for k in range(0, dim):
                c = 0
                for i in range(max(0, k - s), min(r, k) + 1):
                    c += binom(r, i) * binom(s, k - i) / binom(r + s, k) * self.coeff[i] * other.coeff[k - i]
                coeff[k] = c
            return PolynomialBernstein(coeff, r + s, m)
        else:
            k = 0
            for tau in multiindex.generate_all(m, r + s):
                c = 0
                for nu in multiindex.generate_all_multi_cap(tau):
                    if multiindex.norm(tau - nu) > s:
                        continue
                    if multiindex.norm(nu) > r:
                        continue
                    i = multiindex.get_index(nu, r)
                    j = multiindex.get_index(tau - nu, s)
                    c += (multiindex.multinom_general(r, nu)
                          * multiindex.multinom_general(s, tau - nu)
                          / multiindex.multinom_general(r + s, tau)
                          * self.coeff[i] * other.coeff[j])
                coeff[k] = c
                k += 1
            return PolynomialBernstein(coeff, r + s, m)

    def __pow__(self, exp):
        r"""
        Raise the polynomial to a power.

        .. math::

            (b^{\mu})(x) = b(x)^{\mu} =  b_1(x)^{\mu_1} b_2(x)^{\mu_2} \ldots b_n(x)^{\mu_n}.

        :param exp: Power we want the raise the polynomial to (natural number or multi-index depending on the dimension
            of the target of the polynomial).
        :type exp: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
        :return: This polynomial raised to the given power.
        :rtype: :class:`PolynomialBernstein`.
        """
        if isinstance(exp, numbers.Integral):
            assert exp >= 0
            assert self.target_dimension() == 1
            if exp == 0:
                return unit_polynomial(0, self.m)
            if exp == 1:
                return PolynomialBernstein(self.coeff, self.r, self.m)
            return self * self**(exp - 1)
        else:
            n = self.target_dimension()
            assert len(exp) == n
            assert [entry >= 0 for entry in exp]
            norm_exp = multiindex.norm(exp)
            if norm_exp == 0:
                return PolynomialBernstein(np.array([1]), 0, self.m)
            for i in range(len(exp)):
                if exp[i] == 0:
                    continue
                ei = multiindex.unit_multiindex(n, i)
                return PolynomialBernstein(self.coeff[:, i], self.r, self.m) * self ** (exp - ei)

    def partial_derivative(self, i=0):
        """
        Compute the i:th partial derivative of the polynomial.

        :param int i: Index of partial derivative.
        :return: i:th partial derivative of this polynomial.
        :rtype: :class:`PolynomialBernstein`.
        """
        assert isinstance(i, numbers.Integral)
        assert i >= 0
        m = self.domain_dimension()
        n = self.target_dimension()
        assert i < m
        r = self.degree()
        if r == 0:
            return zero_polynomial(0, m, n)

        if m == 1:
            if n == 1:
                coeff = np.empty(r)
            else:
                coeff = np.empty((r, n))
            for k in range(0, r):
                coeff[k] = self.coeff[k + 1] - self.coeff[k]
            coeff *= r
        else:
            dim = get_dimension(r - 1, m)
            if n == 1:
                coeff = np.empty(dim)
            else:
                coeff = np.empty((dim, n))
            k = 0
            for mu in multiindex.generate_all(m, r - 1):
                k1 = multiindex.get_index(mu + multiindex.unit_multiindex(m, i), r)
                k2 = multiindex.get_index(mu, r)
                coeff[k] = self.coeff[k1] - self.coeff[k2]
                k += 1
            coeff *= r
        return PolynomialBernstein(coeff, r - 1, m)

    def degree_elevate(self, s=1):
        r"""
        Express the polynomial using a higher degree basis.

        Let :math:`p(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} b_{\nu, r}(x)` be this
        polynomial, where :math:`\{ b_{\nu, r} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}` is the Bernstein
        basis for :math:`\mathcal{P}_r (\mathbb{R}^m)`. Let :math:`\{ b_{\nu, s} \}_{\substack{\nu \in \mathbb{N}_0^m
        \\ |\nu| \leq s}}, s \geq r` be the Bernstein basis for :math:`\mathcal{P}_s (\mathbb{R}^m)`. Then this function
        returns a polynomial :math:`q(x)`

        .. math:: q(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq s}} \tilde{a}_{\nu} b_{\nu, s}(x),

        such that :math:`p(x) = q(x) \, \forall x \in \Delta_c^n`.

        :param int s: New degree for the polynomial basis the polynomial should be expressed in.
        :return: Elevation of this polynomial to the higher degree basis.
        :rtype: :class:`PolynomialBernstein`.
        """
        assert s >= self.degree()
        m = self.domain_dimension()
        n = self.target_dimension()
        r = self.degree()
        if s == r:
            return PolynomialBernstein(self.coeff, r, m)
        dim = get_dimension(s, m)
        if n == 1:
            coeff = np.zeros(dim)
        else:
            coeff = np.zeros((dim, n))
        if m == 1:
            for k in range(dim):
                c = 1 / binom(s, k)
                for i in range(max(0, k - (s - r)), min(r, k) + 1):
                    coeff[k] += binom(r, i) * binom(s - r, k - i) * c * self.coeff[i]
            return PolynomialBernstein(coeff, s, m)
        else:
            k = 0
            for mu in multiindex.generate_all(m, s):
                c = 1 / multiindex.multinom_general(s, mu)
                for sigma in multiindex.generate_all_multi_cap(mu):
                    if multiindex.norm(sigma) < multiindex.norm(mu) - (s - r):
                        continue
                    if multiindex.norm(sigma) > r:
                        continue
                    idx = multiindex.get_index(sigma, r)
                    coeff[k] += (multiindex.multinom_general(r, sigma) * binom(s - r, multiindex.norm(mu - sigma))
                                 * multiindex.multinom(mu - sigma) * c * self.coeff[idx])
                k += 1
            return PolynomialBernstein(coeff, s, m)

    def to_monomial_basis(self):
        """
        Compute the monomial representation of this polynomial.

        :return: This polynomial expressed in the monomial basis.
        :rtype: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`.
        """
        basis_polynomials_monomial_form = bernstein_basis_monomial(self.r, self.m)
        return sum([b * a for (a, b) in zip(self.coeff, basis_polynomials_monomial_form)],
                   zero_polynomial_monomial(0, self.m, self.n))

    def latex_str(self):
        r"""
        Generate a Latex string for this polynomial.

        :return: Latex string for this polynomial.
        :rtype: str
        """
        try:
            len(self.coeff[0])
            coeff_strs = [str_number_array(c, latex=True) for c in self.coeff]
            basis_strs = bernstein_basis_latex_compact(self.r, self.m)
            return str_dot_product(coeff_strs, basis_strs)
        except TypeError:
            coeff_strs = [str_number(c, latex_fraction=True) for c in self.coeff]
            basis_strs = bernstein_basis_latex_compact(self.r, self.m)
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
            basis_strs = bernstein_basis_latex(self.r, self.m)
            res = "0"
            for i in range(len(coeff_strs)):
                if basis_strs[i][0].isnumeric():
                    p = str_product(coeff_strs[i], basis_strs[i], r"\cdot")
                else:
                    p = str_product(coeff_strs[i], basis_strs[i])
                res = str_sum(res, p)
            return res
        except TypeError:
            coeff_strs = [str_number(c, latex_fraction=True) for c in self.coeff]
            basis_strs = bernstein_basis_latex(self.r, self.m)
            res = "0"
            for i in range(len(coeff_strs)):
                if basis_strs[i][0].isnumeric():
                    p = str_product(coeff_strs[i], basis_strs[i], r"\cdot")
                else:
                    p = str_product(coeff_strs[i], basis_strs[i])
                res = str_sum(res, p)
            return res

    def code_str(self, fn_name):
        r"""
        Generate a function code string for evaluating this polynomial.

        :param str fn_name: Name for the function in the generated code.
        :return: Code string for evaluating this polynomial.
        :rtype: str
        """
        return self._eval_code.replace(self._eval_fn_name, fn_name)


def bernstein_basis_fn(nu, r):
    r"""
    Generate a Bernstein basis polynomial on the n-dimensional unit simplex (:math:`\Delta_c^n`)

    .. math:: b_{\nu, r} (x) = \binom{r}{\nu} x^{\nu} (1 - |x|)^{r - |\nu|},

    where n is equal to the length of nu.

    :param nu: Multi-index indicating which Bernstein basis polynomial should be generated.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: The Bernstein base polynomial as specified by nu and r.
    :rtype: :class:`PolynomialBernstein`.

    .. rubric:: Examples

    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> bernstein_basis_fn(0, 1)(x1) - (-x1 + 1)
    0
    >>> bernstein_basis_fn(1, 1)(x1) - x1
    0
    >>> sp.simplify(bernstein_basis_fn((1, 0), 2)((x1, x2)) - 2*x1*(-x1 - x2 + 1))
    0
    >>> sp.simplify(bernstein_basis_fn((1, 1), 3)((x1, x2)) - 6*x1*x2*(-x1 - x2 + 1))
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
    return PolynomialBernstein(coeff, r, m)


def bernstein_basis(r, n):
    r"""
    Generate all Bernstein base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the unit simplex.
    :return: List of base polynomials.
    :rtype: List[:class:`PolynomialBernstein`].
    """
    basis = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis.append(bernstein_basis_fn(mi, r))
    return basis


def vector_valued_bernstein_basis_fn(nu, r, i, n):
    r"""
    Generate a vector valued Bernstein basis polynomial on the m-dimensional unit simplex,
    :math:`b_{\nu, r, i} : \Delta_c^m \to \mathbb{R}^n`.

    The vector valued basis polynomial is generated by specifying a scalar valued basis polynomial and the component
    of the vector valued basis polynomial that should be equal to the scalar valued basis polynomial. All other
    components of the vector valued basis polynomial will be zero, i.e.

    .. math:: b_{\nu, r, i}^j (x) = \begin{cases} b_{\nu, r} (x), & i = j \\ 0, & \text{else} \end{cases},

    where m is equal to the length of nu.

    :param nu: Multi-index indicating which scalar valued Bernstein basis polynomial should be generated for the
        non-zero component.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param int i: Index of the vector component that is non-zero.
    :param int n: Dimension of the target.
    :return: The Bernstein base polynomial as specified by nu, r, i and n.
    :rtype: :class:`PolynomialBernstein`.

    .. rubric:: Examples

    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> vector_valued_bernstein_basis_fn(0, 1, 0, 2)(x1)
    array([-x1 + 1, 0], dtype=object)
    >>> vector_valued_bernstein_basis_fn(1, 1, 1, 2)(x1)
    array([0, x1], dtype=object)
    >>> vector_valued_bernstein_basis_fn((1, 0), 2, 0, 2)((x1, x2))
    array([2*x1*(-x1 - x2 + 1), 0], dtype=object)
    >>> vector_valued_bernstein_basis_fn((1, 1), 3, 1, 3)((x1, x2))
    array([0, 6*x1*x2*(-x1 - x2 + 1), 0], dtype=object)
    """
    if n == 1:
        assert i == 0
        return bernstein_basis_fn(nu, r)
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
    return PolynomialBernstein(coeff, r, m)


def vector_valued_bernstein_basis(r, m, n, ordering="interleaved"):
    r"""
    Generate all Bernstein base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)`.

    :param int r: Degree of the polynomial space.
    :param int m: Dimension of the unit simplex.
    :param int n: Dimension of the target.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: List of base polynomials.
    :rtype: List[:class:`PolynomialBernstein`].
    """
    basis = []
    if ordering == "interleaved":
        for mi in multiindex.MultiIndexIterator(m, r):
            for i in range(n):
                basis.append(vector_valued_bernstein_basis_fn(mi, r, i, n))
    else:
        for i in range(n):
            for mi in multiindex.MultiIndexIterator(m, r):
                basis.append(vector_valued_bernstein_basis_fn(mi, r, i, n))
    return basis


def bernstein_basis_fn_monomial(nu, r):
    r"""
    Generate a Bernstein basis polynomial on the n-dimensional unit simplex (:math:`\Delta_c^n`)

    .. math:: b_{\nu, r} (x) = \binom{r}{\nu} x^{\nu} (1 - |x|)^{r - |\nu|},

    where n is equal to the length of nu, expanded in the monomial basis.

    This is the same polynomial as given by the :func:`bernstein_basis_fn` function, but expressed in the monomial
    basis.

    :param nu: Multi-index indicating which Bernstein basis polynomial should be generated.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: The Bernstein base polynomial as specified by nu and r.
    :rtype: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`.

    .. rubric:: Examples

    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> bernstein_basis_fn_monomial(0, 1)(x1)
    -x1 + 1
    >>> bernstein_basis_fn_monomial(1, 1)(x1)
    x1
    >>> bernstein_basis_fn_monomial((1, 0), 2)((x1, x2))
    -2*x1**2 - 2*x1*x2 + 2*x1
    >>> bernstein_basis_fn_monomial((1, 1), 3)((x1, x2))
    -6*x1**2*x2 - 6*x1*x2**2 + 6*x1*x2
    """
    try:
        m = len(nu)
    except TypeError:
        # Univariate case is a special case of the multivariate case
        return bernstein_basis_fn((nu,), r)
    dim = get_dimension(r, m)
    coeff = np.zeros(dim, dtype=int)

    # Using the multinomial theorem
    s = r - multiindex.norm(nu)
    r_fac = math.factorial(r)
    nu_fac = multiindex.factorial(nu)
    norm_nu = multiindex.norm(nu)
    c0 = int(r_fac / nu_fac)
    for mu in multiindex.generate_all(m, s):
        norm_mu = multiindex.norm(mu)
        mu_fac = multiindex.factorial(mu)
        c = c0 / (mu_fac * math.factorial(r - norm_nu - norm_mu))
        if norm_mu % 2 == 1:
            c *= -1
        j = multiindex.get_index(mu + nu, r)
        coeff[j] = c
    return Polynomial(coeff, r, m)


def bernstein_basis_monomial(r, n):
    r"""
    Generate all Bernstein base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`, expanded in the monomial
    basis.

    This is the same set of polynomials as given by the :func:`bernstein_basis` function, but expressed in the
    monomial basis.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the unit simplex.
    :return: List of base polynomials.
    :rtype: List[:class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`].
    """
    basis = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis.append(bernstein_basis_fn_monomial(mi, r))
    return basis


def dual_bernstein_basis_polynomial(mu, r):
    r"""
    Generate a dual Bernstein basis polynomial on the n-dimensional unit simplex (:math:`\Delta_c^n`)
    :math:`d_{\mu, r} (x)`, where n is equal to the length of mu.

    The dual Bernstein basis :math:`d_{\mu, r} (x)` is the unique polynomial that satisfies

    .. math::

        \int_{\Delta_c^n} d_{\mu, r}(x) b_{\nu, r} (x) \, dx = \delta_{\mu, \nu}.

    :param mu: Multi-index indicating which dual Bernstein basis polynomial should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: The dual Bernstein base polynomial as specified by mu and r.
    :rtype: :class:`PolynomialBernstein`.

    .. rubric:: Examples

    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> d = dual_bernstein_basis_polynomial((0, 0), 2)((x1, x2))
    >>> b = bernstein_basis_fn((0, 0), 2)((x1, x2))
    >>> abs(sp.integrate(d * b, (x2, 0, -x1 + 1), (x1, 0, 1)) - 1.0) < 1e-10
    True
    >>> import sympy as sp
    >>> x1, x2 = sp.symbols('x1 x2')
    >>> d = dual_bernstein_basis_polynomial((0, 0), 2)((x1, x2))
    >>> b = bernstein_basis_fn((1, 0), 2)((x1, x2))
    >>> abs(sp.integrate(d * b, (x2, 0, -x1 + 1), (x1, 0, 1))) < 1e-10
    True
    """
    try:
        m = len(mu)
        if m == 1:
            mu = mu[0]
    except TypeError:
        m = 1
    if r == 0:
        return unit_polynomial(0, m) / volume_unit(m)
    dim = get_dimension(r, m)
    coeff = np.empty(dim)
    if m == 1:
        j = mu
        for k in range(dim):
            factor = (-1)**(j + k) / (binom(r, j) * binom(r, k))
            c = 0
            for l in range(0, min(j, k) + 1):
                c += (2 * l + 1) * (binom(r + l + 1, r - j) * binom(r - l, r - j)
                                    * binom(r + l + 1, r - k) * binom(r - l, r - k))
            c *= factor
            coeff[k] = c
        return PolynomialBernstein(coeff, r, m)
    else:
        # TODO: Derive closed form formula for the coefficients, or at least cache the computed coefficients for
        # low degree and dimension cases

        # Assemble linear system
        dim = get_dimension(r, m)
        a = np.empty((dim, dim))
        from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_bernstein_basis_calculus import \
            integrate_bernstein_basis_fn_unit_simplex
        factor = integrate_bernstein_basis_fn_unit_simplex(2 * r, m)
        mis = multiindex.generate_all(m, r)
        for i in range(dim):
            nu = mis[i]
            f1 = multiindex.multinom_general(r, nu)
            for j in range(i, dim):
                tau = mis[j]
                f2 = multiindex.multinom_general(r, tau)
                a[i][j] = f1 * f2 / multiindex.multinom_general(2 * r, nu + tau) * factor
                if j != i:
                    a[j][i] = a[i][j]
        a_inv = np.linalg.inv(a)
        v = np.zeros(dim)
        v[multiindex.get_index(mu, r)] = 1
        coeff = np.dot(a_inv, v)
        return PolynomialBernstein(coeff, r, m)


def dual_bernstein_polynomial_basis(r, n):
    r"""
    Generate all dual Bernstein base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`.

    See :func:`dual_bernstein_basis_polynomial`.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the unit simplex.
    :return: List of dual base polynomials.
    :rtype: List[:class:`PolynomialBernstein`].
    """
    basis = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis.append(dual_bernstein_basis_polynomial(mi, r))
    return basis


def dual_bernstein_basis_fn(mu, r):
    r"""
    Generate a dual basis function to the Bernstein polynomial basis, i.e. the linear map
    :math:`q_{\mu, r} : \mathcal{P}_r(\Delta_c^n) \to \mathbb{R}` that satisfies

    .. math::

        q_{\mu, r}(b_{\nu, r}) = \delta_{\mu, \nu},

    where :math:`b_{\nu, r}` is the degree r Bernstein basis polynomial indexed by the multi-index :math:`\nu`
    (see :func:`bernstein_basis_fn`) and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual Bernstein basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...].
    :param int r: Degree of polynomial space.
    :return: The dual Bernstein basis function as specified by mu and r.
    :rtype: Callable :math:`q_{\mu, r}(b)`.
    """
    from polynomials_on_simplices.calculus.polynomial.polynomials_simplex_bernstein_basis_calculus import \
        integrate_bernstein_polynomial_unit_simplex
    d = dual_bernstein_basis_polynomial(mu, r)
    try:
        m = len(mu)
    except TypeError:
        m = 1

    def q(p):
        if isinstance(p, PolynomialBernstein):
            return integrate_bernstein_polynomial_unit_simplex(2 * r, (d * p).coeff, m)
        else:
            return integrate_polynomial_unit_simplex(2 * r, m, (d.to_monomial_basis() * p.to_monomial_basis()).coeff)
    return q


def dual_bernstein_basis(r, n):
    r"""
    Generate all dual Bernstein base functions for the space :math:`\mathcal{P}_r(\Delta_c^n)` (i.e. the Bernstein basis
    for :math:`\mathcal{P}_r(\Delta_c^n)^*`).

    See :func:`dual_bernstein_basis_fn`.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the unit simplex.
    :return: List of dual base functions.
    :rtype: List[callable `q(b)`].
    """
    dual_basis = []
    for mi in multiindex.MultiIndexIterator(n, r):
        dual_basis.append(dual_bernstein_basis_fn(mi, r))
    return dual_basis


def dual_vector_valued_bernstein_basis_fn(mu, r, i, n):
    r"""
    Generate a dual basis function to the vector valued Bernstein polynomial basis, i.e. the linear map
    :math:`q_{\mu, r, i} : \mathcal{P}_r(\Delta_c^m, \mathbb{R}^n) \to \mathbb{R}` that satisfies

    .. math::

        q_{\mu, r, i}(b_{\nu, r, j}) = \delta_{\mu, \nu} \delta_{i, j},

    where :math:`b_{\nu, r, j}` is the degree r vector valued Bernstein basis polynomial indexed by the
    multi-index :math:`\nu` with a non-zero i:th component (see :func:`vector_valued_bernstein_basis_fn`) and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual Bernstein basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...].
    :param int r: Degree of polynomial space.
    :param int i: Integer indicating which dual Bernstein basis function should be generated.
    :param int n: Dimension of the target.
    :return: The dual Bernstein basis function as specified by mu, r and i.
    :rtype: Callable :math:`q_{\mu, r, i}(b)`.
    """
    if n == 1:
        assert i == 0
        return dual_bernstein_basis_fn(mu, r)
    assert i >= 0
    assert i < n

    qs = dual_bernstein_basis_fn(mu, r)

    def q(p):
        assert p.target_dimension() == n
        return qs(p[i])

    return q


def dual_vector_valued_bernstein_basis(r, m, n, ordering="interleaved"):
    r"""
    Generate all dual Bernstein base functions for the space :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)` (i.e. the
    Bernstein basis for :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)^*`).

    See :func:`dual_vector_valued_bernstein_basis_fn`.

    :param int r: Degree of the polynomial space.
    :param int m: Dimension of the unit simplex.
    :param int n: Dimension of the target.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: List of dual base functions.
    :rtype: List[callable `q(b)`].
    """
    dual_basis = []
    if ordering == "interleaved":
        for mi in multiindex.MultiIndexIterator(m, r):
            for i in range(n):
                dual_basis.append(dual_vector_valued_bernstein_basis_fn(mi, r, i, n))
    else:
        for i in range(n):
            for mi in multiindex.MultiIndexIterator(m, r):
                dual_basis.append(dual_vector_valued_bernstein_basis_fn(mi, r, i, n))
    return dual_basis


def degree_elevated_bernstein_basis_fn(nu, r, s):
    r"""
    Generate a Bernstein basis polynomial on the n-dimensional unit simplex (:math:`\Delta_c^n`)
    :math:`b_{\nu, r} (x) = \binom{r}{\nu} x^{\nu} (1 - |x|)^{r - |\nu|}`,
    where n is equal to the length of nu, and degree elevate it to a degree s Bernstein polynomial.

    :param nu: Multi-index indicating which Bernstein basis polynomial should be generated.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param int s: Degree for the polynomial basis the polynomial should be expressed in.
    :return: The Bernstein base polynomial as specified by nu and r, expressed as a degree s Bernstein polynomial.
    :rtype: :class:`PolynomialBernstein`.
    """
    assert s > r
    try:
        m = len(nu)
        if m == 1:
            nu = nu[0]
    except TypeError:
        m = 1
    dim = get_dimension(s, m)
    coeff = np.zeros(dim)
    if m == 1:
        i = nu
        c = binom(r, i)
        for k in range(i, i + s - r + 1):
            coeff[k] = c * binom(s - r, k - i) / binom(s, k)
        return PolynomialBernstein(coeff, s, m)
    else:
        c = multiindex.multinom_general(r, nu)
        for mu in multiindex.MultiIndexIterator(m, s - r):
            idx = multiindex.get_index(nu + mu, s)
            coeff[idx] = (c * binom(s - r, multiindex.norm(mu)) * multiindex.multinom(mu)
                          / multiindex.multinom_general(s, nu + mu))
        return PolynomialBernstein(coeff, s, m)


def bernstein_basis_fn_latex(nu, r):
    r"""
    Generate Latex string for a Bernstein basis polynomial on the n-dimensional unit simplex (:math:`\Delta_c^n`)

    .. math:: b_{\nu, r} (x) = \binom{r}{\nu} x^{\nu} (1 - |x|)^{r - |\nu|},

    where n is equal to the length of nu.

    :param nu: Multi-index indicating which Bernstein basis polynomial we should generate Latex string for.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: Latex string for the Bernstein base polynomial as specified by nu and r.
    :rtype: str

    .. rubric:: Examples

    >>> bernstein_basis_fn_latex(2, 3)
    '3 x^2 (1 - x)'
    >>> bernstein_basis_fn_latex((1, 1), 3)
    '6 x_1 x_2 (1 - x_1 - x_2)'
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
    e = r - multiindex.norm(nu)
    if e != 0:
        part = "(1"
        for i in range(len(variables)):
            part += " - " + variables[i]
        part += ")"
        if e == 1:
            latex_str_parts.append(part)
        else:
            latex_str_parts.append(part + "^" + str(e))
    if n == 1:
        c = int(binom(r, nu[0]))
    else:
        c = int(multiindex.multinom_general(r, nu))
    if c == 1:
        return " ".join(latex_str_parts)
    else:
        return str(c) + " " + " ".join(latex_str_parts)


def bernstein_basis_fn_latex_compact(nu, r):
    r"""
    Generate compact Latex string for a Bernstein basis polynomial on the n-dimensional unit simplex
    (:math:`\Delta_c^n`)

    .. math:: b_{\nu, r} (x) = \binom{r}{\nu} x^{\nu} (1 - |x|)^{r - |\nu|},

    where n is equal to the length of nu.

    :param nu: Multi-index indicating which Bernstein basis polynomial we should generate Latex string for.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :return: Latex string for the Bernstein base polynomial as specified by nu and r.
    :rtype: str

    .. rubric:: Examples

    >>> bernstein_basis_fn_latex_compact(2, 3)
    'b_{2, 3}(x)'
    >>> bernstein_basis_fn_latex_compact((1, 1), 3)
    'b_{(1, 1), 3}(x)'
    """
    try:
        n = len(nu)
    except TypeError:
        n = 1
        nu = multiindex.MultiIndex(nu)

    if n == 1:
        return "b_{" + str(nu[0]) + ", " + str(r) + "}(x)"
    else:
        return "b_{" + str(nu) + ", " + str(r) + "}(x)"


def bernstein_basis_latex(r, n):
    r"""
    Generate Latex strings for all Bernstein base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the unit simplex.
    :return: List of Latex strings for each Bernstein base polynomial.
    :rtype: List[str]

    .. rubric:: Examples

    >>> bernstein_basis_latex(2, 1)
    ['(1 - x)^2', '2 x (1 - x)', 'x^2']
    >>> bernstein_basis_latex(2, 2)
    ['(1 - x_1 - x_2)^2', '2 x_1 (1 - x_1 - x_2)', 'x_1^2', '2 x_2 (1 - x_1 - x_2)', '2 x_1 x_2', 'x_2^2']
    """
    basis_latex_strings = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis_latex_strings.append(bernstein_basis_fn_latex(mi, r))
    return basis_latex_strings


def bernstein_basis_latex_compact(r, n):
    r"""
    Generate compact Latex strings for all Bernstein base polynomials for the space :math:`\mathcal{P}_r(\Delta_c^n)`.

    :param int r: Degree of the polynomial space.
    :param int n: Dimension of the unit simplex.
    :return: List of Latex strings for each Bernstein base polynomial.
    :rtype: List[str]

    .. rubric:: Examples

    >>> bernstein_basis_latex_compact(2, 1)
    ['b_{0, 2}(x)', 'b_{1, 2}(x)', 'b_{2, 2}(x)']
    >>> bernstein_basis_latex_compact(1, 2)
    ['b_{(0, 0), 1}(x)', 'b_{(1, 0), 1}(x)', 'b_{(0, 1), 1}(x)']
    """
    basis_latex_strings = []
    for mi in multiindex.MultiIndexIterator(n, r):
        basis_latex_strings.append(bernstein_basis_fn_latex_compact(mi, r))
    return basis_latex_strings


def zero_polynomial(r=0, m=1, n=1):
    r"""
    Get the Bernstein polynomial :math:`b \in \mathcal{P}(\Delta_c^m, \mathbb{R}^n)` which is identically zero.

    :param int r: The zero polynomial will be expressed in the Bernstein basis for
        :math:`\mathcal{P}_r(\Delta_c^m, \mathbb{R}^n)`.
    :param int m: Dimension of the polynomial domain.
    :param int n: Dimension of the polynomial target.
    :return: The zero polynomial.
    :rtype: :class:`PolynomialBernstein`.
    """
    dim = get_dimension(r, m)
    if n == 1:
        coeff = np.zeros(dim)
    else:
        coeff = np.zeros((dim, n))
    return PolynomialBernstein(coeff, r, m)


def unit_polynomial(r=0, m=1, n=1):
    r"""
    Get the Bernstein polynomial :math:`b \in \mathcal{P}(\Delta_c^m, \mathbb{R}^n)` which is identically one.

    :param int r: The unit polynomial will be expressed in the Bernstein basis for
        :math:`\mathcal{P}_r(\mathbb{R}^m, \mathbb{R}^n)`.
    :param int m: Dimension of the polynomial domain.
    :param int n: Dimension of the polynomial target.
    :return: The unit polynomial.
    :rtype: :class:`PolynomialBernstein`.
    """
    dim = get_dimension(r, m)
    if n == 1:
        coeff = np.ones(dim)
    else:
        coeff = np.ones((dim, n))
    return PolynomialBernstein(coeff, r, m)


def barycentric_polynomial_general(r, a, x):
    r"""
    Evaluate a degree r barycentric polynomial on the m-dimensional unit simplex.

    .. math:: b(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} x^{\nu} (1 - |x|)^{r - |\nu|}.

    :param int r: Degree of the polynomial.
    :param a: Coefficient in front of each barycentric base polynomial.
    :param x: Point where the polynomial should be evaluated.
    :return: Value of the polynomial.
    """
    try:
        m = len(x)
    except TypeError:
        m = 1
    value = 0
    if m == 1:
        for i in range(len(a)):
            value += a[i] * x**i * (1 - x)**(r - i)
    else:
        # TODO: Potentially improve performance by caching multiindices
        i = 0
        for nu in multiindex.generate_all(m, r):
            value += a[i] * multiindex.power(x, nu) * (1 - multiindex.norm(x))**(r - multiindex.norm(nu))
            i += 1
    return value


def get_associated_sub_simplex(nu, r, simplex=None):
    r"""
    Get the sub simplex associated with a Bernstein basis polynomial.

    For a Bernstein basis polynomial p on a simplex T there exist a unique sub simplex f of T such that p vanishes
    to degree r on on f*, where f* is the sub simplex opposite to the simplex f (see
    :func:`polynomials_on_simplices.geometry.mesh.simplicial_complex.opposite_sub_simplex`). A polynomial vanishes to degree
    r on a simplex f* if

    .. math::

        (\partial_{\alpha} p) = 0, \, \forall x \in f^*, \, \forall \alpha \in \mathbb{N}_0^n, |\alpha| \leq r - 1.

    :param nu: Multi-index indicating for which Bernstein basis polynomial we should get the associated sub simplex.
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
