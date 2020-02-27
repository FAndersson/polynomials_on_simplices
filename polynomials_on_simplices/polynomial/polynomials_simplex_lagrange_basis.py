r"""Polynomials on an m-dimensional simplex T with values in :math:`\mathbb{R}^n`, expressed using the
Lagrange basis.

.. math::

    l(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} l_{\nu, r}(x)
    = \sum_{\nu} a_{\nu} (\bar{l}_{\nu, r} \circ \Phi^{-1})(x),

where :math:`a_{\nu} \in \mathbb{R}^n, \bar{l}_{\nu, r}` is the Lagrange basis on the unit simplex and :math:`\Phi`
is the unique affine map which maps the unit simplex onto the simplex T (the i:th vertex of the unit simplex is mapped
to the i:th vertex of the simplex T).

The basis polynomials :math:`l_{\nu, r} = \bar{l}_{\nu, r} \circ \Phi^{-1}` satisfies

.. math:: l_{\nu, r}(\Phi(x_{\mu}) = \delta_{\mu, \nu},

where :math:`x_{\mu}` are the Lagrange points on the unit simplex.

The set :math:`\{ l_{\nu, r} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}` is a basis for the space
of all polynomials of degree less than or equal to r on the simplex T, :math:`\mathcal{P}_r (T)`.
"""

import numbers

import numpy as np

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.generic_tools.code_generation_utils import CodeWriter
from polynomials_on_simplices.generic_tools.str_utils import str_dot_product, str_number, str_number_array
from polynomials_on_simplices.geometry.primitives.simplex import (
    affine_map_from_unit, affine_map_to_unit, affine_transformation_to_unit, dimension)
from polynomials_on_simplices.polynomial.polynomials_base import get_dimension
from polynomials_on_simplices.polynomial.polynomials_monomial_basis import Polynomial, dual_monomial_basis
from polynomials_on_simplices.polynomial.polynomials_simplex_base import PolynomialSimplexBase
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import (
    PolynomialLagrange, generate_lagrange_point, generate_lagrange_points, lagrange_basis_latex_compact)


def unique_identifier_lagrange_basis_simplex(vertices):
    """
    Get unique identifier for the Lagrange polynomial basis on a simplex T.

    :param vertices: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
        simplex).
    :return: Unique identifier.
    :rtype: str
    """
    from polynomials_on_simplices.generic_tools.code_generation_utils import CodeWriter
    identifier = CodeWriter()
    identifier.wl("Lagrange(")
    identifier.inc_indent()
    identifier.wc(str(vertices))
    identifier.dec_indent()
    identifier.wl(")")
    return identifier.code


def generate_lagrange_point_simplex(vertices, r, nu):
    r"""
    Generate a Lagrange point indexed by a multi-index on an n-dimensional simplex T from the set
    :math:`\{ \bar{x}_nu \}` of evenly spaced Lagrange points on the m-dimensional unit simplex (:math:`\Delta_c^n`)
    (Lagrange basis points are constructed so that each basis function has the value 1 at one of the points,
    and 0 at all the other points).

    .. math:: \bar{x}_{\nu} = \frac{\nu}{r},

    .. math:: x_{\nu} = \Phi(\bar{x}_{\nu},

    where :math:`\Phi` is the unique affine map which maps the unit simplex to the simplex T.

    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :param int r: Degree of the polynomial.
    :param nu: Multi-index :math:`\nu` indexing the Lagrange point, where :math:`\frac{\nu_i}{r}` gives
        the i:th coordinate of the corresponding Lagrange point in the unit simplex.
    :return: Point in the n-dimensional simplex T.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    n = dimension(vertices)
    x = generate_lagrange_point(n, r, nu)
    phi = affine_map_from_unit(vertices)
    return phi(x)


def generate_lagrange_points_simplex(vertices, r):
    r"""
    Generate evenly spaced Lagrange points on an n-dimensional simplex T
    (Lagrange basis points are constructed so that each basis function has the value 1 at one of the points,
    and 0 at all the other points).

    .. math:: \{ x_{\nu} \}_{\substack{\nu \in \mathbb{N}_0^n \\ |\nu| \leq r}}, x_{\nu} = x_{\nu} = \Phi(\bar{x}_{\nu},

    where :math:`\{ \bar{x}_{\nu} \}` is the set of evenly spaced Lagrange points on the unit simplex, and :math:`\Phi`
    is the unique affine map which maps the unit simplex to the simplex T.

    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :param int r: Degree of the polynomial.
    :return: List of points in the n-dimensional simplex T.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    phi = affine_map_from_unit(vertices)
    n = len(vertices[0])
    if n == 1:
        x = np.empty(r + 1)
    else:
        x = np.empty((get_dimension(r, n), n))
    xbar = generate_lagrange_points(n, r)
    for i in range(len(x)):
        x[i] = phi(xbar[i])
    return x


class PolynomialLagrangeSimplex(PolynomialSimplexBase):
    r"""
    Implementation of the abstract polynomial base class for a polynomial on an m-dimensional simplex T,
    expressed in the Lagrange basis.

    .. math:: l(x) = \sum_{i = 0}^{\dim(\mathcal{P}_r(\mathbb{R}^m)) - 1} a_{\nu_i} l_{\nu_i, r}(x).
    """

    def __init__(self, coeff, vertices, r=None):
        r"""
        :param coeff: Coefficients for the polynomial in the Lagrange basis for :math:`\mathcal{P}_r (T,
            \mathbb{R}^n). \text{coeff}[i] = a_{\nu_i}`, where :math:`\nu_i` is the i:th multi-index in the sequence
            of all multi-indices of dimension m with norm :math:`\leq r`
            (see :func:`polynomials_on_simplices.algebra.multiindex.generate` function).
            Array of scalars for a scalar valued polynomial (n = 1) and array of n-dimensional vectors for a vector
            valued polynomial (:math:`n \geq 2`).
        :param vertices: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
            simplex).
        :param int r: Degree of the polynomial space. Optional, will be inferred from the number of polynomial
            coefficients if not specified.
        """
        m = len(vertices[0])
        PolynomialSimplexBase.__init__(self, coeff, vertices, r)
        self.vertices = vertices
        self._unit_simplex_polynomial = PolynomialLagrange(coeff, r, m)
        self._a, self._b = affine_transformation_to_unit(vertices)
        self._phi_inv = affine_map_to_unit(vertices)

    def __repr__(self):
        return "polynomials_on_simplices.algebra.polynomial.polynomials_simplex_lagrange_basis.PolynomialLagrangeSimplex("\
               + str(self.coeff) + ", " + str(self.vertices) + ", " + str(self.degree()) + ")"

    def basis(self):
        r"""
        Get basis for the space :math:`\mathcal{P}_r (\mathbb{R}^m)` used to express this polynomial.

        :return: Unique identifier for the basis used.
        :rtype: str
        """
        return unique_identifier_lagrange_basis_simplex(self.vertices)

    def __call__(self, x):
        r"""
        Evaluate the polynomial at a point :math:`x \in \mathbb{R}^m`.

        :param x: Point where the polynomial should be evaluated.
        :type x: float or length m :class:`Numpy array <numpy.ndarray>`
        :return: Value of the polynomial.
        :rtype: float or length n :class:`Numpy array <numpy.ndarray>`.
        """
        return self._unit_simplex_polynomial(self._phi_inv(x))

    def __mul__(self, other):
        """
        Multiplication of this polynomial with another polynomial, a scalar, or a vector (for a scalar valued
        polynomial), self * other.

        :param other: Polynomial, scalar or vector we should multiply this polynomial with.
        :type: PolynomialLagrangeSimplex, scalar or vector
        :return: Product of this polynomial with other.
        :rtype: :class:`PolynomialLagrangeSimplex`.
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
        x = generate_lagrange_points_simplex(self.vertices, r)
        for i in range(len(x)):
            coeff[i] = self(x[i]) * other(x[i])
        return PolynomialLagrangeSimplex(coeff, self.vertices, r)

    def __pow__(self, exp):
        r"""
        Raise the polynomial to a power.

        .. math::

            (l^{\mu})(x) = l(x)^{\mu} =  l_1(x)^{\mu_1} l_2(x)^{\mu_2} \ldots l_n(x)^{\mu_n}.

        :param exp: Power we want the raise the polynomial to (natural number or multi-index depending on the dimension
            of the target of the polynomial).
        :type exp: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
        :return: This polynomial raised to the given power.
        :rtype: :class:`PolynomialLagrangeSimplex`.
        """
        if isinstance(exp, numbers.Integral):
            assert exp >= 0
            assert self.target_dimension() == 1
            if exp == 0:
                return unit_polynomial_simplex(self.vertices, 1)
            if exp == 1:
                return PolynomialLagrangeSimplex(self.coeff, self.vertices, self.r)
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
            x = generate_lagrange_points_simplex(self.vertices, r)
            for i in range(len(x)):
                coeff[i] = multiindex.power(self(x[i]), exp)
            return PolynomialLagrangeSimplex(coeff, self.vertices, r)

    def partial_derivative(self, i=0):
        """
        Compute the i:th partial derivative of the polynomial.

        :param int i: Index of partial derivative.
        :return: i:th partial derivative of this polynomial.
        :rtype: :class:`PolynomialLagrangeSimplex`.
        """
        assert isinstance(i, numbers.Integral)
        assert i >= 0
        m = self.domain_dimension()
        n = self.target_dimension()
        assert i < m
        r = self.degree()
        if r == 0:
            return zero_polynomial_simplex(self.vertices, 0, n)

        # Compute derivative using the chain rule
        # We have D(l)(x) = D((lb o pi)(x) = D(lb)(pi(x)) * D(pi)(x)
        from polynomials_on_simplices.calculus.polynomial.polynomials_calculus import gradient, jacobian
        if m == 1:
            if n == 1:
                db = self._unit_simplex_polynomial.partial_derivative()
                return PolynomialLagrangeSimplex(db.coeff, self.vertices, self.r - 1) * self._a
            else:
                jb = jacobian(self._unit_simplex_polynomial)
                coeff = np.empty((len(jb[0][0].coeff), n))
                for j in range(n):
                    coeff[:, j] = jb[j][0].coeff * self._a
                return PolynomialLagrangeSimplex(coeff, self.vertices, self.r - 1)
        else:
            if n == 1:
                gb = gradient(self._unit_simplex_polynomial)
                d = PolynomialLagrangeSimplex(gb[0].coeff, self.vertices, self.r - 1) * self._a[0, i]
                for k in range(1, m):
                    d += PolynomialLagrangeSimplex(gb[k].coeff, self.vertices, self.r - 1) * self._a[k, i]
                return d
            else:
                jb = jacobian(self._unit_simplex_polynomial)
                coeff = np.empty((len(jb[0][0].coeff), n))
                for j in range(n):
                    coeff[:, j] = jb[j][0].coeff * self._a[0, i]
                    for k in range(1, m):
                        coeff[:, j] += jb[j][k].coeff * self._a[k, i]
                return PolynomialLagrangeSimplex(coeff, self.vertices, self.r - 1)

    def degree_elevate(self, s):
        r"""
        Express the polynomial using a higher degree basis.

        Let :math:`p(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}} a_{\nu} l_{\nu, r}(x)` be this
        polynomial, where :math:`\{ l_{\nu, r} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}` is the Lagrange
        basis for :math:`\mathcal{P}_r (T)`. Let :math:`\{ l_{\nu, s} \}_{\substack{\nu \in \mathbb{N}_0^m
        \\ |\nu| \leq s}}, s \geq r` be the Lagrange basis for :math:`\mathcal{P}_s (T)`. Then this function
        returns a polynomial :math:`q(x)`

        .. math:: q(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq s}} \tilde{a}_{\nu} l_{\nu, s}(x),

        such that :math:`p(x) = q(x) \, \forall x \in T`.

        :param int s: New degree for the polynomial basis the polynomial should be expressed in.
        :return: Elevation of this polynomial to the higher degree basis.
        :rtype: :class:`PolynomialLagrangeSimplex`.
        """
        assert s >= self.degree()
        if s == self.degree():
            return PolynomialLagrangeSimplex(self.coeff, self.vertices, self.r)
        p = self._unit_simplex_polynomial.degree_elevate(s)
        return PolynomialLagrangeSimplex(p.coeff, self.vertices, s)

    def to_monomial_basis(self):
        """
        Compute the monomial representation of this polynomial.

        :return: This polynomial expressed in the monomial basis.
        :rtype: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`.
        """
        if self.n == 1:
            a = np.empty(get_dimension(self.r, self.m))
        else:
            a = np.empty((get_dimension(self.r, self.m), self.n))

        q = dual_monomial_basis(self.r, self.m)
        for i in range(len(q)):
            a[i] = q[i](self)

        return Polynomial(a, self.r, self.m)

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
            basis_strs = lagrange_basis_simplex_latex(self.r, self.vertices)
            for i in range(len(basis_strs)):
                if len(basis_strs[i]) > 3:
                    basis_strs[i] = "(" + basis_strs[i] + ")"
            return str_dot_product(coeff_strs, basis_strs)
        except TypeError:
            coeff_strs = [str_number(c, latex_fraction=True) for c in self.coeff]
            basis_strs = lagrange_basis_simplex_latex(self.r, self.vertices)
            for i in range(len(basis_strs)):
                if len(basis_strs[i]) > 3:
                    basis_strs[i] = "(" + basis_strs[i] + ")"
            return str_dot_product(coeff_strs, basis_strs)

    @staticmethod
    def _generate_function_specific_name(a, vertices):
        """
        Generate name for a general function evaluating a polynomial.

        :param a: Coefficients for the polynomial used to generate a unique name.
        :return: Name for the function.
        :rtype: str
        """
        coeff_hash = hash(str(a))
        if coeff_hash < 0:
            # Cannot have minus sign in name
            coeff_hash *= -1
        vertices_hash = hash(str(vertices))
        if vertices_hash < 0:
            # Cannot have minus sign in name
            vertices_hash *= -1
        return str(coeff_hash) + "_" + str(vertices_hash)

    def code_str(self, fn_name):
        r"""
        Generate a function code string for evaluating this polynomial.

        :param str fn_name: Name for the function in the generated code.
        :return: Code string for evaluating this polynomial.
        :rtype: str
        """
        code = CodeWriter()
        code.wl("def " + fn_name + "(y):")
        code.inc_indent()
        if self.m == 1:
            code.wl("x = " + str(self._a) + " * y + " + str(self._b))
        else:
            code.wl("a = np." + self._a.__repr__())
            code.wl("b = np." + self._b.__repr__())
            code.wl("x = np.dot(a, y) + b")
        poly_eval_code = self._unit_simplex_polynomial.code_str("temp")
        poly_eval_code = poly_eval_code.split('\n')[1:]
        poly_eval_code = "\n".join(poly_eval_code)
        code.verbatim(poly_eval_code)
        code.dec_indent()
        return code.code


def lagrange_basis_fn_simplex(nu, r, vertices):
    r"""
    Generate a Lagrange basis polynomial on an n-dimensional simplex T,
    where n is equal to the length of nu.

    .. math:: l_{\nu, r}(x) = (\bar{l}_{\nu, r} \circ \Phi^{-1})(x),

    where :math:`\bar{l}_{\nu, r}` is the corresponding Lagrange basis polynomial on the (n-dimensional) unit simplex,
    and :math:`\Phi` is the unique affine map which maps the unit simplex to the simplex T.

    :param nu: Multi-index indicating which Lagrange basis polynomial should be generated.
        The polynomial will have the value 1 at the point associated with the multi-index,
        and value 0 at all other points.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: The Lagrange base polynomial on the simplex T, as specified by nu and r.
    :rtype: :class:`PolynomialLagrangeSimplex`.
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
    return PolynomialLagrangeSimplex(coeff, vertices, r)


def lagrange_basis_simplex(r, vertices):
    r"""
    Generate all Lagrange base polynomials for the space :math:`\mathcal{P}_r(T)` where T is an n-dimensional simplex.

    :param int r: Degree of the polynomial space.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: List of base polynomials.
    :rtype: List[:class:`PolynomialLagrangeSimplex`].
    """
    basis = []
    n = dimension(vertices)
    for mi in multiindex.MultiIndexIterator(n, r):
        basis.append(lagrange_basis_fn_simplex(mi, r, vertices))
    return basis


def vector_valued_lagrange_basis_fn_simplex(nu, r, i, vertices, n):
    r"""
    Generate a vector valued Lagrange basis polynomial on an m-dimensional simplex T,
    :math:`l_{\nu, r, i} : T \to \mathbb{R}^n`.

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
    :param vertices: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
        simplex).
    :param int n: Dimension of the target.
    :return: The Lagrange base polynomial on the simplex T as specified by nu, r, i and n.
    :rtype: :class:`PolynomialLagrangeSimplex`.
    """
    if n == 1:
        assert i == 0
        return lagrange_basis_fn_simplex(nu, r, vertices)
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
    return PolynomialLagrangeSimplex(coeff, vertices, r)


def vector_valued_lagrange_basis_simplex(r, vertices, n, ordering="interleaved"):
    r"""
    Generate all Lagrange base polynomials for the space :math:`\mathcal{P}_r(T, \mathbb{R}^n)`, where T is an
    m-dimensional simplex.

    :param int r: Degree of the polynomial space.
    :param vertices: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
        simplex).
    :param int n: Dimension of the target.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: List of base polynomials.
    :rtype: List[:class:`PolynomialLagrangeSimplex`].
    """
    basis = []
    m = dimension(vertices)
    if ordering == "interleaved":
        for mi in multiindex.MultiIndexIterator(m, r):
            for i in range(n):
                basis.append(vector_valued_lagrange_basis_fn_simplex(mi, r, i, vertices, n))
    else:
        for i in range(n):
            for mi in multiindex.MultiIndexIterator(m, r):
                basis.append(vector_valued_lagrange_basis_fn_simplex(mi, r, i, vertices, n))
    return basis


def lagrange_basis_fn_simplex_monomial(nu, r, vertices):
    r"""
    Generate a Lagrange basis polynomial on an n-dimensional simplex T,
    where n is equal to the length of nu, expanded in the monomial basis.

    This is the same polynomial as given by the :func:`lagrange_basis_fn_simplex` function, but expressed in the
    monomial basis.

    :param nu: Multi-index indicating which Lagrange basis polynomial should be generated
        The polynomial will have the value 1 at the point associated with the multi-index,
        and value 0 at all other points.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: The Lagrange base polynomial on the simplex T, as specified by nu and r.
    :rtype: :class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`.
    """
    return lagrange_basis_fn_simplex(nu, r, vertices).to_monomial_basis()


def lagrange_basis_simplex_monomial(r, vertices):
    r"""
    Generate all Lagrange base polynomials for the space :math:`\mathcal{P}_r(T)` where T is an n-dimensional simplex,
    expanded in the monomial basis.

    This is the same set of polynomials as given by the :func:`lagrange_basis_simplex` function, but expressed in the
    monomial basis.

    :param int r: Degree of the polynomial space.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: List of base polynomials.
    :rtype: List[:class:`~polynomials_on_simplices.polynomial.polynomials_monomial_basis.Polynomial`].
    """
    basis = []
    n = dimension(vertices)
    for mi in multiindex.MultiIndexIterator(n, r):
        basis.append(lagrange_basis_fn_simplex_monomial(mi, r, vertices))
    return basis


def dual_lagrange_basis_fn_simplex(mu, r, vertices):
    r"""
    Generate a dual basis function to the Lagrange polynomial basis, i.e. the linear map
    :math:`q_{\mu, r} : \mathcal{P}_r(T) \to \mathbb{R}` that satisfies

    .. math::

        q_{\mu, r}(l_{\nu, r}) = \delta_{\mu, \nu},

    where :math:`l_{\nu, r}` is the degree r Lagrange basis polynomial on T indexed by the multi-index :math:`\nu`
    (see :func:`lagrange_basis_fn_simplex`) and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual Lagrange basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...].
    :param int r: Degree of polynomial space.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: The dual Lagrange basis function as specified by mu and r.
    :rtype: Callable :math:`q_{\mu, r}(l)`.
    """
    try:
        m = len(mu)
    except TypeError:
        m = 1
    x_nu = generate_lagrange_point_simplex(vertices, r, mu)
    if m == 1:
        x_nu = x_nu[0]

    def q(p):
        return p(x_nu)
    return q


def dual_lagrange_basis_simplex(r, vertices):
    r"""
    Generate all dual Lagrange base functions for the space :math:`\mathcal{P}_r(T)`, where T is an n-dimensional
    simplex (i.e. the Lagrange basis for :math:`\mathcal{P}_r(T)^*`).

    See :func:`dual_lagrange_basis_fn_simplex`.

    :param int r: Degree of the polynomial space.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: List of dual base functions.
    :rtype: List[callable `q(l)`].
    """
    dual_basis = []
    n = dimension(vertices)
    for mi in multiindex.MultiIndexIterator(n, r):
        dual_basis.append(dual_lagrange_basis_fn_simplex(mi, r, vertices))
    return dual_basis


def dual_vector_valued_lagrange_basis_fn_simplex(mu, r, i, vertices, n):
    r"""
    Generate a dual basis function to the vector valued Lagrange polynomial basis, i.e. the linear map
    :math:`q_{\mu, r, i} : \mathcal{P}_r(T, \mathbb{R}^n) \to \mathbb{R}` that satisfies

    .. math::

        q_{\mu, r, i}(l_{\nu, r, j}) = \delta_{\mu, \nu} \delta_{i, j},

    where :math:`l_{\nu, r, j}` is the degree r vector valued Lagrange basis polynomial indexed by the
    multi-index :math:`\nu` with a non-zero i:th component (see :func:`vector_valued_lagrange_basis_fn_simplex`) and

    .. math::

        \delta_{\mu, \nu} = \begin{cases}
            1 & \mu = \nu \\
            0 & \text{else}
            \end{cases}.

    :param mu: Multi-index indicating which dual Lagrange basis function should be generated.
    :type mu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...].
    :param int r: Degree of polynomial space.
    :param int i: Integer indicating which dual Lagrange basis function should be generated.
    :param vertices: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
        simplex).
    :param int n: Dimension of the target.
    :return: The dual Lagrange basis function as specified by mu, r and i.
    :rtype: Callable :math:`q_{\mu, r, i}(l)`.
    """
    if n == 1:
        assert i == 0
        return dual_lagrange_basis_fn_simplex(mu, r, vertices)
    assert i >= 0
    assert i < n

    qs = dual_lagrange_basis_fn_simplex(mu, r, vertices)

    def q(p):
        assert p.target_dimension() == n
        return qs(p[i])

    return q


def dual_vector_valued_lagrange_basis_simplex(r, vertices, n, ordering="interleaved"):
    r"""
    Generate all dual Lagrange base functions for the space :math:`\mathcal{P}_r(T, \mathbb{R}^n)`, where T is an
    m-dimensional simplex (i.e. the Lagrange basis for :math:`\mathcal{P}_r(T, \mathbb{R}^n)^*`).

    See :func:`dual_vector_valued_lagrange_basis_fn_simplex`.

    :param int r: Degree of the polynomial space.
    :param vertices: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
        simplex).
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
    m = dimension(vertices)
    if ordering == "interleaved":
        for mi in multiindex.MultiIndexIterator(m, r):
            for i in range(n):
                dual_basis.append(dual_vector_valued_lagrange_basis_fn_simplex(mi, r, i, vertices, n))
    else:
        for i in range(n):
            for mi in multiindex.MultiIndexIterator(m, r):
                dual_basis.append(dual_vector_valued_lagrange_basis_fn_simplex(mi, r, i, vertices, n))
    return dual_basis


def lagrange_basis_fn_simplex_latex(nu, r, vertices):
    r"""
    Generate Latex string for a Lagrange basis polynomial on an n-dimensional simplex T,
    where n is equal to the length of nu.

    :param nu: Multi-index indicating which Lagrange basis polynomial we should generate Latex string for.
    :type nu: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
    :param int r: Degree of polynomial.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: Latex string for the Lagrange base polynomial on T, as specified by nu and r.
    :rtype: str
    """
    return lagrange_basis_fn_simplex(nu, r, vertices).to_monomial_basis().latex_str()


def lagrange_basis_simplex_latex(r, vertices):
    r"""
    Generate Latex strings for all Lagrange base polynomials for the space :math:`\mathcal{P}_r(T)` where T is an
    n-dimensional simplex.

    :param int r: Degree of the polynomial space.
    :param vertices: Vertices of the simplex T ((n + 1) x n matrix where row i contains the i:th vertex of the
        simplex).
    :return: List of Latex strings for each Lagrange base polynomial.
    :rtype: List[str]
    """
    basis_latex_strings = []
    n = dimension(vertices)
    for mi in multiindex.MultiIndexIterator(n, r):
        basis_latex_strings.append(lagrange_basis_fn_simplex_latex(mi, r, vertices))
    return basis_latex_strings


def zero_polynomial_simplex(vertices, r=0, n=1):
    r"""
    Get the Lagrange polynomial :math:`l \in \mathcal{P}(T, \mathbb{R}^n)` which is identically zero, where T is an
    m-dimensional simplex.

    :param vertices: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
        simplex).
    :param int r: The zero polynomial will be expressed in the Lagrange basis for
        :math:`\mathcal{P}_r(T, \mathbb{R}^n)`.
    :param int n: Dimension of the polynomial target.
    :return: The zero polynomial.
    :rtype: :class:`PolynomialLagrangeSimplex`.
    """
    try:
        m = len(vertices[0])
    except TypeError:
        m = 1
    dim = get_dimension(r, m)
    if n == 1:
        coeff = np.zeros(dim)
    else:
        coeff = np.zeros((dim, n))
    return PolynomialLagrangeSimplex(coeff, vertices, r)


def unit_polynomial_simplex(vertices, r=0, n=1):
    r"""
    Get the Lagrange polynomial :math:`l \in \mathcal{P}(T, \mathbb{R}^n)` which is identically one, where T is an
    m-dimensional simplex.

    :param vertices: Vertices of the simplex T ((m + 1) x m matrix where row i contains the i:th vertex of the
        simplex).
    :param int r: The unit polynomial will be expressed in the Lagrange basis for
        :math:`\mathcal{P}_r(T, \mathbb{R}^n)`.
    :param int n: Dimension of the polynomial target.
    :return: The unit polynomial.
    :rtype: :class:`PolynomialLagrangeSimplex`.
    """
    # The Lagrange basis forms a partition of unity, so we just need to set all coefficients to 1
    try:
        m = len(vertices[0])
    except TypeError:
        m = 1
    dim = get_dimension(r, m)
    if n == 1:
        coeff = np.ones(dim)
    else:
        coeff = np.ones((dim, n))
    return PolynomialLagrangeSimplex(coeff, vertices, r)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
