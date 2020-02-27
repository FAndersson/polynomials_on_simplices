r"""Discontinuous Galerkin finite elements (piecewise polynomials) on a simplicial domain (triangulation)
:math:`\mathcal{T}`, i.e. elements of :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` or :math:`D\mathcal{P}_r (\mathcal{T})`,
expressed using the Lagrange polynomial basis.
"""

import numbers

import numpy as np

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.geometry.mesh.simplicial_complex import simplex_vertices
from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial import (
    PiecewisePolynomialBase, generate_inverse_local_to_global_map, generate_local_to_global_map)
from polynomials_on_simplices.polynomial.polynomials_simplex_lagrange_basis import (
    PolynomialLagrangeSimplex, generate_lagrange_point_simplex, get_dimension, lagrange_basis_simplex,
    zero_polynomial_simplex)
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import unique_identifier_lagrange_basis


class PiecewisePolynomialLagrange(PiecewisePolynomialBase):
    r"""
    Implementation of the abstract piecewise polynomial base class using the Lagrange polynomial basis on the unit
    simplex.

    .. math:: p(x) = \sum_{i = 1}^N a_i \phi_i(x),

    where the basis :math:`\{ \phi_i \}_{i = 1}^N` for the space of piecewise polynomials is constructed from the
    Lagrange polynomial basis and the local-to-global map. See
    :class:`~polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial.PiecewisePolynomialBase` and
    :func:`~polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial.generate_local_to_global_map` for details.
    """

    def __init__(self, coeff, triangles, vertices, r, tau=None, boundary_simplices=None, keep_boundary_dofs_last=False,
                 support=None, bsp_tree=None, basis_polynomials=None):
        r"""
        :param coeff: Coefficients for the piecewise polynomial in the :math:`\{ \phi_i \}_{i = 1}^N` basis derived
            from the Lagrange basis for :math:`\mathcal{P}_r (\Delta_c^m)` and the local-to-global map :math:`\tau`.
        :type coeff: List[Union[Scalar, Vector]]
        :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
            array of indices).
        :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
        :param int r: Degree of each polynomial in the piecewise polynomial.
        :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
            basis function. Will be generated if not supplied.
        :type tau: Optional[Callable :math:`\tau(j, \nu)`]
        :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial function should
            vanish (for an element of :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately
            (if `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
            vertex indices of the vertices that form the simplex.
        :type boundary_simplices: List[List[int]]
        :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
            boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
            boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
            :math:`D\mathcal{P}_r (\mathcal{T})` in a practical way.
        :param Optional[Set[int]] support: Indices of the triangles in the triangulation where the piecewise polynomial
            is supported. Will be generated if not supplied.
        :param bsp_tree: Optional implementation detail. A binary space partitioning tree built
            around the triangulation for quicker lookup of triangle a point lies in. Will be generated if not supplied.
        :param basis_polynomials: Precomputed basis polynomials for each triangle in the given triangulation. Will be
            generated where necessary if not supplied.
        :type basis_polynomials: Optional[Dict[int, List[
            :class:`~polynomials_on_simplices.polynomial.polynomials_simplex_lagrange_basis.PolynomialLagrangeSimplex`]]]
        """
        PiecewisePolynomialBase.__init__(self, coeff, triangles, vertices, r, tau, boundary_simplices,
                                         keep_boundary_dofs_last, support, bsp_tree)

        # Compute basis polynomials
        if basis_polynomials is not None:
            for j in self.triangles_in_the_support:
                self._basis_polynomials[j] = basis_polynomials[j]
        else:
            for j in self.triangles_in_the_support:
                self._basis_polynomials[j] = lagrange_basis_simplex(self.r,
                                                                    simplex_vertices(triangles[j], self.vertices))

    def basis(self):
        r"""
        Get basis for the space :math:`\mathcal{P}_r (\Delta_c^m)` used to express the piecewise polynomial.

        :return: Unique identifier for the basis used.
        :rtype: str
        """
        return unique_identifier_lagrange_basis()

    def __mul__(self, other):
        """
        Multiplication of this piecewise polynomial with another piecewise polynomial (only if n = 1), a scalar, or
        a vector (only if n = 1), self * other.

        :param other: Piecewise polynomial, scalar or vector we should multiply this piecewise polynomial with.
        :return: Product of this piecewise polynomial with other.
        :rtype: Instance of self.__class__
        """
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            return self.multiply_with_constant(other)
        # Multiplication of two piecewise polynomials
        # Multiplied piecewise polynomials need to have the same domain dimension
        assert self.domain_dimension() == other.domain_dimension()
        # And the same underlying triangle mesh
        assert self.triangles.data == other.triangles.data
        assert self.vertices.data == other.vertices.data
        assert ((self.boundary_simplices is None and other.boundary_simplices is None)
                or (self.boundary_simplices.data == other.boundary_simplices.data))
        assert self.keep_boundary_dofs_last == other.keep_boundary_dofs_last
        # Cannot multiply two vector valued piecewise polynomials
        assert self.target_dimension() == 1
        assert other.target_dimension() == 1
        m = self.domain_dimension()
        r = self.degree() + other.degree()
        # New local-to-global map for the product piecewise polynomial
        if self.keep_boundary_dofs_last:
            tau, num_dofs, num_internal_dofs = self.__class__.generate_local_to_global_map(self.triangles, r,
                                                                                           self.boundary_simplices,
                                                                                           self.keep_boundary_dofs_last)
        else:
            tau, num_dofs = self.__class__.generate_local_to_global_map(self.triangles, r, self.boundary_simplices)
        coeff = np.zeros(num_dofs)
        support = self.support().intersection(other.support())
        for j in support:
            tri_vertices = simplex_vertices(self.triangles[j], self.vertices)
            for nu in multiindex.generate_all(m, r):
                x_nu = generate_lagrange_point_simplex(tri_vertices, r, nu)
                i = tau(j, nu)
                # Note: Need to make sure that we evaluate on the j:th triangle here, since the value of the
                # piecewise polynomial is ambiguous on the boundary between two neighbouring triangles
                coeff[i] = self.evaluate_on_simplex(j, x_nu) * other.evaluate_on_simplex(j, x_nu)
        return self.__class__(coeff, self.triangles, self.vertices, r, tau, self.boundary_simplices,
                              self.keep_boundary_dofs_last, support, self._bsp_tree)

    def __pow__(self, exp):
        r"""
        Raise the piecewise polynomial to a power.

        .. math::

            (p^{\mu})(x) = p(x)^{\mu} =  p_1(x)^{\mu_1} p_2(x)^{\mu_2} \ldots p_n(x)^{\mu_n}.

        :param exp: Power we want the raise the piecewise polynomial to (natural number or multi-index depending on
            the dimension of the target of the piecewise polynomial).
        :type exp: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
        :return: This piecewise polynomial raised to the given power.
        :rtype: Instance of self.__class__
        """
        if isinstance(exp, numbers.Integral):
            assert exp >= 0
            assert self.target_dimension() == 1
            if exp == 0:
                return self.get_unit_piecewise_polynomial_lagrange(self.triangles, self.vertices, 1, None,
                                                                   self.boundary_simplices,
                                                                   self.keep_boundary_dofs_last, self._bsp_tree)
            if exp == 1:
                return self.__class__(self.coeff, self.triangles, self.vertices, self.r, self.tau,
                                      self.boundary_simplices, self.keep_boundary_dofs_last, self.support(),
                                      self._bsp_tree, self._basis_polynomials)
            return self * self**(exp - 1)
        else:
            assert len(exp) == self.target_dimension()
            assert [entry >= 0 for entry in exp]
            m = self.domain_dimension()
            r = self.degree() * multiindex.norm(exp)
            if r == 0:
                return self.get_unit_piecewise_polynomial_lagrange(self.triangles, self.vertices, 1, None,
                                                                   self.boundary_simplices,
                                                                   self.keep_boundary_dofs_last, self._bsp_tree)
            # New local-to-global map for the resulting piecewise polynomial
            if self.keep_boundary_dofs_last:
                tau, num_dofs, num_internal_dofs = self.__class__.generate_local_to_global_map(
                    self.triangles, r, self.boundary_simplices, self.keep_boundary_dofs_last)
            else:
                tau, num_dofs = self.__class__.generate_local_to_global_map(self.triangles, r, self.boundary_simplices)
            coeff = np.zeros(num_dofs)
            for j in self.support():
                tri_vertices = simplex_vertices(self.triangles[j], self.vertices)
                for nu in multiindex.generate_all(m, r):
                    x_nu = generate_lagrange_point_simplex(tri_vertices, r, nu)
                    i = tau(j, nu)
                    coeff[i] = multiindex.power(self.evaluate_on_simplex(j, x_nu), exp)
            return self.__class__(coeff, self.triangles, self.vertices, r, tau, self.boundary_simplices,
                                  self.keep_boundary_dofs_last, self.support(), self._bsp_tree)

    def degree_elevate(self, s):
        r"""
        Express the piecewise polynomial using a higher degree polynomial basis.

        Let :math:`p(x)` be this piecewise polynomial.
        Let :math:`\{ \bar{\varphi}_{\nu, r} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq r}}` be the
        polynomial basis for :math:`\mathcal{P}_r(\Delta_c^m)` used for this piecewise polynomial, and let
        :math:`\{ \bar{\varphi}_{\nu, s} \}_{\substack{\nu \in \mathbb{N}_0^m \\ |\nu| \leq s}}, s \geq r` be the
        corresponding higher degree polynomial basis for :math:`\mathcal{P}_s(\Delta_c^m)`. Then this function
        returns a piecewise polynomial :math:`q` using the basis :math:`\{ \bar{\varphi}_{\nu, s} \}` such that
        :math:`p(x) = q(x) \, \forall x \in \mathbb{R}^m`.

        :param int s: New degree for the basic polynomial basis the piecewise polynomial should use.
        :return: Elevation of this piecewise polynomial to the higher degree basis.
        :rtype: Instance of self.__class__
        """
        assert s >= self.degree()
        if s == self.degree():
            return self.__class__(self.coeff, self.triangles, self.vertices, self.r, self.tau, self.boundary_simplices,
                                  self.keep_boundary_dofs_last, self.support(), self._bsp_tree, self._basis_polynomials)
        # New local-to-global map for the higher degree polynomial
        if self.keep_boundary_dofs_last:
            tau, num_dofs, num_internal_dofs = self.__class__.generate_local_to_global_map(self.triangles, s,
                                                                                           self.boundary_simplices,
                                                                                           self.keep_boundary_dofs_last)
        else:
            tau, num_dofs = self.__class__.generate_local_to_global_map(self.triangles, s, self.boundary_simplices)
        m = self.domain_dimension()
        n = self.target_dimension()
        if n == 1:
            coeff = np.zeros(num_dofs)
        else:
            coeff = np.zeros((num_dofs, n))
        mis = multiindex.generate_all(m, s)
        for j in range(len(self.triangles)):
            p = self.restrict_to_simplex(j).degree_elevate(s)
            for k in range(len(p.coeff)):
                i = tau(j, mis[k])
                coeff[i] = p.coeff[k]
        return self.__class__(coeff, self.triangles, self.vertices, s, tau, self.boundary_simplices,
                              self.keep_boundary_dofs_last, self.support(), self._bsp_tree)

    def restrict_to_simplex(self, i):
        r"""
        Restriction of the piecewise polynomial to a specified simplex :math:`T_i \in \mathcal{T}`.

        :param int i: Index of the simplex we want to restrict the piecewise polynomial to (in :math:`0, 1, \ldots,
            | \mathcal{T} | - 1`).
        :return: Polynomial which agrees with the piecewise polynomial on the simplex :math:`T_i`.
        :rtype: :class:`~.polynomial.polynomials_simplex_lagrange_basis.PolynomialLagrangeSimplex`.
        """
        assert i < len(self.triangles)
        tri_vertices = simplex_vertices(self.triangles[i], self.vertices)
        m = self.domain_dimension()
        n = self.target_dimension()
        r = self.degree()
        if i not in self.support():
            return zero_polynomial_simplex(tri_vertices, r, n)
        dim = get_dimension(r, m)
        if n == 1:
            coeff = np.zeros(dim)
        else:
            coeff = np.zeros((dim, n))
        mis = multiindex.generate_all(m, r)
        for k in range(len(mis)):
            idx = self.tau(i, mis[k])
            coeff[k] = self.coeff[idx]
        return PolynomialLagrangeSimplex(coeff, tri_vertices, r)

    @staticmethod
    def get_unit_piecewise_polynomial_lagrange(triangles, vertices, n=1, tau=None, boundary_simplices=None,
                                               keep_boundary_dofs_last=False, bsp_tree=None):
        r"""
        Get the piecewise polynomial of degree r on the given triangulation :math:`\mathcal{T}`, where the polynomials
        on each simplex is expressed in the Lagrange basis, which is identically one. See
        :func:`unit_piecewise_polynomial_lagrange`.
        """
        return unit_piecewise_polynomial_lagrange(triangles, vertices, n, tau, boundary_simplices,
                                                  keep_boundary_dofs_last, bsp_tree)


def piecewise_polynomial_lagrange_basis_fn(triangles, vertices, r, i, tau=None, num_dofs=None, boundary_simplices=None,
                                           keep_boundary_dofs_last=False, bsp_tree=None, basis_polynomials=None):
    r"""
    Generate a basis function for the space of piecewise polynomials of degree r on the given triangulation
    :math:`\mathcal{T}`, where the polynomials on each simplex is expressed in the Lagrange basis.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int r: Degree of each polynomial in the piecewise polynomial.
    :param int i: Index of the basis function that should be generated.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param Optional[int] num_dofs: Dimension of the space of piecewise polynomials on the given triangulation. Will
        be computed if not supplied.
    :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial functions should
        vanish (for :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`D\mathcal{P}_r (\mathcal{T})` in a practical way.
    :param bsp_tree: Optional implementation detail. A binary space partitioning tree built around the triangulation
        :math:`\mathcal{T}` for quicker lookup of triangle a point lies in. Will be generated if not supplied.
    :param basis_polynomials: Precomputed basis polynomials for each triangle in the given triangulation. Will be
        generated where necessary if not supplied.
    :type basis_polynomials: Optional[Dict[int, List[
        :class:`~polynomials_on_simplices.polynomial.polynomials_simplex_lagrange_basis.PolynomialLagrangeSimplex`]]]
    :return: Basis function.
    :rtype: :class:`PiecewisePolynomialLagrange`.
    """
    if tau is None or num_dofs is None:
        if keep_boundary_dofs_last:
            tau, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                            keep_boundary_dofs_last)
        else:
            tau, num_dofs = generate_local_to_global_map(triangles, r, boundary_simplices)

    coeff = np.zeros(num_dofs)
    coeff[i] = 1
    return PiecewisePolynomialLagrange(coeff, triangles, vertices, r, tau, boundary_simplices, keep_boundary_dofs_last,
                                       bsp_tree=bsp_tree, basis_polynomials=basis_polynomials)


def piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau=None, num_dofs=None, boundary_simplices=None,
                                        keep_boundary_dofs_last=False, bsp_tree=None):
    r"""
    Generate all basis functions for the space of piecewise polynomials of degree r on the given
    triangulation :math:`\mathcal{T}`, where the polynomials on each simplex is expressed in the Lagrange basis.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int r: Degree of each polynomial in the piecewise polynomial.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param Optional[int] num_dofs: Dimension of the space of piecewise polynomials on the given triangulation. Will
        be computed if not supplied.
    :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial functions should
        vanish (for :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`D\mathcal{P}_r (\mathcal{T})` in a practical way.
    :param bsp_tree: Optional implementation detail. A binary space partitioning tree built around the triangulation
        :math:`\mathcal{T}` for quicker lookup of triangle a point lies in. Will be generated if not supplied.
    :rtype: List[:class:`PiecewisePolynomialLagrange`].
    """
    basis = []
    if tau is None or num_dofs is None:
        if keep_boundary_dofs_last:
            tau, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                            keep_boundary_dofs_last)
        else:
            tau, num_dofs = generate_local_to_global_map(triangles, r, boundary_simplices)
    basis_polynomials = {}
    for j in range(len(triangles)):
        tri_vertices = simplex_vertices(triangles[j], vertices)
        basis_polynomials[j] = lagrange_basis_simplex(r, tri_vertices)
    for i in range(num_dofs):
        basis.append(piecewise_polynomial_lagrange_basis_fn(triangles, vertices, r, i, tau, num_dofs,
                                                            boundary_simplices, keep_boundary_dofs_last,
                                                            bsp_tree=bsp_tree, basis_polynomials=basis_polynomials))
    return basis


def dual_piecewise_polynomial_lagrange_basis_fn(triangles, vertices, r, i, tau=None, num_dofs=None, tau_inv=None,
                                                boundary_simplices=None, keep_boundary_dofs_last=False):
    r"""
    Generate a dual basis function to the basis for the space of piecewise polynomials of degree r on the given
    triangulation :math:`\mathcal{T}, D \mathcal{P}_r(\mathcal{T})` or :math:`D \mathcal{P}_{r, 0}(\mathcal{T})`, where
    the polynomials on each simplex is expressed in the Lagrange basis.
    I.e. the linear map :math:`\phi_i^* : D \mathcal{P}_r(\mathcal{T}) \to \mathbb{R}` that satisfies

    .. math:: \phi_i^* (\phi_j) = \delta_{ij},

    where :math:`\phi_j` is the j:th Lagrange basis function for the space of piecewise polynomials of degree r
    (see :func:`piecewise_polynomial_lagrange_basis_fn`).

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int r: Degree of each polynomial in the piecewise polynomial.
    :param int i: Index of the dual basis function that should be generated.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param Optional[int] num_dofs: Dimension of the space of piecewise polynomials on the given triangulation. Will
        be computed if not supplied.
    :param tau_inv: Inverse of the local-to-global map. Will be generated if not supplied.
    :type tau_inv: Optional[Callable :math:`\tau^{-1}(i)`]
    :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial functions should
        vanish (for :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`D\mathcal{P}_r (\mathcal{T})` in a practical way.
    :return: The i:th dual Lagrange basis function as specified by mu and r.
    :rtype: Callable :math:`\phi_i^*(p)`.
    """
    if tau is None or num_dofs is None:
        if keep_boundary_dofs_last:
            tau, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                            keep_boundary_dofs_last)
        else:
            tau, num_dofs = generate_local_to_global_map(triangles, r, boundary_simplices)
    if tau_inv is None:
        tau_inv = generate_inverse_local_to_global_map(tau, len(triangles), num_dofs, r, len(vertices[0]))
    j, nu = tau_inv(i)
    tri_vertices = simplex_vertices(triangles[j], vertices)
    x_nu = generate_lagrange_point_simplex(tri_vertices, r, nu)

    def phi_star(p):
        if isinstance(p, PiecewisePolynomialBase):
            # More efficient than simply evaluating the piecewise polynomial at x_nu
            return p.evaluate_on_simplex(j, x_nu)
        else:
            return p(x_nu)

    return phi_star


def dual_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau=None, num_dofs=None, tau_inv=None,
                                             boundary_simplices=None, keep_boundary_dofs_last=False):
    r"""
    Generate all dual basis functions to the basis of piecewise polynomials of degree r on the given
    triangulation :math:`\mathcal{T}, D \mathcal{P}_r(\mathcal{T})` or :math:`D \mathcal{P}_{r, 0}(\mathcal{T})`, where
    the polynomials on each simplex is expressed in the Lagrange basis.

    See :func:`dual_piecewise_polynomial_lagrange_basis_fn`.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int r: Degree of each polynomial in the piecewise polynomial.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param Optional[int] num_dofs: Dimension of the space of piecewise polynomials on the given triangulation. Will
        be computed if not supplied.
    :param tau_inv: Inverse of the local-to-global map. Will be generated if not supplied.
    :type tau_inv: Optional[Callable :math:`\tau^{-1}(i)`]
    :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial functions should
        vanish (for :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`D\mathcal{P}_r (\mathcal{T})` in a practical way.
    :return: List of dual base functions.
    :rtype: List[Callable :math:`\phi_i^*(p)`].
    """
    basis = []
    if tau is None or num_dofs is None:
        if keep_boundary_dofs_last:
            tau, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                            keep_boundary_dofs_last)
        else:
            tau, num_dofs = generate_local_to_global_map(triangles, r, boundary_simplices)
    if tau_inv is None:
        try:
            m = len(vertices[0])
        except TypeError:
            m = 1
        tau_inv = generate_inverse_local_to_global_map(tau, len(triangles), num_dofs, r, m)
    for i in range(num_dofs):
        basis.append(dual_piecewise_polynomial_lagrange_basis_fn(triangles, vertices, r, i, tau, num_dofs, tau_inv,
                                                                 boundary_simplices, keep_boundary_dofs_last))
    return basis


def zero_piecewise_polynomial_lagrange(triangles, vertices, n=1, tau=None, boundary_simplices=None,
                                       keep_boundary_dofs_last=False, bsp_tree=None):
    r"""
    Get the piecewise polynomial of degree r on the given triangulation :math:`\mathcal{T}`, where the polynomials on
    each simplex is expressed in the Lagrange basis, which is identically zero.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int n: Dimension of the target of the piecewise polynomial.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial functions should
        vanish (for :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`D\mathcal{P}_r (\mathcal{T})` in a practical way.
    :param bsp_tree: Optional implementation detail. A binary space partitioning tree built around the triangulation
        :math:`\mathcal{T}` for quicker lookup of triangle a point lies in. Will be generated if not supplied.
    :return: The zero piecewise polynomial.
    :rtype: :class:`PiecewisePolynomialLagrange`.
    """
    dim = len(triangles)
    if n == 1:
        coeff = np.zeros(dim)
    else:
        coeff = np.zeros((dim, n))
    support = set()
    return PiecewisePolynomialLagrange(coeff, triangles, vertices, 0, tau, boundary_simplices, keep_boundary_dofs_last,
                                       support, bsp_tree)


def unit_piecewise_polynomial_lagrange(triangles, vertices, n=1, tau=None, boundary_simplices=None,
                                       keep_boundary_dofs_last=False, bsp_tree=None):
    r"""
    Get the piecewise polynomial of degree r on the given triangulation :math:`\mathcal{T}`, where the polynomials
    on each simplex is expressed in the Lagrange basis, which is identically one.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int n: Dimension of the target of the piecewise polynomial.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial functions should
        vanish (for :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`D\mathcal{P}_r (\mathcal{T})` in a practical way.
    :param bsp_tree: Optional implementation detail. A binary space partitioning tree built around the triangulation
        :math:`\mathcal{T}` for quicker lookup of triangle a point lies in. Will be generated if not supplied.
    :return: The unit piecewise polynomial.
    :rtype: :class:`PiecewisePolynomialLagrange`.
    """
    dim = len(triangles)
    if n == 1:
        coeff = np.ones(dim)
    else:
        coeff = np.ones((dim, n))
    support = set(range(len(triangles)))
    return PiecewisePolynomialLagrange(coeff, triangles, vertices, 0, tau, boundary_simplices, keep_boundary_dofs_last,
                                       support, bsp_tree)
