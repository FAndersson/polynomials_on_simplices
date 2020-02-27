r"""Lagrange finite elements (continuous piecewise polynomials) on a simplicial domain (triangulation)
:math:`\mathcal{T}`, i.e. elements of :math:`C\mathcal{P}_{r, 0} (\mathcal{T})` or :math:`C\mathcal{P}_r (\mathcal{T})`,
expressed using the Lagrange polynomial basis.
"""

import numpy as np

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.geometry.mesh.simplicial_complex import simplex_vertices
from polynomials_on_simplices.piecewise_polynomial.continuous_piecewise_polynomial import (
    ContinuousPiecewisePolynomialBase, generate_local_to_global_map, generate_local_to_global_preimage_map)
from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial import PiecewisePolynomialBase
from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial_lagrange_basis import (
    PiecewisePolynomialLagrange)
from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial_lagrange_basis import \
    generate_local_to_global_map as generate_local_to_global_map_dp
from polynomials_on_simplices.polynomial.polynomials_simplex_lagrange_basis import (
    generate_lagrange_point_simplex, lagrange_basis_simplex)


class ContinuousPiecewisePolynomialLagrange(PiecewisePolynomialLagrange, ContinuousPiecewisePolynomialBase):
    r"""
    Implementation of the abstract continuous piecewise polynomial base class using the Bernstein polynomial basis on
    the unit simplex.

    .. math:: p(x) = \sum_{i = 1}^N a_i \phi_i(x),

    where the basis :math:`\{ \phi_i \}_{i = 1}^N` for the space of continuous piecewise polynomials is constructed
    from the Lagrange polynomial basis and the local-to-global map. See
    :class:`~polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial.PiecewisePolynomialBase` and
    :func:`~polynomials_on_simplices.piecewise_polynomial.continuous_piecewise_polynomial.generate_local_to_global_map`
    for details.
    """

    def __init__(self, coeff, triangles, vertices, r, tau=None, boundary_simplices=None, keep_boundary_dofs_last=False,
                 support=None, bsp_tree=None, basis_polynomials=None):
        r"""
        :param coeff: Coefficients for the continuous piecewise polynomial in the :math:`\{ \phi_i \}_{i = 1}^N` basis
            derived from the Lagrange basis for :math:`\mathcal{P}_r (\Delta_c^m)` and the local-to-global map
            :math:`\tau`.
        :type coeff: List[Union[Scalar, Vector]]
        :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
            array of indices).
        :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
        :param int r: Degree of each polynomial in the continuous piecewise polynomial.
        :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
            basis function, in a way that makes sure that the piecewise polynomial is continuous. Will be generated
            if not supplied.
        :type tau: Optional[Callable :math:`\tau(j, \nu)`]
        :param boundary_simplices: List of simplices or subsimplices on which the continuous piecewise polynomial
            function should vanish (for an element of :math:`C\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be
            treated separately (if `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is
            specified as a list of vertex indices of the vertices that form the simplex.
        :type boundary_simplices: List[List[int]]
        :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
            boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
            boundary simplices last is useful for handling :math:`C\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
            :math:`C\mathcal{P}_r (\mathcal{T})` in a practical way.
        :param Optional[Set[int]] support: Indices of the triangles in the triangulation where the continuous piecewise
            polynomial is supported. Will be generated if not supplied.
        :param bsp_tree: Optional implementation detail. A binary space partitioning tree built
            around the triangulation for quicker lookup of triangle a point lies in. Will be generated if not supplied.
        :param basis_polynomials: Precomputed basis polynomials for each triangle in the given triangulation. Will be
            generated where necessary if not supplied.
        :type basis_polynomials: Optional[Dict[int, List[
            :class:`~.polynomial.polynomials_simplex_lagrange_basis.PolynomialLagrangeSimplex`]]]
        """
        # TODO: Verify that the local-to-global map generates a single-valued (continuous) piecewise polynomial?
        if tau is None:
            if keep_boundary_dofs_last:
                tau, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                                keep_boundary_dofs_last)
            else:
                tau, num_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                             keep_boundary_dofs_last)
            assert num_dofs == len(coeff)
        PiecewisePolynomialLagrange.__init__(self, coeff, triangles, vertices, r, tau, boundary_simplices,
                                             keep_boundary_dofs_last, support, bsp_tree, basis_polynomials)

    def weak_partial_derivative(self, i=0):
        """
        Compute the i:th weak partial derivative of the continuous piecewise polynomial.

        :param int i: Index of partial derivative.
        :return: i:th weak partial derivative of this continuous piecewise polynomial.
        :rtype: :class:`~.piecewise_polynomial.piecewise_polynomial_lagrange_basis.PiecewisePolynomialLagrange`.
        """
        # New local-to-global map for the resulting piecewise polynomial
        r = self.r - 1
        if self.keep_boundary_dofs_last:
            tau, num_dofs, num_internal_dofs = generate_local_to_global_map_dp(self.triangles, r,
                                                                               self.boundary_simplices,
                                                                               self.keep_boundary_dofs_last)
        else:
            tau, num_dofs = generate_local_to_global_map_dp(self.triangles, r, self.boundary_simplices,
                                                            self.keep_boundary_dofs_last)

        if self.n == 1:
            coeff = np.empty(num_dofs)
        else:
            coeff = np.empty((num_dofs, self.n))

        mis = multiindex.generate_all(self.m, r)
        for j in range(len(self.triangles)):
            p = self.restrict_to_simplex(j)
            v = simplex_vertices(self.triangles[j], self.vertices)
            for nu in mis:
                k = tau(j, nu)
                if k == -1:
                    continue
                x_nu = generate_lagrange_point_simplex(v, r, nu)
                if self.m == 1:
                    x_nu = x_nu[0]
                coeff[k] = p.partial_derivative(i)(x_nu)

        return PiecewisePolynomialLagrange(coeff, self.triangles, self.vertices, r, tau,
                                           boundary_simplices=self.boundary_simplices,
                                           keep_boundary_dofs_last=self.keep_boundary_dofs_last, support=self.support(),
                                           bsp_tree=self._bsp_tree)

    @staticmethod
    def get_unit_piecewise_polynomial_lagrange(triangles, vertices, n=1, tau=None, boundary_simplices=None,
                                               keep_boundary_dofs_last=False, bsp_tree=None):
        r"""
        Get the continuous piecewise polynomial of degree r on the given triangulation :math:`\mathcal{T}`, where the
        polynomials on each simplex is expressed in the Lagrange basis, which is identically one. See
        :func:`unit_continuous_piecewise_polynomial_lagrange`.
        """
        return unit_continuous_piecewise_polynomial_lagrange(triangles, vertices, n, tau, boundary_simplices,
                                                             keep_boundary_dofs_last, bsp_tree)


def continuous_piecewise_polynomial_lagrange_basis_fn(triangles, vertices, r, i, tau=None, num_dofs=None,
                                                      boundary_simplices=None, keep_boundary_dofs_last=False,
                                                      bsp_tree=None, basis_polynomials=None):
    r"""
    Generate a basis function for the space of continuous piecewise polynomials of degree r on the given triangulation
    :math:`\mathcal{T}`, where the polynomials on each simplex is expressed in the Lagrange basis.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int r: Degree of each polynomial in the continuous piecewise polynomial.
    :param int i: Index of the basis function that should be generated.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param Optional[int] num_dofs: Dimension of the space of continuous piecewise polynomials on the given
        triangulation. Will be computed if not supplied.
    :param boundary_simplices: List of simplices or subsimplices on which the continuous piecewise polynomial functions
        should vanish (for :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`C\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`C\mathcal{P}_r (\mathcal{T})` in a practical way.
    :param bsp_tree: Optional implementation detail. A binary space partitioning tree built around the triangulation
        :math:`\mathcal{T}` for quicker lookup of triangle a point lies in. Will be generated if not supplied.
    :param basis_polynomials: Precomputed basis polynomials for each triangle in the given triangulation. Will be
        generated where necessary if not supplied.
    :type basis_polynomials: Optional[Dict[int, List[
        :class:`~polynomials_on_simplices.polynomial.polynomials_simplex_lagrange_basis.PolynomialLagrangeSimplex`]]]
    :return: Basis function.
    :rtype: :class:`ContinuousPiecewisePolynomialLagrange`.
    """
    if tau is None or num_dofs is None:
        if keep_boundary_dofs_last:
            tau, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                            keep_boundary_dofs_last)
        else:
            tau, num_dofs = generate_local_to_global_map(triangles, r, boundary_simplices)

    coeff = np.zeros(num_dofs)
    coeff[i] = 1
    return ContinuousPiecewisePolynomialLagrange(coeff, triangles, vertices, r, tau, boundary_simplices,
                                                 keep_boundary_dofs_last, bsp_tree=bsp_tree,
                                                 basis_polynomials=basis_polynomials)


def continuous_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau=None, num_dofs=None,
                                                   boundary_simplices=None, keep_boundary_dofs_last=False,
                                                   bsp_tree=None):
    r"""
    Generate all basis functions for the space of continuous piecewise polynomials of degree r on the given
    triangulation :math:`\mathcal{T}`, where the polynomials on each simplex is expressed in the Lagrange basis.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int r: Degree of each polynomial in the continuous piecewise polynomial.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param Optional[int] num_dofs: Dimension of the space of continuous piecewise polynomials on the given
        triangulation. Will be computed if not supplied.
    :param boundary_simplices: List of simplices or subsimplices on which the continuous piecewise polynomial functions
        should vanish (for :math:`C\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`C\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`C\mathcal{P}_r (\mathcal{T})` in a practical way.
    :param bsp_tree: Optional implementation detail. A binary space partitioning tree built around the triangulation
        :math:`\mathcal{T}` for quicker lookup of triangle a point lies in. Will be generated if not supplied.
    :rtype: List[:class:`ContinuousPiecewisePolynomialLagrange`].
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
        basis.append(continuous_piecewise_polynomial_lagrange_basis_fn(triangles, vertices, r, i, tau, num_dofs,
                                                                       boundary_simplices, keep_boundary_dofs_last,
                                                                       bsp_tree=bsp_tree,
                                                                       basis_polynomials=basis_polynomials))
    return basis


def dual_continuous_piecewise_polynomial_lagrange_basis_fn(triangles, vertices, r, i, tau=None, num_dofs=None,
                                                           tau_preim=None, boundary_simplices=None,
                                                           keep_boundary_dofs_last=False):
    r"""
    Generate a dual basis function to the basis for the space of continuous piecewise polynomials of degree r on the
    given triangulation :math:`\mathcal{T}, C \mathcal{P}_r(\mathcal{T})` or :math:`C \mathcal{P}_{r, 0}(\mathcal{T})`,
    where the polynomials on each simplex is expressed in the Lagrange basis.
    I.e. the linear map :math:`\phi_i^* : C \mathcal{P}_r(\mathcal{T}) \to \mathbb{R}` that satisfies

    .. math:: \phi_i^* (\phi_j) = \delta_{ij},

    where :math:`\phi_j` is the j:th Lagrange basis function for the space of continuous piecewise polynomials of
    degree r (see :func:`continuous_piecewise_polynomial_lagrange_basis_fn`).

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int r: Degree of each polynomial in the continuous piecewise polynomial.
    :param int i: Index of the dual basis function that should be generated.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param Optional[int] num_dofs: Dimension of the space of continuous piecewise polynomials on the given
        triangulation. Will be computed if not supplied.
    :param tau_preim: Preimage of the local-to-global map. Will be generated if not supplied.
    :type tau_preim: Optional[Callable :math:`\operatorname{preim}_{\tau}(i)`]
    :param boundary_simplices: List of simplices or subsimplices on which the continuous piecewise polynomial functions
        should vanish (for :math:`C\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`C\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`C\mathcal{P}_r (\mathcal{T})` in a practical way.
    :return: The i:th dual Lagrange basis function as specified by mu and r.
    :rtype: Callable :math:`\phi_i^*(p)`.
    """
    if tau is None or num_dofs is None:
        if keep_boundary_dofs_last:
            tau, num_dofs, num_interior_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                            keep_boundary_dofs_last)
        else:
            tau, num_dofs = generate_local_to_global_map(triangles, r, boundary_simplices)
    if tau_preim is None:
        tau_preim = generate_local_to_global_preimage_map(tau, len(triangles), num_dofs, r, len(vertices[0]))
    assert len(tau_preim({i})) > 0
    j, nu = _get_any_element_from_set(tau_preim({i}))
    tri_vertices = simplex_vertices(triangles[j], vertices)
    x_nu = generate_lagrange_point_simplex(tri_vertices, r, nu)

    def phi_star(p):
        if isinstance(p, PiecewisePolynomialBase):
            # More efficient than simply evaluating the piecewise polynomial at x_nu
            return p.evaluate_on_simplex(j, x_nu)
        else:
            return p(x_nu)

    return phi_star


def dual_continuous_piecewise_polynomial_lagrange_basis(triangles, vertices, r, tau=None, num_dofs=None, tau_preim=None,
                                                        boundary_simplices=None, keep_boundary_dofs_last=False):
    r"""
    Generate all dual basis functions to the basis of continuous piecewise polynomials of degree r on the given
    triangulation :math:`\mathcal{T}, C \mathcal{P}_r(\mathcal{T})` or :math:`C \mathcal{P}_{r, 0}(\mathcal{T})`, where
    the polynomials on each simplex is expressed in the Lagrange basis.

    See :func:`dual_continuous_piecewise_polynomial_lagrange_basis_fn`.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int r: Degree of each polynomial in the continuous piecewise polynomial.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param Optional[int] num_dofs: Dimension of the space of continuous piecewise polynomials on the given
        triangulation. Will be computed if not supplied.
    :param tau_preim: Preimage of the local-to-global map. Will be generated if not supplied.
    :type tau_preim: Optional[Callable :math:`\operatorname{preim}_{\tau}(i)`]
    :param boundary_simplices: List of simplices or subsimplices on which the continuous piecewise polynomial functions
        should vanish (for :math:`C\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`C\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`C\mathcal{P}_r (\mathcal{T})` in a practical way.
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
    if tau_preim is None:
        try:
            m = len(vertices[0])
        except TypeError:
            m = 1
        tau_preim = generate_local_to_global_preimage_map(tau, len(triangles), num_dofs, r, m)
    for i in range(num_dofs):
        basis.append(dual_continuous_piecewise_polynomial_lagrange_basis_fn(triangles, vertices, r, i, tau, num_dofs,
                                                                            tau_preim, boundary_simplices,
                                                                            keep_boundary_dofs_last))
    return basis


def zero_continuous_piecewise_polynomial_lagrange(triangles, vertices, n=1, tau=None, boundary_simplices=None,
                                                  keep_boundary_dofs_last=False, bsp_tree=None):
    r"""
    Get the continuous piecewise polynomial of degree r on the given triangulation :math:`\mathcal{T}`, where the
    polynomials on each simplex is expressed in the Lagrange basis, which is identically zero.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int n: Dimension of the target of the continuous piecewise polynomial.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param boundary_simplices: List of simplices or subsimplices on which the continuous piecewise polynomial functions
        should vanish (for :math:`C\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`C\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`C\mathcal{P}_r (\mathcal{T})` in a practical way.
    :param bsp_tree: Optional implementation detail. A binary space partitioning tree built around the triangulation
        :math:`\mathcal{T}` for quicker lookup of triangle a point lies in. Will be generated if not supplied.
    :return: The zero piecewise polynomial.
    :rtype: :class:`ContinuousPiecewisePolynomialLagrange`.
    """
    dim = len(vertices)
    if n == 1:
        coeff = np.zeros(dim)
    else:
        coeff = np.zeros((dim, n))
    support = set()
    return ContinuousPiecewisePolynomialLagrange(coeff, triangles, vertices, 1, tau, boundary_simplices,
                                                 keep_boundary_dofs_last, support, bsp_tree)


def unit_continuous_piecewise_polynomial_lagrange(triangles, vertices, n=1, tau=None, boundary_simplices=None,
                                                  keep_boundary_dofs_last=False, bsp_tree=None):
    r"""
    Get the continuous piecewise polynomial of degree r on the given triangulation :math:`\mathcal{T}`, where the
    polynomials on each simplex is expressed in the Lagrange basis, which is identically one.

    :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
        array of indices).
    :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
    :param int n: Dimension of the target of the continuous piecewise polynomial.
    :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
        basis function. Will be generated if not supplied.
    :type tau: Optional[Callable :math:`\tau(j, \nu)`]
    :param boundary_simplices: List of simplices or subsimplices on which the continuous piecewise polynomial functions
        should vanish (for :math:`C\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately (if
        `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`C\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`C\mathcal{P}_r (\mathcal{T})` in a practical way.
    :param bsp_tree: Optional implementation detail. A binary space partitioning tree built around the triangulation
        :math:`\mathcal{T}` for quicker lookup of triangle a point lies in. Will be generated if not supplied.
    :return: The unit continuous piecewise polynomial.
    :rtype: :class:`ContinuousPiecewisePolynomialLagrange`.
    """
    dim = len(vertices)
    if n == 1:
        coeff = np.ones(dim)
    else:
        coeff = np.ones((dim, n))
    support = set(range(len(triangles)))
    return ContinuousPiecewisePolynomialLagrange(coeff, triangles, vertices, 1, tau, boundary_simplices,
                                                 keep_boundary_dofs_last, support, bsp_tree)


def _get_any_element_from_set(s):
    """
    Get any element from the given set.

    :param set s: Set from which we want to get any element.
    :return: Any element from the given set, or None if the set is empty.
    """
    e = None
    for e in s:
        break
    return e
