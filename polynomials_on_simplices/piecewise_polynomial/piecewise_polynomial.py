r"""Functionality for discontinuous Galerkin finite elements (piecewise polynomials) on a simplicial domain
(triangulation) :math:`\mathcal{T}`.

The space of piecewise polynomials of degree r on :math:`\mathcal{T}, D\mathcal{P}_r (\mathcal{T})
= D\mathcal{P}_r (\mathcal{T}, \mathbb{R}^n) \subset L^2(\mathcal{T}, \mathbb{R}^n)` is defined as

.. math::

    D\mathcal{P}_r (\mathcal{T}) = \{ v \in L^2(\mathcal{T}, \mathbb{R}^n) \big| v|_T \in
    \mathcal{P}_r(T, \mathbb{R}^n) \, \forall T \in \mathcal{T} \}.

Correspondingly the space of piecewise polynomials of degree r on :math:`\mathcal{T}` which are zero on specified set
of simplices or subsimplices of :math:`\mathcal{T}, D\mathcal{P}_{r, 0} (\mathcal{T}` is defined as

.. math::

    D\mathcal{P}_{r, 0} (\mathcal{T}) = \{ v \in L^2_0(\mathcal{T}, \mathbb{R}^n) \big| v|_T \in
    \mathcal{P}_r(T, \mathbb{R}^n) \, \forall T \in \mathcal{T} \}.
"""

import abc
import numbers

import numpy as np

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.geometry.mesh.simplicial_complex import simplex_sub_simplices, simplex_vertices
from polynomials_on_simplices.geometry.primitives.simplex import inside_simplex
from polynomials_on_simplices.geometry.space_partitioning.kd_mesh_tree.simplicial_mesh_kd_tree import create_kd_tree
from polynomials_on_simplices.geometry.space_partitioning.kd_tree import find_leaf_containing_point
from polynomials_on_simplices.polynomial.polynomials_simplex_base import polynomials_equal_on_simplex
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import get_associated_sub_simplex


def generate_local_to_global_map(triangles, r, boundary_simplices=None, keep_boundary_dofs_last=False):
    r"""
    Generate a local-to-global map :math:`\tau` for the space of piecewise polynomial functions of degree r on a
    triangulation :math:`\mathcal{T}, D \mathcal{P}_r (\mathcal{T})` or :math:`D \mathcal{P}_{r, 0} (\mathcal{T})`.

    We have :math:`\tau : \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m \to \mathbb{N}_0 \cup \{ -1 \}`.

    A local-to-global map maps a local polynomial basis function on a simplex in the triangulation to a global basis
    index for a basis for the space of piecewise polynomials.
    The value -1 is used to indicate that a local basis function has no corresponding global basis function (see later).

    Let :math:`d = \dim \mathcal{P}_r (\Delta_c^m)`, and let the triangles in :math:`\mathcal{T}` be enumerated so that
    we have :math:`\mathcal{T} = \{ T_0, T_1, \ldots, T_{| \mathcal{T} | - 1} \}`.
    Given a basis :math:`\{ \bar{\varphi}_{\nu} \}_{\substack{\nu \in \mathbb{N}_0^m \\ | \nu | \leq r}}` for
    :math:`\mathcal{P}_r (\Delta_c^m)`, the functions :math:`\{ \bar{\varphi}_{\nu} \circ \Phi_{T_j}^{-1} \}_{
    \substack{\nu \in \mathbb{N}_0^m \\ | \nu | \leq r}}` is a basis for :math:`\mathcal{P}_r(T_j)`, where
    :math:`\Phi_{T_j}, j = 0, 1, \ldots, | \mathcal{T} | - 1` is the unique affine map which maps the unit simplex
    :math:`\Delta_c^m` to the simplex :math:`T_j`.
    From these local bases and the local-to-global map :math:`\tau` a basis :math:`\{ \phi_i \}_{i = 1}^N` for
    :math:`D \mathcal{P}_r (\mathcal{T})` is constructed, with

    .. math::

        \phi_{\tau(j, \nu)}(x) = \begin{cases}
        \bar{\varphi}_{\nu} \circ \Phi_{T_j}^{-1}(x), & x \in T_j \\
        0, & \text{else}
        \end{cases}.

    Optionally a set of boundary simplices can be prescribed where the piecewise polynomial function should vanish.
    This is achieved by associating an invalid global index (-1) for all local basis functions supported on any of the
    boundary simplices). Alternatively local basis functions supported on boundary simplices can be associated with
    global indices placed last in the enumeration of global basis functions (by setting the
    `keep_boundary_dofs_last` to true).

    :param triangles: Triangles (or in general simplices) in the mesh
        (num_triangles by n + 1 list of indices, where n is the dimension of each simplex).
    :param int r: Polynomial degree for the piecewise polynomial functions.
    :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial functions should
        vanish (for an element of :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately
        (if `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`D\mathcal{P}_r (\mathcal{T})` in a practical way.
    :return: Tuple containing the local to global map :math:`\tau` and the number of global basis functions.
        If `keep_boundary_dofs_last` is true then the number of global basis functions not supported on the
        boundary is also returned.
    :rtype: Tuple[Callable :math:`\tau(j, \nu)`, int, Optional[int]].
    """
    num_non_boundary_basis_fns = 0
    if keep_boundary_dofs_last:
        local_to_global_map, global_basis_fn_counter, num_non_boundary_basis_fns\
            = generate_local_to_global_map_as_dictionaries(triangles, r, boundary_simplices, keep_boundary_dofs_last)
    else:
        local_to_global_map, global_basis_fn_counter\
            = generate_local_to_global_map_as_dictionaries(triangles, r, boundary_simplices, keep_boundary_dofs_last)

    # Create callable local-to-global map (tau)
    def tau(j, nu):
        assert j >= 0
        assert j < len(triangles)
        if nu in local_to_global_map[j]:
            return local_to_global_map[j][nu]
        else:
            # No corresponding global basis index for local basis function
            return -1

    if keep_boundary_dofs_last:
        return tau, global_basis_fn_counter, num_non_boundary_basis_fns
    else:
        return tau, global_basis_fn_counter


def generate_local_to_global_map_as_dictionaries(triangles, r, boundary_simplices=None, keep_boundary_dofs_last=False):
    r"""
    Generate a local-to-global map for the space of piecewise polynomial functions of degree r on a triangulation
    :math:`\mathcal{T}, D \mathcal{P}_r (\mathcal{T})` or :math:`D \mathcal{P}_{r, 0} (\mathcal{T})`. For details
    see :func:`generate_local_to_global_map`. This function differs from that function in that a list of dictionaries
    (one for each triangle in the mesh) with multi-indices (second function argument) as keys and global
    DOF indices (function value) as values is returned instead of a callable, which might be preferable in some cases.

    :param triangles: Triangles (or in general simplices) in the mesh
        (num_triangles by n + 1 list of indices, where n is the dimension of each simplex).
    :param int r: Polynomial degree for the piecewise polynomial functions.
    :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial functions should
        vanish (for an element of :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated separately
        (if `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a list of
        vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T})` as a subset of
        :math:`D\mathcal{P}_r (\mathcal{T})` in a practical way.
    :return: Tuple containing the local to global map dictionaries, one for each triangle, and the number of global
        basis functions. If `keep_boundary_dofs_last` is true then the number of global basis functions not
        supported on the boundary is also returned.
    :rtype: Tuple[List[Dict[Tuple[int], int]], int, Optional[int]]
    """
    n = len(triangles[0]) - 1
    # Map containing global basis function associated with each sub simplex of the mesh
    boundary_sub_simplices = set()
    if boundary_simplices is not None:
        for boundary_simplex in boundary_simplices:
            boundary_sub_simplices = boundary_sub_simplices.union(simplex_sub_simplices(boundary_simplex))
    global_basis_fn_counter = 0
    boundary_basis_fn_counter = -1
    local_to_global_map = []
    mis = multiindex.generate_all(n, r)
    for triangle in triangles:
        # Triangle local-to-global mapping
        tri_local_to_global = {}
        # Iterate over all multi-indices
        for mi in mis:
            # Compute sub simplex associated with the multi-index
            if r > 0:
                sub_simplex, sub_mi = get_associated_sub_simplex(mi, r, triangle)
                # Sort sub simplex and the associated multi-index to be able to compare simplices
                # regardless of orientation
                sub_simplex_sorted, mi_sorted = zip(*sorted(zip(sub_simplex, sub_mi)))
            else:
                sub_simplex_sorted = tuple(sorted(triangle))
            if sub_simplex_sorted in boundary_sub_simplices:
                if not keep_boundary_dofs_last:
                    # No corresponding global basis index for local basis function
                    continue
                else:
                    # Degree of freedom associated with a boundary simplex, so we give it a placeholder
                    # enumeration for now, which later will be updated to make sure that these DOFs are
                    # enumerated last
                    tri_local_to_global[tuple(mi)] = boundary_basis_fn_counter
                    boundary_basis_fn_counter -= 1
            else:
                # Associate simplex and multi-index with the next available global basis fn
                tri_local_to_global[tuple(mi)] = global_basis_fn_counter
                global_basis_fn_counter += 1
        local_to_global_map.append(tri_local_to_global)

    # Handle boundary basis functions which should be placed last
    num_non_boundary_basis_fns = None
    if keep_boundary_dofs_last:
        num_non_boundary_basis_fns = global_basis_fn_counter
        num_boundary_basis_fns = 0
        for tri_idx in range(len(triangles)):
            for key, value in local_to_global_map[tri_idx].items():
                if value < 0:
                    local_to_global_map[tri_idx][key] = num_non_boundary_basis_fns + (-value) - 1
                    num_boundary_basis_fns = max(num_boundary_basis_fns, -value)
        global_basis_fn_counter += num_boundary_basis_fns

    if num_non_boundary_basis_fns is not None:
        return local_to_global_map, global_basis_fn_counter, num_non_boundary_basis_fns
    else:
        return local_to_global_map, global_basis_fn_counter


def generate_vector_valued_local_to_global_map(triangles, r, n, boundary_simplices=None, keep_boundary_dofs_last=False,
                                               ordering="interleaved"):
    r"""
    Generate a local-to-global map :math:`\tau` for the space of vector valued piecewise polynomial functions of degree
    r on a triangulation :math:`\mathcal{T}` with values in :math:`\mathbb{R}^n`,
    D \mathcal{P}_r (\mathcal{T}, \mathbb{R}^n)` or :math:`D \mathcal{P}_{r, 0} (\mathcal{T}, \mathbb{R}^n)`.

    We have :math:`\tau : \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m \times \{ 0, 1, \ldots, n - 1 \}
    \to \mathbb{N}_0 \cup \{ -1 \}`.

    A local-to-global map maps a local polynomial basis function on a simplex in the triangulation to a global basis
    index for a basis for the space of vector valued piecewise polynomials.
    The value -1 is used to indicate that a local basis function has no corresponding global basis function (see later).

    Let :math:`d = \dim \mathcal{P}_r (\Delta_c^m, \mathbb{R}^n)`, and let the triangles in :math:`\mathcal{T}` be
    enumerated so that we have :math:`\mathcal{T} = \{ T_0, T_1, \ldots, T_{| \mathcal{T} | - 1} \}`.
    Given a basis :math:`\{ \bar{\varphi}_{\nu, i} \}_{\substack{\nu \in \mathbb{N}_0^m \\ | \nu | \leq r \\
    i \in \{ 0, 1, \ldots, n - 1 \}}}` for :math:`\mathcal{P}_r (\Delta_c^m, \mathbb{R}^n)`, the functions
    :math:`\{ \bar{\varphi}_{\nu, i} \circ \Phi_{T_j}^{-1} \}_{\substack{\nu \in \mathbb{N}_0^m \\ | \nu | \leq r \\
    i \in \{ 0, 1, \ldots, n - 1 \}}}` is a basis for :math:`\mathcal{P}_r(T_j, \mathbb{R}^n)`, where
    :math:`\Phi_{T_j}, j = 0, 1, \ldots, | \mathcal{T} | - 1` is the unique affine map which maps the unit simplex
    :math:`\Delta_c^m` to the simplex :math:`T_j`.
    From these local bases and the local-to-global map :math:`\tau` a basis :math:`\{ \phi_i \}_{i = 1}^N` for
    :math:`D \mathcal{P}_r (\mathcal{T}, \mathbb{R}^n)` is constructed, with

    .. math::

        \phi_{\tau(j, \nu, k)}(x) = \begin{cases}
        \bar{\varphi}_{\nu, k} \circ \Phi_{T_j}^{-1}(x), & x \in T_j \\
        0, & \text{else}
        \end{cases}.

    Optionally a set of boundary simplices can be prescribed where the piecewise polynomial function should vanish.
    This is achieved by associating an invalid global index (-1) for all local basis functions supported on any of the
    boundary simplices). Alternatively local basis functions supported on boundary simplices can be associated with
    global indices placed last in the enumeration of global basis functions (by setting the
    `keep_boundary_dofs_last` to true).

    :param triangles: Triangles (or in general simplices) in the mesh
        (num_triangles by n + 1 list of indices, where n is the dimension of each simplex).
    :param int r: Polynomial degree for the piecewise polynomial functions.
    :param int n: Dimension of the target.
    :param boundary_simplices: List of simplices or subsimplices on which the piecewise polynomial functions should
        vanish (for an element of :math:`D\mathcal{P}_{r, 0} (\mathcal{T}, \mathbb{R}^n)`) or which should be treated
        separately (if `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a
        list of vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last is useful for handling :math:`D\mathcal{P}_{r, 0} (\mathcal{T}, \mathbb{R}^n)` as a
        subset of :math:`D\mathcal{P}_r (\mathcal{T}, \mathbb{R}^n)` in a practical way.
    :param str ordering: How the vector valued basis functions are ordered. Can be "sequential" or "interleaved".
        For sequential, sorting is first done on the index of the component that is non-zero, and then the non-zero
        component is sorted in the same way as the scalar valued basis functions. For "interleaved" basis functions
        are first sorted on their non-zero component in the same way as scalar valued basis functions, and then they
        are sorted on the index of the component that is non-zero.
    :return: Tuple containing the local to global map :math:`\tau` and the number of global basis functions.
        If `keep_boundary_dofs_last` is true then the number of global basis functions not supported on the
        boundary is also returned.
    :rtype: Tuple[Callable :math:`\tau(j, \nu, k)`, int, Optional[int]].
    """
    # For a scalar valued piecewise polynomial, generate_local_to_global_map should be used instead
    assert n > 1

    # Create callable local-to-global map (tau_scalar) for a scalar valued piecewise polynomial
    num_scalar_non_boundary_basis_fns = 0
    if keep_boundary_dofs_last:
        tau_scalar, global_scalar_basis_fn_counter, num_scalar_non_boundary_basis_fns =\
            generate_local_to_global_map(triangles, r, boundary_simplices, keep_boundary_dofs_last)
    else:
        tau_scalar, global_scalar_basis_fn_counter = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                                  keep_boundary_dofs_last)

    # Create callable local-to-global map (tau) for a vector valued piecewise polynomial from the local-to-global
    # map for a scalar valued piecewise polynomial and the specified ordering of vector valued basis functions
    if ordering == "interleaved":
        def tau(j, nu, k):
            assert j >= 0
            assert j < len(triangles)
            assert k >= 0
            assert k < n
            vs = tau_scalar(j, nu)
            if vs == -1:
                return -1
            return n * vs + k
    else:
        def tau(j, nu, k):
            assert j >= 0
            assert j < len(triangles)
            assert k >= 0
            assert k < n
            vs = tau_scalar(j, nu)
            if vs == -1:
                return -1
            return k * global_scalar_basis_fn_counter + vs

    if keep_boundary_dofs_last:
        return tau, n * global_scalar_basis_fn_counter, n * num_scalar_non_boundary_basis_fns
    else:
        return tau, n * global_scalar_basis_fn_counter


def generate_inverse_local_to_global_map(tau, num_triangles, num_dofs, r, m):
    r"""
    Generate the inverse of a local-to-global map (i.e. a global-to-local map).

    Let :math:`\tau : \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m \to \mathbb{N}_0 \cup \{ -1 \}`
    be a local-to-global map for the space of piecewise polynomials of degree r on a triangulation
    (see :func:`generate_local_to_global_map`).
    Then this function generates the inverse of :math:`\tau`, i.e. the map :math:`\tau^{-1} : \mathbb{N}_0 \to
    \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m` (:math:`\tau` is not invertible where it has the
    value -1) such that

    .. math:: \tau(\tau^{-1} (i)) = i, \, i = 0, 1, \ldots, N - 1

    where :math:`N` is the number of degrees of freedom for the piecewise polynomial.

    :param tau: Local-to-global map that we want to invert.
    :type tau: Callable :math:`\tau(j, \nu)`
    :param int num_triangles: Number of triangles in the triangulation :math:`\mathcal{T}` on which the piecewise
        polynomials associated with the local-to-global map are defined.
    :param int num_dofs: Number of degrees of freedom (dimension) for the space of piecewise polynomials.
    :param int r: Polynomial degree for the piecewise polynomials.
    :param int m: Dimension of the domain of the piecewise polynomials.
    :return: Inverse local-to-global map.
    :rtype: Callable :math:`\tau^{-1}(i)`.
    """
    global_to_local_map = {}
    mis = multiindex.generate_all(m, r)
    for j in range(num_triangles):
        for nu in mis:
            dof = tau(j, nu)
            if dof == -1:
                continue
            global_to_local_map[dof] = j, nu

    # Create callable global-to-local map (tau^{-1})
    def tau_inv(i):
        assert i >= 0
        assert i < num_dofs
        return global_to_local_map[i]

    return tau_inv


def generate_inverse_vector_valued_local_to_global_map(tau, num_triangles, num_dofs, r, m, n):
    r"""
    Generate the inverse of a local-to-global map (i.e. a global-to-local map).

    Let :math:`\tau : \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m \times \{ 0, 1, \ldots, n - 1 \}
    \to \mathbb{N}_0 \cup \{ -1 \}` be a local-to-global map for the space of vector valued piecewise polynomials
    of degree r on a triangulation (see :func:`generate_vector_valued_local_to_global_map`).
    Then this function generates the inverse of :math:`\tau`, i.e. the map :math:`\tau^{-1} : \mathbb{N}_0 \to
    \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m \times \{ 0, 1, \ldots, n - 1 \}` (:math:`\tau` is
    not invertible where it has the value -1) such that

    .. math:: \tau(\tau^{-1} (i)) = i, \, i = 0, 1, \ldots, N - 1

    where :math:`N` is the number of degrees of freedom for the vector valued piecewise polynomial.

    :param tau: Local-to-global map that we want to invert.
    :type tau: Callable :math:`\tau(j, \nu, k)`
    :param int num_triangles: Number of triangles in the triangulation :math:`\mathcal{T}` on which the piecewise
        polynomials associated with the local-to-global map are defined.
    :param int num_dofs: Number of degrees of freedom (dimension) for the space of piecewise polynomials.
    :param int r: Polynomial degree for the piecewise polynomials.
    :param int m: Dimension of the domain of the piecewise polynomials.
    :param int n: Dimension of the target of the piecewise polynomials.
    :return: Inverse local-to-global map.
    :rtype: Callable :math:`\tau^{-1}(i)`.
    """
    global_to_local_map = {}
    mis = multiindex.generate_all(m, r)
    for j in range(num_triangles):
        for nu in mis:
            for k in range(n):
                dof = tau(j, nu, k)
                if dof == -1:
                    continue
                global_to_local_map[dof] = j, nu, k

    # Create callable global-to-local map (tau^{-1})
    def tau_inv(i):
        assert i >= 0
        assert i < num_dofs
        return global_to_local_map[i]

    return tau_inv


def piecewise_polynomials_equal(p1, p2, rel_tol=1e-9, abs_tol=1e-7):
    r"""
    Check if two piecewise polynomials p1 and p2 are approximately equal by comparing their values on each simplex
    in the domain of the first piecewise polynomial.

    For equality on a simplex the
    :func:`~polynomials_on_simplices.polynomial.polynomials_simplex_base.polynomials_equal_on_simplex` function is
    used.

    :param p1: First piecewise polynomial.
    :type p1: Implementation of the :class:`PiecewisePolynomialBase` interface
    :param p2: Second piecewise polynomial.
    :type p2: Callable p2(x)
    :param float rel_tol: Tolerance for the relative error. See :func:`math.isclose <python:math.isclose>` for details.
    :param float abs_tol: Tolerance for the absolute error. See :func:`math.isclose <python:math.isclose>` for details.
    :return: Whether or not the two piecewise polynomials are approximately equal.
    :rtype: bool
    """
    # Note: This function takes callables as second input instead of an instance of the PiecewisePolynomialBase
    # abstract base class. The reason for this is that the former is more general. It allows us to check for equality
    # with a callable that supposedly is a piecewise polynomials but doesn't implement the PiecewisePolynomialBase
    # interface. The first input however is still an instance of the PiecewisePolynomialBase interface, to be able
    # to get the domain and degree of the piecewise polynomial (which otherwise would have had to be specified as
    # extra arguments).

    for triangle in p1.triangles:
        tri_vertices = simplex_vertices(triangle, p1.vertices)
        if not polynomials_equal_on_simplex(p1, p2, p1.r, tri_vertices, rel_tol, abs_tol):
            return False
    return True


class PiecewisePolynomialBase(abc.ABC):
    r"""
    Abstract base class for a piecewise polynomial function of degree r on a triangulation :math:`\mathcal{T}`, i.e.
    an element of :math:`D\mathcal{P}_r (\mathcal{T})` or :math:`D\mathcal{P}_{r, 0} (\mathcal{T})`.

    The domain dimension m = :math:`\dim \mathcal{T}` and the target dimension n of the piecewise polynomial is given
    by the :meth:`domain_dimension` and :meth:`target_dimension` functions respectively.

    The degree r of the piecewise polynomial is given by the :meth:`degree` method.

    Let :math:`N = \dim D\mathcal{P}_r (\mathcal{T}, \mathbb{R}^n)`. This value is given by the :meth:`dimension`
    method.

    .. rubric:: Restriction to a simplex

    On each simplex :math:`T \in \mathcal{T}` the piecewise polynomial is given by a polynomial of degree r.
    This polynomial can be acquired using the :meth:`restrict_to_simplex` method.

    .. rubric:: Basis

    This class assumes that a basis :math:`\{ \bar{\varphi}_{\nu} \}_{\substack{\nu \in \mathbb{N}_0^m
    \\ | \nu | \leq r}}` for :math:`\mathcal{P}_r (\Delta_c^m)` has been chosen.
    From this bases :math:`\{ \varphi_{\nu, j} \}_{\substack{\nu \in \mathbb{N}_0^m \\ | \nu | \leq r}}`
    for :math:`\mathcal{P}_r (T_j, \mathbb{R}^n)` are constructed by

    .. math:: \varphi_{\nu, j}(x) = (\bar{\varphi}_{\nu} \circ \Phi_{T_j}^{-1})(x),

    where :math:`\Phi_{T_j}` is the unique affine map which maps the unit simplex :math:`\Delta_c^m` to the simplex
    :math:`T_j` (the i:th vertex of the unit simplex is mapped to the i:th vertex of :math:`T_j`).

    This class also assumes that a local-to-global map :math:`\tau, \tau : \{ 0, 1, \ldots, | \mathcal{T} | - 1 \}
    \times \mathbb{N}_0^m \to \mathbb{N}_0 \cup \{ -1 \}` is available which maps local (simplex) basis functions to
    the index of the corresponding global basis function (or to -1 if the local basis function doesn't correspond to
    a global basis function (which is the case if the piecewise polynomial has a fixed (prescribed) value on a
    (sub)simplex)). See :func:`generate_local_to_global_map`.
    Then a basis :math:`\{ \phi_i \}_{i = 1}^N` is given by

    .. math::

        \phi_i(x) = \sum_{(j, \nu) \in \operatorname{preim}_{\tau}(i)} \chi_{T_j}(x) \cdot \varphi_{\nu, j}(x),

    And an arbitrary piecewise polynomial :math:`p(x)` is given by specifying coefficients :math:`a_i \in \mathbb{R}^n,
    i = 1, 2, \ldots, N` in front of each basis function.

    .. math::

        p(x) = \sum_{i = 1}^N a_i \phi_i(x).

    Hence the value of the piecewise polynomial function on a simplex :math:`T_j` is given by

    .. math::

        p(x) = \sum_{\substack{\nu \in \mathbb{N}_0^m \\ | \nu | \leq r \\ \tau(j, \nu) \neq -1}} a_{\tau(j, \nu)}
        \cdot \varphi_{\nu, j}(x), x \in T_j.

    The basis chosen for :math:`\mathcal{P}_r (\Delta_c^m)` is given by the :meth:`basis` method.

    .. rubric:: Algebraic structure

    This class also defines the basic algebraic structures of the space of piecewise polynomials.

    **Ring structure:**

    Addition: :math:`+ : D\mathcal{P}_r (\mathcal{T}) \times D\mathcal{P}_r (\mathcal{T})
    \to D\mathcal{P}_r (\mathcal{T}), (p_1 + p_2)(x) = p_1(x) + p_2(x)`.

    Multiplication: :math:`\cdot : D\mathcal{P} (\mathcal{T}) \times D\mathcal{P} (\mathcal{T})
    \to D\mathcal{P} (\mathcal{T}), (p_1 \cdot p_2)(x) = p_1(x) \cdot p_2(x)` (only for n = 1).

    **Vector space structure:**

    Scalar multiplication: :math:`\cdot : \mathbb{R} \times D\mathcal{P}_r (\mathcal{T})
    \to D\mathcal{P}_r (\mathcal{T}), (s \cdot p)(x) = s \cdot p(x)`.
    """

    def __init__(self, coeff, triangles, vertices, r, tau=None, boundary_simplices=None, keep_boundary_dofs_last=False,
                 support=None, bsp_tree=None):
        r"""
        :param coeff: Coefficients for the piecewise polynomial in the :math:`\{ \phi_i \}_{i = 1}^N` basis derived
            from the basis for :math:`\mathcal{P}_r (\Delta_c^m)` used (see :meth:`basis`) and the local-to-global
            map :math:`\tau`.
        :type coeff: List[Union[Scalar, Vector]]
        :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
            array of indices).
        :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
        :param int r: Degree of each polynomial in the piecewise polynomial.
        :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
            basis function. Will be generated if not supplied.
        :type tau: Callable :math:`\tau(j, \nu)`
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
        """
        assert r >= 0
        self.coeff = _to_numpy_array(coeff)
        if tau is not None:
            self.tau = tau
        else:
            self.tau, num_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                              keep_boundary_dofs_last)
            assert num_dofs == len(self.coeff)
        self.triangles = triangles
        self.vertices = vertices
        self.boundary_simplices = boundary_simplices
        self.keep_boundary_dofs_last = keep_boundary_dofs_last
        self.r = r
        try:
            self.n = len(self.coeff[0])
        except TypeError:
            self.n = 1
        try:
            self.m = len(vertices[0])
        except TypeError:
            self.m = 1

        # Compute triangles where the piecewise polynomial is non-zero
        if self.n == 1:
            def is_zero(x):
                return x == 0
        else:
            def is_zero(x):
                return np.all(x == 0)
        if support is not None:
            self.triangles_in_the_support = support
        else:
            self.triangles_in_the_support = set()
            mis = multiindex.generate_all(self.domain_dimension(), self.degree())
            for j in range(len(self.triangles)):
                for k in range(len(mis)):
                    i = self.tau(j, mis[k])
                    if not is_zero(coeff[i]):
                        self.triangles_in_the_support.add(j)
                        break

        # Create binary space partitioning tree around the triangulation (for quicker lookup of which triangle a
        # point x lies in in the call method)
        if bsp_tree is not None:
            self._bsp_tree = bsp_tree
        else:
            self._bsp_tree = create_kd_tree(self.triangles, self.vertices, fixed_depth=False, max_depth=10,
                                            max_simplices_in_leaf=5,
                                            simplices_of_interest=self.triangles_in_the_support)

        # Basis polynomials for each simplex where the piecewise polynomial is supported.
        # Dictionary with simplex index j as key and a polynomial basis for P_r(T_j) as value.
        # Need to be populated by concrete subclasses
        self._basis_polynomials = {}

    def domain_dimension(self):
        """
        Get dimension of the domain of the piecewise polynomial.

        :return: Dimension of the domain of the piecewise polynomial.
        :rtype: int
        """
        return self.m

    def target_dimension(self):
        """
        Get dimension of the target of the piecewise polynomial.

        :return: Dimension of the target of the piecewise polynomial.
        :rtype: int
        """
        return self.n

    def degree(self):
        """
        Get degree of each polynomial in the piecewise polynomial.

        :return: Polynomial degree.
        :rtype: int
        """
        return self.r

    def dimension(self):
        """
        Get dimension of the space of piecewise polynomial the function belongs to.

        :return: Dimension of the function space.
        :rtype: int
        """
        return len(self.coeff)

    @abc.abstractmethod
    def basis(self):
        r"""
        Get basis for the space :math:`\mathcal{P}_r (\Delta_c^m)` used to express the piecewise polynomial.

        :return: Unique identifier for the basis used.
        :rtype: str
        """
        pass

    def __call__(self, x):
        r"""
        Evaluate the piecewise polynomial at a point :math:`x \in \mathbb{R}^m`.

        :param x: Point where the piecewise polynomial should be evaluated.
        :type x: float or length m :class:`Numpy array <numpy.ndarray>`
        :return: Value of the piecewise polynomial.
        :rtype: float or length n :class:`Numpy array <numpy.ndarray>`.
        """
        # Find simplex x belongs to
        bsp_tree_node = find_leaf_containing_point(self._bsp_tree, x)
        j = None
        for tri_idx in bsp_tree_node.potentially_intersecting_simplices:
            if inside_simplex(x, simplex_vertices(self.triangles[tri_idx], self.vertices)):
                j = tri_idx
                break
        if j is None:
            # x doesn't lie inside any simplex where the polynomial is supported, so return value zero
            # FIXME: Currently we don't distinguish if x lies inside the triangle mesh, but where the piecewise
            # polynomial is not supported, so that zero value is correct, or if x lies outside of the triangle mesh
            # so that the value of the piecewise polynomial is actually undefined
            if self.n == 1:
                return 0.0
            else:
                return np.zeros(self.n)

        return self.evaluate_on_simplex(j, x)

    def evaluate_on_simplex(self, j, x):
        r"""
        Evaluate the piecewise polynomial at a point :math:`x \in \mathbb{R}^m` in the j:th simplex of the
        triangulation the piecewise polynomial is defined on.

        In other words the polynomial at the j:th simplex is evaluated at the point x.

        :param int j: Index of simplex where the piecewise polynomial should be evaluated.
        :param x: Point where the piecewise polynomial should be evaluated.
        :type x: float or length m :class:`Numpy array <numpy.ndarray>`
        :return: Value of the piecewise polynomial.
        :rtype: float or length n :class:`Numpy array <numpy.ndarray>`.
        """
        if self.n == 1:
            value = 0.0
        else:
            value = np.zeros(self.n)

        if j not in self.triangles_in_the_support:
            return value

        # The evaluation of the function at the point is given by summing the contribution from each basis
        # function on the simplex
        k = 0
        for nu in multiindex.generate_all(self.m, self.r):
            i = self.tau(j, nu)
            if i == -1:
                k += 1
                continue
            value += self.coeff[i] * self._basis_polynomials[j][k](x)
            k += 1

        return value

    def __getitem__(self, i):
        """
        Get the i:th component of the piecewise polynomial (for a vector valued piecewise polynomial).

        :param int i: Component to get.
        :return: The i:th component of the vector valued piecewise polynomial (real valued piecewise polynomial).
        :rtype: Instance of self.__class__
        """
        assert i >= 0
        assert i < self.target_dimension()
        if self.target_dimension() == 1:
            return self.__class__(self.coeff, self.triangles, self.vertices, self.r, self.tau, self.boundary_simplices,
                                  self.keep_boundary_dofs_last, self.support(), self._bsp_tree, self._basis_polynomials)
        else:
            return self.__class__(self.coeff[:, i], self.triangles, self.vertices, self.r, self.tau,
                                  self.boundary_simplices, self.keep_boundary_dofs_last, None, self._bsp_tree,
                                  self._basis_polynomials)

    def __add__(self, other):
        """
        Addition of this piecewise polynomial with another piecewise polynomial, self + other.

        :param other: Other piecewise polynomial.
        :return: Sum of the two piecewise polynomials.
        :rtype: Instance of self.__class__
        """
        # Added piecewise polynomials need to have the same domain and target dimension
        assert self.domain_dimension() == other.domain_dimension()
        assert self.target_dimension() == other.target_dimension()
        # For now require that both piecewise polynomials are expressed in the same basis.
        # If not we would need to transform them to some common basis, and what basis
        # this is would need to be specified by the user.
        assert self.basis() == other.basis()
        assert self.triangles.data == other.triangles.data
        assert self.vertices.data == other.vertices.data
        assert ((self.boundary_simplices is None and other.boundary_simplices is None)
                or (self.boundary_simplices.data == other.boundary_simplices.data))
        if self.degree() == other.degree():
            return self.__class__(self.coeff + other.coeff, self.triangles, self.vertices, self.r, self.tau,
                                  self.boundary_simplices, self.keep_boundary_dofs_last, None, self._bsp_tree,
                                  self._basis_polynomials)
        if self.degree() > other.degree():
            return self + other.degree_elevate(self.degree())
        else:
            return self.degree_elevate(other.degree()) + other

    def __sub__(self, other):
        """
        Subtraction of this piecewise polynomial with another piecewise polynomial, self - other.

        :param other: Other piecewise polynomial.
        :return: Difference of the two piecewise polynomials.
        :rtype: Instance of self.__class__
        """
        # Subtracted piecewise polynomials need to have the same domain and target dimension
        assert self.domain_dimension() == other.domain_dimension()
        assert self.target_dimension() == other.target_dimension()
        # For now require that both piecewise polynomials are expressed in the same basis.
        # If not we would need to transform them to some common basis, and what basis
        # this is would need to be specified by the user.
        assert self.basis() == other.basis()
        assert self.triangles.data == other.triangles.data
        assert self.vertices.data == other.vertices.data
        assert ((self.boundary_simplices is None and other.boundary_simplices is None)
                or (self.boundary_simplices.data == other.boundary_simplices.data))
        if self.degree() == other.degree():
            return self.__class__(self.coeff - other.coeff, self.triangles, self.vertices, self.r, self.tau,
                                  self.boundary_simplices, self.keep_boundary_dofs_last, None, self._bsp_tree,
                                  self._basis_polynomials)
        if self.degree() > other.degree():
            return self - other.degree_elevate(self.degree())
        else:
            return self.degree_elevate(other.degree()) - other

    @abc.abstractmethod
    def __mul__(self, other):
        """
        Multiplication of this piecewise polynomial with another piecewise polynomial (only if n = 1), a scalar, or
        a vector (only if n = 1), self * other.

        :param other: Piecewise polynomial, scalar or vector we should multiply this piecewise polynomial with.
        :return: Product of this piecewise polynomial with other.
        """
        pass

    def __rmul__(self, other):
        """
        Multiplication of this piecewise polynomial with another piecewise polynomial (only if n = 1) or a scalar,
        other * self.

        :param other: Other piecewise polynomial or scalar.
        :return: Product of this piecewise polynomial with other.
        """
        return self * other

    def multiply_with_constant(self, c):
        """
        Multiplication of this piecewise polynomial with a constant scalar or a vector (only if n = 1), self * c.

        :param c: Scalar or vector we should multiply this polynomial with.
        :type c: Union[float, :class:`Numpy array <numpy.ndarray>`]
        :return: Product of this piecewise polynomial with the constant.
        :rtype: Instance of self.__class__
        """
        if isinstance(c, numbers.Number):
            # Multiplication of the piecewise polynomial with a scalar
            if c == 0:
                support = set()
            else:
                support = self.support()
            return self.__class__(self.coeff * c, self.triangles, self.vertices, self.r, self.tau,
                                  self.boundary_simplices, self.keep_boundary_dofs_last, support, self._bsp_tree,
                                  self._basis_polynomials)
        if isinstance(c, np.ndarray):
            # Multiplication of the piecewise polynomial with a vector
            # Can only multiply scalar valued piecewise polynomials with a vector
            assert self.n == 1
            if np.linalg.norm(c) == 0:
                support = set()
            else:
                support = self.support()
            return self.__class__(np.outer(self.coeff, c), self.triangles, self.vertices, self.r, self.tau,
                                  self.boundary_simplices, self.keep_boundary_dofs_last, support, self._bsp_tree,
                                  self._basis_polynomials)

    def __truediv__(self, s):
        """
        Division of this piecewise polynomial with a scalar, self / s.

        :param float s: Scalar to divide with.
        :return: Division of this piecewise polynomial with s.
        :rtype: Instance of self.__class__
        """
        assert isinstance(s, numbers.Number)
        return self.__class__(self.coeff / s, self.triangles, self.vertices, self.r, self.tau, self.boundary_simplices,
                              self.keep_boundary_dofs_last, self.support(), self._bsp_tree, self._basis_polynomials)

    @abc.abstractmethod
    def __pow__(self, exp):
        r"""
        Raise the piecewise polynomial to a power.

        .. math::

            (p^{\mu})(x) = p(x)^{\mu} =  p_1(x)^{\mu_1} p_2(x)^{\mu_2} \ldots p_n(x)^{\mu_n}.

        :param exp: Power we want the raise the piecewise polynomial to (natural number or multi-index depending on
            the dimension of the target of the piecewise polynomial).
        :type exp: int or :class:`~polynomials_on_simplices.algebra.multiindex.MultiIndex` or Tuple[int, ...]
        :return: This piecewise polynomial raised to the given power.
        """
        pass

    @abc.abstractmethod
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
        """
        pass

    @abc.abstractmethod
    def restrict_to_simplex(self, i):
        r"""
        Restriction of the piecewise polynomial to a specified simplex :math:`T_i \in \mathcal{T}`.

        :param int i: Index of the simplex we want to restrict the piecewise polynomial to (in :math:`0, 1, \ldots,
            | \mathcal{T} | - 1`).
        :return: Polynomial which agrees with the piecewise polynomial on the simplex :math:`T_i`.
        """
        pass

    def support(self):
        r"""
        Get the indices of the simplices in :math:`\mathcal{T}` where this piecewise polynomial is non-zero.

        :return: Indices of simplices whose intersection with the support of this piecewise polynomial is non-empty.
        :rtype: Set[int]
        """
        return self.triangles_in_the_support

    @staticmethod
    def generate_local_to_global_map(triangles, r, boundary_simplices=None, keep_boundary_dofs_last=False):
        r"""
        Generate a local-to-global map :math:`\tau` for the space of piecewise polynomial functions of degree r on a
        triangulation :math:`\mathcal{T}, D \mathcal{P}_r (\mathcal{T})` or :math:`D \mathcal{P}_{r, 0} (\mathcal{T})`.
        See :func:`generate_local_to_global_map`.
        """
        return generate_local_to_global_map(triangles, r, boundary_simplices, keep_boundary_dofs_last)


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
