r"""Functionality for Lagrange (continuous Galerkin) finite elements (continuous piecewise polynomials) on a simplicial
domain (triangulation) :math:`\mathcal{T}`.

The space of continuous piecewise polynomials of degree r on :math:`\mathcal{T}, C\mathcal{P}_r (\mathcal{T})
= C\mathcal{P}_r (\mathcal{T}, \mathbb{R}^n) \subset H^1(\mathcal{T}, \mathbb{R}^n)` is defined as

.. math::

    C\mathcal{P}_r (\mathcal{T}) = \{ v \in C(\mathcal{T}, \mathbb{R}^n) \big| v|_T \in
    \mathcal{P}_r(T, \mathbb{R}^n) \, \forall T \in \mathcal{T} \}.

Correspondingly the space of continuous piecewise polynomials of degree r on :math:`\mathcal{T}` which are zero on
specified set of simplices or subsimplices of :math:`\mathcal{T}, C\mathcal{P}_{r, 0} (\mathcal{T}))` is defined as

.. math::

    C\mathcal{P}_{r, 0} (\mathcal{T}) = \{ v \in C_0(\mathcal{T}, \mathbb{R}^n) \big| v|_T \in
    \mathcal{P}_r(T, \mathbb{R}^n) \, \forall T \in \mathcal{T} \}.
"""

import abc

import polynomials_on_simplices.algebra.multiindex as multiindex
from polynomials_on_simplices.geometry.mesh.simplicial_complex import simplex_sub_simplices
from polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial import PiecewisePolynomialBase
from polynomials_on_simplices.polynomial.polynomials_unit_simplex_lagrange_basis import get_associated_sub_simplex


def generate_local_to_global_map(triangles, r, boundary_simplices=None, keep_boundary_dofs_last=False):
    r"""
    Generate a local-to-global map :math:`\tau` for the space of continuous piecewise polynomial functions of degree
    r on a triangulation :math:`\mathcal{T}, C \mathcal{P}_r (\mathcal{T})` or
    :math:`C \mathcal{P}_{r, 0} (\mathcal{T})`.

    We have :math:`\tau : \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m \to \mathbb{N}_0 \cup \{ -1 \}`.

    A local-to-global map maps a local polynomial basis function on a simplex in the triangulation to a global basis
    index for a basis for the space of continuous piecewise polynomials.
    The value -1 is used to indicate that a local basis function has no corresponding global basis function (see later).

    Let :math:`d = \dim \mathcal{P}_r (\Delta_c^m)`, and let the triangles in :math:`\mathcal{T}` be enumerated so that
    we have :math:`\mathcal{T} = \{ T_0, T_1, \ldots, T_{| \mathcal{T} | - 1} \}`.
    Given a basis :math:`\{ \bar{\varphi}_{\nu} \}_{\substack{\nu \in \mathbb{N}_0^m \\ | \nu | \leq r}}` for
    :math:`\mathcal{P}_r (\Delta_c^m)`, the functions :math:`\{ \varphi_{\nu, j}
    = \bar{\varphi}_{\nu} \circ \Phi_{T_j}^{-1} \}_{\substack{\nu \in \mathbb{N}_0^m \\ | \nu | \leq r}}` is a basis
    for :math:`\mathcal{P}_r(T_j)`, where :math:`\Phi_{T_j}, j = 0, 1, \ldots, | \mathcal{T} | - 1` is the unique
    affine map which maps the unit simplex :math:`\Delta_c^m` to the simplex :math:`T_j`.
    From these local bases and the local-to-global map :math:`\tau` a basis :math:`\{ \phi_i \}_{i = 1}^N` for
    :math:`C \mathcal{P}_r (\mathcal{T})` is constructed, with

    .. math::

        \phi_i(x) = \sum_{(j, \nu) \in \operatorname{preim}_{\tau}(i)} \chi_{T_j}(x) \cdot \varphi_{\nu, j}(x).

    For each triangle :math:`T_j` a basis function :math:`\phi_i` either agrees with a basis function
    :math:`\varphi_{\nu, j}` or it's zero. I.e. the set :math:`v_j = \{ (k, \nu) \in \operatorname{preim}_{\tau}(i)
    \big| k = j \}` either contains zero or one element.

    .. math::

        \phi_i(x) = \begin{cases}
        \bar{\varphi}_{\nu} \circ \Phi_{T_j}^{-1}(x), & (\nu, j) \text{ unique element in } v_j \\
        0, & v_j \text{ is empty}
        \end{cases}.

    Optionally a set of boundary simplices can be prescribed where the piecewise polynomial function should vanish.
    This is achieved by associating an invalid global index (-1) for all local basis functions supported on any of the
    boundary simplices). Alternatively local basis functions supported on boundary simplices can be associated with
    global indices placed last in the enumeration of global basis functions (by setting the
    `keep_boundary_dofs_last` to true).

    :param triangles: Triangles (or in general simplices) in the mesh
        (num_triangles by n + 1 list of indices, where n is the dimension of each simplex).
    :param int r: Polynomial degree for the continuous piecewise polynomial functions.
    :param boundary_simplices: List of simplices or subsimplices on which the continuous piecewise polynomial functions
        should vanish (for an element of :math:`C\mathcal{P}_{r, 0} (\mathcal{T})`) or which should be treated
        separately (if `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified as a
        list of vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last makes :math:`C\mathcal{P}_{r, 0} (\mathcal{T})` a subset of
        :math:`C\mathcal{P}_r (\mathcal{T})` in a practical way.
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
    Generate a local-to-global map for the space of continuous piecewise polynomial functions of degree r on a
    triangulation :math:`\mathcal{T}, C \mathcal{P}_r (\mathcal{T})` or :math:`C \mathcal{P}_{r, 0} (\mathcal{T})`.
    For details see :func:`generate_local_to_global_map`. This function differs from that function in that a list of
    dictionaries (one for each triangle in the mesh) with multi-indices (second function argument) as keys and global
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
    :rtype: Tuple[List[dict[Tuple[int], int]], int, Optional[int]]
    """
    assert r >= 1
    n = len(triangles[0]) - 1
    # Map containing global basis function associated with each sub simplex of the mesh
    sub_simplex_basis_fns = {}
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
            sub_simplex, sub_mi = get_associated_sub_simplex(mi, r, triangle)
            # Sort sub simplex and the associated multi-index to be able to compare simplices
            # regardless of orientation
            sub_simplex_sorted, mi_sorted = zip(*sorted(zip(sub_simplex, sub_mi)))
            if sub_simplex_sorted in boundary_sub_simplices and not keep_boundary_dofs_last:
                continue
            if len(sub_simplex) == n + 1:
                # Interior basis function, hence cannot be associated with an existing global basis function
                tri_local_to_global[tuple(mi)] = global_basis_fn_counter
                global_basis_fn_counter += 1
            else:
                # Add sub simplex to simplex basis function dict if it doesn't already exist
                if sub_simplex_sorted not in sub_simplex_basis_fns:
                    sub_simplex_basis_fns[sub_simplex_sorted] = {}
                # Check if the simplex and multi-index have already been associated with a global basis function
                if mi_sorted in sub_simplex_basis_fns[sub_simplex_sorted]:
                    tri_local_to_global[tuple(mi)] = sub_simplex_basis_fns[sub_simplex_sorted][mi_sorted]
                else:
                    if sub_simplex_sorted not in boundary_sub_simplices:
                        # Associate simplex and multi-index with the next available global basis fn
                        tri_local_to_global[tuple(mi)] = global_basis_fn_counter
                        sub_simplex_basis_fns[sub_simplex_sorted][mi_sorted] = global_basis_fn_counter
                        global_basis_fn_counter += 1
                    else:
                        # Degree of freedom associated with a boundary simplex, so we give it a placeholder
                        # enumeration for now, which later will be updated to make sure that these DOFs are
                        # enumerated last
                        tri_local_to_global[tuple(mi)] = boundary_basis_fn_counter
                        sub_simplex_basis_fns[sub_simplex_sorted][mi_sorted] = boundary_basis_fn_counter
                        boundary_basis_fn_counter -= 1
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
    Generate a local-to-global map :math:`\tau` for the space of vector valued continuous piecewise polynomial
    functions of degree r on a triangulation :math:`\mathcal{T}, C \mathcal{P}_r (\mathcal{T}, \mathbb{R}^n)` or
    :math:`C \mathcal{P}_{r, 0} (\mathcal{T}, \mathbb{R}^n)`.

    We have :math:`\tau : \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m \times \{ 0, 1, \ldots, n - 1\}
    \to \mathbb{N}_0 \cup \{ -1 \}`.

    A local-to-global map maps a local polynomial basis function on a simplex in the triangulation to a global basis
    index for a basis for the space of vector valued continuous piecewise polynomials.
    The value -1 is used to indicate that a local basis function has no corresponding global basis function (see later).

    Let :math:`d = \dim \mathcal{P}_r (\Delta_c^m, \mathbb{R}^n)`, and let the triangles in :math:`\mathcal{T}` be
    enumerated so that we have :math:`\mathcal{T} = \{ T_0, T_1, \ldots, T_{| \mathcal{T} | - 1} \}`.
    Given a basis :math:`\{ \bar{\varphi}_{\nu, i} \}_{\substack{\nu \in \mathbb{N}_0^m \\ | \nu | \leq r \\
    i \in \{ 0, 1, \ldots, n - 1 \}}}` for :math:`\mathcal{P}_r (\Delta_c^m, \mathbb{R}^n)`, the functions
    :math:`\{ \varphi_{\nu, j, i} = \bar{\varphi}_{\nu, i} \circ \Phi_{T_j}^{-1} \}_{\substack{\nu \in \mathbb{N}_0^m \\
    | \nu | \leq r \\ i \in \{ 0, 1, \ldots, n - 1 \}}}` is a basis for :math:`\mathcal{P}_r(T_j, \mathbb{R}^n)`,
    where :math:`\Phi_{T_j}, j = 0, 1, \ldots, | \mathcal{T} | - 1` is the unique affine map which maps the unit
    simplex :math:`\Delta_c^m` to the simplex :math:`T_j`.
    From these local bases and the local-to-global map :math:`\tau` a basis :math:`\{ \phi_i \}_{i = 1}^N` for
    :math:`C \mathcal{P}_r (\mathcal{T}, \mathbb{R}^n)` is constructed, with

    .. math::

        \phi_i(x) = \sum_{(j, \nu, k) \in \operatorname{preim}_{\tau}(i)} \chi_{T_j}(x) \cdot \varphi_{\nu, j, k}(x).

    For each triangle :math:`T_j` a basis function :math:`\phi_i` either agrees with a basis function
    :math:`\varphi_{\nu, j, k}` or it's zero. I.e. the set :math:`v_j = \{ (l, \nu, k) \in
    \operatorname{preim}_{\tau}(i) \big| l = j \}` either contains zero or one element.

    .. math::

        \phi_i(x) = \begin{cases}
        \bar{\varphi}_{\nu, k} \circ \Phi_{T_j}^{-1}(x), & (j, \nu, k) \text{ unique element in } v_j \\
        0, & v_j \text{ is empty}
        \end{cases}.

    Optionally a set of boundary simplices can be prescribed where the piecewise polynomial function should vanish.
    This is achieved by associating an invalid global index (-1) for all local basis functions supported on any of the
    boundary simplices). Alternatively local basis functions supported on boundary simplices can be associated with
    global indices placed last in the enumeration of global basis functions (by setting the
    `keep_boundary_dofs_last` to true).

    :param triangles: Triangles (or in general simplices) in the mesh
        (num_triangles by n + 1 list of indices, where n is the dimension of each simplex).
    :param int r: Polynomial degree for the continuous piecewise polynomial functions.
    :param int n: Dimension of the target.
    :param boundary_simplices: List of simplices or subsimplices on which the continuous piecewise polynomial functions
        should vanish (for an element of :math:`C\mathcal{P}_{r, 0} (\mathcal{T}, \mathbb{R}^n)`) or which should be
        treated separately (if `keep_boundary_dofs_last` is set to True). Each simplex or subsimplex is specified
        as a list of vertex indices of the vertices that form the simplex.
    :type boundary_simplices: List[List[int]]
    :param bool keep_boundary_dofs_last: Whether or not to collect all global basis functions associated with any
        boundary simplex last in the enumeration of all basis functions. Enumerating basis functions associated with
        boundary simplices last makes :math:`C\mathcal{P}_{r, 0} (\mathcal{T}, \mathbb{R}^n)` a subset of
        :math:`C\mathcal{P}_r (\mathcal{T}, \mathbb{R}^n)` in a practical way.
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

    # Create callable local-to-global map (tau_scalar) for a scalar valued continuous piecewise polynomial
    num_scalar_non_boundary_basis_fns = 0
    if keep_boundary_dofs_last:
        tau_scalar, global_scalar_basis_fn_counter, num_scalar_non_boundary_basis_fns = \
            generate_local_to_global_map(triangles, r, boundary_simplices, keep_boundary_dofs_last)
    else:
        tau_scalar, global_scalar_basis_fn_counter = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                                                  keep_boundary_dofs_last)

    # Create callable local-to-global map (tau) for a vector valued continuous piecewise polynomial from the
    # local-to-global map for a scalar valued piecewise polynomial and the specified ordering of vector valued
    # basis functions
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


def generate_local_to_global_preimage_map(tau, num_triangles, num_dofs, r, m):
    r"""
    Generate the preimage map of a local-to-global map.

    Let :math:`\tau : \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m \to \mathbb{N}_0 \cup \{ -1 \}` be
    a local-to-global map (see :func:`generate_local_to_global_map`). Then the preimage map of :math:`\tau` is the
    map

    .. math::

        \operatorname{preim}_{\tau} : \mathcal{P} \mathbb{N}_0 \to \mathcal{P} \left(
        \{ 0, 1, \ldots, |\mathcal{T}| - 1 \} \times \mathbb{N}_0^N \right)

    such that

    .. math:: \tau(j, \nu) = i \quad \forall \, (j, \nu) \in \operatorname{preim}_{\tau}(\{i\}).

    :param tau: Local-to-global map that we want to create the preimage for.
    :type tau: Callable :math:`\tau(j, \nu)`
    :param int num_triangles: Number of triangles in the triangulation :math:`\mathcal{T}` on which the continuous
        piecewise polynomials associated with the local-to-global map are defined.
    :param int num_dofs: Number of degrees of freedom (dimension) for the space of continuous piecewise polynomials.
    :param int r: Polynomial degree for the continuous piecewise polynomials.
    :param int m: Dimension of the domain of the continuous piecewise polynomials.
    :return: Preimage of the local-to-global map.
    :rtype: Callable :math:`\operatorname{preim}_{\tau}(A)`.
    """
    preimage_map_dict = {}
    mis = multiindex.generate_all(m, r)
    for j in range(num_triangles):
        for nu in mis:
            dof = tau(j, nu)
            if dof == -1:
                continue
            if dof in preimage_map_dict:
                preimage_map_dict[dof].add((j, nu))
            else:
                preimage_map_dict[dof] = {(j, nu)}

    # Create callable preimage map
    def preimage_map(si):
        value = set()
        for i in si:
            assert i >= 0
            assert i < num_dofs
            value = value.union(preimage_map_dict[i])
        return value

    return preimage_map


def generate_vector_valued_local_to_global_preimage_map(tau, num_triangles, num_dofs, r, m, n):
    r"""
    Generate the preimage map of a local-to-global map.

    Let :math:`\tau : \{ 0, 1, \ldots, | \mathcal{T} | - 1 \} \times \mathbb{N}_0^m \times \{ 0, 1, \ldots, n - 1\}
    \to \mathbb{N}_0 \cup \{ -1 \}` be a local-to-global map (see :func:`generate_vector_valued_local_to_global_map`).
    Then the preimage map of :math:`\tau` is the map

    .. math::

        \operatorname{preim}_{\tau} : \mathcal{P} \mathbb{N}_0 \to \mathcal{P} \left(
        \{ 0, 1, \ldots, |\mathcal{T}| - 1 \} \times \mathbb{N}_0^N \times \{ 0, 1, \ldots, n - 1 \} \right)

    such that

    .. math:: \tau(j, \nu, k) = i \quad \forall \, (j, \nu, k) \in \operatorname{preim}_{\tau}(\{i\}).

    :param tau: Local-to-global map that we want to create the preimage for.
    :type tau: Callable :math:`\tau(j, \nu, k)`
    :param int num_triangles: Number of triangles in the triangulation :math:`\mathcal{T}` on which the continuous
        piecewise polynomials associated with the local-to-global map are defined.
    :param int num_dofs: Number of degrees of freedom (dimension) for the space of continuous piecewise polynomials.
    :param int r: Polynomial degree for the continuous piecewise polynomials.
    :param int m: Dimension of the domain of the continuous piecewise polynomials.
    :param int n: Dimension of the domain of the continuous piecewise polynomials.
    :return: Preimage of the local-to-global map.
    :rtype: Callable :math:`\operatorname{preim}_{\tau}(A)`.
    """
    preimage_map_dict = {}
    mis = multiindex.generate_all(m, r)
    for j in range(num_triangles):
        for nu in mis:
            for k in range(n):
                dof = tau(j, nu, k)
                if dof == -1:
                    continue
                if dof in preimage_map_dict:
                    preimage_map_dict[dof].add((j, nu, k))
                else:
                    preimage_map_dict[dof] = {(j, nu, k)}

    # Create callable preimage map
    def preimage_map(si):
        value = set()
        for i in si:
            assert i >= 0
            assert i < num_dofs
            value = value.union(preimage_map_dict[i])
        return value

    return preimage_map


class ContinuousPiecewisePolynomialBase(PiecewisePolynomialBase, abc.ABC):
    r"""
    Abstract base class for a continuous piecewise polynomial function of degree r on a triangulation
    :math:`\mathcal{T}`, i.e. an element of :math:`C\mathcal{P}_r (\mathcal{T})` or
    :math:`C\mathcal{P}_{r, 0} (\mathcal{T})`. The space of continuous piecewise polynomials is a subspace of the space
    of piecewise polynomials, see :mod:`polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial` and in particular
    :class:`~polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial.PiecewisePolynomialBase`.

    .. rubric:: Differentiable structure

    Besides the basic algebraic structures of the space of piecewise polynomials this class also defines the (weakly)
    differentiable structure of the space of continuous piecewise polynomials.

    `i`:th partial weak derivative: :math:`\partial_i : C\mathcal{P}_r (\mathcal{T}) \to
    D\mathcal{P}_{r - 1} (\mathcal{T})`.
    """

    def __init__(self, coeff, triangles, vertices, r, tau=None, boundary_simplices=None, keep_boundary_dofs_last=False,
                 support=None, bsp_tree=None):
        r"""
        :param coeff: Coefficients for the piecewise polynomial in the :math:`\{ \phi_i \}_{i = 1}^N` basis derived
            from the basis for :math:`\mathcal{P}_r (\Delta_c^m)` used (see
            :meth:`polynomials_on_simplices.piecewise_polynomial.piecewise_polynomial.PiecewisePolynomialBase.basis`) and the
            local-to-global map :math:`\tau`.
        :type coeff: List[Union[Scalar, Vector]]
        :param triangles: Triangles (or in general simplices) in the mesh :math:`\mathcal{T}` (num_triangles by m + 1
            array of indices).
        :param vertices: Vertices in the mesh :math:`\mathcal{T}` (num_vertices by m array of scalars).
        :param int r: Degree of each polynomial in the piecewise polynomial.
        :param tau: Local-to-global map for mapping local basis functions to the index of the corresponding global
            basis function, in a way that makes sure that the piecewise polynomial is continuous. Will be generated
            if not supplied.
        :type tau: Callable :math:`\tau(j, \nu)`
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
        """
        # TODO: Verify that the local-to-global map generates a single-valued (continuous) piecewise polynomial?
        if tau is None:
            tau, num_dofs = generate_local_to_global_map(triangles, r, boundary_simplices,
                                                         keep_boundary_dofs_last)
            assert num_dofs == len(coeff)
        PiecewisePolynomialBase.__init__(self, coeff, triangles, vertices, r, tau, boundary_simplices,
                                         keep_boundary_dofs_last, support, bsp_tree)

    @abc.abstractmethod
    def weak_partial_derivative(self, i=0):
        """
        Compute the i:th weak partial derivative of the continuous piecewise polynomial.

        :param int i: Index of partial derivative.
        :return: i:th weak partial derivative of this continuous piecewise polynomial.
        """
        pass

    @staticmethod
    def generate_local_to_global_map(triangles, r, boundary_simplices=None, keep_boundary_dofs_last=False):
        r"""
        Generate a local-to-global map :math:`\tau` for the space of continuous piecewise polynomial functions of
        degree r on a triangulation :math:`\mathcal{T}, C \mathcal{P}_r (\mathcal{T})` or
        :math:`C \mathcal{P}_{r, 0} (\mathcal{T})`. See :func:`generate_local_to_global_map`.
        """
        return generate_local_to_global_map(triangles, r, boundary_simplices, keep_boundary_dofs_last)
