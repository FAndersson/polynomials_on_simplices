"""Calculations on a simplicial complex."""

import numpy as np
from scipy.sparse import coo_matrix

from polynomials_on_simplices.algebra.permutations import construct_permutation, construct_permutation_general, sign


def simplex_boundary(simplex):
    """
    Compute the boundary simplices of a simplex.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :return: List of boundary simplices.
    :rtype: :class:`Numpy array <numpy.ndarray>` of ints.

    .. rubric:: Examples

    >>> simplex_boundary([0, 1, 2])
    array([[1, 2],
           [2, 0],
           [0, 1]])
    >>> simplex_boundary([0, 1, 2, 3])
    array([[1, 2, 3],
           [0, 3, 2],
           [0, 1, 3],
           [0, 2, 1]])
    >>> simplex_boundary([0, 1, 12, 4])
    array([[ 1, 12,  4],
           [ 0,  4, 12],
           [ 0,  1,  4],
           [ 0, 12,  1]])
    """
    # Special handling of vertices
    if len(simplex) == 1:
        return np.array([])
    boundary_simplices = np.empty((len(simplex), len(simplex) - 1), dtype=int)
    for i in range(len(simplex)):
        boundary_simplex = np.delete(simplex, i)
        if i % 2 == 1 and len(boundary_simplex) > 1:
            boundary_simplex[-1], boundary_simplex[-2] = boundary_simplex[-2], boundary_simplex[-1]
        boundary_simplices[i] = boundary_simplex
    return boundary_simplices


def simplex_boundary_map(simplex):
    """
    Compute the (k-1)-chain which is the boundary of the given k-simplex.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :return: Boundary chain (pair of two arrays, the first containing the boundary simplices and the second containing
        the chain coefficients).
    """
    # Special handling of vertices
    if len(simplex) == 1:
        return np.array([]), 0
    boundary_simplices = np.empty((len(simplex), len(simplex) - 1), dtype=int)
    coefficients = np.empty(len(simplex), dtype=int)
    for i in range(len(simplex)):
        boundary_simplices[i] = np.delete(simplex, i)
        coefficients[i] = (-1)**i
    return boundary_simplices, coefficients


def simplex_sub_simplices(simplex, include_self=True):
    """
    Get the set of all sub simplices of a simplex.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :param bool include_self: Whether or not to include the simplex itself in the set of sub simplices.
    :return: Set of sub simplices, where each sub simplex is defined by a tuple of sorted vertex indices.
    :rtype: Set[Tuple[int]]

    .. rubric:: Examples

    >>> simplex_sub_simplices([0, 1, 2])
    {(1, 2), (0, 1), (0,), (1,), (0, 1, 2), (2,), (0, 2)}
    >>> simplex_sub_simplices([0, 1, 2], include_self=False)
    {(1, 2), (0, 1), (0,), (1,), (2,), (0, 2)}
    """
    sub_simplices = set()
    if include_self:
        sub_simplices.add(tuple(sorted(simplex)))
    if len(simplex) > 1:
        for boundary_simplex in simplex_boundary(simplex):
            nested_sub_simplices = simplex_sub_simplices(boundary_simplex)
            sub_simplices = sub_simplices.union(nested_sub_simplices)
    return sub_simplices


def simplex_sub_simplices_fixed_dimension(simplex, k):
    """
    Get the set of all k-dimensional sub simplices of a simplex T.

    :param simplex: Simplex T defined by a list of vertex indices.
    :type simplex: List[int]
    :param int k: Dimension of the sub simplices (in 0, 1, ..., dim(T)).
    :return: Set of sub simplices, where each sub simplex is defined by a tuple of sorted vertex indices.
    :rtype: Set[Tuple[int]]

    .. rubric:: Examples

    >>> simplex_sub_simplices_fixed_dimension([0, 1, 2], 0)
    {(2,), (0,), (1,)}
    >>> simplex_sub_simplices_fixed_dimension([0, 1, 2], 1)
    {(1, 2), (0, 1), (0, 2)}
    """
    assert k >= 0
    assert k <= simplex_dimension(simplex)
    sub_simplices = {sub_simplex for sub_simplex in simplex_sub_simplices(simplex)
                     if simplex_dimension(sub_simplex) == k}
    return sub_simplices


def opposite_sub_simplex(simplex, sub_simplex):
    """
    Get the opposite sub simplex of a given sub simplex in a simplex.

    The opposite sub simplex of a sub simplex f in a simplex T is the simplex consisting of all the vertices
    of T not in f.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :param sub_simplex: Sub simplex defined by a list of vertex indices.
    :type sub_simplex: List[int]
    :return: Opposite sub simplex defined by a list of vertex indices.
    :rtype: List[int]

    .. rubric:: Examples

    >>> opposite_sub_simplex([0, 1, 2], [1])
    [0, 2]
    >>> opposite_sub_simplex([0, 1, 2], [0, 1])
    [2]
    >>> opposite_sub_simplex([0, 1, 2], [0, 1, 2])
    []
    """
    assert(i in simplex for i in sub_simplex)
    opposite = []
    for i in simplex:
        if i not in sub_simplex:
            opposite.append(i)
    return opposite


def simplex_dimension(simplex):
    """
    Get the dimension of a simplex.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :return: Dimension of the simplex.
    """
    return len(simplex) - 1


def num_simplex_vertices(n):
    """
    Get the number of vertices in an n-dimensional simplex.

    :param n: Dimension of the simplex.
    :return: Number of vertices of an n-dimensional simplex.
    """
    return n + 1


def num_simplex_boundary_simplices(n):
    """
    Get the number of boundary simplices of an n-dimensional simplex.

    :param n: Dimension of the simplex.
    :return: Number of boundary simplices.

    .. rubric:: Examples

    >>> num_simplex_boundary_simplices(2)
    3
    >>> num_simplex_boundary_simplices(3)
    4
    """
    return n + 1


def boundary(simplices):
    """
    Compute the boundary simplices of a simplicial complex.

    .. note::

        Assumes that the simplicial complex is a manifold (with boundary), i.e., each boundary simplex is shared by
        at most two simplices in the complex.

    :param simplices: Simplicial complex (list of list of vertex indices for each simplex in the complex).
    :return: List of list of boundary simplices.

    .. rubric:: Examples

    >>> boundary([[0, 1, 2], [1, 3, 2]])
    array([[2, 0],
           [0, 1],
           [3, 2],
           [1, 3]])
    >>> boundary([[0, 1, 2, 3], [1, 4, 2, 3]])
    array([[0, 3, 2],
           [0, 1, 3],
           [0, 2, 1],
           [4, 2, 3],
           [1, 4, 3],
           [1, 2, 4]])
    """
    # Set of boundary simplices, with vertex indices sorted
    boundary_set = set()
    # Dictionary mapping sorted boundary simplices to the original boundary simplex, and its ordering number
    boundary_dictionary = {}
    idx = 0
    for simplex in simplices:
        for bs in simplex_boundary(simplex):
            sbs = tuple(sorted(bs))
            if sbs in boundary_set:
                # Simplex occurs twice, so it's not a boundary simplex
                boundary_set.remove(sbs)
                boundary_dictionary.pop(sbs)
            else:
                # Simplex not in the boundary set, so it's a potential boundary simplex
                boundary_set.add(sbs)
                boundary_dictionary[sbs] = (bs, idx)
            idx += 1
    sorted_boundary_simplices = sorted(boundary_dictionary.values(), key=lambda bsi: bsi[1])
    boundary_list = np.array([bsi[0] for bsi in sorted_boundary_simplices], dtype=int)
    return boundary_list


def boundary_map_matrix(simplices, k):
    r"""
    Compute the matrix representation of the boundary map :math:`\mathcal{C}^k \to \mathcal{C}^{k-1}` for the given
    simplicial complex.

    :param simplices: Simplicial complex. Multi-array where the k:th entry contains an inner array with all
        k-dimensional simplices in the simplicial complex.
    :param k: Determines which boundary map we want to compute the matrix representation of.
    :return: Matrix representation of the boundary map (scipy sparse matrix in coordinate format (coo_matrix).
    """
    # Create simplex to index map (maps a (k-1)-simplex to its index in the
    # list of (k-1)-simplices in the simplicial complex simplices)
    simplex_to_index_map = {}
    for i in range(len(simplices[k - 1])):
        simplex_to_index_map[tuple(sorted(simplices[k - 1][i]))] = i

    row = []
    col = []
    data = []

    for j in range(len(simplices[k])):
        boundary_chain = simplex_boundary_map(simplices[k][j])
        for bs, a in zip(boundary_chain[0], boundary_chain[1]):
            # Compute orientation of boundary simplex with respect to the simplex in the simplicial complex
            i = simplex_to_index_map[tuple(sorted(bs))]
            permutation = construct_permutation_general(tuple(simplices[k - 1][i]), tuple(bs))
            data.append(sign(permutation) * a)
            row.append(i)
            col.append(j)

    return coo_matrix((data, (row, col)), shape=(len(simplices[k - 1]), len(simplices[k])))


def simplex_vertices(simplex, vertex_list):
    """
    Get the vertices of a simplex.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :param vertex_list: List of vertices in the simplicial complex.
    :return: List of vertices in the simplex.
    """
    # Dimension of the embedding
    try:
        dim = len(vertex_list[0])
    except TypeError:
        dim = 1
    # Number of vertices of the simplex
    n = len(simplex)
    p = np.zeros((n, dim))
    for j in range(n):
        p[j] = vertex_list[simplex[j]]
    return p


def simplex_boundary_orientation(simplex, boundary_simplex):
    """
    Compute the orientation of a boundary simplex of the given simplex.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :param boundary_simplex: Boundary sub simplex (list of vertex indices).
    :return: Orientation of the boundary simplex (-1/1).

    .. rubric:: Examples

    >>> simplex_boundary_orientation([0, 1, 2], [0, 1])
    1
    >>> simplex_boundary_orientation([0, 1, 2], [1, 0])
    -1
    """
    assert len(boundary_simplex) + 1 == len(simplex)
    if not isinstance(boundary_simplex, list):
        boundary_simplex = list(boundary_simplex)
    # Find given boundary simplex in the simplex boundary
    b = _find_boundary_simplex(simplex, boundary_simplex)
    if b is None:
        raise ValueError("Boundary simplex is not in the boundary of the given simplex")
    # Construct permutation which maps the given boundary simplex to the simplex in the boundary
    substitution_map = list(zip(sorted(boundary_simplex), range(len(boundary_simplex))))
    b = _substitute_vertex_indices(b, substitution_map)
    boundary_simplex = _substitute_vertex_indices(boundary_simplex, substitution_map)
    p = construct_permutation(boundary_simplex, b, len(simplex))
    return sign(p)


def swap_simplex_orientation(simplex):
    """
    Swap orientation of a simplex.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :return: Same simplex but with opposite orientation.
    """
    opposite_simplex = np.copy(simplex)
    opposite_simplex[-1], opposite_simplex[-2] = opposite_simplex[-2], opposite_simplex[-1]
    return opposite_simplex


def swap_orientation(simplices):
    """
    Swap orientation for all simplices in a simplicial complex.

    :param simplices: Simplicial complex (list of list of vertex indices for each simplex in the complex).
        Modified in place.
    """
    for i in range(len(simplices)):
        simplices[i] = swap_simplex_orientation(simplices[i])


def _find_boundary_simplex(simplex, boundary_simplex):
    """
    Search for simplex in the boundary of a simplex which contains the same vertices as the given boundary simplex.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :param boundary_simplex: Simplex to look for in the boundary (list of vertex indices).
    :return: Boundary simplex (list of vertex indices), or None if no matching boundary simplex was found.
    """
    for b in simplex_boundary(simplex).tolist():
        equal = True
        for i in range(len(b)):
            if b[i] not in boundary_simplex:
                equal = False
                break
        if equal:
            return b
    return None


def _substitute_vertex_indices(simplex, substitution_map):
    """
    Substitute all indices in a simplex according to the given substitution map.

    :param simplex: Simplex defined by a list of vertex indices.
    :type simplex: List[int]
    :param substitution_map: List of (old value, new value) tuples.
    :return: Simplex with indices substituted (list of vertex indices).

    .. rubric:: Examples

    >>> _substitute_vertex_indices([3, 5, 2], [(3, 4), (5, 3)])
    [4, 3, 2]
    """
    updated_simplex = list(simplex)
    for old_value, new_value in substitution_map:
        for n, v in enumerate(simplex):
            if v == old_value:
                updated_simplex[n] = new_value
                break
    return updated_simplex


if __name__ == "__main__":
    import doctest
    doctest.testmod()
