r"""Compute different simplex properties.

An n-simplex is the convex hull of n + 1 vertices (points in :math:`\mathbb{R}^d, d \geq n`).

In this module a simplex is usually defined by specifying its vertices as rows in a matrix ((n + 1) x n for an
n-simplex in :math:`\mathbb{R}^n` and (n + 1) x d, d > n for an n-simplex embedded in a higher dimensional space).
"""

import math

import numpy as np

from polynomials_on_simplices.algebra.permutations import (
    increasing_subset_permutations, num_increasing_subset_permutations)
from polynomials_on_simplices.calculus.affine_map import (
    affine_composition, create_affine_map, inverse_affine_transformation, pseudoinverse_affine_transformation)
from polynomials_on_simplices.calculus.real_interval import in_closed_range, in_open_range
from polynomials_on_simplices.linalg.basis import gram_schmidt_orthonormalization_rn, transform_to_basis
from polynomials_on_simplices.linalg.vector_space_projection import subspace_projection_map


def basis(vertices, base_vertex=0):
    """
    Compute the simplex basis consisting of the simplex edges emanating from a given vertex.

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :param base_vertex: Base vertex for the basis.
    :return: Basis vectors (matrix with the basis vectors as columns).
    """
    n = dimension(vertices)
    if n == 0:
        raise TypeError("Simplex has no basis")
    m = vertices.shape[1]
    basis_vectors = np.empty((m, n))
    for i in range(n):
        basis_vectors[:, i] = vertices[(base_vertex + i + 1) % (n + 1)] - vertices[base_vertex]
    return basis_vectors


def orthonormal_basis(vertices, base_vertex=0):
    """
    Compute the orthonormalization of the simplex basis returned from the :func:`basis` function.

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :param base_vertex: Base vertex for the basis.
    :return: Orthonormal basis vectors (matrix with the basis vectors as columns).
    """
    b = basis(vertices, base_vertex)
    # Orthonormalize the basis
    b = gram_schmidt_orthonormalization_rn(b)
    return b


def unit(n, ne=None):
    r"""
    Create the vertices of the n-dimensional unit simplex :math:`\Delta_c^n`.

    :param int n: Dimension of the simplex.
    :param int ne: Dimension of the space the simplex is embedded in. Equal to n if None.
    :return: Array of vertices in the simplex.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> unit(2)
    array([[0., 0.],
           [1., 0.],
           [0., 1.]])

    >>> unit(2, 3)
    array([[0., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
    """
    if ne is None:
        ne = n
    vertices = np.zeros((n + 1, ne))
    for i in range(1, n + 1):
        vertices[i][i - 1] = 1.0
    return vertices


def equilateral(n, d=1, ne=None):
    r"""
    Create the vertices of a n-dimensional equilateral simplex (simplex where all the edges have the same length),
    with edge length d.

    The n-dimensional equilateral simplex returned here is created from the n-1 dimensional equilateral simplex by
    adding a new vertex above the centroid (with positive n:th coordinate), with an equal distance to all the
    vertices of the n-1 dimensional simplex.

    :param int n: Dimension of the simplex.
    :param float d: Length of each edge in the simplex.
    :param int ne: Dimension of the space the simplex is embedded in. Equal to n if None.
    :return: Array of vertices in the simplex.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> equilateral(1)
    array([[0.],
           [1.]])
    >>> equilateral(2, ne=3)
    array([[0.       , 0.       , 0.       ],
           [1.       , 0.       , 0.       ],
           [0.5      , 0.8660254, 0.       ]])
    """
    assert n >= 1
    if ne is None:
        ne = n
    if n == 1:
        return d * unit(n, ne)
    vertices = np.zeros((n + 1, ne))
    vertices[0:n] = equilateral(n - 1, d, ne)
    v = np.zeros(n - 1)
    c = centroid(vertices[0:n, 0:n - 1])
    vertices[n, 0:n - 1] = c
    vertices[n, n - 1] = np.sqrt(d**2 - np.linalg.norm(c - v)**2)
    return vertices


def dimension(vertices):
    """
    Get the dimension n of a simplex.

    :param vertices: Vertices of the simplex ((n + 1) x m matrix where row i contains the i:th vertex of the simplex).
    :return: Dimension n of the simplex.
    :rtype: int
    """
    return len(vertices) - 1


def embedding_dimension(vertices):
    """
    Get the dimension of the space a simplex is embedded in.

    :param vertices: Vertices of the simplex ((n + 1) x m matrix where row i contains the i:th vertex of the simplex).
    :return: Dimension m of the space the simplex is embedded in.
    :rtype: int
    """
    return len(vertices[0])


def affine_transformation_to_unit(vertices):
    r"""
    Generate the affine transformation Ax + b which transforms the given n-dimensional simplex embedded in
    :math:`\mathbb{R}^m, m \geq n`, to the n-dimensional unit simplex :math:`\Delta_c^n`.

    :param vertices: Vertices of the simplex ((n + 1) x m matrix where row i contains the i:th vertex of the simplex).
    :return: Tuple of A and b.
    """
    n = dimension(vertices)
    m = embedding_dimension(vertices)
    a_inv, b_inv = affine_transformation_from_unit(vertices)
    if m != n:
        return pseudoinverse_affine_transformation(a_inv, b_inv)
    return inverse_affine_transformation(a_inv, b_inv)


def affine_map_to_unit(vertices):
    r"""
    Generate the affine map :math:`\Phi : \mathbb{R}^m \to \mathbb{R}^m` which maps the given n-dimensional simplex
    embedded in :math:`\mathbb{R}^m, m \geq n,` to the n-dimensional unit simplex :math:`\Delta_c^n`.

    :param vertices: Vertices of the simplex ((n + 1) x m matrix where row i contains the i:th vertex of the simplex).
    :return: Function which takes an n-dimensional vector as input and returns an n-dimensional vector.
    :rtype: Callable :math:`\Phi(x)`.
    """
    n = dimension(vertices)
    m = embedding_dimension(vertices)
    a, b = affine_transformation_to_unit(vertices)
    if m != n:
        phi_local = create_affine_map(a, b)

        def phi(x):
            # Make sure that the point x lies in the subspace spanned by the simplex
            assert in_subspace(x, vertices, 1e-14)
            return phi_local(x)
        return phi
    return create_affine_map(a, b)


def affine_transformation_from_unit(vertices):
    r"""
    Generate the affine transformation Ax + b which transforms the unit simplex :math:`\Delta_c^n` to the given
    n-dimensional simplex embedded in :math:`\mathbb{R}^m, n \leq m`.

    :param vertices: Vertices of the simplex ((n + 1) x m matrix where row i contains the i:th vertex of the simplex).
    :return: Tuple of A and b.
    """
    n = dimension(vertices)
    m = embedding_dimension(vertices)
    assert n <= m
    # Handle 1d case separately
    if n == 1 and m == 1:
        b = vertices[0][0]
        a = vertices[1][0] - vertices[0][0]
        return a, b
    b = vertices[0]
    a = np.empty((m, n))
    for i in range(n):
        a[:, i] = vertices[i + 1] - vertices[0]
    return a, b


def affine_map_from_unit(vertices):
    r"""
    Generate the affine map :math:`\Phi : \mathbb{R}^n \to \mathbb{R}^m` which maps the n-dimensional unit simplex
    :math:`\Delta_c^n` to the given n-dimensional simplex embedded in :math:`\mathbb{R}^m, n \leq m`.

    :param vertices: Vertices of the simplex ((n + 1) x m matrix where row i contains the i:th vertex of the simplex).
    :return: Function which takes an n-dimensional vector as input and returns an m-dimensional vector.
    :rtype: Callable :math:`\Phi(x)`.
    """
    a, b = affine_transformation_from_unit(vertices)
    return create_affine_map(a, b)


def affine_transformation(vertices1, vertices2):
    """
    Generate the affine transformation Ax + b which transforms from one simplex to another.

    :param vertices1: Vertices of the first simplex (the domain).
    :param vertices2: Vertices of the second simplex (the codomain).
    :return: Tuple of A and b.
    """
    # Go via the unit simplex
    phi1 = affine_transformation_to_unit(vertices1)
    phi2 = affine_transformation_from_unit(vertices2)
    return affine_composition(phi1, phi2)


def local_coordinates(vertices):
    """
    Express the vertices of a simplex in the local orthonormal simplex basis (as given by the
    :func:`orthonormal_basis` function).

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: Coordinates of the simplex in the local orthonormal basis.
    """
    if dimension(vertices) == 0:
        # Special case of 0-dimensional simplex
        return np.array([[0.0]])
    # Translate the simplex to the origin
    vertices = vertices - vertices[0]
    # Create orthonormal basis for the simplex
    b = orthonormal_basis(vertices)
    # Express the simplex in the local orthonormal basis
    vertices = transform_to_basis(vertices.T, b).T
    return vertices


def volume(vertices):
    """
    Compute the volume of a simplex, defined by a list of vertices.

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: Volume of the simplex.
    """
    # Dimension of the simplex
    k = dimension(vertices)
    if k == 0:
        # The volume of a point is 1
        return 1
    # Dimension of the embedding
    n = embedding_dimension(vertices)
    if n > k:
        # The k-dimensional simplex is embedded in a higher-dimensional space.
        # We need to express the simplex in the local k-dimensional orthonormal basis, rather than the
        # n-dimensional Cartesian basis
        vertices = local_coordinates(vertices)
        return signed_volume(vertices)
    return np.abs(signed_volume(vertices))


def volume_unit(n):
    r"""
    Compute the volume of the n-dimensional unit simplex :math:`\Delta_c^n`.

    :param int n: Dimension of the simplex.
    :return: Volume of the unit simplex.
    :rtype: float

    .. rubric:: Examples

    >>> volume_unit(1)
    1.0
    >>> volume_unit(2)
    0.5
    >>> from fractions import Fraction
    >>> Fraction(volume_unit(3)).limit_denominator()
    Fraction(1, 6)
    """
    return 1 / math.factorial(n)


def signed_volume(vertices):
    """
    Compute the signed volume of a simplex, defined by a list of vertices.

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: Signed volume of the simplex.
    """
    # Compute dimension of the simplex
    n = dimension(vertices)
    if n == 0:
        return 1
    if embedding_dimension(vertices) > n:
        raise ValueError("Signed volume is not meaningful for a simplex embedded in a higher-dimensional space")
    # Compute parallelepiped spanned by the simplex vertices
    p = np.empty((n, n))
    # Vertices data structure need to support subtraction of rows (vertices)
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)
    for i in range(n):
        p[:, i] = vertices[i + 1] - vertices[0]
    return np.linalg.det(p) / math.factorial(n)


def orientation(vertices):
    """
    Compute the orientation of a simplex.

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: Orientation of the simplex (-1/1).
    """
    if signed_volume(vertices) >= 0.0:
        return 1
    return -1


def _cartesian_to_barycentric_volume(point, vertices):
    r"""
    Compute the barycentric coordinates of a point in a simplex.
    Computed by evaluating the volume of simplices formed by the point and all but one of the vertices of the simplex.

    :param point: Cartesian point for which the barycentric coordinates should be computed.
    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: barycentric coordinates of the point
        (:math:`b_i` such that :math:`p = \sum b_i v_i` and :math:`\sum b_i = 1`).
    """
    k = dimension(vertices)
    n = embedding_dimension(vertices)
    if n > k:
        assert in_subspace(point, vertices, 1e-14)
        # The k-dimensional simplex is embedded in a higher-dimensional space.
        # We need to express the simplex and the point in the local k-dimensional orthonormal basis, rather than the
        # n-dimensional Cartesian basis
        b = orthonormal_basis(vertices)
        point = transform_to_basis(point - vertices[0], b)
        vertices = local_coordinates(vertices)
    bary = np.empty(len(vertices))
    vol = volume(vertices)
    for i in range(len(vertices)):
        v_i = vertices[i].copy()
        vertices[i] = point
        vol_i = signed_volume(vertices)
        bary[i] = vol_i / vol
        vertices[i] = v_i
    return bary


def _cartesian_to_barycentric_basis(point, vertices):
    r"""
    Compute the barycentric coordinates of a point in a simplex.
    Computed by transforming the point to the simplex basis.

    :param point: Cartesian point for which the barycentric coordinates should be computed.
    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: barycentric coordinates of the point
        (:math:`b_i` such that :math:`p = \sum b_i v_i` and :math:`\sum b_i = 1`).
    """
    b = basis(vertices)
    bary = np.empty(len(vertices))
    bary[1:] = transform_to_basis(point - vertices[0], b)
    bary[0] = 1 - sum(bary[1:])
    return bary


def cartesian_to_barycentric(point, vertices):
    r"""
    Compute the barycentric coordinates of a point in a simplex.

    :param point: Cartesian point for which the barycentric coordinates should be computed.
    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: barycentric coordinates of the point
        (:math:`b_i` such that :math:`p = \sum b_i v_i` and :math:`\sum b_i = 1`).
    """
    return _cartesian_to_barycentric_volume(point, vertices)


def barycentric_to_cartesian(barycentric, vertices):
    """
    Compute the Cartesian coordinates for a point given by barycentric coordinates in a simplex.

    :param barycentric: Barycentric coordinates of the point.
    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: Cartesian coordinates of the point.
    """
    p = np.zeros(len(vertices[0]))
    for i in range(len(vertices)):
        p += barycentric[i] * vertices[i]
    return p


def cartesian_to_barycentric_unit(point):
    r"""
    Compute the barycentric coordinates of a point in the unit simplex :math:`\Delta_c^n`.

    :param point: Cartesian point for which the barycentric coordinates should be computed.
    :return: barycentric coordinates of the point
        (:math:`b_i` such that :math:`p = \sum b_i v_i` and :math:`\sum b_i = 1`).
    """
    return np.append(1.0 - sum(point), point)


def barycentric_to_cartesian_unit(barycentric):
    r"""
    Compute the Cartesian coordinates for a point given by barycentric coordinates in the unit
    simplex :math:`\Delta_c^n`.

    :param barycentric: Barycentric coordinates of the point.
    :return: Cartesian coordinates of the point.
    """
    return barycentric[1:]


def barycentric_to_trilinear(barycentric, vertices):
    """
    Compute the trilinear coordinates (triangle), quadriplanar coordinates (tetrahedra) or the higher dimensional
    analog thereof (ratios between the distances from a point to the simplex (hyper-)faces) of a point with given
    barycentric coordinates.
    .. note::

        The computed coordinates are normalized (or exact), so that the i:th entry is the actual distance to the
        i:th face.

    :param barycentric: Barycentric coordinates of the point.
    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: Trilinear coordinates of the point.
    """
    v = volume(vertices)
    n = dimension(vertices)
    trilinear = np.zeros(n + 1)
    for i in range(n + 1):
        # si is the sub-simplex (face) formed by removing vertex i from the simplex
        si = face(vertices, i)
        vi = volume(si)
        # This formula comes from V = V_f * h / n, where V_f is the volume of the face
        trilinear[i] = n * v * barycentric[i] / vi
    return trilinear


def trilinear_to_barycentric(trilinear, vertices):
    """
    Compute the barycentric coordinates of a point with given trilinear coordinates (ratios between the distances
    to the simplex (hyper-)faces).

    :param trilinear: Trilinear coordinates of the point.
    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :returns: Barycentric coordinates of the point.
    """
    v = volume(vertices)
    n = dimension(vertices)
    barycentric = np.empty(n + 1)
    for i in range(n + 1):
        # si is the sub-simplex (face) formed by removing vertex i from the simplex
        si = face(vertices, i)
        vi = volume(si)
        barycentric[i] = trilinear[i] * vi / v
    # Normalize the barycentric coordinates (so that they sum to 1, required since the input trilinear coordinates
    # can be a multiple of the distances to faces hi
    barycentric *= 1 / sum(b for b in barycentric)
    return barycentric


def centroid(vertices):
    """
    Compute the centroid of a simplex (center of gravity).

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: The centroid of the simplex.
    """
    return vertices.sum(0) / len(vertices)


def edges(vertices):
    """
    Get all edges of a simplex.

    This returns edges pointing from a vertex with lower index to a vertex with higher index, e.g. the edge from
    vertex 0 to vertex 1, from vertex 2 to vertex 4, etc. The order of the returned edges is as follows: First all edges
    containing vertex 0, with the second vertex in increasing order. Then all edges containing vertex 1, excluding the
    edge (v0, v1), with the second vertex in increasing order etc.

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: The edges of the simplex as rows in a matrix.
    """
    n = dimension(vertices)
    ne = embedding_dimension(vertices)
    e = np.empty((num_increasing_subset_permutations(n + 1, 2), ne))
    i = 0
    for v0, v1 in increasing_subset_permutations(n + 1, 2):
        e[i] = vertices[v1] - vertices[v0]
        i += 1
    return e


def face(vertices, i):
    """
    Get the vertices of the i:th face of a simplex. The i:th face is the sub simplex containing all but the i:th
    vertex of the simplex.

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :param i: Index of face.
    :return: Vertices of the face.
    """
    from polynomials_on_simplices.geometry.mesh.simplicial_complex import simplex_boundary, simplex_vertices
    f = simplex_boundary(range(len(vertices)))[i]
    return simplex_vertices(f, vertices)


def face_normal(vertices, i):
    """
    Compute normal to the i:th face of a simplex. The i:th face is the sub simplex containing all but the i:th
    vertex of the simplex. The normal is oriented so that it points out of the simplex (away from omitted vertex).

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :param i: Index of face whose normal should be computed.
    :return: Face normal vector (unit length vector).
    """
    b = orthonormal_basis(vertices, (i + 1) % len(vertices))
    return -b[:, -1]


def altitude(vertices, i):
    """
    Compute an altitude in a simplex (i.e. the shortest distance from a vertex to the plane spanned by the opposite
    face).

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :param i: Index of vertex/face whose altitude should be computed.
    :returns: The i:th altitude of the simplex.
    """
    si = face(vertices, i)
    n = dimension(vertices)
    # From V = V_f * h / n => h = V * n / V_f
    h = n * volume(vertices) / volume(si)
    return h


def circumcenter(vertices):
    """
    Compute the circumcenter of a simplex (the center of the n-sphere which passes through all the
    n + 1 vertices of the simplex).

    :param vertices: Vertices of the simplex (n + 1 x n matrix where row i contains the i:th vertex of the simplex).
    :return: The circumcenter of the simplex.
    """
    if len(vertices) < 3:
        # Point or line
        return centroid(vertices)
    # Compute circumcenter and normal of first face
    c = circumcenter(vertices[1:])
    n = face_normal(vertices, 0)
    # Find t such that |c + t*n - v0|^2 = |c + t*n - v1|^2
    # (the points on the line c + t*n are already on the same distance from all of the vertices except v0)
    v0 = vertices[0]
    v1 = vertices[1]
    num = np.dot(v0, v0) - np.dot(v1, v1) + 2 * np.dot(c, v1) - 2 * np.dot(c, v0)
    denom = 2 * (np.dot(n, v0) - np.dot(n, v1))
    t = num / denom
    return c + t * n


def circumradius(vertices):
    r"""
    Compute the circumradius of a simplex (the radius of the n-sphere which passes through all the n + 1 vertices
    of the simplex).

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: The circumradius of the simplex.
    """
    return np.linalg.norm(circumcenter(vertices) - vertices[0])


def incenter(vertices):
    r"""
    Compute the incenter of a simplex (the center of the largest inscribed n-sphere).

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: The incenter of the simplex.
    """
    trilinear = np.array(len(vertices) * [1])
    barycentric = trilinear_to_barycentric(trilinear, vertices)
    return barycentric_to_cartesian(barycentric, vertices)


def inradius(vertices):
    r"""
    Compute the inradius of a simplex (the radius of the largest inscribed n-sphere).

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: The inradius of the simplex.
    """
    n = dimension(vertices)
    trilinear = np.array((n + 1) * [1])
    exact_trilinear = barycentric_to_trilinear(trilinear_to_barycentric(trilinear, vertices), vertices)
    assert all([abs(exact_trilinear[i] - exact_trilinear[(i + 1) % (n + 1)]) < 1e-10 for i in range(n + 1)])
    return exact_trilinear[0]


def diameter(vertices):
    r"""
    Compute largest distance between any two points in a simplex, i.e. the length of the longest simplex edge.

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :return: Simplex diameter.
    """
    h = 0.0
    for i in range(len(vertices) - 1):
        for j in range(i + 1, len(vertices)):
            h = max(h, np.linalg.norm(vertices[i] - vertices[j]))
    return h


def inside_simplex(point, vertices, include_boundary=True, tol=0.0):
    """
    Check whether or not a point lies inside a simplex.

    A point lies inside a simplex if all its barycentric coordinates are in the range [0, 1] (or (0, 1) if the
    boundary is not included.
    However these kind of sharp checks often doesn't make sense for floating point values. So for this a tolerance
    can be specified, in which case the point is considered to be inside the simplex if all its barycentric coordinates
    lies in the range (0 - tol, 1 + tol).

    :param point: Point which we want to check.
    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :param include_boundary: Whether or not to consider points on the boundary as inside the simplex.
    :param float tol: Tolerance used for fuzzy inside simplex checks.
    :return: Whether or not the point lies inside the simplex.
    :rtype: bool
    """
    bary = cartesian_to_barycentric(point, vertices)
    # Point is inside the simplex if all barycentric coordinates are in the range [0, 1]
    # (or (0, 1) if we exclude the boundary)
    # With a fuzzy boundary, points are considered inside the simplex if all barycentric coordinates are in the range
    # (0 - tol, 1 + tol)
    if tol == 0.0:
        if include_boundary:
            in_range = in_closed_range
        else:
            in_range = in_open_range
        return all([in_range(b, 0, 1) for b in bary])
    else:
        return all([in_open_range(b, 0 - tol, 1 + tol) for b in bary])


def in_subspace(point, vertices, tol=0.0):
    r"""
    Check whether or not a point in :math:`\mathbb{R}^n` lies in the subspace spanned by the edges of an m-dimensional
    simplex (:math:`m \leq n`).

    A point p is considered to lie in the simplex subspace if

    .. math:: \| p - p_{\text{proj}} \| leq \text{tol},

    where :math:`p_{\text{proj}}` is the projection of p onto the subspace spanned by the simplex.

    :param point: Point which we want to check.
    :type point: Element in :math:`\mathbb{R}^n`
    :param vertices: Vertices of the simplex ((m + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :param float tol: Tolerance for the distance check.
    :return: Whether or not the point lies inside the subspace spanned by the edges of the simplex.
    :rtype: bool
    """
    assert len(point) == embedding_dimension(vertices)
    if len(point) == dimension(vertices):
        return True
    # Handle special case of a 0-dimensional simplex (a point)
    if dimension(vertices) == 0:
        return np.linalg.norm(point - vertices[0]) <= tol
    subspace_basis = basis(vertices)
    subspace_origin = vertices[0]
    point_projected = subspace_projection_map(subspace_basis, subspace_origin)(point)
    return np.linalg.norm(point - point_projected) <= tol


def sub_simplex(vertices, f):
    """
    Get the vertices of a sub simplex of a simplex.

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :param f: Sub simplex f of the input simplex, defined as a list of indices (in [0, 1, ..., n]) to vertices that
        are contained in f.
    :type f: List[int]
    :return: Array of vertices in the sub simplex f.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    Sub simplices of the unit triangle:

    >>> sub_simplex(unit(2), [0, 1])
    array([[0., 0.],
           [1., 0.]])
    >>> sub_simplex(unit(2), [1, 2])
    array([[1., 0.],
           [0., 1.]])
    >>> sub_simplex(unit(2), [2, 0])
    array([[0., 1.],
           [0., 0.]])
    >>> sub_simplex(unit(2), [1])
    array([[1., 0.]])

    Line sub simplex of the unit tetrahedra:

    >>> sub_simplex(unit(3), [1, 3])
    array([[1., 0., 0.],
           [0., 0., 1.]])
    """
    sub_vertices = np.empty((len(f), embedding_dimension(vertices)))
    for i in range(len(f)):
        sub_vertices[i] = vertices[f[i]]
    return sub_vertices


def is_degenerate(vertices, eps=1e-4):
    r"""
    Check if a simplex is degenerate.

    A simplex is considered degenerate if the ratio of any height of the simplex to the simplex diameter is smaller
    than the specified threshold :math:`\varepsilon`,

    .. math:: \min_{i = 1, 2, \ldots, n + 1} \frac{h_i}{d} < \varepsilon,

    where :math:`h_i` is the i:th height of the simplex (see :func:`altitude`) and d is the diameter of the simplex
    (see :func:`diameter`).

    :param vertices: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    :param float eps: Triangle angle tolerance in the degeneracy check.
    :return: Whether or not the triangle is degenerate (has an angle that is too small).
    :rtype: bool
    """
    d = diameter(vertices)
    hs = [altitude(vertices, i) for i in range(len(vertices))]
    return any([h / d < eps for h in hs])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
