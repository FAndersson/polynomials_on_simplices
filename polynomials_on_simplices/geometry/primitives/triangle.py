"""Compute different triangle properties. A triangle is defined by its 3 vertices, given as rows in a matrix."""

import numpy

from polynomials_on_simplices.calculus.angle import compute_angle
from polynomials_on_simplices.calculus.real_interval import in_closed_range, in_open_range
from polynomials_on_simplices.linalg.vector_space_projection import vector_oblique_projection_2


def edges(vertices):
    """
    Compute the edge vectors of a triangle. The i:th edge is the edge opposite the i:th vertex.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle edge vectors (3 by n matrix with the edges as rows (where n is the dimension of the space)).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    n = len(vertices[0])
    v = numpy.zeros((3, n))
    for i in range(3):
        v[i] = vertices[(i + 2) % 3] - vertices[(i + 1) % 3]
    return v


def edge_lengths(vertices):
    """
    Compute the length of each edge in a triangle (i.e. the length of each vector returned by the :func:`edges`
    function.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: Length 3 array of edge lengths.
    :rtype: List[float]
    """
    return [numpy.linalg.norm(edge) for edge in edges(vertices)]


def diameter(vertices):
    """
    Return largest distance between two points in a triangle, i.e. the length of the longest triangle edge.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :return: Triangle diameter.
    :rtype: float
    """
    el = edge_lengths(vertices)
    return max(
        el[0],
        max(
            el[1],
            el[2]
        )
    )


def edges_2(vertices):
    """
    Compute the edge vectors of a triangle, expressed in the triangle
    plane orthonormal basis.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle edge vectors (3 by 2 matrix with the coordinates for edge i in row i).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    v = edges(vertices)
    b = basis(vertices)

    v2 = numpy.zeros((3, 2))
    for i in range(3):
        for j in range(2):
            v2[i, j] = numpy.dot(v[i], b[j])
    return v2


def dual_edges(vertices):
    r"""
    Compute the dual edge vectors of a triangle.
    Dual edge vectors are normal to the edge vectors and points out of the triangle, :math:`d_i = v_i \times n`,
    where :math:`v_i` are the triangle edges and :math:`n` is the triangle normal.

    :param vertices: The triangle vertices (3 by 2 or 3 by 3 matrix with the vertices as rows).
    :returns: The triangle dual edge vectors.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    # For now only works in 2d and 3d
    assert len(vertices[0]) == 2 or len(vertices[0]) == 3
    if len(vertices[0]) == 2:
        # Triangle in 2d
        v = edges(vertices)
        t = numpy.empty((3, 2))
        for i in range(3):
            t[i][1] = v[i][0]
            t[i][0] = -v[i][1]
            if numpy.dot(t[i], v[(i + 1) % 3]) > 0.0:
                t[i] *= -1
        return t
    # Triangle in 3d
    v = edges(vertices)
    n = normal(vertices)
    t = numpy.empty((3, 3))
    for i in range(0, 3):
        t[i] = numpy.cross(v[i], n)
    return t


def altitude_vectors(vertices):
    r"""
    Compute the altitude vectors of a triangle.

    An altitude vector is a line segment from a vertex to the opposite edge, such that it is perpendicular to the edge.
    An altitude vector is parallel with the corresponding dual edge vector. But whereas the dual edge vector has
    the same length as the edge, the length of the altitude vector is the altitude or height of the triangle.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :return: The triangle altitude vectors.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    a = area(vertices)
    hv = dual_edges(vertices)
    for i in range(3):
        h = 2 * a / numpy.linalg.norm(vertices[(i + 1) % 3] - vertices[(i + 2) % 3])
        hv[i] = h * hv[i] / numpy.linalg.norm(hv[i])
    return hv


def altitudes(vertices):
    """
    Compute the altitudes in a triangle (i.e. the shortest distance from each vertex to the line spanned by the
    opposite edge).

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: Length 3 array of triangle altitudes.
    :rtype: List[float]
    """
    a = area(vertices)
    heights_values = 3 * [0]
    for i in range(3):
        # From the formula A = b * h / 2 => h = 2 * A / b
        heights_values[i] = 2 * a / numpy.linalg.norm(vertices[(i + 1) % 3] - vertices[(i + 2) % 3])
    return heights_values


def altitude_feet(vertices):
    """
    Compute the foot of each altitude (the point where the altitude vector intersects the base vector).

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :return: 3 by n matrix with the altitude feet as rows.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    n = len(vertices[0])
    feet = numpy.empty((3, n))
    for i in range(3):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % 3]
        v2 = vertices[(i + 2) % 3]
        s = -numpy.dot(v2 - v1, v1 - v0) / numpy.dot(v2 - v1, v2 - v1)
        feet[i] = (1 - s) * v1 + s * v2
    return feet


def dual_edges_2(vertices):
    """
    Compute the dual edge vectors of a triangle, expressed in the
    triangle plane orthonormal basis.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle dual edge vectors (3 by 2 matrix with the coordinates for edge i in row i).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    t = dual_edges(vertices)

    t2 = numpy.zeros((3, 2))
    for i in range(3):
        t2[i] = in_triangleplane_coords(vertices, t[i])
    return t2


def basis(vertices):
    r"""
    Compute an orthonormal basis :math:`{b_0, b_1}` in the triangle plane,
    such that :math:`b_0 \times b_1 = n` and :math:`b_0 || v_0`.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The orthonormal basis.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    # For now only works in 2d and 3d
    assert len(vertices[0]) == 2 or len(vertices[0]) == 3
    if len(vertices[0]) == 2:
        # Triangle in 2d
        b = numpy.zeros((2, 2))
        v = edges(vertices)
        b[0] = v[0] / numpy.linalg.norm(v[0])
        b[1] = v[1] - numpy.dot(v[1], b[0]) * b[0]
        b[1] /= numpy.linalg.norm(b[1])
        return b
    # Triangle in 3d
    b = numpy.zeros((2, 3))
    v = edges(vertices)
    b[0] = v[0] / numpy.linalg.norm(v[0])
    n = normal(vertices)
    b[1] = numpy.cross(n, b[0])

    return b


def in_triangleplane_coords(vertices, v):
    """
    Compute the representation of the vector v in the triangle
    plane orthonormal basis.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :param v: Vector whose coordinates we are interested in.
    :type v: n-dimensional vector
    :returns: Vector expressed in the triangle plane orthonormal basis (2d array with vector components).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    b = basis(vertices)
    v2 = numpy.zeros(2)
    for i in range(2):
        v2[i] = numpy.dot(v, b[i])
    return v2


def area_weighted_normal(vertices):
    r"""
    Compute the area weighted normal of a triangle (:math:`2 A N = (p_1 - p_0) \times (p_2 - p_1)`), where
    :math:`p_0, p_1, p_2` are the vertices of the triangle.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The area weighted triangle normal.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    v = edges(vertices)
    n = numpy.cross(v[0], v[1])
    return n


def normal(vertices):
    """Compute the normal of a triangle.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The normalized triangle normal.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    v = edges(vertices)
    n = numpy.cross(v[0], v[1])
    n /= numpy.linalg.norm(n)
    return n


def area(vertices):
    """
    Compute the area of a triangle.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle area.
    :rtype: float
    """
    v = edges(vertices)
    v0xv1 = numpy.cross(v[0], v[1])
    return 0.5 * numpy.linalg.norm(v0xv1)


def perimeter(vertices):
    """
    Compute the perimeter of a triangle.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle perimeter.
    :rtype: float
    """
    return sum(edge_lengths(vertices))


def angle(vertices, i):
    """
    Compute the angle of triangle at a vertex of the triangle.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :param i: Vertex at which we compute the triangle angle.
    :returns: The triangle angle at the supplied vertex.
    :rtype: float
    """
    v = edges(vertices)
    u0 = -v[(i + 1) % 3]
    u1 = v[(i + 2) % 3]
    return compute_angle(u0, u1)


def medians(vertices):
    """
    Compute the median vectors of a triangle, pointing from each
    vertex into the triangle to the midpoint of the opposite edge.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle median vectors (3 by n matrix with the median vectors as rows).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    m = numpy.zeros((3, len(vertices[0])))
    for i in range(3):
        m[i] = 0.5 * (vertices[(i + 1) % 3] + vertices[(i + 2) % 3]) - vertices[i]
    return m


def medians_2(vertices):
    """
    Compute the median vectors of a triangle, pointing from each
    vertex into the triangle.
    Expressed in the triangle plane orthonormal basis.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle median vectors (3 by 2 matrix with the median vectors as rows).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    m = medians(vertices)

    m2 = numpy.zeros((3, 2))
    for i in range(3):
        m2[i] = in_triangleplane_coords(vertices, m[i])
    return m2


def barycentric_to_cartesian(bary, vertices):
    """
    Compute the Cartesian coordinates of a point with given barycentric coordinates.

    :param bary: The barycentric coordinates.
    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The Cartesian coordinates vector.
    :rtype: n-dimensional vector
    """
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2]


def cartesian_to_barycentric(cartesian, vertices):
    """
    Compute the barycentric coordinates of a point with given
    Cartesian coordinates.

    :param cartesian: The Cartesian coordinates.
    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The barycentric coordinates (length 3 array).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    # Compute triangle edges and medians
    edge_vectors = edges_2(vertices)
    median_vectors = medians_2(vertices)
    # Project the Cartesian point along the medians onto the opposing edges
    v1 = in_triangleplane_coords(vertices, cartesian - vertices[0])
    v1 = vector_oblique_projection_2(v1, edge_vectors[2], median_vectors[2])
    v2 = in_triangleplane_coords(vertices, cartesian - vertices[1])
    v2 = vector_oblique_projection_2(v2, edge_vectors[0], median_vectors[0])
    r1 = numpy.dot(v1, edge_vectors[2]) / numpy.dot(edge_vectors[2], edge_vectors[2])
    r2 = numpy.dot(v2, edge_vectors[0]) / numpy.dot(edge_vectors[0], edge_vectors[0])
    # Compute barycentric coordinates based on the position of the projected points
    A = numpy.array([[0.5, -0.5], [0.5, 1]])
    b = numpy.array([0.5 - r1, 1 - r2])
    x = numpy.linalg.solve(A, b)
    bary = numpy.zeros(3)
    bary[0] = x[0]
    bary[1] = x[1]
    bary[2] = 1 - bary[0] - bary[1]
    return bary


def barycentric_to_trilinear(bary, vertices):
    """
    Compute the trilinear coordinates (ratios between the distances to the triangle edges) of a point with given
    barycentric coordinates.

    :param bary: The barycentric coordinates.
    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The trilinear coordinates (length 3 array).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    e = edges(vertices)
    trilinear = numpy.zeros(3)
    for i in range(3):
        trilinear[i] = bary[i] / numpy.linalg.norm(e[i])
    return normalize_trilinear_coordinates(trilinear, vertices)


def trilinear_to_barycentric(trilinear, vertices):
    """
    Compute the barycentric coordinates of a point with given trilinear coordinates (ratios between the distances
    to the triangle edges).

    :param trilinear: The trilinear coordinates.
    :type trilinear: 3d vector
    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The barycentric coordinates (length 3 array).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    e = edges(vertices)
    bary = numpy.zeros(3)
    for i in range(3):
        bary[i] = trilinear[i] * numpy.linalg.norm(e[i])
    # Normalize the barycentric coordinates (so that they sum to 1)
    bary *= 1 / sum(b for b in bary)
    return bary


def trilinear_to_side_distances(trilinear, vertices):
    """
    Compute actual distances to the triangle edges for a point with given trilinear coordinates (ratios between
    the distances to the triangle edges).

    :param trilinear: The trilinear coordinates.
    :type trilinear: 3d vector
    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :return: Distance from the point to each edge of the triangle (length 3 array).
    :rtype: 3d vector
    """
    # See https://en.wikipedia.org/wiki/Trilinear_coordinates#Conversions
    el = edge_lengths(vertices)
    denominator = 0.0
    for i in range(3):
        denominator += el[i] * trilinear[i]
    k = 2 * area(vertices) / denominator
    return k * trilinear


def side_distances_to_trilinear(side_distances):
    """
    Compute trilinear coordinates (ratios between the distances to the triangle edges) for a point with given
    distances to each edge of the triangle.

    :param side_distances: Distance from the point to each edge of the triangle.
    :type side_distances: 3d vector
    :return: Trilinear coordinates.
    :rtype: 3d vector
    """
    # Already normalized (exact) trilinear coordinates
    return side_distances


def normalize_trilinear_coordinates(trilinear, vertices):
    """
    Compute normalized (or exact) trilinear coordinates (ratios between the distances to the triangle edges).

    Two trilinear coordinates x:y:z and kx:ky:kz describe the same point. Here normalized means that the three
    entries in the trilinear coordinates is exactly the distance from the point to the three edges.
    See http://mathworld.wolfram.com/ExactTrilinearCoordinates.html.

    :param trilinear: Trilinear coordinates.
    :type trilinear: 3d vector
    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :return: Equivalent normalized trilinear coordinates.
    :rtype: 3d vector
    """
    return trilinear_to_side_distances(trilinear, vertices)


def centroid(vertices):
    """
    Compute the centroid of a triangle (center of gravity).

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle centroid.
    :rtype: n-dimensional vector
    """
    return (vertices[0] + vertices[1] + vertices[2]) / 3


def circumcenter(vertices):
    """
    Compute the circumcenter of a triangle (the center of the circle which passes through all the vertices of the
    triangle).

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle circumcenter.
    :rtype: n-dimensional vector
    """
    # Compute trilinear coordinates
    trilinear = numpy.zeros(3)
    for i in range(3):
        trilinear[i] = numpy.cos(angle(vertices, i))
    bary = trilinear_to_barycentric(trilinear, vertices)
    return barycentric_to_cartesian(bary, vertices)


def circumradius(vertices):
    """
    Compute the circumradius of a triangle (the radius of the circle which passes through all the vertices of the
    triangle). See http://mathworld.wolfram.com/Circumradius.html.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle circumradius.
    :rtype: float
    """
    el = edge_lengths(vertices)
    a = el[0]
    b = el[1]
    c = el[2]
    r = a * b * c / numpy.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))
    return r


def orthocenter(vertices):
    """
    Compute the orthocenter of a triangle (the point where all the triangle heights intersect).

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle orthocenter.
    :rtype: n-dimensional vector
    """
    # Compute trilinear coordinates
    trilinear = numpy.zeros(3)
    for i in range(3):
        alpha = angle(vertices, i)
        if alpha != numpy.pi / 2:
            trilinear[i] = 1 / numpy.cos(angle(vertices, i))
        else:
            trilinear = numpy.zeros(3)
            trilinear[i] = 1
            break
    bary = trilinear_to_barycentric(trilinear, vertices)
    return barycentric_to_cartesian(bary, vertices)


def incenter(vertices):
    """
    Compute the incenter of a triangle (the center of the largest inscribed circle or the intersection of all the angle
    bisectors).

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle incenter.
    :rtype: n-dimensional vector
    """
    trilinear = numpy.array([1, 1, 1])
    bary = trilinear_to_barycentric(trilinear, vertices)
    return barycentric_to_cartesian(bary, vertices)


def inradius(vertices):
    """
    Compute the inradius of a triangle (the radius of the of the largest inscribed circle). See
    http://mathworld.wolfram.com/Inradius.html.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :returns: The triangle inradius.
    :rtype: float
    """
    a = area(vertices)
    s = perimeter(vertices) / 2
    return a / s


def inside_triangle(point, vertices, include_boundary=True):
    """
    Check whether or not a point lies inside a triangle.

    :param point: Point which we want to check.
    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :param include_boundary: Whether or not to consider points on the boundary as inside the triangle.
    :return: Whether or not the points lies inside the triangle.
    :rtype: bool
    """
    bary = cartesian_to_barycentric(point, vertices)
    # Point is inside the triangle if all barycentric coordinates are in the range [0, 1]
    # (or (0, 1) if we exclude the boundary)
    if include_boundary:
        in_range = in_closed_range
    else:
        in_range = in_open_range
    return all([in_range(b, 0, 1) for b in bary])


def is_degenerate(vertices, eps=0.0002):
    r"""
    Check if a triangle is degenerate.

    A triangle is considered degenerate if any of its angles is smaller than :math:`\varepsilon` radians,

    .. math:: \min_{i = 1, 2, 3} \alpha_i < \varepsilon.

    :param vertices: The triangle vertices (3 by n matrix with the vertices as rows (where n is the dimension of the
        space)).
    :param float eps: Triangle angle tolerance in the degeneracy check.
    :return: Whether or not the triangle is degenerate (has an angle that is too small).
    :rtype: bool
    """
    d = numpy.cos(eps)**2
    e = edges(vertices)
    squared_edge_lengths = [numpy.dot(e[i], e[i]) for i in range(3)]
    edge_dot_products2 = [numpy.dot(e[(i + 1) % 3], e[(i + 2) % 3])**2 for i in range(3)]
    for i in range(3):
        if edge_dot_products2[i] / (squared_edge_lengths[(i + 1) % 3] * squared_edge_lengths[(i + 2) % 3]) > d:
            return True
    return False
