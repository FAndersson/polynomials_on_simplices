"""Calculations on a triangle mesh."""

import math

import numpy

import polynomials_on_simplices.geometry.primitives.triangle as triangle


def get_num_vertices(triangles):
    """
    Get the number of vertices in a triangulation.

    :param triangles: List of triangles in the triangulation.
    :returns: Number of vertices in the triangulation.
    """
    return numpy.amax(numpy.reshape(triangles, -1)) + 1


def vertices(tri, vertex_list):
    """
    Get the vertices of a triangle in the mesh.

    :param tri: Triangle (list of 3 vertex indices).
    :param vertex_list: List of vertices in the mesh. Size (m, dim) where m is the number of vertices.
    :returns: 3 by dim matrix containing the triangle vertices as row vectors.
    """
    dim = len(vertex_list[0])
    p = numpy.zeros((3, dim))
    for j in range(3):
        p[j] = vertex_list[tri[j]]
    return p


def normals(t, v):
    """
    Compute triangle normals for each triangle in the mesh.

    :param t: List of triangle indices in the mesh. Size (n, 3) where n is the number of triangles.
    :param v: List of vertices in the mesh. Size (m, 3) where m is the number of vertices.
    :returns: List of normalized triangle normals.
    """
    n = numpy.zeros((len(t), 3))
    for i in range(0, len(t)):
        p = vertices(t[i], v)
        n[i] = triangle.normal(p)
    return n


def edge_index(indexed_triangle, edge):
    """
    Find the index of an edge in a triangle.

    .. note:: The i:th edge of a triangle is the edge opposite to the i:th vertex, i in {0, 1, 2}.

    :param indexed_triangle: Indices for the three vertices in the triangle.
    :type indexed_triangle: Length 3 array of ints
    :param edge: Indices for the two vertices of the edge.
    :type edge: Pair of ints
    :return: Index (in {0, 1, 2}) of the edge in the triangle.
    :rtype: int

    .. rubric:: Examples

    >>> edge_index([0, 1, 2], (0, 1))
    2
    >>> edge_index([3, 8, 11], (3, 11))
    1
    """
    for i in range(3):
        triangle_edge = indexed_triangle[(i + 1) % 3], indexed_triangle[(i + 2) % 3]
        if triangle_edge == edge:
            return i
        triangle_edge = triangle_edge[1], triangle_edge[0]
        if triangle_edge == edge:
            return i
    # Edge not found in triangle
    assert False


def centroids(t, v):
    """
    Compute triangle centroids for each triangle in the mesh.

    :param t: List of triangle indices in the mesh. Size (n, 3) where n is the number of triangles.
    :param v: List of vertices in the mesh. Size (m, 3) where m is the number of vertices.
    :return: List of triangle centroids.
    """
    c = numpy.zeros((len(t), 3))
    for i in range(len(t)):
        p = vertices(t[i], v)
        c[i] = triangle.centroid(p)
    return c


def centroid(t, v):
    """
    Compute the centroid of a triangle mesh
    (the centroid is the center of gravity if all triangles have uniform and identical density).

    :param t: List of triangle indices in the mesh. Size (n, 3) where n is the number of triangles.
    :param v: List of vertices in the mesh. Size (m, 3) where m is the number of vertices.
    :return: The centroid of the triangle mesh.
    """
    c = numpy.zeros(v[0].shape)
    total_area = 0
    for i in range(len(t)):
        p = vertices(t[i], v)
        ct = triangle.centroid(p)
        area = triangle.area(p)
        c += area * ct
        total_area += area
    c /= total_area
    return c


def vertex_normals_cotangent(t, v):
    """
    Compute a normal at each vertex in the mesh using the cotan formula.
    This expression arises when you identify the vertex normal with the
    surface area gradient with respect to the vertex coordinates.
    """
    n = numpy.zeros(v.shape)

    for i in range(0, t.shape[0]):
        p = vertices(t[i], v)
        e = triangle.edges(p)
        for j in range(3):
            alpha = triangle.angle(p, j)
            cota = 1 / math.tan(alpha)
            v0 = t[i][(j + 1) % 3]
            v1 = t[i][(j + 2) % 3]
            n[v0] += e[j] * cota
            n[v1] -= e[j] * cota

    for i in range(0, n.shape[0]):
        n[i] /= numpy.linalg.norm(n[i])

    return -n


def vertex_normals(t, v, weight="area"):
    """
    Compute vertex normals for each vertex in the mesh by computing
    a weighted sum of the triangle normals surrounding each vertex.

    The area weighted vertex normal arises when you identify the vertex
    normal with the volume gradient with respect to the vertex coordinates.

    :param t: List of triangle indices in the mesh. Size (n, 3) where n is the number of triangles.
    :param v: List of vertices in the mesh. Size (m, 3) where m is the number of vertices.
    :param str weight: Weight factor used when computing the weighted sum of triangle normals. Possible values:

        - None: 1 is used as weight for each triangle normal.
        - "area": Each triangle normal is weighted with the area of the triangle.
        - "angle": Each triangle normal is weighted with the triangle angle at the vertex.

    :returns: List of normalized vertex normals.
    """
    # Compute triangle normals
    tn = normals(t, v)

    n = numpy.zeros(v.shape)

    for i in range(0, t.shape[0]):
        p = vertices(t[i], v)
        for j in range(0, 3):
            w = 1
            if weight == "area":
                w = triangle.area(p)
            if weight == "angle":
                w = triangle.angle(p, j)
            if weight == "centroid":
                c = triangle.centroid(p)
                w = 1 / numpy.linalg.norm(c - v[t[i][j]])
            n[t[i][j]] += w * tn[i]

    # Normalize vertex normals
    for i in range(0, v.shape[0]):
        n[i] /= numpy.linalg.norm(n[i])

    return n


def neighbour(t, i, j):
    """
    Get the triangle edge neighbour of a triangle.

    :param t: List of triangle indices in the mesh. Size (n, 3) where n is the number of triangles.
    :param i: Triangle index.
    :param j: Edge index.
    :returns: Index of triangle neighbouring triangle i along edge j, or None if no neighbour exists.
    """
    v0 = t[i][(j + 1) % 3]
    v1 = t[i][(j + 2) % 3]

    for k in range(len(t)):
        if k != i:
            if v0 in t[k] and v1 in t[k]:
                return k

    return None


def has_vertex(t, tri, vertex):
    """
    Whether or not a triangle contains a specific vertex.

    :param t: List of triangle indices in the mesh. Size (n, 3) where n is the number of triangles.
    :param tri: Triangle index.
    :param vertex: Vertex index.
    :returns: True if the triangle contains the vertex, otherwise False.
    """
    for i in range(3):
        if t[tri][i] == vertex:
            return True
    return False


def vertex_position(indexed_triangle, vertex_index):
    """
    Position of a specific vertex in an indexed triangle.

    :param indexed_triangle: List of three vertex indices.
    :param vertex_index: Vertex index to look for in the triangle.
    :return: 0, 1 or 2 depending on whether the vertex is the first, second or third vertex in the triangle.
        Returns None if the vertex does not exist in the triangle.
    """
    for i in range(3):
        if indexed_triangle[i] == vertex_index:
            return i
    return None


def opposite_vertex(t, tri, edge):
    """
    Get the opposite vertex of a triangle across one of its edges.

    :param t: List of triangle indices in the mesh. Size (n, 3) where n is the number of triangles.
    :param tri: Triangle index.
    :param edge: Edge index.
    :returns: Index of vertex opposite edge j of triangle i, or None if no neighbour exists.
    """
    tn = neighbour(t, tri, edge)
    if tn is None:
        return None

    for i in range(3):
        if not has_vertex(t, tri, t[tn][i]):
            return t[tn][i]

    return None


def has_same_orientation(triangle1, triangle2):
    """
    Check if two neighbouring indexed triangles have the same orientation.

    :param triangle1: First indexed triangle (list of three vertex indices).
    :param triangle2: Second indexed triangle (list of three vertex indices).
    :return: True if the orientation is the same, False otherwise.
    """
    # Find the two common vertices
    common_vertices = list(set(triangle1).intersection(triangle2))
    # Find the position of the two common vertices in the two triangles
    idx10 = vertex_position(triangle1, common_vertices[0])
    idx11 = vertex_position(triangle1, common_vertices[1])
    idx20 = vertex_position(triangle2, common_vertices[0])
    idx21 = vertex_position(triangle2, common_vertices[1])
    # Compute the order of the two common vertices
    order1 = (idx10 - idx11) % 3
    order2 = (idx20 - idx21) % 3
    if order1 != order2:
        return True
    return False


def has_same_edge_orientation(triangles, tri_idx, edge):
    """
    Check if a triangle and its neighbour across a given edge have the same orientation.

    :param triangles: List of triangles in the triangulation.
    :param tri_idx: Triangle we originate from.
    :param edge: Edge index in {0, 1, 2}. We compare the orientation with the triangle across this edge.
    :return: True if the orientation is the same, or if the triangle don't have any neighbour
        across the given edge. False otherwise.
    """
    neigh_idx = neighbour(triangles, tri_idx, edge)
    if neigh_idx is None:
        return True
    # Find position of first edge vertex in neighbouring triangle
    v0 = triangles[tri_idx][(edge + 1) % 3]
    idx = -1
    for i in range(3):
        if triangles[neigh_idx][i] == v0:
            idx = i
            break
    # If the second edge vertex comes before the first one in the neighbouring triangle, the orientations are the same
    v1 = triangles[tri_idx][(edge + 2) % 3]
    if triangles[neigh_idx][(idx - 1) % 3] == v1:
        return True
    return False


def swap_orientation(triangles, tri_idx):
    """
    Change the orientation of a triangle.

    :param triangles: List of triangles in the triangulation.
    :param tri_idx: Triangle which should have its orientation swapped.
    :return: Nothing.
    """
    triangles[tri_idx][1], triangles[tri_idx][2] = triangles[tri_idx][2], triangles[tri_idx][1]


def has_consistent_orientation(triangles):
    """
    Check if all triangles in a triangulation have the same orientation.

    :param triangles: List of triangles in the triangulation.
    :return: True if all the triangles have the same orientation. False otherwise.
    """
    num_triangles = len(triangles)
    for tri_idx in range(num_triangles):
        for edge in range(3):
            if not has_same_edge_orientation(triangles, tri_idx, edge):
                return False
    return True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
