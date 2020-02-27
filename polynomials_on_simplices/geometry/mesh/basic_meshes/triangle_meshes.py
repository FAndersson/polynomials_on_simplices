"""Triangulated geometries."""

import numpy as np

from polynomials_on_simplices.algebra.multiindex import MultiIndexIterator, general_to_exact_norm


def rectangle_triangulation(width_resolution=2, height_resolution=2):
    """
    Create a triangulation of a rectangle.

    :param width_resolution: Number of vertices along the width of the rectangle.
    :param height_resolution: Number of vertices along the height of the rectangle.
    :returns: n by 3 list of triangles.
    """
    assert width_resolution > 1 and height_resolution > 1

    num_triangles = 2 * (width_resolution - 1) * (height_resolution - 1)
    triangles = np.empty((num_triangles, 3), dtype=int)

    idx = 0
    for i in range(height_resolution - 1):
        for j in range(width_resolution - 1):
            triangles[idx][0] = i * width_resolution + j
            triangles[idx][1] = i * width_resolution + j + 1
            triangles[idx][2] = (i + 1) * width_resolution + j
            idx += 1

            triangles[idx][0] = i * width_resolution + j + 1
            triangles[idx][1] = (i + 1) * width_resolution + j + 1
            triangles[idx][2] = (i + 1) * width_resolution + j
            idx += 1

    return triangles


def rectangle_vertices(width, height, width_resolution=2, height_resolution=2):
    """
    Create the vertices of a rectangle in the xy-plane centered in the origin,
    and with edges aligned with the x- and y-axis.

    :param width: Width of the rectangle.
    :param height: Height of the rectangle.
    :param width_resolution: Number of vertices along the width of the rectangle.
    :param height_resolution: Number of vertices along the height of the rectangle.
    :returns: n by 3 list of vertices.
    """
    assert width_resolution > 1 and height_resolution > 1

    num_vertices = width_resolution * height_resolution
    vertices = np.empty((num_vertices, 3))

    idx = 0
    for i in range(height_resolution):
        s = i / (height_resolution - 1)
        y = (1 - s) * -height / 2 + s * height / 2
        for j in range(width_resolution):
            t = j / (width_resolution - 1)
            x = (1 - t) * -width / 2 + t * width / 2
            vertices[idx][0] = x
            vertices[idx][1] = y
            vertices[idx][2] = 0
            idx += 1

    return vertices


def unit_square_vertices(width_resolution=2, height_resolution=2):
    """
    Create the vertices of the unit square in the xy-plane.

    :param width_resolution: Number of vertices along the width of the unit square.
    :param height_resolution: Number of vertices along the height of the unit square.
    :returns: n by 3 list of vertices.
    """
    assert width_resolution > 1 and height_resolution > 1

    num_vertices = width_resolution * height_resolution
    vertices = np.empty((num_vertices, 3))

    idx = 0
    for i in range(height_resolution):
        s = i / (height_resolution - 1)
        y = s
        for j in range(width_resolution):
            t = j / (width_resolution - 1)
            x = t
            vertices[idx][0] = x
            vertices[idx][1] = y
            vertices[idx][2] = 0
            idx += 1

    return vertices


def disc_triangulation(angular_resolution=20, radial_resolution=2):
    """
    Create a triangulation of a disc.

    :param angular_resolution: Number of vertices in the triangulation, going around the disc.
    :param radial_resolution: Number of vertices in the triangulation, from the center of the disc to the perimeter.
    :returns: n by 3 list of triangles.
    """
    if angular_resolution < 3:
        raise ValueError("Angular resolution need to be greater than or equal to three")
    if radial_resolution < 2:
        raise ValueError("Radial resolution need to be greater than or equal to two")

    num_triangles = angular_resolution + 2 * (radial_resolution - 2) * angular_resolution
    triangles = np.empty((num_triangles, 3), dtype=int)

    # Add triangles around the center vertex
    for i in range(angular_resolution):
        triangles[i][0] = 0
        triangles[i][1] = 1 + i
        triangles[i][2] = 1 + (i + 1) % angular_resolution

    # Add triangles outwards
    if radial_resolution > 2:
        triangles[angular_resolution:] = annulus_triangulation(angular_resolution, radial_resolution - 1)
        # Offset the triangles to take the center vertex into account
        offset_triangulation_vertices(triangles[angular_resolution:], 1)

    return triangles


def annulus_triangulation(angular_resolution, radial_resolution):
    """
    Create a triangulation of an annulus.

    :param angular_resolution: Number of vertices in the triangulation, going around the annulus.
    :param radial_resolution: Number of vertices in the triangulation, from the inner perimeter to the outer perimeter.
    :returns: n by 3 list of triangles.
    """
    if angular_resolution < 3:
        raise ValueError("Angular resolution need to be greater than or equal to three")
    if radial_resolution < 2:
        raise ValueError("Radial resolution need to be greater than or equal to two")

    num_triangles = 2 * (radial_resolution - 1) * angular_resolution
    triangles = np.empty((num_triangles, 3), dtype=int)

    # Add triangles for each ring
    idx = 0
    for i in range(radial_resolution - 1):
        for j in range(angular_resolution):
            triangles[idx][0] = i * angular_resolution + j
            triangles[idx][1] = (i + 1) * angular_resolution + j
            triangles[idx][2] = (i + 1) * angular_resolution + (j + 1) % angular_resolution
            idx += 1

            triangles[idx][0] = i * angular_resolution + j
            triangles[idx][1] = (i + 1) * angular_resolution + (j + 1) % angular_resolution
            triangles[idx][2] = i * angular_resolution + (j + 1) % angular_resolution
            idx += 1

    return triangles


def triangle_triangulation(edge_resolution):
    """
    Create a triangulation of a triangle.

    :param edge_resolution: Number of vertices along each edge of the triangle.
    :return: n by 3 list of triangles.
    """
    num_triangles = (edge_resolution - 1)**2
    triangles = np.empty((num_triangles, 3), dtype=int)

    idx = 0
    from polynomials_on_simplices.algebra.multiindex import MultiIndexIterator, norm
    for mi in MultiIndexIterator(2, edge_resolution - 1):
        if norm(mi) < edge_resolution - 1:
            # Index of current vertex and the vertex one row up
            i0 = (2 * edge_resolution - mi[1] + 1) / 2 * mi[1] + mi[0]
            i1 = (2 * edge_resolution - (mi[1] + 1) + 1) / 2 * (mi[1] + 1) + mi[0]

            triangles[idx][0] = i0
            triangles[idx][1] = i0 + 1
            triangles[idx][2] = i1
            idx += 1

            if norm(mi) < edge_resolution - 2:
                triangles[idx][0] = i0 + 1
                triangles[idx][1] = i1 + 1
                triangles[idx][2] = i1
                idx += 1

    return triangles


def triangle_vertices(width=1.0, height=1.0, edge_resolution=3):
    """
    Create the vertices of a triangle in the xy-plane with corners in (0, 0),
    (width, 0) and (0, height).

    :param width: 'Width' of the triangle.
    :param height: 'Height' of the triangle.
    :param edge_resolution: Number of vertices along each edge of the triangle.
    :return: n by 3 list of vertices.
    """
    num_vertices = int(edge_resolution * (edge_resolution + 1) / 2)
    vertices = np.empty((num_vertices, 3))

    idx = 0
    for row in range(edge_resolution):
        y = (row / (edge_resolution - 1)) * height
        for col in range(edge_resolution - row):
            x = (col / (edge_resolution - 1)) * width
            vertices[idx] = np.array([x, y, 0.0])
            idx += 1

    return vertices


def equilateral_triangle_vertices(edge_length=1.0, edge_resolution=3):
    """
    Create the vertices of an equilateral triangle in the xy-plane with
    two corners along the x-axis and the third corner in the first quadrant.

    :param edge_length: Length of each edge in the triangle.
    :param edge_resolution: Number of vertices along each edge of the triangle.
    :return: n by 3 list of vertices.
    """
    num_vertices = int(edge_resolution * (edge_resolution + 1) / 2)
    vertices = np.empty((num_vertices, 3))

    width = edge_length
    height = np.sqrt(3) / 2 * edge_length

    idx = 0
    for row in range(edge_resolution):
        y = (row / (edge_resolution - 1)) * height
        for col in range(edge_resolution - row):
            x = (col / (edge_resolution - 1)) * width + row * width / (edge_resolution - 1) / 2
            vertices[idx] = np.array([x, y, 0.0])
            idx += 1

    return vertices


def general_triangle_vertices(tri_vertices, edge_resolution=3):
    """
    Create the vertices of a triangle defined by its three vertices.

    :param tri_vertices: The three vertices of the triangle.
    :param edge_resolution: Number of vertices along each edge of the triangle.
    :return: n by 3 or n by 2 list of vertices (depending on the shape of the input triangle vertices).
    """
    dim = tri_vertices.shape[1]
    num_vertices = int(edge_resolution * (edge_resolution + 1) / 2)
    vertices = np.empty((num_vertices, dim))

    idx = 0
    for row in range(edge_resolution):
        w = row / (edge_resolution - 1)
        for col in range(edge_resolution - row):
            v = col / (edge_resolution - 1)
            u = 1 - v - w
            vertices[idx] = u * tri_vertices[0] + v * tri_vertices[1] + w * tri_vertices[2]
            idx += 1

    return vertices


def triangle_mesh_triangulation(original_triangles, edge_resolution):
    """
    Create a triangulation of a triangle mesh, where each original triangle is subdivided with the given number
    of vertices along each edge.

    :param original_triangles: Triangle mesh to be subdivided.
    :param edge_resolution: Number of vertices along each edge in the original triangle mesh.
    :return: n by 3 list of triangles.
    """
    from python_math_library.finite_element.lagrange import generate_local_to_global_map
    num_triangle_triangles = (edge_resolution - 1)**2
    num_triangles = len(original_triangles) * num_triangle_triangles
    triangles = np.empty((num_triangles, 3), dtype=int)

    local_to_global_map, _ = generate_local_to_global_map(original_triangles, edge_resolution - 1)
    general_triangle_triangles = triangle_triangulation(edge_resolution)

    for i in range(len(original_triangles)):
        triangle_triangles = np.copy(general_triangle_triangles)
        local_vi = 0
        for mi in MultiIndexIterator(2, edge_resolution - 1):
            global_vi = local_to_global_map(i, tuple(mi))
            for j in range(len(triangle_triangles)):
                for k in range(3):
                    if general_triangle_triangles[j][k] == local_vi:
                        triangle_triangles[j][k] = global_vi
            local_vi += 1
        triangles[i * num_triangle_triangles:(i + 1) * num_triangle_triangles, :] = triangle_triangles

    return triangles


def triangle_mesh_vertices(original_triangles, original_vertices, edge_resolution):
    """
    Create the vertices of a triangle mesh, where each original triangle is subdivided with the given number
     of vertices along each edge.

    :param original_triangles: Triangle mesh to be subdivided.
    :param original_vertices: Vertices in the original triangle mesh.
    :param edge_resolution: Number of vertices along each edge in the original triangle mesh.
    :return: n by 3 list of vertices.
    """
    from python_math_library.finite_element.lagrange import generate_local_to_global_map
    dim = len(original_vertices[0])
    local_to_global_map, num_dofs = generate_local_to_global_map(original_triangles, edge_resolution - 1)
    vertices = np.empty((num_dofs, dim))
    vertex_evaluated = np.zeros(num_dofs)

    for i in range(len(original_triangles)):
        for mi in MultiIndexIterator(2, edge_resolution - 1):
            global_vi = local_to_global_map(i, tuple(mi))
            if vertex_evaluated[global_vi]:
                continue
            mi_exact = general_to_exact_norm(mi, edge_resolution - 1)
            v = np.zeros(dim)
            for j in range(3):
                v += mi_exact[j] * original_vertices[original_triangles[i][j]]
            v /= (edge_resolution - 1)
            vertices[global_vi, :] = v
            vertex_evaluated[global_vi] = 1

    return vertices


def offset_triangulation_vertices(triangles, offset):
    """
    Offset the index of each vertex in a triangulation. The triangles list is modified in-place.

    :param triangles: n by 3 list of indexed triangles which will be edited.
    :param offset: Offset for each vertex index.
    """
    for triangle in triangles:
        for j in range(3):
            triangle[j] += offset
    return triangles
