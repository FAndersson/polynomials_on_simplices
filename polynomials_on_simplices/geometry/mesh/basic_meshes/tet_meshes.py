"""Geometries made up of tetrahedrons (polyhedron with four vertices)."""

import numpy as np

from polynomials_on_simplices.algebra.multiindex import MultiIndexIterator, get_index, norm
from polynomials_on_simplices.geometry.primitives.simplex import affine_transformation_from_unit


def rectangular_box_triangulation(width_resolution, height_resolution, depth_resolution):
    """
    Create a triangulation of a rectangular box (subdivision of the geometry into a set of tetrahedrons).

    :param width_resolution: Number of vertices along the width of the rectangular box.
    :param height_resolution: Number of vertices along the height of the rectangular box.
    :param depth_resolution: Number of vertices along the depth of the rectangular box.
    :return: n by 4 list of tetrahedrons.
    """
    if width_resolution < 2:
        raise ValueError("Width resolution need to be greater than or equal to two")
    if height_resolution < 2:
        raise ValueError("Height resolution need to be greater than or equal to two")
    if depth_resolution < 2:
        raise ValueError("Depth resolution need to be greater than or equal to two")

    num_tetrahedrons = 6 * (width_resolution - 1) * (height_resolution - 1) * (depth_resolution - 1)
    tetrahedrons = np.empty((num_tetrahedrons, 4), dtype=int)
    num_plane_vertices = width_resolution * height_resolution

    idx = 0
    for i in range(depth_resolution - 1):
        for j in range(height_resolution - 1):
            for k in range(width_resolution - 1):
                # Compute the eight vertices of the current cube
                a1 = i * num_plane_vertices + j * width_resolution + k
                a2 = a1 + 1
                a3 = a1 + width_resolution
                a4 = a1 + width_resolution + 1
                a5 = a1 + num_plane_vertices
                a6 = a1 + num_plane_vertices + 1
                a7 = a1 + num_plane_vertices + width_resolution
                a8 = a1 + num_plane_vertices + width_resolution + 1
                tetrahedrons[idx] = [a1, a2, a3, a5]
                tetrahedrons[idx + 1] = [a5, a6, a2, a3]
                tetrahedrons[idx + 2] = [a5, a6, a3, a7]
                tetrahedrons[idx + 3] = [a2, a4, a3, a6]
                tetrahedrons[idx + 4] = [a6, a4, a3, a7]
                tetrahedrons[idx + 5] = [a6, a8, a4, a7]
                idx += 6
    return tetrahedrons


def unit_cube_vertices(width_resolution, height_resolution, depth_resolution):
    """
    Create the vertices of the unit cube.

    :param width_resolution: Number of vertices along the width of the unit cube.
    :param height_resolution: Number of vertices along the height of the unit cube.
    :param depth_resolution: Number of vertices along the depth of the unit cube.
    :return: n by 3 list of vertices.
    """
    if width_resolution < 2:
        raise ValueError("Width resolution need to be greater than or equal to two")
    if height_resolution < 2:
        raise ValueError("Height resolution need to be greater than or equal to two")
    if depth_resolution < 2:
        raise ValueError("Depth resolution need to be greater than or equal to two")

    num_vertices = width_resolution * height_resolution * depth_resolution
    vertices = np.empty((num_vertices, 3))

    idx = 0
    for i in range(depth_resolution):
        z = i / (depth_resolution - 1)
        for j in range(height_resolution):
            y = j / (height_resolution - 1)
            for k in range(width_resolution):
                x = k / (width_resolution - 1)
                vertices[idx] = [x, y, z]
                idx += 1
    return vertices


def tetrahedron_triangulation(edge_resolution):
    """
    Create a triangulation of a tetrahedron (subdivision of the geometry into a set of tetrahedrons).

    :param edge_resolution: Number of vertices along each edge of the tetrahedron.
    :return: n by 4 list of tetrahedrons.
    """
    assert edge_resolution >= 2

    num_tetrahedrons = (edge_resolution - 1)**3
    tetrahedrons = np.empty((num_tetrahedrons, 4), dtype=int)

    idx = 0
    for mi in MultiIndexIterator(3, edge_resolution - 1):
        if norm(mi) < edge_resolution - 3:
            i0 = get_index(mi, edge_resolution - 1)
            i1 = i0 + 1
            i2 = get_index(mi + (0, 1, 0), edge_resolution - 1)
            i3 = i2 + 1
            i4 = get_index(mi + (0, 0, 1), edge_resolution - 1)
            i5 = i4 + 1
            i6 = get_index(mi + (0, 1, 1), edge_resolution - 1)
            i7 = i6 + 1
            tetrahedrons[idx:idx + 6, :] = _create_rectangular_box_triangulation(i0, i1, i2, i3, i4, i5, i6, i7)
            idx += 6
        if norm(mi) == edge_resolution - 3:
            i0 = get_index(mi, edge_resolution - 1)
            i1 = i0 + 1
            i2 = get_index(mi + (0, 1, 0), edge_resolution - 1)
            i3 = i2 + 1
            i4 = get_index(mi + (0, 0, 1), edge_resolution - 1)
            i5 = i4 + 1
            i6 = get_index(mi + (0, 1, 1), edge_resolution - 1)
            tetrahedrons[idx:idx + 3, :] = _create_triangular_prism_triangulation(i0, i1, i2, i4, i5, i6)
            idx += 3
            tetrahedrons[idx:idx + 2, :] = _create_cut_triangular_prism_triangulation(i1, i2, i3, i5, i6)
            idx += 2
        if norm(mi) == edge_resolution - 2:
            i0 = get_index(mi, edge_resolution - 1)
            i1 = i0 + 1
            i2 = get_index(mi + (0, 1, 0), edge_resolution - 1)
            i3 = get_index(mi + (0, 0, 1), edge_resolution - 1)
            tetrahedrons[idx] = np.array([i0, i1, i2, i3])
            idx += 1

    return tetrahedrons


def tetrahedron_vertices(edge_resolution):
    """
    Create the vertices of the unit tetrahedron subdivided into a number of smaller tetrahedrons.

    :param edge_resolution: Number of vertices along each edge of the tetrahedron.
    :return: n by 3 list of vertices.
    """
    assert edge_resolution >= 2

    num_vertices = int((edge_resolution + 2) * (edge_resolution + 1) * edge_resolution / 6)
    vertices = np.empty((num_vertices, 3))

    idx = 0
    for w in range(edge_resolution):
        z = w / (edge_resolution - 1)
        for v in range(edge_resolution - w):
            y = v / (edge_resolution - 1)
            for u in range(edge_resolution - v - w):
                x = u / (edge_resolution - 1)
                vertices[idx] = np.array([x, y, z])
                idx += 1

    return vertices


def general_tetrahedron_vertices(tet_vertices, edge_resolution):
    """
    Create the vertices of the a tetrahedron defined by its four vertices,
    subdivided into a number of smaller tetrahedrons.

    :param tet_vertices: The four vertices of the tetrahedron.
    :param edge_resolution: Number of vertices along each edge of the tetrahedron.
    :return: n by 3 list of vertices.
    """
    assert edge_resolution >= 2

    vertices = tetrahedron_vertices(edge_resolution)
    a, b = affine_transformation_from_unit(tet_vertices)
    for i in range(len(vertices)):
        vertices[i] = np.dot(a, vertices[i]) + b
    return vertices


def _create_cut_triangular_prism_triangulation(i0, i1, i2, i3, i4):
    """
    Create a triangulation of a cut triangular prism (helper function).

    :param i0: Index of the first vertex in the boundary triangle.
    :param i1: Index of the second vertex in the boundary triangle.
    :param i2: Index of the third vertex in the boundary triangle.
    :param i3: Index of the first vertex in the boundary line.
    :param i4: Index of the second vertex in the boundary line.
    :return: Triangulation of the cut triangular prism (2 by 4 list of indexed tetrahedrons).
    """
    return np.array([
        [i0, i2, i1, i3],
        [i3, i2, i1, i4]
    ])


def _create_triangular_prism_triangulation(i0, i1, i2, i3, i4, i5):
    """
    Create a triangulation of a triangular prism (helper function).

    :param i0: Index of the first vertex in the first boundary triangle.
    :param i1: Index of the second vertex in the first boundary triangle.
    :param i2: Index of the third vertex in the first boundary triangle.
    :param i3: Index of the first vertex in the second boundary triangle.
    :param i4: Index of the second vertex in the second boundary triangle.
    :param i5: Index of the third vertex in the second boundary triangle.
    :return: Triangulation of the triangular prism (3 by 4 list of indexed tetrahedrons).
    """
    return np.array([
        [i0, i1, i2, i3],
        [i3, i4, i1, i2],
        [i3, i4, i2, i5],
    ])


def _create_rectangular_box_triangulation(i0, i1, i2, i3, i4, i5, i6, i7):
    """
    Create a triangulation of a rectangular box (helper function).

    :param i0: Index of the first vertex in the first boundary rectangle.
    :param i1: Index of the second vertex in the first boundary rectangle.
    :param i2: Index of the third vertex in the first boundary rectangle.
    :param i3: Index of the forth vertex in the first boundary rectangle.
    :param i4: Index of the first vertex in the second boundary rectangle.
    :param i5: Index of the second vertex in the second boundary rectangle.
    :param i6: Index of the third vertex in the second boundary rectangle.
    :param i7: Index of the forth vertex in the second boundary rectangle.
    :return: Triangulation of the rectangular box (6 by 4 list of indexed tetrahedrons).
    """
    return np.array([
        [i0, i1, i2, i4],
        [i4, i5, i1, i2],
        [i4, i5, i2, i6],
        [i1, i3, i2, i5],
        [i5, i3, i2, i6],
        [i5, i7, i3, i6],
    ])
