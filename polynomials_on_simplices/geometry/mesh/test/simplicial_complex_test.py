import unittest

import numpy as np

from polynomials_on_simplices.geometry.mesh.basic_meshes.tet_meshes import tetrahedron_triangulation
from polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes import disc_triangulation
from polynomials_on_simplices.geometry.mesh.simplicial_complex import (
    boundary, boundary_map_matrix, simplex_boundary, simplex_boundary_map, simplex_boundary_orientation,
    simplex_sub_simplices, simplex_vertices, swap_orientation)
from polynomials_on_simplices.geometry.primitives.simplex import unit


class TestSimplexBoundary(unittest.TestCase):
    def test_vertex(self):
        simplex = np.array([0])
        boundary_simplices_ref = np.array([])
        boundary_simplices = simplex_boundary(simplex)

        self.assertTrue(np.array_equal(boundary_simplices_ref, boundary_simplices))

    def test_line(self):
        simplex = np.array([3, 6])
        boundary_simplices_ref = np.array([[6], [3]])
        boundary_simplices = simplex_boundary(simplex)

        self.assertTrue(np.array_equal(boundary_simplices_ref, boundary_simplices))

    def test_triangle(self):
        simplex = np.array([3, 7, 6])
        boundary_simplices_ref = np.array([[7, 6], [6, 3], [3, 7]])
        boundary_simplices = simplex_boundary(simplex)

        self.assertTrue(np.array_equal(boundary_simplices_ref, boundary_simplices))

    def test_tetrahedron(self):
        simplex = np.array([3, 9, 7, 6])
        boundary_simplices_ref = np.array([[9, 7, 6], [3, 6, 7], [3, 9, 6], [3, 7, 9]])
        boundary_simplices = simplex_boundary(simplex)

        self.assertTrue(np.array_equal(boundary_simplices_ref, boundary_simplices))


class TestSimplexBoundaryMap(unittest.TestCase):
    def test_vertex(self):
        simplex = np.array([0])
        boundary_chain_ref = np.array([]), 0
        boundary_chain = simplex_boundary_map(simplex)

        self.assertTrue(np.array_equal(boundary_chain_ref[0], boundary_chain[0]))
        self.assertTrue(np.array_equal(boundary_chain_ref[1], boundary_chain[1]))

    def test_line(self):
        simplex = np.array([3, 6])
        boundary_chain_ref = np.array([[6], [3]]), np.array([1, -1])
        boundary_chain = simplex_boundary_map(simplex)

        self.assertTrue(np.array_equal(boundary_chain_ref[0], boundary_chain[0]))
        self.assertTrue(np.array_equal(boundary_chain_ref[1], boundary_chain[1]))

    def test_triangle(self):
        simplex = np.array([3, 7, 6])
        boundary_chain_ref = np.array([[7, 6], [3, 6], [3, 7]]), np.array([1, -1, 1])
        boundary_chain = simplex_boundary_map(simplex)

        self.assertTrue(np.array_equal(boundary_chain_ref[0], boundary_chain[0]))
        self.assertTrue(np.array_equal(boundary_chain_ref[1], boundary_chain[1]))

    def test_tetrahedron(self):
        simplex = np.array([3, 9, 7, 6])
        boundary_chain_ref = np.array([[9, 7, 6], [3, 7, 6], [3, 9, 6], [3, 9, 7]]), np.array([1, -1, 1, -1])
        boundary_chain = simplex_boundary_map(simplex)

        self.assertTrue(np.array_equal(boundary_chain_ref[0], boundary_chain[0]))
        self.assertTrue(np.array_equal(boundary_chain_ref[1], boundary_chain[1]))


class TestSimplexSubSimplices(unittest.TestCase):
    def test_vertex(self):
        simplex = [0]
        sub_simplices_ref = {(0,)}
        sub_simplices = simplex_sub_simplices(simplex)
        self.assertEqual(sub_simplices_ref, sub_simplices)

        sub_simplices_ref = set()
        sub_simplices = simplex_sub_simplices(simplex, include_self=False)
        self.assertEqual(sub_simplices_ref, sub_simplices)

    def test_line(self):
        simplex = [3, 6]
        sub_simplices_ref = {(3, 6), (3,), (6,)}
        sub_simplices = simplex_sub_simplices(simplex)
        self.assertEqual(sub_simplices_ref, sub_simplices)

        sub_simplices_ref = {(3,), (6,)}
        sub_simplices = simplex_sub_simplices(simplex, include_self=False)
        self.assertEqual(sub_simplices_ref, sub_simplices)

    def test_triangle(self):
        simplex = [3, 7, 6]
        sub_simplices_ref = {(3, 6, 7), (6, 7), (3, 6), (3, 7), (3,), (7,), (6,)}
        sub_simplices = simplex_sub_simplices(simplex)
        self.assertEqual(sub_simplices_ref, sub_simplices)

        sub_simplices_ref = {(6, 7), (3, 6), (3, 7), (3,), (7,), (6,)}
        sub_simplices = simplex_sub_simplices(simplex, include_self=False)
        self.assertEqual(sub_simplices_ref, sub_simplices)

    def test_tetrahedron(self):
        simplex = [3, 9, 7, 6]
        sub_simplices_ref = {(3, 6, 7, 9), (6, 7, 9), (3, 6, 7), (3, 6, 9), (3, 7, 9), (6, 7), (3, 6), (3, 7), (3, 9),
                             (6, 9), (7, 9), (3,), (6,), (7,), (9,)}
        sub_simplices = simplex_sub_simplices(simplex)
        self.assertEqual(sub_simplices_ref, sub_simplices)

        simplex = [3, 9, 7, 6]
        sub_simplices_ref = {(6, 7, 9), (3, 6, 7), (3, 6, 9), (3, 7, 9), (6, 7), (3, 6), (3, 7), (3, 9), (6, 9),
                             (7, 9), (3,), (6,), (7,), (9,)}
        sub_simplices = simplex_sub_simplices(simplex, include_self=False)
        self.assertEqual(sub_simplices_ref, sub_simplices)


class TestBoundary(unittest.TestCase):
    def test_disc(self):
        simplices = disc_triangulation(4, 2)
        b = boundary(simplices)
        expected_b = np.array([
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1]
        ])
        self.assertTrue(np.array_equal(expected_b, b))

    def test_tetrahedron(self):
        simplices = tetrahedron_triangulation(2)
        b = boundary(simplices)
        expected_b = np.array([
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1]
        ])
        self.assertTrue(np.array_equal(expected_b, b))

    def test_chain_complex(self):
        # The composition of two boundary operations should give the empty set
        simplices = disc_triangulation(4, 2)
        b = boundary(boundary(simplices))
        self.assertEqual(0, len(b))

        simplices = tetrahedron_triangulation(2)
        b = boundary(boundary(simplices))
        self.assertEqual(0, len(b))


class TestVertices(unittest.TestCase):
    def test_1d(self):
        # Vertex array as a 2 x 1 array
        vertex_list = unit(1)
        simplex = [0]
        vertices = simplex_vertices(simplex, vertex_list)
        vertices_ref = [[0]]
        self.assertTrue(np.array_equal(vertices_ref, vertices))

        # For 1D we also have the corner case where each vertex is a scalar instead of a length 1 array
        # Test by having the vertex array as a length 2 array
        vertex_list = [0, 1]
        vertices = simplex_vertices(simplex, vertex_list)
        self.assertTrue(np.array_equal(vertices_ref, vertices))

    def test_2d(self):
        vertex_list = unit(2)
        simplex = [0, 1]
        vertices = simplex_vertices(simplex, vertex_list)
        vertices_ref = np.array([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        self.assertTrue(np.array_equal(vertices_ref, vertices))

        simplex = [1, 0]
        vertices = simplex_vertices(simplex, vertex_list)
        vertices_ref = np.array([
            [1.0, 0.0],
            [0.0, 0.0]
        ])
        self.assertTrue(np.array_equal(vertices_ref, vertices))

        vertex_list = unit(2, 3)
        simplex = [0, 1]
        vertices = simplex_vertices(simplex, vertex_list)
        vertices_ref = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        self.assertTrue(np.array_equal(vertices_ref, vertices))


class TestBoundaryOrientation(unittest.TestCase):
    def test_4d(self):
        simplex = [0, 1, 2, 3, 4]
        boundary_simplices = [
            (1, 2, 4, 0),
            (2, 1, 4, 3),
            (1, 0, 2, 3),
            (2, 0, 3, 1),
            (0, 2, 3, 4)
        ]
        expected_signs = [
            1,
            1,
            -1,
            -1,
            -1
        ]
        i = 0
        for boundary_simplex in boundary_simplices:
            sign = simplex_boundary_orientation(simplex, boundary_simplex)
            expected_sign = expected_signs[i]
            self.assertEqual(expected_sign, sign)
            i += 1

    def test_3d(self):
        # Basic sign check
        simplex = [0, 1, 2, 3]
        boundary_simplices = [
            [0, 1, 3],
            [0, 3, 1],
            [1, 2, 3],
            [1, 3, 2],
            [0, 3, 2],
            [0, 2, 3],
            [0, 2, 1],
            [0, 1, 2],
            [1, 0, 2],
            [2, 0, 1]
        ]
        i = 0
        for boundary_simplex in boundary_simplices:
            sign = simplex_boundary_orientation(simplex, boundary_simplex)
            expected_sign = (-1)**(i % 2)
            self.assertEqual(expected_sign, sign)
            i += 1

        # Handle arbitrary vertex indices
        simplex = [2, 5, 8, 12]
        boundary_simplices = [
            [2, 5, 12],
            [2, 12, 5]
        ]
        for boundary_simplex in boundary_simplices:
            sign = simplex_boundary_orientation(simplex, boundary_simplex)
            expected_sign = (-1)**(i % 2)
            self.assertEqual(expected_sign, sign)
            i += 1

        # Handle np.array
        simplex = np.array([2, 5, 8, 12])
        sign = simplex_boundary_orientation(simplex, np.array([2, 5, 12]))
        self.assertEqual(1, sign)
        sign = simplex_boundary_orientation(simplex, np.array([2, 12, 5]))
        self.assertEqual(-1, sign)

        # Handle tuple
        sign = simplex_boundary_orientation(simplex, (2, 5, 12))
        self.assertEqual(1, sign)
        sign = simplex_boundary_orientation(simplex, (2, 12, 5))
        self.assertEqual(-1, sign)


class TestSwapOrientation(unittest.TestCase):
    def test_swap_orientation(self):
        a = np.array([[0, 1, 2], [1, 3, 2]])
        swap_orientation(a)
        self.assertTrue(np.array_equal(a, np.array([[0, 2, 1], [1, 2, 3]])))


class TestBoundaryMapMatrix(unittest.TestCase):
    def test_2d_mesh(self):
        # Example taken from Desbrun et. al. Discrete Differential Forms for Computational Modeling
        simplicial_complex = [
            [
                [0],
                [1],
                [2],
                [3],
                [4]
            ],
            [
                [0, 1],
                [1, 2],
                [3, 2],
                [0, 3],
                [3, 1]
            ],
            [
                [0, 1, 3],
                [1, 2, 3]
            ]
        ]
        matrix = boundary_map_matrix(simplicial_complex, 1).toarray()
        expected_matrix = np.array([
            [-1, 0, 0, -1, 0],
            [1, -1, 0, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 0, -1, 1, -1],
            [0, 0, 0, 0, 0]
        ])
        self.assertTrue(np.array_equal(expected_matrix, matrix))

        matrix = boundary_map_matrix(simplicial_complex, 2).toarray()
        expected_matrix = np.array([
            [1, 0],
            [0, 1],
            [0, -1],
            [-1, 0],
            [-1, 1]
        ])
        self.assertTrue(np.array_equal(expected_matrix, matrix))

    def test_3d_tetrahedron(self):
        simplicial_complex = [
            [
                [0],
                [1],
                [2],
                [3],
            ],
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [0, 3],
                [1, 3],
                [2, 3],
            ],
            [
                [0, 1, 3],
                [1, 2, 3],
                [2, 0, 3],
                [0, 2, 1],
            ],
            [
                [0, 1, 2, 3]
            ]
        ]
        matrix = boundary_map_matrix(simplicial_complex, 1).toarray()
        expected_matrix = np.array([
            [-1, 0, 1, -1, 0, 0],
            [1, -1, 0, 0, -1, 0],
            [0, 1, -1, 0, 0, -1],
            [0, 0, 0, 1, 1, 1],
        ])
        self.assertTrue(np.array_equal(expected_matrix, matrix))

        matrix = boundary_map_matrix(simplicial_complex, 2).toarray()
        expected_matrix = np.array([
            [1, 0, 0, -1],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [-1, 0, 1, 0],
            [1, -1, 0, 0],
            [0, 1, -1, 0],
        ])
        self.assertTrue(np.array_equal(expected_matrix, matrix))

        matrix = boundary_map_matrix(simplicial_complex, 3).toarray()
        expected_matrix = np.array([
            [1],
            [1],
            [1],
            [1],
        ])
        self.assertTrue(np.array_equal(expected_matrix, matrix))


if __name__ == "__main__":
    unittest.main()
