import unittest

import numpy as np

from polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes import (
    rectangle_triangulation, rectangle_vertices)
from polynomials_on_simplices.geometry.mesh.triangle_mesh import (
    centroid, has_consistent_orientation, has_same_edge_orientation, has_same_orientation)


class TestOrientation(unittest.TestCase):
    def test_has_same_orientation(self):
        triangle1 = [0, 1, 2]
        triangle2 = [1, 3, 2]
        triangle3 = [3, 1, 2]
        self.assertTrue(has_same_orientation(triangle1, triangle2))
        self.assertFalse(has_same_orientation(triangle1, triangle3))

    def test_has_same_edge_orientation(self):
        triangles1 = [[0, 1, 2], [1, 3, 2]]
        self.assertTrue(has_same_edge_orientation(triangles1, 0, 0))
        triangles2 = [[0, 1, 2], [1, 2, 3]]
        self.assertTrue(not has_same_edge_orientation(triangles2, 0, 0))

    def test_has_consistent_orientation(self):
        triangles1 = [[0, 1, 2], [1, 3, 2]]
        self.assertTrue(has_consistent_orientation(triangles1))
        triangles2 = [[0, 1, 2], [1, 2, 3]]
        self.assertTrue(not has_consistent_orientation(triangles2))
        triangles3 = [[0, 1, 2],
                      [1, 3, 2],
                      [1, 4, 5],
                      [1, 5, 3],
                      [3, 5, 6]]
        self.assertTrue(has_consistent_orientation(triangles3))
        triangles4 = [[0, 1, 2],
                      [1, 3, 2],
                      [1, 5, 4],
                      [1, 5, 3],
                      [6, 5, 3]]
        self.assertTrue(not has_consistent_orientation(triangles4))


class TestCentroid(unittest.TestCase):
    def test_square_centroid(self):
        t = rectangle_triangulation(4, 3)
        v = rectangle_vertices(1.0, 1.0, 4, 3)
        c = centroid(t, v)
        c_expected = np.array([0.0, 0.0, 0.0])

        self.assertTrue(np.allclose(c_expected, c))

        # Making the triangulaion irregular should not affect the result
        v[1][0] += 0.1
        v[2][0] += 0.2
        c = centroid(t, v)
        c_expected = np.array([0.0, 0.0, 0.0])

        self.assertTrue(np.allclose(c_expected, c))


if __name__ == "__main__":
    unittest.main()
