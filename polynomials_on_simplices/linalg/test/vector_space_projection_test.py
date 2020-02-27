import unittest

import numpy as np

from polynomials_on_simplices.linalg.vector_space_projection import (
    subspace_projection_map, vector_oblique_projection_2, vector_projection, vector_rejection)


class TestVectorProjection(unittest.TestCase):
    def test_projection(self):
        a = np.array([1.3, 1.5, 0])
        b = np.array([1, 0, 0])
        self.assertTrue(np.array_equal(vector_projection(a, b), np.array([1.3, 0.0, 0.0])))
        self.assertTrue(np.array_equal(vector_rejection(a, b), np.array([0.0, 1.5, 0.0])))

    def test_oblique_projection(self):
        a = np.array([1.0, 1.0])
        b = np.array([2.0, 0.0])
        c = np.array([1.0, 1.0])
        self.assertTrue(np.array_equal(vector_oblique_projection_2(a, b, c), np.array([0.0, 0.0])))

        a = np.array([1.0, 1.0])
        b = np.array([2.0, 0.0])
        c = np.array([1.0, 2.0])
        self.assertTrue(np.array_equal(vector_oblique_projection_2(a, b, c), np.array([0.5, 0.0])))


class TestSubspaceProjection(unittest.TestCase):
    def test_1d_in_2d(self):
        # Test projection onto a 1-dimensional subspace in 2D
        origin = np.random.rand(2)
        basis = np.random.random_sample((2, 1))
        point = np.random.rand(2)

        point_projected = subspace_projection_map(basis, origin)(point)
        # The difference between the point and it's projection should be orthogonal to all
        # basis vectors spanning the subspace
        v = point_projected - point
        assert np.dot(v, basis[:, 0]) < 1e-12

    def test_1d_in_3d(self):
        # Test projection onto a 1-dimensional subspace in 3D
        origin = np.random.rand(3)
        basis = np.random.random_sample((3, 1))
        point = np.random.rand(3)

        point_projected = subspace_projection_map(basis, origin)(point)
        # The difference between the point and it's projection should be orthogonal to all
        # basis vectors spanning the subspace
        v = point_projected - point
        assert np.dot(v, basis[:, 0]) < 1e-12

    def test_2d_in_3d(self):
        # Test projection onto a 2-dimensional subspace in 3D
        origin = np.random.rand(3)
        basis = np.random.random_sample((3, 2))
        point = np.random.rand(3)

        point_projected = subspace_projection_map(basis, origin)(point)
        # The difference between the point and it's projection should be orthogonal to all
        # basis vectors spanning the subspace
        v = point_projected - point
        assert np.dot(v, basis[:, 0]) < 1e-12
        assert np.dot(v, basis[:, 1]) < 1e-12

    def test_3d_in_5d(self):
        # Test projection onto a 3-dimensional subspace in 5D
        origin = np.random.rand(5)
        basis = np.random.random_sample((5, 3))
        point = np.random.rand(5)

        point_projected = subspace_projection_map(basis, origin)(point)
        # The difference between the point and it's projection should be orthogonal to all
        # basis vectors spanning the subspace
        v = point_projected - point
        assert np.dot(v, basis[:, 0]) < 1e-12
        assert np.dot(v, basis[:, 1]) < 1e-12
        assert np.dot(v, basis[:, 2]) < 1e-12


if __name__ == '__main__':
    unittest.main()
