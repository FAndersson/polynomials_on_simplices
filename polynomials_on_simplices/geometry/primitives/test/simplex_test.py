import numbers
import unittest

import numpy as np
import pytest

from polynomials_on_simplices.geometry.primitives.simplex import (
    affine_map_from_unit, affine_map_to_unit, affine_transformation, affine_transformation_from_unit,
    affine_transformation_to_unit, altitude, barycentric_to_cartesian, barycentric_to_cartesian_unit,
    barycentric_to_trilinear, basis, cartesian_to_barycentric, cartesian_to_barycentric_unit, centroid, circumcenter,
    circumradius, edges, equilateral, face, face_normal, in_subspace, incenter, inradius, inside_simplex, is_degenerate,
    local_coordinates, orientation, orthonormal_basis, signed_volume, trilinear_to_barycentric, unit, volume)
from polynomials_on_simplices.linalg.rigid_motion import move, random_rigid_motion
from polynomials_on_simplices.probability_theory.uniform_sampling import nsimplex_sampling


def _random_oriented_simplex(n):
    """
    Help function for generating a random positively oriented n-dimensional simplex.

    :param int n: Dimension of the simplex.
    :return: Vertices of the simplex ((n + 1) x n matrix where row i contains the i:th vertex of the simplex).
    """
    vertices = np.random.rand(n + 1, n)
    while orientation(vertices) < 0.0:
        vertices = np.random.rand(n + 1, n)
    return vertices


class TestBasis(unittest.TestCase):
    def test_3d(self):
        vertices = unit(3)
        b = basis(vertices)
        expected_basis = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.assertTrue(np.array_equal(expected_basis, b))

        b = basis(vertices, 2)
        expected_basis = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
        ])
        self.assertTrue(np.array_equal(expected_basis, b))


class TestUnit(unittest.TestCase):
    def test_0d(self):
        vertices = unit(0, 1)
        vertices_ref = np.array([
            [0.0],
        ])
        self.assertTrue(np.all(vertices == vertices_ref))

        vertices = unit(0, 2)
        vertices_ref = np.array([
            [0.0, 0.0],
        ])
        self.assertTrue(np.all(vertices == vertices_ref))

    def test_1d(self):
        vertices = unit(1)
        vertices_ref = np.array([[0.0], [1.0]])
        self.assertTrue(np.all(vertices == vertices_ref))

        vertices = unit(1, 2)
        vertices_ref = np.array([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        self.assertTrue(np.all(vertices == vertices_ref))

    def test_2d(self):
        vertices = unit(2)
        vertices_ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(np.all(vertices == vertices_ref))

        vertices = unit(2, 3)
        vertices_ref = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]])
        self.assertTrue(np.all(vertices == vertices_ref))

    def test_3d(self):
        vertices = unit(3)
        vertices_ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(np.all(vertices == vertices_ref))

        vertices = unit(3, 4)
        vertices_ref = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]])
        self.assertTrue(np.all(vertices == vertices_ref))


def test_equilateral_2d():
    vertices = equilateral(2)
    for edge in edges(vertices):
        assert abs(np.linalg.norm(edge) - 1) < 1e-12

    vertices = equilateral(2, d=2)
    for edge in edges(vertices):
        assert abs(np.linalg.norm(edge) - 2) < 1e-12


def test_equilateral_3d():
    vertices = equilateral(3)
    for edge in edges(vertices):
        assert abs(np.linalg.norm(edge) - 1) < 1e-12

    vertices = equilateral(3, d=2)
    for edge in edges(vertices):
        assert abs(np.linalg.norm(edge) - 2) < 1e-12


def test_equilateral_4d():
    vertices = equilateral(4)
    for edge in edges(vertices):
        assert abs(np.linalg.norm(edge) - 1) < 1e-12

    vertices = equilateral(4, d=2)
    for edge in edges(vertices):
        assert abs(np.linalg.norm(edge) - 2) < 1e-12

    vertices = equilateral(4, d=2, ne=6)
    for edge in edges(vertices):
        assert abs(np.linalg.norm(edge) - 2) < 1e-12


class TestAffineTransformation(unittest.TestCase):
    def evaluate_nd(self, n):
        # Generate random n-dimensional simplex
        vertices = np.random.rand(n + 1, n)
        # Test transformation from the unit simplex
        vertices_unit = unit(n)
        a, b = affine_transformation_from_unit(vertices)
        # Transform each vertex of the unit simplex and make sure that we get the expected vertex
        vertices_transformed = np.empty((n + 1, n))
        for i in range(n + 1):
            vertices_transformed[i, :] = np.dot(a, vertices_unit[i, :]) + b
        self.assertTrue(np.allclose(vertices_transformed, vertices))
        # Test transformation to the unit simplex
        a_inv, b_inv = affine_transformation_to_unit(vertices)
        # Transform each vertex of the simplex and make sure that we get the unit simplex
        vertices_inverse_transformed = np.empty((n + 1, n))
        for i in range(n + 1):
            vertices_inverse_transformed[i, :] = np.dot(a_inv, vertices_transformed[i, :]) + b_inv
        self.assertTrue(np.allclose(vertices_inverse_transformed, vertices_unit))

    def test_1d(self):
        self.evaluate_nd(1)

    def test_2d(self):
        self.evaluate_nd(2)

    def test_3d(self):
        self.evaluate_nd(3)

    def test_general_simplices(self):
        for n in range(1, 4):
            # Generate two random simplices
            vertices1 = np.random.rand(n + 1, n)
            vertices2 = np.random.rand(n + 1, n)
            a, b = affine_transformation(vertices1, vertices2)
            vertices_transformed = np.empty((n + 1, n))
            for i in range(n + 1):
                vertices_transformed[i, :] = np.dot(a, vertices1[i, :]) + b
            self.assertTrue(np.allclose(vertices_transformed, vertices2))

    def test_integer_simplex(self):
        vertices = np.array([[1], [3]])
        a, b = affine_transformation_to_unit(vertices)
        self.assertTrue(0.5, a)
        self.assertEqual(-0.5, b)

        vertices = np.array([
            [0, 0],
            [2, 0],
            [0, 2]
        ])
        a, b = affine_transformation_to_unit(vertices)
        expected_a = np.array([
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        expected_b = np.zeros(2)
        self.assertTrue(np.array_equal(expected_a, a))
        self.assertTrue(np.array_equal(expected_b, b))

    def test_list_simplex(self):
        vertices = [[1], [3]]
        a, b = affine_transformation_to_unit(vertices)
        self.assertTrue(0.5, a)
        self.assertEqual(-0.5, b)

        vertices = [
            [0, 0],
            [2, 0],
            [0, 2]
        ]
        a, b = affine_transformation_to_unit(vertices)
        expected_a = np.array([
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        expected_b = np.zeros(2)
        self.assertTrue(np.array_equal(expected_a, a))
        self.assertTrue(np.array_equal(expected_b, b))

    def test_affine_transformation_to_higher_dimensional_space(self):
        # Test affine map from the n-dimensional unit simplex to an n-dimensional simplex in R^m, m > n.

        # 1D to 2D
        vertices = np.array([[-1, -1], [1, 0]])
        a, b = affine_transformation_from_unit(vertices)
        expected_a = np.array([[2], [1]])
        expected_b = np.array([-1, -1])
        self.assertTrue(np.array_equal(expected_a, a))
        self.assertTrue(np.array_equal(expected_b, b))
        phi = affine_map_from_unit(vertices)
        self.assertTrue(np.linalg.norm(phi(0) - np.array([-1, -1])) < 1e-12)
        self.assertTrue(np.linalg.norm(phi(1) - np.array([1, 0])) < 1e-12)

        # 2D to 3D
        vertices = np.array([
            [1, 0, 1],
            [2, -1, 1],
            [3 / 2, 1, 2]
        ])
        a, b = affine_transformation_from_unit(vertices)
        expected_a = np.array([[1, 1 / 2], [-1, 1], [0, 1]])
        expected_b = np.array([1, 0, 1])
        self.assertTrue(np.array_equal(expected_a, a))
        self.assertTrue(np.array_equal(expected_b, b))
        phi = affine_map_from_unit(vertices)
        self.assertTrue(np.linalg.norm(phi([0, 0]) - np.array([1, 0, 1])) < 1e-12)
        self.assertTrue(np.linalg.norm(phi([1, 0]) - np.array([2, -1, 1])) < 1e-12)
        self.assertTrue(np.linalg.norm(phi([0, 1]) - np.array([3 / 2, 1, 2])) < 1e-12)

        # Test affine map from an n-dimensional simplex in R^m, m > n, to the n-dimensional unit simplex.

        # 2D to 1D
        vertices = np.array([[-1, -1], [1, 0]])
        phi = affine_map_to_unit(vertices)
        # Verify that the vertices of the 1D simplex is mapped to the vertices of the 1D unit simplex
        self.assertTrue(isinstance(phi([-1, -1]), numbers.Number))
        self.assertTrue(isinstance(phi([1, 0]), numbers.Number))
        self.assertTrue(abs(phi([-1, -1]) - 0) < 1e-12)
        self.assertTrue(abs(phi([1, 0]) - 1) < 1e-12)
        # Verify that evaluating the map at a point not in the simplex subspace raises an assert
        x = np.array([1, 1])
        with pytest.raises(Exception):
            phi(x)

        # 3D to 2D
        vertices = np.array([
            [1, 0, 1],
            [2, -1, 1],
            [3 / 2, 1, 2]
        ])
        phi = affine_map_to_unit(vertices)
        # Verify that the vertices of the 2D simplex is mapped to the vertices of the 2D unit simplex
        vu = unit(2)
        for i in range(len(vu)):
            assert np.linalg.norm(phi(vertices[i]) - vu[i]) < 1e-12
        # Verify that evaluating the map at a point not in the simplex subspace raises an assert
        x = np.array([1, 1, 1])
        with pytest.raises(Exception):
            phi(x)


class TestLocalCoordinates(unittest.TestCase):
    def test_1d(self):
        vertices = np.array([[1.0], [3.0]])
        local_vertices = local_coordinates(vertices)
        expected_local_vertices = np.array([[0.0], [2.0]])
        self.assertTrue(np.array_equal(expected_local_vertices, local_vertices))

        # 1D simplex embedded in 3D
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        local_vertices = local_coordinates(vertices)
        expected_local_vertices = np.array([[0.0], [np.sqrt(3)]])
        self.assertTrue(np.allclose(expected_local_vertices, local_vertices))

    def test_2d(self):
        vertices = np.array([
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        local_vertices = local_coordinates(vertices)
        expected_local_vertices = np.array([
            [0.0, 0.0],
            [np.sqrt(2.0), 0.0],
            [np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]
        ])
        self.assertTrue(np.allclose(expected_local_vertices, local_vertices))

        # 2D simplex embedded in 3D
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        local_vertices = local_coordinates(vertices)
        expected_local_vertices = np.array([
            [0.0, 0.0],
            [np.sqrt(2.0), 0.0],
            [np.sqrt(2.0) / 2.0, np.sqrt(3.0) / np.sqrt(2.0)]
        ])
        self.assertTrue(np.allclose(expected_local_vertices, local_vertices))

    def test_3d(self):
        vertices = np.random.rand(4, 3)
        local_vertices = local_coordinates(vertices)
        vertices_check = np.zeros((4, 3))
        b = orthonormal_basis(vertices)
        for i in range(4):
            vertices_check[i] = vertices[0]
            for j in range(3):
                vertices_check[i] += local_vertices[i][j] * b.T[j]
        self.assertTrue(np.allclose(vertices, vertices_check))


class TestVolume(unittest.TestCase):
    def test_0d(self):
        vertices = np.array([[0]])
        vol = volume(vertices)
        vol_ref = 1
        self.assertEqual(vol, vol_ref)

        vertices = np.array([[1]])
        vol = volume(vertices)
        vol_ref = 1
        self.assertEqual(vol, vol_ref)

        vertices = np.array([[0.0, 0.0]])
        vol = volume(vertices)
        vol_ref = 1
        self.assertEqual(vol, vol_ref)

        vertices = np.array([[0.0, 0.0, 0.0]])
        vol = volume(vertices)
        vol_ref = 1
        self.assertEqual(vol, vol_ref)

    def test_1d(self):
        vertices = unit(1)
        vol = volume(vertices)
        vol_ref = 1
        self.assertEqual(vol, vol_ref)

        vertices = np.array([[3], [4.5]])
        vol = volume(vertices)
        vol_ref = 1.5
        self.assertEqual(vol, vol_ref)

        vertices = np.array([[1], [0]])
        vol = volume(vertices)
        vol_ref = 1
        self.assertEqual(vol, vol_ref)

        vertices = np.array([[0.0, 0.0], [1.0, 1.0]])
        vol = volume(vertices)
        vol_ref = np.sqrt(2)
        self.assertAlmostEqual(vol, vol_ref)

        vertices = np.array([[1.0, 1.0], [2.0, 2.0]])
        vol = volume(vertices)
        vol_ref = np.sqrt(2)
        self.assertAlmostEqual(vol, vol_ref)

        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        vol = volume(vertices)
        vol_ref = np.sqrt(3)
        self.assertAlmostEqual(vol, vol_ref)

        rot, trans = random_rigid_motion()
        vertices = move(rot, trans, vertices.T).T
        vol = volume(vertices)
        vol_ref = np.sqrt(3)
        self.assertAlmostEqual(vol, vol_ref)

    def test_2d(self):
        vertices = unit(2)
        vol = volume(vertices)
        vol_ref = 0.5
        self.assertEqual(vol, vol_ref)

        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        vol = volume(vertices)
        vol_ref = 0.25
        self.assertEqual(vol, vol_ref)

        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        vol = volume(vertices)
        vol_ref = 0.5
        self.assertEqual(vol, vol_ref)

        vertices = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        vol = volume(vertices)
        vol_ref = 0.5
        self.assertEqual(vol, vol_ref)

        rot, trans = random_rigid_motion()
        vertices = move(rot, trans, vertices.T).T
        vol = volume(vertices)
        vol_ref = 0.5
        self.assertAlmostEqual(vol, vol_ref)

    def test_3d(self):
        vertices = unit(3)
        vol = volume(vertices)
        vol_ref = 1 / 6
        self.assertEqual(vol, vol_ref)

        rot, trans = random_rigid_motion()
        vertices = move(rot, trans, vertices.T).T
        vol = volume(vertices)
        vol_ref = 1 / 6
        self.assertAlmostEqual(vol, vol_ref)

        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        vertices = move(rot, trans, vertices.T).T
        vol = volume(vertices)
        vol_ref = 1 / 2
        self.assertAlmostEqual(vol, vol_ref)


class TestSignedVolume(unittest.TestCase):
    def test_0d(self):
        vertices = np.array([[0]])
        vol = signed_volume(vertices)
        vol_ref = 1
        self.assertEqual(vol, vol_ref)

    def test_1d(self):
        vertices = np.array([[1], [0]])
        vol = signed_volume(vertices)
        vol_ref = -1
        self.assertEqual(vol, vol_ref)

    def test_2d(self):
        vertices = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        vol = signed_volume(vertices)
        vol_ref = -0.5
        self.assertEqual(vol, vol_ref)

    def test_3d(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        vol = signed_volume(vertices)
        vol_ref = -1 / 6
        self.assertEqual(vol, vol_ref)


class TestOrientation(unittest.TestCase):
    def test_1d(self):
        vertices = np.array([[0], [1]])
        self.assertEqual(orientation(vertices), 1)
        vertices = np.array([[1], [0]])
        self.assertEqual(orientation(vertices), -1)

    def test_2d(self):
        vertices = unit(2)
        self.assertEqual(orientation(vertices), 1)
        vertices = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(orientation(vertices), -1)

    def test_3d(self):
        vertices = unit(3)
        self.assertEqual(orientation(vertices), 1)
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.assertEqual(orientation(vertices), -1)


class TestBarycentricCoordinates(unittest.TestCase):
    def test_1d_conversion(self):
        vertices = unit(1)
        point = np.random.rand(1)
        bary = cartesian_to_barycentric(point, vertices)
        point2 = barycentric_to_cartesian(bary, vertices)
        self.assertTrue(np.allclose(point, point2))

        vertices = np.array([[0.0, 0.0], [1.0, 1.0]])
        point = np.empty(2)
        point[0] = point[1] = np.random.rand()
        bary = cartesian_to_barycentric(point, vertices)
        point2 = barycentric_to_cartesian(bary, vertices)
        self.assertTrue(np.allclose(point, point2))

        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        point = np.empty(3)
        point[0] = point[1] = point[2] = np.random.rand()
        bary = cartesian_to_barycentric(point, vertices)
        point2 = barycentric_to_cartesian(bary, vertices)
        self.assertTrue(np.allclose(point, point2))

        rot, trans = random_rigid_motion()
        vertices = move(rot, trans, vertices.T).T
        point = np.empty(3)
        point[0] = point[1] = point[2] = np.random.rand()
        point = move(rot, trans, point)
        bary = cartesian_to_barycentric(point, vertices)
        point2 = barycentric_to_cartesian(bary, vertices)
        self.assertTrue(np.allclose(point, point2))

    def test_1d_unit_conversion(self):
        point = nsimplex_sampling(1, 1)[0]
        bary = cartesian_to_barycentric(point, unit(1))
        bary2 = cartesian_to_barycentric_unit(point)
        self.assertTrue(np.allclose(bary, bary2))

        point2 = barycentric_to_cartesian_unit(bary)
        self.assertTrue(np.allclose(point, point2))

    def test_2d_conversion(self):
        vertices = unit(2)
        point = np.random.rand(2)
        bary = cartesian_to_barycentric(point, vertices)
        point2 = barycentric_to_cartesian(bary, vertices)
        self.assertTrue(np.allclose(point, point2))

        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        point = np.zeros(3)
        point[0:2] = np.random.rand(2)
        bary = cartesian_to_barycentric(point, vertices)
        point2 = barycentric_to_cartesian(bary, vertices)
        self.assertTrue(np.allclose(point, point2))

        rot, trans = random_rigid_motion()
        vertices = move(rot, trans, vertices.T).T
        point = np.zeros(3)
        point[0:2] = np.random.rand(2)
        point = move(rot, trans, point)
        bary = cartesian_to_barycentric(point, vertices)
        point2 = barycentric_to_cartesian(bary, vertices)
        self.assertTrue(np.allclose(point, point2))

    def test_2d_unit(self):
        point = nsimplex_sampling(2, 1)[0]
        bary = cartesian_to_barycentric(point, unit(2))
        bary2 = cartesian_to_barycentric_unit(point)
        self.assertTrue(np.allclose(bary, bary2))

        point2 = barycentric_to_cartesian_unit(bary)
        self.assertTrue(np.allclose(point, point2))

    def test_2d_outside(self):
        # Verify that barycentric coordinates outside of the unit triangle are correct
        points = np.array([
            [-1.0, 3.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [3.0, -1.0],
            [1.0, 1.0]
        ])
        for point in points:
            expected_bary = np.insert(point, 0, 1.0 - point[0] - point[1])
            bary = cartesian_to_barycentric(point, unit(2))
            self.assertTrue(np.allclose(bary, expected_bary))

            bary = cartesian_to_barycentric_unit(point)
            self.assertTrue(np.allclose(bary, expected_bary))

        # Verify that coordinates are still correct after conversion to and from trilinear coordinates
        vertices = unit(2)
        for point in points:
            bary = cartesian_to_barycentric(point, vertices)
            trilinear = barycentric_to_trilinear(bary, vertices)
            bary2 = trilinear_to_barycentric(trilinear, vertices)
            point2 = barycentric_to_cartesian(bary2, vertices)
            self.assertTrue(np.allclose(point, point2))

    def test_3d_conversion(self):
        vertices = unit(3)
        point = np.random.rand(3)
        bary = cartesian_to_barycentric(point, vertices)
        point2 = barycentric_to_cartesian(bary, vertices)
        self.assertTrue(np.allclose(point, point2))

        rot, trans = random_rigid_motion()
        vertices = move(rot, trans, vertices.T).T
        point = np.random.rand(3)
        point = move(rot, trans, point)
        bary = cartesian_to_barycentric(point, vertices)
        point2 = barycentric_to_cartesian(bary, vertices)
        self.assertTrue(np.allclose(point, point2))

    def test_3d_unit(self):
        point = nsimplex_sampling(3, 1)[0]
        bary = cartesian_to_barycentric(point, unit(3))
        bary2 = cartesian_to_barycentric_unit(point)
        self.assertTrue(np.allclose(bary, bary2))

        point2 = barycentric_to_cartesian_unit(bary)
        self.assertTrue(np.allclose(point, point2))

    def test_vertices_coordinates(self):
        for n in [2, 3, 4]:
            vertices = _random_oriented_simplex(n)

            for i in range(n + 1):
                bary = cartesian_to_barycentric(vertices[i], vertices)
                expected_bary = np.zeros(n + 1)
                expected_bary[i] = 1.0
                self.assertTrue(np.allclose(bary, expected_bary))


class TestTrilinearCoordinates(unittest.TestCase):
    @staticmethod
    def _random_barycentric_coordinates(n):
        return cartesian_to_barycentric_unit(nsimplex_sampling(n, 1)[0])

    @staticmethod
    def _random_trilinear_coordinates(n):
        t = np.random.random_sample((n,))
        return np.append(t, 1.0)

    def test_conversions(self):
        for n in [2, 3, 4]:
            vertices = _random_oriented_simplex(n)

            bary = TestTrilinearCoordinates._random_barycentric_coordinates(n)
            trilinear = barycentric_to_trilinear(bary, vertices)
            bary2 = trilinear_to_barycentric(trilinear, vertices)
            self.assertTrue(np.allclose(bary, bary2))

            trilinear = TestTrilinearCoordinates._random_trilinear_coordinates(n)
            bary = trilinear_to_barycentric(trilinear, vertices)
            trilinear2 = barycentric_to_trilinear(bary, vertices)
            # Trilinear coordinates are unique up to multiplication with a constant
            k = trilinear[0] / trilinear2[0]
            self.assertTrue(np.allclose(trilinear, k * trilinear2))

    def test_vertices_coordinates(self):
        for n in [2, 3, 4]:
            vertices = _random_oriented_simplex(n)

            for i in range(n + 1):
                h = altitude(vertices, i)

                bary = cartesian_to_barycentric(vertices[i], vertices)
                trilinear = barycentric_to_trilinear(bary, vertices)
                self.assertAlmostEqual(trilinear[i], h)
                for j in range(1, n + 1):
                    self.assertAlmostEqual(trilinear[(i + j) % (n + 1)], 0.0)


class TestCentroid(unittest.TestCase):
    def test_point(self):
        vertices = np.array([[0.1, 0.2]])
        c = centroid(vertices)
        c_expected = np.array([0.1, 0.2])

        self.assertTrue(np.array_equal(c_expected, c))

    def test_line(self):
        vertices = np.array([[0.0, 0.0], [1.0, 1.0]])
        c = centroid(vertices)
        c_expected = np.array([0.5, 0.5])

        self.assertTrue(np.array_equal(c_expected, c))

    def test_tetrahedron(self):
        vertices = unit(3)
        c = centroid(vertices)
        c_expected = np.array([1 / 4, 1 / 4, 1 / 4])

        self.assertTrue(np.array_equal(c_expected, c))

        # Centroid should be rotation and translation invariant
        rot, trans = random_rigid_motion()
        vertices = move(rot, trans, vertices.T).T

        c = centroid(vertices)
        c_expected = move(rot, trans, np.array([1 / 4, 1 / 4, 1 / 4]))

        self.assertTrue(np.allclose(c_expected, c))


class TestEdges(unittest.TestCase):
    def test_2d(self):
        vertices = unit(2)
        e = edges(vertices)
        expected_edges = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 1.0]
        ])
        self.assertTrue(np.array_equal(expected_edges, e))

    def test_3d(self):
        vertices = unit(3)
        e = edges(vertices)
        expected_edges = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0]
        ])
        self.assertTrue(np.array_equal(expected_edges, e))

    def test_4d(self):
        vertices = unit(4)
        e = edges(vertices)
        expected_edges = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 1.0],
        ])
        self.assertTrue(np.array_equal(expected_edges, e))


class TestFace(unittest.TestCase):
    def test_1d(self):
        n = 1
        vertices = unit(n)
        expected_faces = [
            np.array([[1]]),
            np.array([[0]])
        ]
        for i in range(n + 1):
            f = face(vertices, i)
            self.assertTrue(np.array_equal(expected_faces[i], f))

    def test_2d(self):
        n = 2
        vertices = unit(n)
        expected_faces = [
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]])
        ]
        for i in range(n + 1):
            f = face(vertices, i)
            self.assertTrue(np.array_equal(expected_faces[i], f))

    def test_3d(self):
        n = 3
        vertices = unit(n)
        expected_faces = [
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        ]
        for i in range(n + 1):
            f = face(vertices, i)
            self.assertTrue(np.array_equal(expected_faces[i], f))


class TestFaceNormal(unittest.TestCase):
    def test_2d(self):
        n = 2
        vertices = unit(n)
        expected_face_normals = np.array([
            [np.sqrt(2) / 2, np.sqrt(2) / 2],
            [-1.0, 0.0],
            [0.0, -1.0]
        ])
        for i in range(n + 1):
            fn = face_normal(vertices, i)
            self.assertTrue(np.allclose(expected_face_normals[i], fn))

    def test_3d(self):
        n = 3
        vertices = unit(n)
        expected_face_normals = np.array([
            [np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        for i in range(n + 1):
            fn = face_normal(vertices, i)
            self.assertTrue(np.allclose(expected_face_normals[i], fn))


class TestAltitude(unittest.TestCase):
    def test_2d(self):
        vertices = equilateral(2)
        self.assertTrue(abs(altitude(vertices, 0) - np.sqrt(3) / 2) < 1e-12)
        self.assertTrue(abs(altitude(vertices, 1) - np.sqrt(3) / 2) < 1e-12)
        self.assertTrue(abs(altitude(vertices, 2) - np.sqrt(3) / 2) < 1e-12)

        vertices = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ])
        self.assertTrue(abs(altitude(vertices, 0) - 1 / np.sqrt(2)) < 1e-12)
        self.assertTrue(abs(altitude(vertices, 1) - 1 / np.sqrt(2)) < 1e-12)
        self.assertTrue(abs(altitude(vertices, 2) - 0.5) < 1e-12)

    def test_3d(self):
        vertices = equilateral(3)
        self.assertTrue(abs(altitude(vertices, 0) - np.sqrt(2) / np.sqrt(3)) < 1e-12)
        self.assertTrue(abs(altitude(vertices, 1) - np.sqrt(2) / np.sqrt(3)) < 1e-12)
        self.assertTrue(abs(altitude(vertices, 2) - np.sqrt(2) / np.sqrt(3)) < 1e-12)


class TestCircumcenter(unittest.TestCase):
    def test_2d_random(self):
        n = 2
        vertices = np.random.rand(n + 1, n)
        c = circumcenter(vertices)
        r = circumradius(vertices)
        d = np.empty(n + 1)
        for i in range(n + 1):
            d[i] = np.linalg.norm(c - vertices[i])
        for i in range(n + 1):
            self.assertAlmostEqual(d[i], d[(i + 1) % 3])
            self.assertAlmostEqual(d[i], r)

    def test_3d_random(self):
        n = 3
        vertices = np.random.rand(n + 1, n)
        c = circumcenter(vertices)
        r = circumradius(vertices)
        d = np.empty(n + 1)
        for i in range(n + 1):
            d[i] = np.linalg.norm(c - vertices[i])
        for i in range(n + 1):
            self.assertAlmostEqual(d[i], d[(i + 1) % 3])
            self.assertAlmostEqual(d[i], r)

    def test_4d_random(self):
        n = 4
        vertices = np.random.rand(n + 1, n)
        c = circumcenter(vertices)
        r = circumradius(vertices)
        d = np.empty(n + 1)
        for i in range(n + 1):
            d[i] = np.linalg.norm(c - vertices[i])
        for i in range(n + 1):
            self.assertAlmostEqual(d[i], d[(i + 1) % 3])
            self.assertAlmostEqual(d[i], r)


class TestInCenter(unittest.TestCase):
    def test_maximum_inscribed_nsphere(self):
        for n in [2, 3, 4]:
            # vertices = _random_simplex(n)
            vertices = unit(n)
            eps = 1e-8
            # Going from the incenter in any direction a distance r - eps, where r is the inradius, we should
            # always stay inside the simplex
            v = np.random.rand(n)
            v /= np.linalg.norm(v)
            r = inradius(vertices)
            ic = incenter(vertices)
            p = ic + (r - eps) * v
            self.assertTrue(inside_simplex(p, vertices))

            # Going from the incenter in the direction of the simplex face normal a distance r + eps,
            # where r is the inradius, should take us outside of the simplex for some face normal. Otherwise the
            # maximum inscribed n-sphere property is not satisfied
            outside = False
            for i in range(n + 1):
                n = face_normal(vertices, i)
                p = ic + (r + eps) * n
                if not inside_simplex(p, vertices):
                    outside = True
                    break
            self.assertTrue(outside)


class TestInsideSimplex(unittest.TestCase):
    def test_unit_2(self):
        v = unit(2)
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.5, -0.1],
            [-0.1, 0.5],
            [0.2, 0.2]
        ])
        expected_results_include_boundary = [
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            True
        ]
        expected_results_exclude_boundary = [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True
        ]

        for i in range(len(points)):
            result = inside_simplex(points[i], v)
            self.assertEqual(expected_results_include_boundary[i], result)
            result = inside_simplex(points[i], v, include_boundary=False)
            self.assertEqual(expected_results_exclude_boundary[i], result)

    def test_random_2(self):
        for some_idx in range(200):
            v = np.random.random_sample((3, 2))
            while orientation(v) == -1:
                v = np.random.random_sample((3, 2))
            phi = affine_map_from_unit(v)
            points = np.array([
                phi([0.0, 0.0]),
                phi([1.0, 0.0]),
                phi([0.0, 1.0]),
                phi([0.5, 0.5]),
                phi([0.6, 0.6]),
                phi([0.5, -0.1]),
                phi([-0.1, 0.5]),
                phi([0.2, 0.2])
            ])
            expected_results_include_boundary = [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True
            ]

            for i in range(len(points)):
                result = inside_simplex(points[i], v, tol=1e-10)
                if not expected_results_include_boundary[i] == result:
                    print()
                    print(v)
                self.assertEqual(expected_results_include_boundary[i], result)

    def test_random_3(self):
        # Check if a random selection of points lies inside the unit tetrahedron
        v = unit(3)
        points = np.array([
            [0.16014793, 0.3774361, 0.55660591],
            [-0.18273421, -0.65755768, -0.47891305],
            [-0.19655472, -0.33995204, 0.43864115],
            [0.053419, 0.80788031, -0.91712913],
            [0.01317463, 0.55194183, 0.29637445],
            [0.53349739, 0.96667588, 0.75335823],
            [-0.40444038, 0.53548782, 0.86562701],
            [-0.86006993, 0.55392761, -0.09164995],
            [0.43863676, 0.75285526, 0.13893439],
            [0.67019709, 0.49853792, 0.81513549]
        ])

        expected_result = [
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False
        ]

        for i in range(len(points)):
            result = inside_simplex(points[i], v)
            self.assertEqual(expected_result[i], result)

    def test_eps(self):
        v = np.array([
            [0.5, 0.0, -0.005],
            [0.5, 0.005, -0.005],
            [0.44736842, 0.005, -0.005],
            [0.5, 0.0, 0.0]
        ])
        p = np.array([0.5, 0.005, -0.005])
        result = inside_simplex(p, v)
        self.assertTrue(result)


class TestInSubspace(unittest.TestCase):
    def test_1d_in_2d(self):
        vertices = np.array([
            [0.0, 1.0],
            [3.0, 2.0]
        ])
        assert in_subspace([1.5, 1.5], vertices, 1e-15)
        assert not in_subspace([2.0, 1.5], vertices, 1e-15)

    def test_1d_in_3d(self):
        vertices = np.random.random_sample((2, 3))
        t = nsimplex_sampling(1, 1)[0]
        point = (1 - t) * vertices[0] + t * vertices[1]
        assert in_subspace(point, vertices, 1e-15)
        assert not in_subspace(point + np.random.rand(3), vertices, 1e-15)

    def test_2d_in_3d(self):
        vertices = np.random.random_sample((3, 3))
        t = nsimplex_sampling(2, 1)[0]
        point = (1 - t[0] - t[1]) * vertices[0] + t[0] * vertices[1] + t[1] * vertices[2]
        assert in_subspace(point, vertices, 1e-15)
        assert not in_subspace(point + np.random.rand(3), vertices, 1e-15)


class TestIsDegenerate(unittest.TestCase):
    def test_2d(self):
        # Non-degenerate triangle
        vertices = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.2]
        ])
        self.assertFalse(is_degenerate(vertices))
        # Unless we put the threshold really high
        self.assertTrue(is_degenerate(vertices, eps=0.6))

        # Degenerate triangle
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1e-5]
        ])
        self.assertTrue(is_degenerate(vertices))

    def test_3d(self):
        # Non-degenerate tetrahedron
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        self.assertFalse(is_degenerate(vertices))
        # Unless we put the threshold really high
        self.assertTrue(is_degenerate(vertices, eps=0.6))

        # Degenerate tetrahedron
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1e-5]
        ])
        self.assertTrue(is_degenerate(vertices))


if __name__ == "__main__":
    unittest.main()
