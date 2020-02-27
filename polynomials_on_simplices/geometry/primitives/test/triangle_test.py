import unittest

import numpy as np

from polynomials_on_simplices.calculus.error_measures import relative_error
from polynomials_on_simplices.geometry.mesh.basic_meshes.triangle_meshes import equilateral_triangle_vertices
from polynomials_on_simplices.geometry.primitives.simplex import (
    affine_transformation_from_unit, cartesian_to_barycentric_unit, unit)
import polynomials_on_simplices.geometry.primitives.triangle as triangle
from polynomials_on_simplices.linalg.rigid_motion import move, random_rigid_motion
from polynomials_on_simplices.linalg.vector_space_projection import vector_projection
from polynomials_on_simplices.probability_theory.uniform_sampling import nsimplex_sampling


class TestEdges(unittest.TestCase):
    def test_2d(self):
        vertices = unit(2)
        edges = triangle.edges(vertices)
        expected_edges = np.array([
            [-1.0, 1.0],
            [0.0, -1.0],
            [1.0, 0.0]
        ])
        self.assertTrue(np.allclose(expected_edges, edges))

    def test_3d(self):
        vertices = unit(2, 3)
        edges = triangle.edges(vertices)
        expected_edges = np.array([
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        self.assertTrue(np.allclose(expected_edges, edges))

    def test_diameter(self):
        vertices = unit(2)
        d = triangle.diameter(vertices)
        self.assertEqual(d, np.sqrt(2))

    def test_dual_edges_2d(self):
        vertices = np.random.rand(3, 2)
        edges = triangle.edges(vertices)
        dual_edges = triangle.dual_edges(vertices)
        # Test that dual edges are orthogonal to edges
        for i in range(3):
            self.assertTrue(abs(np.dot(edges[i], dual_edges[i])) < 1e-10)
        # Test that dual edges point out of the triangle by comparing with the vector from the edge midpoint to the
        # triangle centroid
        c = triangle.centroid(vertices)
        for i in range(3):
            edge_midpoint = 0.5 * (vertices[(i + 1) % 3] + vertices[(i + 2) % 3])
            self.assertTrue(np.dot(dual_edges[i], c - edge_midpoint) < 0.0)

    def test_dual_edges_3d(self):
        vertices = np.random.rand(3, 3)
        edges = triangle.edges(vertices)
        dual_edges = triangle.dual_edges(vertices)
        # Test that dual edges are orthogonal to edges
        for i in range(3):
            self.assertTrue(abs(np.dot(edges[i], dual_edges[i])) < 1e-10)
        # Test that dual edges point out of the triangle by comparing with the vector from the edge midpoint to the
        # triangle centroid
        c = triangle.centroid(vertices)
        for i in range(3):
            edge_midpoint = 0.5 * (vertices[(i + 1) % 3] + vertices[(i + 2) % 3])
            self.assertTrue(np.dot(dual_edges[i], c - edge_midpoint) < 0.0)


class TestBasis(unittest.TestCase):
    def test_2d(self):
        p = np.random.rand(3, 2)
        b = triangle.basis(p)
        self.assertTrue(abs(np.dot(b[0], b[1])) < 1e-10)
        self.assertTrue(abs(np.linalg.norm(b[0]) - 1) < 1e-10)
        self.assertTrue(abs(np.linalg.norm(b[1]) - 1) < 1e-10)

    def test_3d(self):
        p = np.random.rand(3, 3)
        b = triangle.basis(p)
        self.assertTrue(abs(np.dot(b[0], b[1])) < 1e-10)
        self.assertTrue(abs(np.linalg.norm(b[0]) - 1) < 1e-10)
        self.assertTrue(abs(np.linalg.norm(b[1]) - 1) < 1e-10)
        n1 = np.cross(b[0], b[1])
        n2 = triangle.normal(p)
        self.assertTrue(np.allclose(n1, n2))


class TestArea(unittest.TestCase):
    def test_unit(self):
        vertices = unit(2)
        a = triangle.area(vertices)
        ea = 0.5
        self.assertEqual(ea, a)

        vertices = unit(2, 3)
        a = triangle.area(vertices)
        self.assertEqual(ea, a)

    def test_arbitrary(self):
        vertices = np.array([
            [0.1, 0.2],
            [0.7, -0.2],
            [0.4, 0.5]
        ])
        a = triangle.area(vertices)
        at, bt = affine_transformation_from_unit(vertices)
        ea = 0.5 * np.abs(np.linalg.det(at))
        self.assertAlmostEqual(ea, a)

    def test_random(self):
        vertices = np.random.rand(3, 2)
        a = triangle.area(vertices)
        at, bt = affine_transformation_from_unit(vertices)
        ea = 0.5 * np.abs(np.linalg.det(at))
        self.assertAlmostEqual(ea, a)

        # Transform triangle to 3D and move arbitrarily in space
        vertices = np.concatenate((vertices, np.zeros((3, 1))), axis=1)
        r, t = random_rigid_motion()
        vertices = move(r, t, vertices.T).T
        a = triangle.area(vertices)
        self.assertAlmostEqual(ea, a)


class TestAngle(unittest.TestCase):
    def test_2d(self):
        vertices = unit(2)
        expected_angles = [np.pi / 2, np.pi / 4, np.pi / 4]
        for i in range(3):
            a = triangle.angle(vertices, i)
            self.assertAlmostEqual(expected_angles[i], a)

    def test_3d(self):
        vertices = equilateral_triangle_vertices(1.0, 2)
        for i in range(3):
            a = triangle.angle(vertices, i)
            self.assertAlmostEqual(np.pi / 3, a)


class TestMedians(unittest.TestCase):
    def test_2d(self):
        vertices = unit(2)
        medians = triangle.medians(vertices)
        expected_medians = np.array([
            [0.5, 0.5],
            [-1.0, 0.5],
            [0.5, -1.0]
        ])
        for i in range(3):
            self.assertTrue(np.allclose(medians[i], expected_medians[i]))

    def test_3d(self):
        vertices = unit(2, 3)
        medians = triangle.medians(vertices)
        expected_medians = np.array([
            [0.5, 0.5, 0.0],
            [-1.0, 0.5, 0.0],
            [0.5, -1.0, 0.0]
        ])
        for i in range(3):
            self.assertTrue(np.allclose(medians[i], expected_medians[i]))


def random_nondegenerate_triangle_2d():
    """Create a random non degenerate triangle in 2d."""
    vertices = np.random.rand(3, 2)
    while triangle.is_degenerate(vertices):
        vertices = np.random.rand(3, 2)
    return vertices


class TestCircumcenter(unittest.TestCase):
    def test_random(self):
        vertices = random_nondegenerate_triangle_2d()
        c = triangle.circumcenter(vertices)
        d = np.empty(3)
        for i in range(3):
            d[i] = np.linalg.norm(c - vertices[i])
        for i in range(3):
            self.assertAlmostEqual(d[i], d[(i + 1) % 3])

    def test_radius(self):
        vertices = random_nondegenerate_triangle_2d()
        c = triangle.circumcenter(vertices)
        r = triangle.circumradius(vertices)
        for i in range(3):
            d = np.linalg.norm(c - vertices[i])
            self.assertAlmostEqual(d, r)


class TestInCenter(unittest.TestCase):
    def test_euler_triangle_formula(self):
        # Verify that Euler's triangle formula holds
        vertices = random_nondegenerate_triangle_2d()
        o = triangle.circumcenter(vertices)
        i = triangle.incenter(vertices)
        d2 = np.dot(i - o, i - o)
        R = triangle.circumradius(vertices)
        r = triangle.inradius(vertices)
        self.assertTrue(relative_error(d2, R * (R - 2 * r)) < 1e-5)

    def test_maximum_inscribed_circle(self):
        vertices = random_nondegenerate_triangle_2d()
        eps = 1e-10
        # Going from the incenter in any direction a distance r - eps, where r is the inradius, we should
        # always stay inside the triangle
        v = np.random.rand(2)
        v /= np.linalg.norm(v)
        r = triangle.inradius(vertices)
        ic = triangle.incenter(vertices)
        p = ic + (r - eps) * v
        self.assertTrue(triangle.inside_triangle(p, vertices))

        # Going from the incenter in the direction of the triangle dual edges a distance r + eps, where r is the
        # inradius, should take us outside of the triangle for some dual edge. Otherwise the maximum inscribed
        # circle property is not satisfied
        e = triangle.dual_edges(vertices)
        p = np.empty((3, 2))
        for i in range(3):
            p[i] = ic + (r + eps) * e[i] / np.linalg.norm(e[i])

        self.assertTrue(
            (not triangle.inside_triangle(p[0], vertices))
            or (not triangle.inside_triangle(p[1], vertices))
            or (not triangle.inside_triangle(p[2], vertices))
        )


class TestAltitudes(unittest.TestCase):
    def test_orthogonality(self):
        # Altitude vectors should be orthogonal to the opposite edge vector
        vertices = np.random.rand(3, 3)
        av = triangle.altitude_vectors(vertices)
        ev = triangle.edges(vertices)
        for i in range(3):
            self.assertAlmostEqual(np.dot(av[i], ev[i]), 0.0)

    def test_direction(self):
        # Altitude vectors should point from a vertex towards the opposite edge
        vertices = np.random.rand(3, 3)
        av = triangle.altitude_vectors(vertices)
        for i in range(3):
            d1 = np.dot(av[i], vertices[(i + 1) % 3] - vertices[i])
            d2 = np.dot(av[i], vertices[(i + 2) % 3] - vertices[i])
            self.assertTrue(d1 > 0.0)
            self.assertTrue(d2 > 0.0)

    def test_height(self):
        # Altitude vectors should have the triangle height as length
        vertices = np.random.rand(3, 3)
        av = triangle.altitude_vectors(vertices)
        ev = triangle.edges(vertices)
        a = triangle.area(vertices)
        for i in range(3):
            a2 = np.linalg.norm(ev[i]) * np.linalg.norm(av[i]) / 2.0
            self.assertAlmostEqual(a, a2)

    def test_altitudes(self):
        # Verify that the altitudes gives the correct triangle area
        vertices = np.random.rand(3, 3)
        a = triangle.area(vertices)
        al = triangle.altitudes(vertices)
        ev = triangle.edges(vertices)
        for i in range(3):
            a2 = np.linalg.norm(ev[i]) * al[i] / 2.0
            self.assertAlmostEqual(a, a2)

    def test_feet(self):
        # Verify that the altitude foot minus the vertex equals the altitude vector
        vertices = np.random.rand(3, 3)
        av = triangle.altitude_vectors(vertices)
        af = triangle.altitude_feet(vertices)
        for i in range(3):
            self.assertTrue(np.allclose(av[i], af[i] - vertices[i]))


class TestCoordinates(unittest.TestCase):
    @staticmethod
    def _random_barycentric_coordinates():
        return cartesian_to_barycentric_unit(nsimplex_sampling(2, 1)[0])

    @staticmethod
    def _random_trilinear_coordinates():
        t1 = np.random.rand()
        t2 = np.random.rand()
        return np.array([1.0, t1, t2])

    def test_conversions(self):
        vertices = np.random.rand(3, 3)

        bary = TestCoordinates._random_barycentric_coordinates()
        trilinear = triangle.barycentric_to_trilinear(bary, vertices)
        bary2 = triangle.trilinear_to_barycentric(trilinear, vertices)
        self.assertTrue(np.allclose(bary, bary2))

        trilinear = TestCoordinates._random_trilinear_coordinates()
        trilinear = triangle.normalize_trilinear_coordinates(trilinear, vertices)
        bary = triangle.trilinear_to_barycentric(trilinear, vertices)
        trilinear2 = triangle.barycentric_to_trilinear(bary, vertices)
        self.assertTrue(np.allclose(trilinear, trilinear2))

    def test_vertices_barycentric_coordinates(self):
        vertices = np.random.rand(3, 3)

        for i in range(3):
            bary = triangle.cartesian_to_barycentric(vertices[i], vertices)
            self.assertAlmostEqual(bary[i], 1.0)
            self.assertAlmostEqual(bary[(i + 1) % 3], 0.0)
            self.assertAlmostEqual(bary[(i + 2) % 3], 0.0)

    def test_centers_barycentric_coordinates(self):
        # Verify that a couple of triangle centers on a random triangle has the expected barycentric coordinates
        vertices = np.random.rand(3, 3)
        el = triangle.edge_lengths(vertices)
        angles = np.array([
            triangle.angle(vertices, 0),
            triangle.angle(vertices, 1),
            triangle.angle(vertices, 2),
        ])

        points = [
            triangle.incenter(vertices),
            triangle.centroid(vertices),
            triangle.circumcenter(vertices),
            triangle.orthocenter(vertices)
        ]
        # From https://en.wikipedia.org/wiki/Trilinear_coordinates
        expected_trilinear_coordinates = [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0 / el[0], 1.0 / el[1], 1.0 / el[2]]),
            np.array([np.cos(angles[0]), np.cos(angles[1]), np.cos(angles[2])]),
            np.array([1.0 / np.cos(angles[0]), 1.0 / np.cos(angles[1]), 1.0 / np.cos(angles[2])]),
        ]
        expected_barycentric_coordinates = np.zeros((len(expected_trilinear_coordinates), 3))
        for i in range(len(expected_trilinear_coordinates)):
            expected_barycentric_coordinates[i] = triangle.trilinear_to_barycentric(expected_trilinear_coordinates[i],
                                                                                    vertices)

        for i in range(len(points)):
            p = points[i]
            bary = triangle.cartesian_to_barycentric(p, vertices)
            self.assertAlmostEqual(bary[0], expected_barycentric_coordinates[i][0])
            self.assertAlmostEqual(bary[1], expected_barycentric_coordinates[i][1])
            self.assertAlmostEqual(bary[2], expected_barycentric_coordinates[i][2])

    def test_vertices_trilinear_coordinates(self):
        vertices = np.random.rand(3, 3)
        hs = triangle.altitudes(vertices)

        for i in range(3):
            bary = triangle.cartesian_to_barycentric(vertices[i], vertices)
            trilinear = triangle.barycentric_to_trilinear(bary, vertices)
            self.assertAlmostEqual(trilinear[i], hs[i])
            self.assertAlmostEqual(trilinear[(i + 1) % 3], 0.0)
            self.assertAlmostEqual(trilinear[(i + 2) % 3], 0.0)

    def test_centers_trilinear_coordinates(self):
        # Verify that a couple of triangle centers on a random triangle has the expected trilinear coordinates
        vertices = np.random.rand(3, 3)
        el = triangle.edge_lengths(vertices)
        angles = np.array([
            triangle.angle(vertices, 0),
            triangle.angle(vertices, 1),
            triangle.angle(vertices, 2),
        ])

        points = [
            triangle.incenter(vertices),
            triangle.centroid(vertices),
            triangle.circumcenter(vertices),
            triangle.orthocenter(vertices)
        ]
        # From https://en.wikipedia.org/wiki/Trilinear_coordinates
        expected_trilinear_coordinates = [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0 / el[0], 1.0 / el[1], 1.0 / el[2]]),
            np.array([np.cos(angles[0]), np.cos(angles[1]), np.cos(angles[2])]),
            np.array([1.0 / np.cos(angles[0]), 1.0 / np.cos(angles[1]), 1.0 / np.cos(angles[2])]),
        ]
        for i in range(len(expected_trilinear_coordinates)):
            expected_trilinear_coordinates[i] = triangle.normalize_trilinear_coordinates(
                expected_trilinear_coordinates[i], vertices)

        for i in range(len(points)):
            p = points[i]
            bary = triangle.cartesian_to_barycentric(p, vertices)
            trilinear = triangle.barycentric_to_trilinear(bary, vertices)
            self.assertAlmostEqual(trilinear[0], expected_trilinear_coordinates[i][0])
            self.assertAlmostEqual(trilinear[1], expected_trilinear_coordinates[i][1])
            self.assertAlmostEqual(trilinear[2], expected_trilinear_coordinates[i][2])

    def test_trilinear_coordinates(self):
        # Verify that the trilinear coordinates is indeed the distances to the triangle edges
        vertices = np.random.rand(3, 3)
        bary = self._random_barycentric_coordinates()
        point = triangle.barycentric_to_cartesian(vertices, bary)
        for i in range(3):
            h1, closest_point = point_line_distance(point, vertices[(i + 1) % 3], vertices[(i + 2) % 3])
            h2 = triangle.barycentric_to_trilinear(bary, vertices)[i]
            self.assertAlmostEqual(h1, h2)


def point_line_distance2(p, p1, p2):
    """
    Squared distance between a point and an infinite line defined by two points.

    :param p: The point.
    :param p1: First point on the line.
    :param p2: Second point on the line.
    :returns: Tuple containing the squared distance and the point on the line closest to p.
    """
    # Line direction vector
    v = p2 - p1
    # Project the point p onto the line
    q = p1 + vector_projection(p - p1, v)
    d2 = np.dot(p - q, p - q)
    return d2, q


def point_line_distance(p, p1, p2):
    """
    Distance between a point and an infinite line defined by two points.

    :param p: The point.
    :param p1: First point on the line.
    :param p2: Second point on the line.
    :returns: Tuple containing the distance and the point on the line closest to p.
    """
    d2, q = point_line_distance2(p, p1, p2)
    return np.sqrt(d2), q


class TestIsDegenerate(unittest.TestCase):
    def test_is_degenerate(self):
        # Three triangles that shouldn't be considered degenerate
        triangles_not_degenerate = [
            np.array([
                [0.24760965, 0.16397004],
                [0.99948244, 0.34642305],
                [0.29575727, 0.24512844]]
            ),
            np.array([
                [0.45904429, 0.99557434],
                [0.59376292, 0.11127516],
                [0.14917516, 0.42804751]
            ]),
            np.array([
                [0.62656294, 0.37315556],
                [0.95052432, 0.79942768],
                [0.19903252, 0.24741029]
            ])
        ]
        for p in triangles_not_degenerate:
            self.assertFalse(triangle.is_degenerate(p))

        # Two triangles that should be considered degenerate
        triangles_degenerate = [
            np.array([
                [0.35665836, 0.1976594],
                [0.3590364, 0.1981898],
                [0.8999754, 0.31050368]
            ]),
            np.array([
                [0.47959977, 0.3451852],
                [0.61701778, 0.19507088],
                [0.27369942, 0.57018227]
            ])
        ]
        for p in triangles_degenerate:
            self.assertTrue(triangle.is_degenerate(p))

        # With lower threshold the two triangles should not be degenerate
        for p in triangles_degenerate:
            self.assertFalse(triangle.is_degenerate(p, eps=2e-5))

        # Whether or not the triangles are degenerate should be scale-invariant
        for p in triangles_not_degenerate:
            self.assertFalse(triangle.is_degenerate(1e-3 * p))
            self.assertFalse(triangle.is_degenerate(1e3 * p))

        for p in triangles_degenerate:
            self.assertTrue(triangle.is_degenerate(1e-3 * p))
            self.assertTrue(triangle.is_degenerate(1e3 * p))


if __name__ == '__main__':
    unittest.main()
