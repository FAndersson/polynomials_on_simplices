import unittest

import numpy as np
import sympy as sp

from polynomials_on_simplices.linalg.basis import (
    basis_matrix, cartesian_basis, dual_basis, gram_schmidt_orthonormalization, gram_schmidt_orthonormalization_rn,
    inverse_metric_tensor, metric_tensor, transform, transform_bilinear_form, transform_from_basis, transform_to_basis)
from polynomials_on_simplices.linalg.rotation import random_rotation_matrix_3
from polynomials_on_simplices.linalg.vector_space_projection import vector_projection


class TestMetricTensor(unittest.TestCase):
    def test_metric_tensor_1d(self):
        g = metric_tensor(cartesian_basis(1))
        self.assertEqual(g, 1)
        g = metric_tensor((2,))
        self.assertEqual(g, 4)
        g = metric_tensor((3,))
        self.assertEqual(g, 9)

    def test_metric_tensor_2d(self):
        g = metric_tensor(cartesian_basis(2))
        g_ref = np.array([[1, 0], [0, 1]])
        np.testing.assert_allclose(g, g_ref, atol=1e-3, rtol=1e-7)

        g = metric_tensor((np.array([2, 1]), np.array([1, 2])))
        g_ref = np.array([[5, 4], [4, 5]])
        np.testing.assert_allclose(g, g_ref, atol=1e-3, rtol=1e-7)

    def test_metric_tensor_3d(self):
        g = metric_tensor(cartesian_basis(3))
        g_ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_allclose(g, g_ref, atol=1e-3, rtol=1e-7)

        g = metric_tensor(basis_matrix((np.array([1, 0, 3]), np.array([1, 3, 0]), np.array([0, 0.5, 4]))))
        g_ref = np.array([[10, 1, 12], [1, 10, 1.5], [12, 1.5, 16.25]])
        np.testing.assert_allclose(g, g_ref, atol=1e-3, rtol=1e-7)


class TestInverseMetricTensor(unittest.TestCase):
    def test_inverse_metric_tensor_1d(self):
        g_inv = inverse_metric_tensor(cartesian_basis(1))
        self.assertEqual(g_inv, 1)
        g_inv = inverse_metric_tensor(2)
        self.assertEqual(g_inv, 1 / 4)
        g_inv = inverse_metric_tensor(3)
        self.assertEqual(g_inv, 1 / 9)

    def test_inverse_metric_tensor_2d(self):
        g_inv = inverse_metric_tensor(cartesian_basis(2))
        g_inv_ref = np.array([[1, 0], [0, 1]])
        np.testing.assert_allclose(g_inv, g_inv_ref, atol=1e-3, rtol=1e-7)

        g_inv = inverse_metric_tensor((np.array([2, 1]), np.array([1, 2])))
        g_inv_ref = np.linalg.inv(np.array([[5, 4], [4, 5]]))
        np.testing.assert_allclose(g_inv, g_inv_ref, atol=1e-3, rtol=1e-7)

    def test_inverse_metric_tensor_3d(self):
        g_inv = inverse_metric_tensor(cartesian_basis(3))
        g_inv_ref = cartesian_basis(3)
        np.testing.assert_allclose(g_inv, g_inv_ref, atol=1e-3, rtol=1e-7)

        g_inv = inverse_metric_tensor(basis_matrix((np.array([1, 0, 3]), np.array([1, 3, 0]), np.array([0, 0.5, 4]))))
        g_inv_ref = np.linalg.inv(np.array([[10, 1, 12], [1, 10, 1.5], [12, 1.5, 16.25]]))
        np.testing.assert_allclose(g_inv, g_inv_ref, atol=1e-3, rtol=1e-7)


class TestDualBasis(unittest.TestCase):
    def test_dual_basis_1d(self):
        basis = cartesian_basis(1)
        dual = dual_basis(basis)
        for i in range(len(dual)):
            for j in range(i + 1, len(dual)):
                self.assertAlmostEqual(np.dot(dual.T[i], basis[j]), 0.0)
            self.assertAlmostEqual(np.dot(dual.T[i], basis[i]), 1.0)

        basis = 2
        dual = dual_basis(basis)
        self.assertEqual(basis * dual, 1.0)

        basis = (3,)
        dual = dual_basis(basis_matrix(basis))
        for i in range(len(dual)):
            for j in range(i + 1, len(dual)):
                self.assertAlmostEqual(np.dot(dual.T[i], basis[j]), 0.0)
            self.assertAlmostEqual(np.dot(dual.T[i], basis[i]), 1.0)

    def test_dual_basis_2d(self):
        basis = cartesian_basis(2)
        dual = dual_basis(basis)
        for i in range(len(dual)):
            for j in range(i + 1, len(dual)):
                self.assertAlmostEqual(np.dot(dual.T[i], basis[j]), 0.0)
            self.assertAlmostEqual(np.dot(dual.T[i], basis[i]), 1.0)

        basis = (np.array([2, 1]), np.array([1, 2]))
        dual = dual_basis(basis_matrix(basis))
        for i in range(len(dual)):
            for j in range(i + 1, len(dual)):
                self.assertAlmostEqual(np.dot(dual.T[i], basis[j]), 0.0)
            self.assertAlmostEqual(np.dot(dual.T[i], basis[i]), 1.0)

    def test_dual_basis_3d(self):
        basis = cartesian_basis(3)
        dual = dual_basis(basis)
        for i in range(len(dual)):
            for j in range(i + 1, len(dual)):
                self.assertAlmostEqual(np.dot(dual[i], basis[j]), 0.0)
            self.assertAlmostEqual(np.dot(dual[i], basis[i]), 1.0)

        basis = (np.array([1, 0, 3]), np.array([1, 3, 0]), np.array([0, 0.5, 4]))
        dual = dual_basis(basis_matrix(basis))
        for i in range(len(dual)):
            for j in range(i + 1, len(dual)):
                self.assertAlmostEqual(np.dot(dual.T[i], basis[j]), 0.0)
            self.assertAlmostEqual(np.dot(dual.T[i], basis[i]), 1.0)


class TestCoordinateChange(unittest.TestCase):
    def test_cartesian_basis(self):
        # Check that the result is the identity when the basis is the Cartesian basis
        v = np.random.rand(3)
        basis = cartesian_basis(3)
        vt = transform_from_basis(v, basis)
        np.testing.assert_allclose(v, vt, atol=1e-3, rtol=1e-7)
        vt = transform_from_basis(v, basis)
        np.testing.assert_allclose(v, vt, atol=1e-3, rtol=1e-7)

    def test_identity(self):
        # Check that the result is identity when transforming back and forth between coordinate systems
        v = np.random.rand(3)
        basis = np.random.rand(3, 3)
        vt = transform_to_basis(v, basis)
        vtt = transform_from_basis(vt, basis)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)
        vt = transform_from_basis(v, basis)
        vtt = transform_to_basis(vt, basis)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)

        # Check that the result is identity when transforming back and forth between coordinate systems (method 2)
        v = np.random.rand(3)
        basis0 = np.random.rand(3, 3)
        basis1 = np.random.rand(3, 3)
        vt = transform(v, basis0, basis1)
        vtt = transform(vt, basis1, basis0)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)
        vt = transform(v, basis1, basis0)
        vtt = transform(vt, basis0, basis1)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)

        # Check that the result is identity when transforming from one coordinate system to itself
        v = np.random.rand(3)
        basis = np.random.rand(3, 3)
        vt = transform(v, basis, basis)
        np.testing.assert_allclose(v, vt, atol=1e-3, rtol=1e-7)

    def test_identity_non_square(self):
        # Check that the result is identity when transforming back and forth between coordinate systems
        basis = np.random.rand(3, 2)
        v = np.random.rand() * basis[:, 0] + np.random.rand() * basis[:, 1]
        vt = transform_to_basis(v, basis)
        vtt = transform_from_basis(vt, basis)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)
        v = np.random.rand(2)
        vt = transform_from_basis(v, basis)
        vtt = transform_to_basis(vt, basis)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)

        # Check that the result is identity when transforming back and forth between coordinate systems (method 2)
        v = np.random.rand(2)
        basis0 = np.random.rand(3, 2)
        basis1 = np.random.rand(3, 3)
        vt = transform(v, basis0, basis1)
        vtt = transform(vt, basis1, basis0)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)

        # Check that the result is identity when transforming from one coordinate system to itself
        v = np.random.rand(2)
        basis = np.random.rand(3, 2)
        vt = transform(v, basis, basis)
        np.testing.assert_allclose(v, vt, atol=1e-3, rtol=1e-7)

    def test_multiple_vectors(self):
        v = np.random.rand(3, 2)
        basis0 = np.random.rand(3, 3)
        basis1 = np.random.rand(3, 3)
        vt = transform(v, basis0, basis1)
        vtt = transform(vt, basis1, basis0)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)
        vt = transform(v, basis1, basis0)
        vtt = transform(vt, basis0, basis1)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)

        v = np.random.rand(2, 2)
        basis0 = np.random.rand(3, 2)
        basis1 = np.random.rand(3, 3)
        vt = transform(v, basis0, basis1)
        vtt = transform(vt, basis1, basis0)
        np.testing.assert_allclose(v, vtt, atol=1e-3, rtol=1e-7)


class TestCoordinateChangeBilinearForm(unittest.TestCase):
    def test_invariant(self):
        # Check that the bilinear form remains invariant when doing coordinate transformations
        bf = np.random.rand(3, 3)
        e = cartesian_basis(3)
        b0 = random_rotation_matrix_3()
        b1 = random_rotation_matrix_3()

        # Compute coordinates for the bilinear form in the 0 and 1 basis
        c0 = transform_bilinear_form(bf, e, b0)
        c1 = transform_bilinear_form(c0, b0, b1)

        # Compute the bilinear forms from the coordinates
        bf0 = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                bf0 += c0[i][j] * np.outer(b0[:, i], b0[:, j])
        bf1 = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                bf1 += c1[i][j] * np.outer(b1[:, i], b1[:, j])

        # Verify that they are the same
        self.assertTrue(np.linalg.norm(bf - bf0) < 1e-10)
        self.assertTrue(np.linalg.norm(bf - bf1) < 1e-10)


class TestGramSchmidtOrthonormalization(unittest.TestCase):
    def test_orthonormal_r3(self):
        # Verity that the output basis is orthonormal
        basis = np.random.rand(3, 3)
        orthonormal_basis = gram_schmidt_orthonormalization_rn(basis)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertAlmostEqual(np.dot(orthonormal_basis[i, :], orthonormal_basis[j, :]), 0.0)
            self.assertAlmostEqual(np.dot(orthonormal_basis[i, :], orthonormal_basis[i, :]), 1.0)

    def test_process_r3(self):
        # Verity that the output basis is the one you obtain using the Gram-Schmidt process
        basis = np.random.rand(3, 3)
        gram_schmidt_basis = gram_schmidt_orthonormalization_rn(basis)
        gram_schmidt_basis_ref = np.empty((3, 3))
        gram_schmidt_basis_ref[:, 0] = basis[:, 0] / np.linalg.norm(basis[:, 0])
        for i in range(1, 3):
            v = basis[:, i]
            for j in range(0, i):
                v -= vector_projection(v, gram_schmidt_basis_ref[:, j])
            v /= np.linalg.norm(v)
            gram_schmidt_basis_ref[:, i] = v
        np.testing.assert_allclose(gram_schmidt_basis, gram_schmidt_basis_ref, atol=1e-3, rtol=1e-7)

    def test_non_square_32(self):
        # Verify that the function can handle also non-square matrices, for example creating an orthonormal
        # basis from two 3d vectors
        basis = np.random.rand(3, 2)
        orthonormal_basis = gram_schmidt_orthonormalization_rn(basis)
        for i in range(2):
            for j in range(i + 1, 2):
                self.assertAlmostEqual(np.dot(orthonormal_basis[:, i], orthonormal_basis[:, j]), 0.0)
            self.assertAlmostEqual(np.dot(orthonormal_basis[:, i], orthonormal_basis[:, i]), 1.0)

    def test_orthonormal_r3_general(self):
        # Verity that the output basis is orthonormal
        basis = np.random.rand(3, 3)
        orthonormal_basis = gram_schmidt_orthonormalization(basis)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertAlmostEqual(np.dot(orthonormal_basis[i], orthonormal_basis[j]), 0.0)
            self.assertAlmostEqual(np.dot(orthonormal_basis[i], orthonormal_basis[i]), 1.0)

    def test_orthonormal_polynomials_3(self):
        x = sp.symbols('x')
        basis = [1, x, x**2, x**3]

        def inner_product(a, b):
            return float(sp.integrate(a * b, (x, 0, 1)))

        orthonormal_basis = gram_schmidt_orthonormalization(basis, inner_product)
        orthonormal_basis_ref = [
            1,
            2 * np.sqrt(3) * x - np.sqrt(3),
            np.sqrt(5) * (6 * x**2 - 6 * x + 1),
            np.sqrt(7) * (20 * x**3 - 30 * x**2 + 12 * x - 1)
        ]
        xv = np.random.rand()
        self.assertTrue(abs((orthonormal_basis[0] - orthonormal_basis_ref[0])) < 1e-10)
        for i in range(1, len(orthonormal_basis)):
            self.assertTrue(abs((orthonormal_basis[i] - orthonormal_basis_ref[i]).subs(x, xv)) < 1e-10)


if __name__ == "__main__":
    unittest.main()
