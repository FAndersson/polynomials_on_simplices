""" Unit tests for the finite_difference module
"""

import unittest

import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess

from polynomials_on_simplices.calculus.finite_difference import (
    central_difference, central_difference_jacobian, forward_difference, forward_difference_jacobian,
    second_central_difference, second_forward_difference)


def is_equal(array1, array2):
    """ Check if two numpy arrays are approximately equal
    """
    try:
        np.testing.assert_allclose(array1, array2, atol=1e-4, rtol=1e-4)
    except AssertionError as ae:
        print(ae)
        return False
    return True


class TestRosenbrockCD(unittest.TestCase):
    def test1(self):
        x = np.zeros(100)
        gradient = rosen_der(x)
        fd_gradient = central_difference(rosen, x)
        self.assertTrue(is_equal(gradient, fd_gradient))

        hessian = rosen_hess(x)
        fd_hessian = second_central_difference(rosen, x)
        self.assertTrue(np.allclose(hessian, fd_hessian, rtol=1e-5, atol=1e-4))

    def test2(self):
        x = np.ones(100)
        gradient = rosen_der(x)
        fd_gradient = central_difference(rosen, x)
        self.assertTrue(is_equal(gradient, fd_gradient))

        hessian = rosen_hess(x)
        fd_hessian = second_central_difference(rosen, x)
        self.assertTrue(np.allclose(hessian, fd_hessian, rtol=1e-5, atol=1e-4))

    def test3(self):
        x = np.random.rand(100)
        gradient = rosen_der(x)
        fd_gradient = central_difference(rosen, x)
        self.assertTrue(is_equal(gradient, fd_gradient))

        hessian = rosen_hess(x)
        fd_hessian = second_central_difference(rosen, x)
        self.assertTrue(np.allclose(hessian, fd_hessian, rtol=1e-5, atol=1e-2))


class TestRosenbrockFD(unittest.TestCase):
    def test1(self):
        x = np.zeros(100)
        gradient = rosen_der(x)
        fd_gradient = forward_difference(rosen, x)
        self.assertTrue(is_equal(gradient, fd_gradient))

        hessian = rosen_hess(x)
        fd_hessian = second_forward_difference(rosen, x)
        self.assertTrue(np.allclose(hessian, fd_hessian, rtol=1e-5, atol=1e-2))

    def test2(self):
        x = np.ones(100)
        gradient = rosen_der(x)
        fd_gradient = forward_difference(rosen, x)
        self.assertTrue(is_equal(gradient, fd_gradient))

        hessian = rosen_hess(x)
        fd_hessian = second_forward_difference(rosen, x)
        self.assertTrue(np.allclose(hessian, fd_hessian, rtol=1e-5, atol=1e-1))

    def test3(self):
        x = np.random.rand(100)
        gradient = rosen_der(x)
        fd_gradient = forward_difference(rosen, x)
        self.assertTrue(is_equal(gradient, fd_gradient))

        hessian = rosen_hess(x)
        fd_hessian = second_forward_difference(rosen, x)
        self.assertTrue(np.allclose(hessian, fd_hessian, rtol=1e-5, atol=1e-1))


class Test1D(unittest.TestCase):
    def test_sin(self):
        f = np.sin
        x = np.random.rand()
        d = forward_difference(f, x)
        self.assertTrue(np.abs(d - np.cos(x)) < 1e-6)
        d = central_difference(f, x)
        self.assertTrue(np.abs(d - np.cos(x)) < 1e-6)

        d2 = second_forward_difference(f, x)
        self.assertTrue(np.abs(d2 - (-np.sin(x))) < 1e-4)
        d2 = second_central_difference(f, x)
        self.assertTrue(np.abs(d2 - (-np.sin(x))) < 1e-5)


class TestJacobian(unittest.TestCase):
    def test_fd_1(self):
        # f : R^2 -> R^2
        def f(x):
            return np.array([x[0]**2 * x[1], 5 * x[0] + np.sin(x[1])])

        p = np.random.rand(2)

        j_expected = np.array([
            [2 * p[0] * p[1], p[0]**2],
            [5, np.cos(p[1])]
        ])
        j_fd = forward_difference_jacobian(f, 2, p)
        assert j_fd.shape == (2, 2)

        self.assertTrue(np.allclose(j_expected, j_fd))

    def test_fd_2(self):
        # f : R^3 -> R^4
        def f(x):
            return np.array([
                x[0],
                5 * x[2],
                4 * x[1]**2 - 2 * x[2],
                x[2] * np.sin(x[0])
            ])

        p = np.random.rand(3)

        j_expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 8.0 * p[1], -2.0],
            [p[2] * np.cos(p[0]), 0.0, np.sin(p[0])]
        ])
        j_fd = forward_difference_jacobian(f, 4, p)
        assert j_fd.shape == (4, 3)

        self.assertTrue(np.allclose(j_expected, j_fd))

    def test_fd_3(self):
        # The Jacobian matrix should still be a matrix, even in the special case where the function is univariate
        # f : R -> R^2
        def f(x):
            return np.array([x, x**2])

        p = np.random.rand()

        j_expected = np.array([[1], [2 * p]])
        j_fd = forward_difference_jacobian(f, 2, p)
        assert j_fd.shape == (2, 1)

        self.assertTrue(np.allclose(j_expected, j_fd))

    def test_fd_4(self):
        # The Jacobian matrix should still be a matrix, even in the special case where the function is scalar valued
        # f : R^2 -> R
        def f(x):
            return x[0] * x[1]**2

        p = np.random.rand(2)

        j_expected = np.array([[p[1]**2, 2 * p[0] * p[1]]])
        j_fd = forward_difference_jacobian(f, 1, p)
        assert j_fd.shape == (1, 2)

        self.assertTrue(np.allclose(j_expected, j_fd))

    def test_cd_1(self):
        # f : R^2 -> R^2
        def f(x):
            return np.array([x[0]**2 * x[1], 5 * x[0] + np.sin(x[1])])

        p = np.random.rand(2)

        j_expected = np.array([
            [2 * p[0] * p[1], p[0]**2],
            [5, np.cos(p[1])]
        ])
        j_fd = central_difference_jacobian(f, 2, p)
        assert j_fd.shape == (2, 2)

        self.assertTrue(np.allclose(j_expected, j_fd))

    def test_cd_2(self):
        # f : R^3 -> R^4
        def f(x):
            return np.array([
                x[0],
                5 * x[2],
                4 * x[1]**2 - 2 * x[2],
                x[2] * np.sin(x[0])
            ])

        p = np.random.rand(3)

        j_expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 8.0 * p[1], -2.0],
            [p[2] * np.cos(p[0]), 0.0, np.sin(p[0])]
        ])
        j_fd = central_difference_jacobian(f, 4, p)
        assert j_fd.shape == (4, 3)

        self.assertTrue(np.allclose(j_expected, j_fd))

    def test_cd_3(self):
        # The Jacobian matrix should still be a matrix, even in the special case where the function is univariate
        # f : R -> R^2
        def f(x):
            return np.array([x, x**2])

        p = np.random.rand()

        j_expected = np.array([[1], [2 * p]])
        j_fd = central_difference_jacobian(f, 2, p)
        assert j_fd.shape == (2, 1)

        self.assertTrue(np.allclose(j_expected, j_fd))

    def test_cd_4(self):
        # The Jacobian matrix should still be a matrix, even in the special case where the function is scalar valued
        # f : R^2 -> R
        def f(x):
            return x[0] * x[1]**2

        p = np.random.rand(2)

        j_expected = np.array([[p[1]**2, 2 * p[0] * p[1]]])
        j_fd = central_difference_jacobian(f, 1, p)
        assert j_fd.shape == (1, 2)

        self.assertTrue(np.allclose(j_expected, j_fd))


if __name__ == "__main__":
    unittest.main()
