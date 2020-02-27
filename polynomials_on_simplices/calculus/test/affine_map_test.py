import numbers
import sys

import numpy as np
import pytest

from polynomials_on_simplices.calculus.affine_map import (
    create_affine_map, inverse_affine_transformation, pseudoinverse_affine_transformation)


def test_create_affine_map():
    # Test 1d case (A, b \in R)
    a = 2
    b = 3
    phi = create_affine_map(a, b)
    x = 1.5
    assert phi(x) == 6
    x = -1
    assert phi(x) == 1

    # Test 2d case (A \in R^{2 x 2}, b \in R^2)
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    phi = create_affine_map(a, b)
    x = np.array([1.5, 2.5])
    assert np.linalg.norm(phi(x) - np.array([11.5, 20.5])) < 1e-10
    x = np.array([-1.0, -2.0])
    assert np.linalg.norm(phi(x) - np.array([0.0, -5.0])) < 1e-10
    phi = create_affine_map(a, b, multiple_arguments=True)
    x = np.array([1.5, 2.5])
    assert np.linalg.norm(phi(x[0], x[1]) - np.array([11.5, 20.5])) < 1e-10
    x = np.array([-1.0, -2.0])
    assert np.linalg.norm(phi(x[0], x[1]) - np.array([0.0, -5.0])) < 1e-10

    # Test 1d to 2d case (A \in R^{2 x 1}, b \in R^2)
    a = np.array([[1.0], [3.0]])
    b = np.array([5.0, 6.0])
    phi = create_affine_map(a, b)
    x = 1.5
    assert np.linalg.norm(phi(x) - np.array([6.5, 10.5])) < 1e-10
    x = -1
    assert np.linalg.norm(phi(x) - np.array([4.0, 3.0])) < 1e-10
    # Should also handle the case where A is a 2d-vector (correct type of affine map can be inferred from b)
    a = np.array([1.0, 3.0])
    b = np.array([5.0, 6.0])
    phi = create_affine_map(a, b)
    x = 1.5
    assert np.linalg.norm(phi(x) - np.array([6.5, 10.5])) < 1e-10
    x = -1
    assert np.linalg.norm(phi(x) - np.array([4.0, 3.0])) < 1e-10

    # Test 2d to 1d case (A \in R^{1 x 2}, b \in R)
    a = np.array([[1.0, 3.0]])
    b = 5.0
    phi = create_affine_map(a, b)
    x = np.array([1.5, 2.5])
    assert isinstance(phi(x), numbers.Number)
    assert abs(phi(x) - 14) < 1e-10
    x = np.array([-1.0, -2.0])
    assert np.linalg.norm(phi(x) - (-2)) < 1e-10
    phi = create_affine_map(a, b, multiple_arguments=True)
    x = np.array([1.5, 2.5])
    assert isinstance(phi(x[0], x[1]), numbers.Number)
    assert abs(phi(x[0], x[1]) - 14) < 1e-10
    x = np.array([-1.0, -2.0])
    assert np.linalg.norm(phi(x[0], x[1]) - (-2)) < 1e-10
    a = np.array([1.0, 3.0])
    b = 5.0
    phi = create_affine_map(a, b)
    x = np.array([1.5, 2.5])
    assert isinstance(phi(x), numbers.Number)
    assert abs(phi(x) - 14) < 1e-10
    x = np.array([-1.0, -2.0])
    assert np.linalg.norm(phi(x) - (-2)) < 1e-10
    phi = create_affine_map(a, b, multiple_arguments=True)
    x = np.array([1.5, 2.5])
    assert isinstance(phi(x[0], x[1]), numbers.Number)
    assert abs(phi(x[0], x[1]) - 14) < 1e-10
    x = np.array([-1.0, -2.0])
    assert np.linalg.norm(phi(x[0], x[1]) - (-2)) < 1e-10


def test_inverse_affine_transformation():
    # Test 1d case
    a = 2
    b = 3
    a_inv, b_inv = inverse_affine_transformation(a, b)
    phi = create_affine_map(a, b)
    phi_inv = create_affine_map(a_inv, b_inv)
    x = 1.5
    assert phi(phi_inv(x)) == x
    assert phi_inv(phi(x)) == x
    x = -1
    assert phi(phi_inv(x)) == x
    assert phi_inv(phi(x)) == x

    # Test 2d case
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    a_inv, b_inv = inverse_affine_transformation(a, b)
    phi = create_affine_map(a, b)
    phi_inv = create_affine_map(a_inv, b_inv)
    x = np.array([1.5, 2.5])
    assert np.linalg.norm(phi(phi_inv(x)) - x) < 1e-10
    assert np.linalg.norm(phi_inv(phi(x)) - x) < 1e-10
    x = np.array([-1.0, -2.0])
    assert np.linalg.norm(phi(phi_inv(x)) - x) < 1e-10
    assert np.linalg.norm(phi_inv(phi(x)) - x) < 1e-10


def test_pseudoinverse_affine_transformation():
    # A specific non-invertible affine map R -> R^2
    a = np.array([2, 1])
    b = np.array([-1, -1])
    a_pi, b_pi = pseudoinverse_affine_transformation(a, b)
    phi_pi = create_affine_map(a_pi, b_pi)
    assert isinstance(phi_pi([-1, -1]), numbers.Number)
    assert isinstance(phi_pi([1, 0]), numbers.Number)
    assert abs(phi_pi([-1, -1])) < 1e-12
    assert abs(phi_pi([1, 0]) - 1) < 1e-12
    a = np.array([[2], [1]])
    b = np.array([-1, -1])
    a_pi, b_pi = pseudoinverse_affine_transformation(a, b)
    phi_pi = create_affine_map(a_pi, b_pi)
    assert isinstance(phi_pi([-1, -1]), numbers.Number)
    assert isinstance(phi_pi([1, 0]), numbers.Number)
    assert abs(phi_pi([-1, -1])) < 1e-12
    assert abs(phi_pi([1, 0]) - 1) < 1e-12

    # Generate a random non-invertible affine map R^2 -> R^3
    a = np.random.rand(3, 2)
    b = np.random.rand(3)
    phi = create_affine_map(a, b)
    a_pi, b_pi = pseudoinverse_affine_transformation(a, b)
    phi_pi = create_affine_map(a_pi, b_pi)

    x = np.random.rand(2)
    assert np.linalg.norm(phi_pi(phi(x)) - x) < 1e-12


if __name__ == '__main__':
    pytest.main(sys.argv)
