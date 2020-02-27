import sys

import numpy as np
import pytest

from polynomials_on_simplices.geometry.mesh.point_clouds import mean, principal_component_axis
from polynomials_on_simplices.linalg.rotation import random_rotation_matrix_2, random_rotation_matrix_3


def test_principal_axis_simple_2():
    points = np.array([[2.0, 0.0],
                       [-2.0, 0.0],
                       [0.0, 1.0],
                       [0.0, -1.0]])
    w = principal_component_axis(points)
    assert np.array_equal(w, np.array([1.0, 0.0]))

    points = np.array([[0.0, 2.0],
                       [0.0, -2.0],
                       [1.0, 0.0],
                       [-1.0, 0.0]])
    w = principal_component_axis(points)
    assert np.array_equal(w, np.array([0.0, 1.0]))

    r = random_rotation_matrix_2()
    for i in range(len(points)):
        points[i] = np.dot(r, points[i])
    w = principal_component_axis(points)
    assert np.allclose(w, np.dot(r, np.array([0.0, 1.0]))) or np.allclose(-w, np.dot(r, np.array([0.0, 1.0])))


def test_principal_axis_simple_3():
    points = np.array([[2.0, 0.0, 0.0],
                       [-2.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, 0.5],
                       [0.0, 0.0, -0.5]])
    w = principal_component_axis(points)
    assert np.array_equal(w, np.array([1.0, 0.0, 0.0]))

    points = np.array([[0.0, 2.0, 0.0],
                       [0.0, -2.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, 0.0, 3.0],
                       [0.0, 0.0, -2.0]])
    w = principal_component_axis(points)
    assert np.array_equal(w, np.array([0.0, 0.0, 1.0]))

    r = random_rotation_matrix_3()
    for i in range(len(points)):
        points[i] = np.dot(r, points[i])
    w = principal_component_axis(points)
    assert np.allclose(w, np.dot(r, np.array([0.0, 0.0, 1.0]))) or np.allclose(-w, np.dot(r, np.array([0.0, 0.0, 1.0])))


def test_principal_axis_random():
    points = np.random.random_sample((10, 2))
    w = principal_component_axis(points)
    u = np.random.rand(2)
    u /= np.linalg.norm(u)

    points -= mean(points)

    sw = 0
    for i in range(10):
        sw += np.dot(points[i], w)**2
    su = 0
    for i in range(10):
        su += np.dot(points[i], u)**2

    assert sw > su


if __name__ == '__main__':
    pytest.main(sys.argv)
