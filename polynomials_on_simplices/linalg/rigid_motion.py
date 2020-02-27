"""Rigid motions (elements of :math:`SE(3)`)."""

import numpy as np

from polynomials_on_simplices.linalg.rotation import random_rotation_matrix_2, random_rotation_matrix_3
from polynomials_on_simplices.probability_theory.uniform_sampling import nsphere_sampling


def move(rotation, translation, v):
    """
    Move a vector or set of vectors using a rigid motion.
    The rigid body motion is composed of a rotation followed by a translation.

    :param rotation: Rotation part of the rigid motion (rotation matrix).
    :param translation: Translation part of the rigid motion (3d vector).
    :param v: Vector(s) to move (specified as columns in a matrix).
    :return: Moved vector(s).
    """
    return translate(translation, np.matmul(rotation, v))


def translate(translation, v):
    """
    Translate a vector or set of vectors.

    :param translation: Translation (3d vector).
    :param v: Vector(s) to translate (specified as columns in a matrix).
    :return: Translated vector(s).
    """
    if len(v.shape) == 1:
        return v + translation
    vt = np.empty(v.shape)
    for i in range(v.shape[1]):
        vt[:, i] = v[:, i] + translation
    return vt


def random_rigid_motion(radius=1.0, n=3):
    """
    Generate a random rigid motion with finite translation.

    :param radius: Translation vector will be uniformly sampled from the ball with this radius.
    :param n: Dimension of the space in which the generated rigid motion acts.
    :return: Tuple of a rotation and a translation (orthogonal 3x3 matrix and 3d vector in 3d,
        orthogonal 2x2 matrix and 2d vector in 2d).
    """
    if n == 3:
        rotation = random_rotation_matrix_3()
        translation = nsphere_sampling(3, 1)[0]
        translation *= radius
        return rotation, translation
    if n == 2:
        rotation = random_rotation_matrix_2()
        translation = nsphere_sampling(2, 1)[0]
        translation *= radius
        return rotation, translation
    raise ValueError("Invalid dimension. Only 2d (n=2) and 3d (n=3) supported.")
