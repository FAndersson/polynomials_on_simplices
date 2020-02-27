"""Basic functionality for rotations."""

import numpy as np

from polynomials_on_simplices.calculus.angle import compute_angle, orthogonal_vector
from polynomials_on_simplices.calculus.real_interval import constrain_to_range
from polynomials_on_simplices.probability_theory.uniform_sampling import nsphere_surface_sampling


def hat(w):
    r"""
    Get the skew-symmetric matrix :math:`\hat{\omega}` corresponding to a rotation vector :math:`\omega`, i.e., the
    matrix :math:`\hat{\omega}` such that

    .. math:: \hat{\omega} v = \omega \times v, \, \forall v \in \mathbb{R}^3.

    Commonly referred to as the "hat" map.

    :param w: Euler vector (rotation vector).
    :return: Equivalent skew-symmetric matrix (:math:`\hat{\omega}`).
    """
    hat_w = np.zeros((3, 3), dtype=type(w[0]))
    hat_w[0][1] = -w[2]
    hat_w[0][2] = w[1]
    hat_w[1][2] = -w[0]
    hat_w[1][0] = -hat_w[0][1]
    hat_w[2][0] = -hat_w[0][2]
    hat_w[2][1] = -hat_w[1][2]
    return hat_w


def rodrigues_formula(v, k, cost, sint):
    r"""
    Rotate a vector in :math:`\mathbb{R}^3` using Rodrigues' formula.

    :param v: Vector to be rotated.
    :param k: Unit vector describing the axis of rotation.
    :param cost: Cosine of the rotation angle.
    :param sint: Sine of the rotation angle.
    :return: Rotated vector.
    """
    # This formula is only valid for unit vectors k
    assert np.abs(np.linalg.norm(k) - 1.0) < 1e-10
    # Rodrigues' formula
    return v * cost + np.cross(k, v) * sint + k * np.dot(k, v) * (1 - cost)


def rodrigues_formula_matrix(k, cost, sint):
    """
    Compute a rotation matrix from a rotation axis and cosine and sine values of the rotation angle using the
    matrix form of Rodrigues' formula.

    :param k: Unit vector describing the axis of rotation.
    :param cost: Cosine of the rotation angle.
    :param sint: Sine of the rotation angle.
    :return: Rotation matrix (3 by 3 Numpy array).
    """
    # This formula is only valid for unit vectors k
    assert np.abs(np.linalg.norm(k) - 1.0) < 1e-10
    return np.identity(3) * cost + sint * hat(k) + (1 - cost) * np.outer(k, k)


def axis_angle_to_rotation_matrix(axis, angle):
    """
    Convert an axis-angle representation of a rotation (exponential coordinates) to the corresponding rotation matrix.

    :param axis: Rotation axis (unit vector).
    :param angle: Rotation angle.
    :return: Rotation matrix (orthogonal matrix).
    """
    # Just to be sure
    axis /= np.linalg.norm(axis)

    cost = np.cos(angle)
    sint = np.sin(angle)

    # Rodrigues' formula
    return rodrigues_formula_matrix(axis, cost, sint)


def rotation_matrix_to_axis_angle(R):
    r"""
    Convert a rotation matrix to the axis-angle representation (exponential coordinates).

    :param R: Rotation matrix (orthogonal matrix).
    :return: Tuple of an axis (3d vector) and an angle (scalar in the range :math:`[0, \pi]`).
    """
    t = np.trace(R)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    if abs(t - 3.0) < 1e-14:
        # Corner case 1: identity matrix
        an = np.linalg.norm(axis)
        if an > 0.0:
            return axis / an, 0.5 * an
        return np.array([1.0, 0.0, 0.0]), 0.0
    if abs(t + 1.0) < 1e-10:
        # Corner case 2: rotation angle ~= pi
        an = np.linalg.norm(axis)
        if abs(an) < 1e-14:
            # Eigenvector corresponding to the one positive (= 1) eigenvalue is always parallel to R[:, i] + e_i
            # = R[i, :] + e_i
            # However it could happen that R[:, i] + e_i = 0. To avoid this we choose i = arg max(R[i, i]).
            # Then R[i, i] need to be greater than -1, otherwise tr(R) cannot be -1, and hence R[:, i] + e_i != 0
            i = np.argmax(np.diag(R))
            axis = R[:, i]
            axis[i] += 1
            axis /= np.linalg.norm(axis)
            return axis, np.pi
        else:
            # Absolute value of the components of the rotation axis extracted from the diagonal elements of R
            # Formula extracted from Rodrigues' formula by Taylor expansion of cos(theta) around theta = pi
            aa0 = np.sqrt(0.5 * (R[0][0] + 1.0))
            aa1 = np.sqrt(0.5 * (R[1][1] + 1.0))
            aa2 = np.sqrt(0.5 * (R[2][2] + 1.0))
            a0 = np.sign(axis[0]) * aa0
            a1 = np.sign(axis[1]) * aa1
            a2 = np.sign(axis[2]) * aa2
            return np.array([a0, a1, a2]), np.pi - 0.5 * an
    axis /= np.linalg.norm(axis)
    angle = np.arccos(constrain_to_range((t - 1) / 2.0, -1.0, 1.0))
    return axis, angle


def random_rotation_matrix_2():
    """
    Generate a random 2d rotation matrix.

    :return: Rotation matrix (orthogonal matrix).
    """
    theta = 2 * np.pi * np.random.rand()
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])


def random_rotation_matrix_3():
    """
    Generate a random 3d rotation matrix.

    :return: Rotation matrix (orthogonal matrix).
    """
    # See http://en.wikipedia.org/wiki/Rotation_matrix#Uniform_random_rotation_matrices
    R = np.zeros((3, 3))
    # Compute random 2 x 2 matrix
    R[0:2, 0:2] = random_rotation_matrix_2()
    R[2, 2] = 1.0
    # Compute random point on the unit sphere
    v = nsphere_surface_sampling(3, 1)[0]
    # Rotate R so that its third column aligns with v
    axis, angle = compute_rotation(R[:, 2], v)
    R = rotate(axis, angle, R)
    return R


def rotate(axis, angle, v):
    """
    Rotate a vector or a set of vectors a specified angle around an axis.

    :param axis: Axis to rotate around (unit vector).
    :param angle: Angle to rotate (in radians).
    :param v: Vector(s) to rotate (specified as columns in a matrix).
    :return: Rotated vector(s).
    """
    # Create rotation matrix from axis-angle representation
    R = axis_angle_to_rotation_matrix(axis, angle)

    # Rotate vector
    v_rot = np.dot(R, v)
    return v_rot


def compute_rotation(v0, v1):
    """
    Compute the rotation in axis-angle representation which would transform one vector into another, while keeping
    orthogonal vectors fixed.

    :param v0: Initial vector (unit vector).
    :param v1: Final vector (unit vector).
    :return: Tuple of an axis (3d vector) and an angle (scalar).
    """
    # Compute rotation axis
    axis = np.cross(v0, v1)
    if np.dot(axis, axis) == 0:
        # Handle corner case when the two vectors are parallel
        if np.dot(v0, v1) > 0:
            return np.array([1.0, 0.0, 0.0]), 0.0
        else:
            axis = orthogonal_vector(v0)
            axis /= np.linalg.norm(axis)
            angle = np.pi
            return axis, angle
    axis /= np.linalg.norm(axis)
    # Compute angle
    angle = compute_angle(v0, v1)
    return axis, angle
