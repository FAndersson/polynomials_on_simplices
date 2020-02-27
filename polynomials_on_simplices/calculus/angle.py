"""Functionality related to angles and for computing angles between vectors."""

import math

import numpy as np

from polynomials_on_simplices.calculus.real_interval import equivalent_periodic_element, in_left_closed_range


def degrees_to_radians(angle):
    """
    Convert an angle from degrees to radians.

    :param angle: Angle in degrees.
    :return: Corresponding angle in radians.
    :rtype: float
    """
    return angle * math.pi / 180


def radians_to_degrees(angle):
    """
    Convert an angle from radians to degrees.

    :param angle: Angle in radians.
    :return: Corresponding angle in degrees.
    :rtype: float
    """
    return angle * 180 / math.pi


def to_positive_angle_interval(angle):
    r"""
    Return the equivalent angle in the interval :math:`[0, 2 \pi)`.

    :param angle: Angle in the range :math:`(-\infty, \infty)`.
    :return: Equivalent angle in the range :math:`[0, 2 \pi)`.
    :rtype: float
    """
    return equivalent_periodic_element(angle, 2 * math.pi)


def to_centered_angle_interval(angle):
    r"""
    Return the equivalent angle in the interval :math:`[-\pi, \pi)`.

    :param angle: Angle in the range :math:`(-\infty, \infty)`.
    :return: Equivalent angle in the range :math:`[-\pi, \pi)`.
    :rtype: float
    """
    # Get equivalent angle in [0, 2 * pi)
    angle = to_positive_angle_interval(angle)
    # Get equivalent angle in [-pi, pi)
    if angle >= math.pi:
        angle -= 2 * math.pi
    return angle


def compute_angle(v0, v1):
    r"""
    Compute the angle between two vectors.

    :param v0: First vector.
    :param v1: Second vector.
    :return: Angle between the two vectors (in :math:`[0, \pi]`).
    """
    # See https://scicomp.stackexchange.com/questions/27689/numerically-stable-way-of-computing-angles-between-vectors
    assert len(v0) == len(v1)
    if len(v0) == 3:
        return np.arctan2(np.linalg.norm(np.cross(v0, v1)), np.dot(v0, v1))
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return 0.0

    x = np.linalg.norm(v0 / n0 + v1 / n1)
    y = np.linalg.norm(v0 / n0 - v1 / n1)
    return 2 * np.arctan2(y, x)


def is_parallel(v0, v1):
    """
    Check if two vectors are parallel.

    :param v0: First vector.
    :param v1: Second vector.
    :return: True/False whether or not the two vectors are (approximately) parallel.
    """
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return True
    cos_angle = np.dot(v0, v1) / (n0 * n1)
    return abs(cos_angle - 1.0) < 1e-12 or abs(cos_angle + 1.0) < 1e-12


def distance(a0, a1):
    r"""
    Compute the distance between two angles (minimum distance along the unit circle).

    :param a0: First angle (in the range :math:`[-\pi, \pi)`).
    :param a1: Second angle (in the range :math:`[-\pi, \pi)`).
    :return: Distance between the angles.
    """
    assert in_left_closed_range(a0, -np.pi, np.pi), "Angle need to be in the interval [-pi, pi)"
    assert in_left_closed_range(a1, -np.pi, np.pi), "Angle need to be in the interval [-pi, pi)"

    delta = abs(a0 - a1)
    return min(delta, 2 * np.pi - delta)


def direction(a0, a1):
    r"""
    Compute the direction to go from a0 to a1.

    :param a0: First angle (in the range :math:`[-\pi, \pi)`).
    :param a1: Second angle (in the range :math:`[-\pi, \pi)`).
    :return: 1 if the shortest path from a0 to a1 goes anti-clockwise around the unit circle. -1 otherwise.
    """
    assert in_left_closed_range(a0, -np.pi, np.pi), "Angle need to be in the interval [-pi, pi)"
    assert in_left_closed_range(a1, -np.pi, np.pi), "Angle need to be in the interval [-pi, pi)"

    delta = abs(a0 - a1)
    if delta <= 2 * np.pi - delta:
        # Closest path between the two angles doesn't pass the cut at -pi
        if a0 > a1:
            return -1
        else:
            return 1
    else:
        # Closest path between the two angles passes the cut at -pi
        # Move angles half a circle so that the closest path doesn't pass the cut
        return direction(to_centered_angle_interval(a0 + np.pi), to_centered_angle_interval(a1 + np.pi))


def orthogonal_vector(v):
    r"""
    Compute a vector which is orthogonal to the input vector :math:`v`.

    .. note::

        The returned orthogonal vector :math:`v^{\perp}` satisfies :math:`\| v^{\perp} \| \geq \frac{2}{n} \| v \|`,
        where :math:`n` is the dimension of the input vector :math:`v`.

    :param v: Non-zero n-dimensional vector.
    :return: n-dimensional vector which is orthogonal to the input vector.
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    assert np.linalg.norm(v) != 0.0, "Input vector cannot have length zero"
    if len(v) == 2:
        return np.array([-v[1], v[0]])
    else:
        # Orthogonal vector created by swapping the two largest (in absolute value) entries
        # and changing sign on one of them, and setting all other entries to 0
        i_max_1 = -1
        i_max_2 = -1
        max_1 = -1
        max_2 = -1
        for i in range(len(v)):
            av = abs(v[i])
            if av > max_1:
                i_max_2 = i_max_1
                max_2 = max_1
                i_max_1 = i
                max_1 = av
            else:
                if av > max_2:
                    max_2 = av
                    i_max_2 = i
        vp = np.zeros(len(v))
        vp[i_max_2] = v[i_max_1]
        vp[i_max_1] = -v[i_max_2]
        return vp


def orthonormal_frame(v, i=0):
    r"""
    Compute an orthonormal frame `R` in 3d such that the i:th (i in {0, 1, 2}) column is parallel to the given vector
    `v` (:math:`R e_i = v, R^T R = R R^T = I`). The two other columns could then be used as an orthonormal basis for
    the plane with normal `v` and which passes through the origin.

    This is the unique rotation matrix `R` which rotates :math:`e_i` to :math:`v` without twist, i.e. vectors parallel
    to :math:`e_x \times v` are kept fixed.

    :param v: Non-zero 3d vector from which we compute the orthonormal frame (point where the derivative is evaluated).
    :param int i: Column of the orthonormal frame which should be parallel to the input vector (i in {0, 1, 2}).
    :return: Orthonormal frame (3 by 3 orthogonal matrix).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    assert np.linalg.norm(v) != 0.0, "Input axis cannot have length zero"
    # Normalize the input vector
    v /= np.linalg.norm(v)
    r0 = np.identity(3)
    angle = compute_angle(v, r0[:, i])
    if angle < 1e-10:
        return r0
    if abs(angle - np.pi) < 1e-10:
        r0 *= -1
        r0[:, (i + 2) % 3] *= -1
        return r0
    # Compute rotation axis and angle for the rotation which takes e_i to v
    axis = np.cross(r0[:, i], v)
    axis /= np.linalg.norm(axis)
    from polynomials_on_simplices.linalg.rotation import axis_angle_to_rotation_matrix
    return axis_angle_to_rotation_matrix(axis, angle)


class Dial:
    """
    A dial pointing at a position on a circle, while remembering the total number of laps it has turned around.
    """

    def __init__(self):
        # Angle position in [-pi, pi)
        self.angle_pos = 0
        # Number of laps the dial has turned around
        self.num_laps = 0

    def update_position(self, new_angle_pos):
        r"""
        Set the new dial position, adjusting the number of laps if the dial crosses the discontinuity at :math:`\pm\pi`.

        :param new_angle_pos: New angle position the dial points at in the range :math:`[-\pi, \pi)`.
        """
        # Compute difference between new and current angle position
        d = to_centered_angle_interval(new_angle_pos - self.angle_pos)
        # Update angle position
        self.angle_pos += d
        # Keep track of number of laps
        if self.angle_pos >= math.pi:
            self.num_laps += 1
        if self.angle_pos < -math.pi:
            self.num_laps -= 1
        # Set angle position to range [-pi, pi)
        self.angle_pos = to_centered_angle_interval(self.angle_pos)

    def get_total_position(self):
        """
        Get position the dial points at, including full revolutions.

        :return: Dial position in radians.
        :rtype: float
        """
        return self.angle_pos + self.num_laps * 2 * math.pi
