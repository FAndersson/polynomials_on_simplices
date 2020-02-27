"""Functionality for working with intervals on the real line."""

import math


def constrain_to_range(s, min_val, max_val):
    """
    Make sure that a value lies in the given (closed) range.

    :param s: Value to check.
    :param min_val: Lower boundary of the interval.
    :param max_val: Upper boundary of the interval.
    :return: Point closest to the input value which lies in the given range.
    :rtype: float
    """
    return max(min(s, max_val), min_val)


def in_closed_range(s, min_val, max_val):
    """
    Check if a value lies in the closed interval [min_val, max_val].

    :param s: Value to check.
    :param min_val: Lower boundary of the interval.
    :param max_val: Upper boundary of the interval.
    :return: True/False whether or not the value lies in the closed interval.
    :rtype: bool
    """
    return min_val <= s <= max_val


def in_open_range(s, min_val, max_val):
    """
    Check if a value lies in the open interval (min_val, max_val).

    :param s: Value to check.
    :param min_val: Lower boundary of the interval.
    :param max_val: Upper boundary of the interval.
    :return: True/False whether or not the value lies in the open interval.
    :rtype: bool
    """
    return min_val < s < max_val


def in_left_closed_range(s, min_val, max_val):
    """
    Check if a value lies in the left-closed interval [min_val, max_val).

    :param s: Value to check.
    :param min_val: Lower boundary of the interval.
    :param max_val: Upper boundary of the interval.
    :return: True/False whether or not the value lies in the left-closed interval.
    :rtype: bool
    """
    return min_val <= s < max_val


def in_right_closed_range(s, min_val, max_val):
    """
    Check if a value lies in the right-closed interval (min_val, max_val].

    :param s: Value to check.
    :param min_val: Lower boundary of the interval.
    :param max_val: Upper boundary of the interval.
    :return: True/False whether or not the value lies in the right-closed interval.
    :rtype: bool
    """
    return min_val < s <= max_val


def equivalent_periodic_element(a, t):
    r"""
    Get the equivalent scalar in [0, T) for an input scalar a, where two scalars are equivalent if they differ by an
    integer translation of T.

    I.e. consider the equivalence relation :math:`a ~ b \leftrightarrow b = a + n \cdot T, n \in \mathbb{Z}`, then
    given a this function returns the unique element b in [0, T) such that a ~ b.

    :param float a: Input scalar in :math:`(-\infty, \infty)`.
    :param float t: Period T.
    :return: Equivalent element b in [0, T).
    :rtype: float
    """
    if in_left_closed_range(a, 0, t):
        return a

    # Get a positive equivalent scalar
    b = a
    if b < 0:
        n = -b // t
        n += 1
        b += n * t
    # Remove multiples of T
    b = math.fmod(b, t)
    return b
