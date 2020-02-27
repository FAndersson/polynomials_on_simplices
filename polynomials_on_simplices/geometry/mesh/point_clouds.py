"""Point cloud operations."""

import numpy as np


def embed_point_in_rn(point, n):
    r"""
    Embed an m-dimensional point in :math:`\mathbb{R}^n, m < n` by adding zeros for the missing coordinates.

    :param point: Point (m:d vector).
    :type point: :class:`Numpy array <numpy.ndarray>`
    :param int n: Dimension of the space we want to embed in.
    :return: The same point embedded in :math:`\mathbb{R}^n`.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> embed_point_in_rn(np.array([1, 1]), 3)
    array([1., 1., 0.])
    """
    assert point.ndim == 1
    m = len(point)
    assert n > m
    return np.concatenate((point, np.zeros(n - m)))


def embed_point_cloud_in_rn(points, n):
    r"""
    Embed an m-dimensional point cloud in :math:`\mathbb{R}^n, m < n` by adding zeros for the missing coordinates.

    :param points: Points in the point cloud (num points by m array).
    :type points: :class:`Numpy array <numpy.ndarray>`
    :param int n: Dimension of the space we want to embed in.
    :return: The same point cloud embedded in :math:`\mathbb{R}^n`.
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> embed_point_cloud_in_rn(np.array([[0, 0], [1, 1]]), 3)
    array([[0., 0., 0.],
           [1., 1., 0.]])
    """
    assert points.ndim == 2
    q, m = points.shape
    assert n > m
    points = np.concatenate((points, np.zeros((q, n - m))), axis=1)
    return np.reshape(points, (-1, n))


def mean(points):
    r"""
    Compute the mean, or centroid, of a point cloud in :math:`\mathbb{R}^m`.

    .. math:: \frac{1}{N} \sum_{i = 1}^N x_i,

    where N is the number of points in the point cloud.

    :param points: Points in the point cloud (N by m array).
    :type points: :class:`Numpy array <numpy.ndarray>`
    :return: Mean of the point cloud along each axis (length m array).
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> mean(np.array([1.0, 3.0, 5.0]))
    3.0
    >>> mean(np.array([[1.0, 2.0], [3.0, 4.0]]))
    array([2., 3.])
    """
    return np.mean(points, axis=0)


def median(points):
    r"""
    Compute the median of a point cloud in :math:`\mathbb{R}^m`.

    The i:th entry of the median is the median of the i:th component of the points in the point cloud.

    :param points: Points in the point cloud (num points by m array).
    :type points: :class:`Numpy array <numpy.ndarray>`
    :return: Median of the point cloud along each axis (length m array).
    :rtype: :class:`Numpy array <numpy.ndarray>`

    .. rubric:: Examples

    >>> median(np.array([1.0, 3.25, 5.0]))
    3.25
    >>> median(np.array([[1.0, 2.0], [3.0, 4.0]]))
    array([2., 3.])
    """
    return np.median(points, axis=0)


def principal_component_axis(points):
    r"""
    Get the principal component axis of a point cloud in :math:`\mathbb{R}^m`.

    The principal component axis is the direction in which the point cloud is most spread out, i.e. the unit vector
    w which maximizes

    .. math:: \sum_{i = 1}^N \langle x_i - \bar{x}, w \rangle,

    where N is the number of points in the point cloud and :math:`\bar{x}` is the mean of the point cloud.

    :param points: Points in the point cloud (N by m array).
    :type points: :class:`Numpy array <numpy.ndarray>`
    :return: Principal component axis of the point cloud (length m array).
    :rtype: :class:`Numpy array <numpy.ndarray>`
    """
    x = points - mean(points)
    xtx = np.dot(x.T, x)
    w, v = np.linalg.eigh(xtx)
    return v[:, -1]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
