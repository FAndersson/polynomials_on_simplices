import sys

import pytest

from polynomials_on_simplices.geometry.proximity.aabb import is_equal
from polynomials_on_simplices.geometry.space_partitioning.kd_tree import (
    create_kd_tree, find_leaf_containing_point, in_node_space, node_aabb)
from polynomials_on_simplices.geometry.space_partitioning.tree_traversal import (
    get_descendant_by_address, get_node_by_address)


def test_node_aabb_1d():
    kd = create_kd_tree(1)
    kd.subdivide(0.0)

    aabb1 = node_aabb(kd.children[0])
    assert is_equal(aabb1, ([[float("-inf")], [0.0]]))
    aabb2 = node_aabb(kd.children[1])
    assert is_equal(aabb2, ([[0.0], [float("inf")]]))

    kd.children[0].subdivide(-1.0)
    kd.children[1].subdivide(1.0)
    aabb1 = node_aabb(kd.children[0].children[0])
    assert is_equal(aabb1, ([[float("-inf")], [-1.0]]))
    aabb2 = node_aabb(kd.children[0].children[1])
    assert is_equal(aabb2, ([[-1.0], [0.0]]))
    aabb3 = node_aabb(kd.children[1].children[0])
    assert is_equal(aabb3, ([[0.0], [1.0]]))
    aabb4 = node_aabb(kd.children[1].children[1])
    assert is_equal(aabb4, ([[1.0], [float("inf")]]))


def test_node_aabb_2d():
    kd = create_kd_tree(2)
    kd.subdivide([0.0, 0.0])

    aabb1 = node_aabb(kd.children[0])
    assert is_equal(aabb1, ([[float("-inf"), float("-inf")], [0.0, float("inf")]]))
    aabb2 = node_aabb(kd.children[1])
    assert is_equal(aabb2, ([[0.0, float("-inf")], [float("inf"), float("inf")]]))

    kd.children[0].subdivide([-1.0, -1.0], 1)
    kd.children[1].subdivide([1.0, 1.0], 1)
    aabb1 = node_aabb(kd.children[0].children[0])
    assert is_equal(aabb1, ([[float("-inf"), float("-inf")], [0.0, -1.0]]))
    aabb2 = node_aabb(kd.children[0].children[1])
    assert is_equal(aabb2, ([[float("-inf"), -1.0], [0.0, float("inf")]]))
    aabb3 = node_aabb(kd.children[1].children[0])
    assert is_equal(aabb3, ([[0.0, float("-inf")], [float("inf"), 1.0]]))
    aabb4 = node_aabb(kd.children[1].children[1])
    assert is_equal(aabb4, ([[0.0, 1.0], [float("inf"), float("inf")]]))

    kd.children[0].children[1].subdivide([-2.0, 0.0], 0)
    aabb5 = node_aabb(kd.children[0].children[1].children[0])
    assert is_equal(aabb5, ([[float("-inf"), -1.0], [-2.0, float("inf")]]))
    aabb6 = node_aabb(kd.children[0].children[1].children[1])
    assert is_equal(aabb6, ([[-2.0, -1.0], [0.0, float("inf")]]))


def test_find_leaf_containing_point_2d():
    kd = create_kd_tree(2)
    kd.subdivide([0.0, 0.0])

    kd.children[0].subdivide([-1.0, -1.0], 1)
    kd.children[1].subdivide([1.0, 1.0], 1)

    kd.children[0].children[1].subdivide([-2.0, 0.0], 0)
    kd.children[1].children[0].subdivide([2.0, 0.0], 0)

    node = find_leaf_containing_point(kd, [-2, -2])
    assert node == get_node_by_address(kd, [0, 0, 0])
    node = find_leaf_containing_point(kd, [-3, 0])
    assert node == get_node_by_address(kd, [0, 0, 1, 0])
    node = find_leaf_containing_point(kd, [-1, 1])
    assert node == get_node_by_address(kd, [0, 0, 1, 1])

    node = find_leaf_containing_point(kd, [1, 0])
    assert node == get_descendant_by_address(kd, [1, 0, 0])
    node = find_leaf_containing_point(kd, [3, 0])
    assert node == get_descendant_by_address(kd, [1, 0, 1])
    node = find_leaf_containing_point(kd, [0.5, 1.2])
    assert node == get_descendant_by_address(kd, [1, 1])


def test_in_node_space_2d():
    kd = create_kd_tree(2)
    kd.subdivide([0.0, 0.0])

    kd.children[0].subdivide([-1.0, -1.0], 1)
    kd.children[1].subdivide([1.0, 1.0], 1)

    kd.children[0].children[1].subdivide([-2.0, 0.0], 0)
    kd.children[1].children[0].subdivide([2.0, 0.0], 0)

    points = [
        [-2, -2],
        [-3, 0],
        [-1, 1],
        [1, 0],
        [3, 0],
        [0.5, 1.2]
    ]

    nodes = [
        get_descendant_by_address(kd, [0, 0]),
        get_descendant_by_address(kd, [0, 1, 0]),
        get_descendant_by_address(kd, [0, 1, 1]),
        get_descendant_by_address(kd, [1, 0, 0]),
        get_descendant_by_address(kd, [1, 0, 1]),
        get_descendant_by_address(kd, [1, 1])
    ]

    expected_results = [
        [True, False, False, False, False, False],
        [False, True, False, False, False, False],
        [False, False, True, False, False, False],
        [False, False, False, True, False, False],
        [False, False, False, False, True, False],
        [False, False, False, False, False, True],
    ]

    i = 0
    for point in points:
        j = 0
        for node in nodes:
            result = in_node_space(node, point)
            assert result == expected_results[i][j]
            j += 1
        i += 1


if __name__ == '__main__':
    pytest.main(sys.argv)
