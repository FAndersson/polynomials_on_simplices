"""
A K-d tree is a special case of a binary space partitioning tree, where each space partitioning (hyper-)plane is
aligned with the coordinate axes (plane normal is some coordinate axis).
"""

import numbers

from polynomials_on_simplices.geometry.proximity.aabb import full


class KdTreeNode:
    r"""
    A node in a K-d tree.

    The node is either a leaf node, or it contains a hyperplane that splits
    space into two parts, each part being associated with one of its two child nodes.
    Hence a node in a K-d tree has a subspace of :math:`\mathbb{R}^k` associated with it.
    In a k-d tree the space partitioning hyperplane is always aligned with the coordinate axes. This means that the
    space associated with a node is an axis aligned box (potentially stretching to infinity), see :func:`in_node_space`
    and :func:`node_aabb`.
    """

    def __init__(self, parent, k=3, side=0):
        """
        :param Optional[KdTreeNode] parent: Parent node. None for the root node.
        :param int k: Dimension of the space the k-d tree exists in.
        :param int side: Indicate if this node is associated with the negative or positive side of the
            parent node hyperplane, indicated by a 0 or 1.
        """
        # Parent node. None for the root node
        self.parent = parent
        # For non-root nodes, this indicate if this node is associated with the negative or positive side of the
        # parent node hyperplane, indicated by a 0 or 1
        self.side = side
        # Pair of child nodes, or None for leaf nodes
        self.children = None
        # Point on the hyperplane splitting this node. None for a leaf node
        self.plane_point = None
        # Index of coordinate axis that is the normal of the hyperplane splitting this node. None for a leaf node
        self.plane_normal_idx = None
        # Dimension of the space the k-dimensional tree exists in
        self.k = k

    def is_root(self):
        """
        :return: Whether or not this node is the root node of the tree.
        :rtype: bool
        """
        return self.parent is None

    def is_leaf(self):
        """
        :return: Whether or not this node is a leaf node in the tree.
        :rtype: bool
        """
        return self.children is None

    def sibling(self):
        """
        Get the sibling node for this node.

        :return: Sibling node.
        :rtype: KdTreeNode
        """
        # The root node has no sibling
        if self.is_root():
            return None
        if self.sibling_index() == 0:
            return self.parent.children[1]
        else:
            return self.parent.children[0]

    def subdivide(self, plane_point, plane_normal_idx=0):
        """
        Subdivide this node into 2 child nodes by splitting this node into two halves using the given (hyper-)plane.

        :param plane_point: A point in the (hyper-)plane used to split this node.
        :param int plane_normal_idx: Index of the coordinate axis that should be the normal of the hyperplane that
            splits this node (in [0, 1, ..., k - 1]).
        """
        if not self.is_leaf():
            raise RuntimeError("K-d tree node tree node have already been subdivided")
        assert plane_normal_idx >= 0
        assert plane_normal_idx < self.k
        if self.k == 1:
            if isinstance(plane_point, numbers.Number):
                plane_point = [plane_point]
        assert len(plane_point) == self.k

        # Set splitting plane
        self.plane_point = plane_point
        self.plane_normal_idx = plane_normal_idx
        # Create child nodes
        self.children = KdTreeNode(self, self.k, 0), KdTreeNode(self, self.k, 1)

    def cut_subtree(self):
        """
        Remove the subtree beneath this node (remove child nodes and plane splitting this node).
        """
        # Remove splitting plane
        self.plane_point = None
        self.plane_normal_idx = None
        # Remove child nodes
        self.children = None

    def recurse(self, fn):
        """
        Recursively call a function on this node and all child nodes.

        :param fn: Function to call.
        :type fn: Callable f(node)
        """
        fn(self)

        if self.children is not None:
            self.children[0].recurse(fn)
            self.children[1].recurse(fn)

    def address(self):
        """
        Address of node in the tree.

        An address in the tree is a list of sibling indices. For example:

        [0] gives the root node.
        [0, 1] gives the second (zero-based indexing) child node of the root node.
        [0, 1, 0] gives the first child node of the second child node of the root node.

        :return: List of nodes to follow to reach to this node.
        :rtype: List[{0, 1}]
        """
        if self.is_root():
            return [0]
        address = self.parent.address()
        address += [self.sibling_index()]
        return address

    def depth(self):
        """
        Get depth of node in the tree (number of parents to traverse to reach the root node).

        :return: Depth of the node in the tree.
        :rtype: int
        """
        if self.is_root():
            return 0
        return self.parent.depth() + 1

    def sibling_index(self):
        """
        Get index of node in the pair of siblings.

        :return: 0 or 1 depending on whether this is the first or second
            child node of the parent.
        """
        return self.side


def create_kd_tree(k):
    """
    Create a k-d tree containing a single root node.

    :param int k: Dimension of the space the k-d tree exists in.
    :return: KdTreeNode tree containing a single root node.
    :rtype: KdTreeNode
    """
    return KdTreeNode(None, k=k)


def find_leaf_containing_point(tree_root, point):
    r"""
    Find a leaf node in a k-d tree which contains the given point.

    :param KdTreeNode tree_root: Root node in the k-d tree.
    :param point: Point in :math:`\mathbb{R}^k`.
    :type point: n-dimensional vector
    :return: Leaf node containing the point.
    :rtype: KdTreeNode
    """
    return _find_leaf_containing_point_recursive(tree_root, point)


def in_node_space(tree_node, point):
    r"""
    Check if the given point lies in the part of space associated with the given node in a k-d tree.

    The root node in a k-d tree is associated with all of space.
    For a non root node the node is associated with the part of the parent node space lying on either the negative
    or positive side of the parent node splitting plane (depending on whether the node is the first or second
    child node of the parent).

    :param KdTreeNode tree_node: Tree node in whose space we want to check if the point lies.
    :param point: Point in :math:`\mathbb{R}^k`.
    :type point: n-dimensional vector
    :return: Whether or not the point lies in the part of space associated with the given node.
    :rtype: bool
    """
    if tree_node.is_root():
        return True
    parent = tree_node.parent
    i = parent.plane_normal_idx
    if tree_node.side == 0:
        in_node_subspace = point[i] - parent.plane_point[i] <= 0
    else:
        in_node_subspace = point[i] - parent.plane_point[i] >= 0
    return in_node_subspace and in_node_space(parent, point)


def node_aabb(tree_node):
    r"""
    Get the space associated with a node in a k-d tree. Since the space partitioning hyperplanes in a k-d tree is
    aligned with the coordinate axis, this space is an axis aligned box.

    :param KdTreeNode tree_node: Tree node whose associated space (axis aligned bounding box) we want to get.
    :return: Axis aligned box for the given tree node, represented with two n:d vectors giving the min and max point
        of the AABB.
    :rtype: Pair of n-dimensional vectors
    """
    if tree_node.is_root():
        k = tree_node.k
        aabb = full(k)
        return aabb

    aabb = node_aabb(tree_node.parent)
    i = tree_node.parent.plane_normal_idx
    if tree_node.side == 0:
        aabb[1][i] = tree_node.parent.plane_point[i]
    else:
        aabb[0][i] = tree_node.parent.plane_point[i]
    return aabb


def _find_leaf_containing_point_recursive(tree_node, point):
    r"""
    Recursively traverse a k-d tree to find a leaf node which contains the given point.

    :param KdTreeNode tree_node: Tree node in which we start the traversal. Assumed to contain the point.
    :param point: Point in :math:`\mathbb{R}^k`.
    :type point: n-dimensional vector
    :return: Leaf node containing the point.
    :rtype: KdTreeNode
    """
    if tree_node.is_leaf():
        return tree_node

    i = tree_node.plane_normal_idx
    if point[i] - tree_node.plane_point[i] < 0:
        return _find_leaf_containing_point_recursive(tree_node.children[0], point)
    else:
        return _find_leaf_containing_point_recursive(tree_node.children[1], point)
