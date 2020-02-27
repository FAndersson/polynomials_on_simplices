"""
Kd-tree built around a simplicial mesh (a mesh consisting of n-dimensional simplices, e.g. a triangle mesh or a
tetrahedral mesh).
"""

import numbers

from polynomials_on_simplices.geometry.mesh.point_clouds import median, principal_component_axis
from polynomials_on_simplices.geometry.mesh.simplicial_complex import simplex_vertices
from polynomials_on_simplices.geometry.primitives.simplex import centroid
from polynomials_on_simplices.geometry.proximity.aabb import create, intersection, is_empty
from polynomials_on_simplices.geometry.space_partitioning.kd_tree import KdTreeNode, node_aabb


class KdTreeSMNode(KdTreeNode):
    """
    A node in a k-d tree, which is augmented with information about the simplices in a simplicial mesh which
    potentially intersects with the space associated with the node.

    The calculation of potentially intersecting simplices is conservative, in that the list of potentially
    intersecting simplices can contain simplices that doesn't intersect with the node. However it will not miss any
    simplices, i.e. there are no simplices in the mesh that intersect a node which are not part of that nodes list of
    potentially intersecting simplices.
    """

    def __init__(self, parent, side, simplices, vertices, simplices_of_interest=None):
        """
        :param Optional[KdTreeSMNode] parent: Parent node. None for the root node.
        :param int side: Indicate if this node is associated with the negative or positive side of the
            parent node hyperplane, indicated by a 0 or 1.
        :param simplices: The simplices in the mesh the k-d tree is built around.
        :type simplices: num simplices by (n + 1) array of integers
        :param vertices: The vertices in the mesh the k-d tree is built around.
        :type vertices: num vertices by n array of floats
        :param simplices_of_interest: Optional subset of simplices in the mesh that we want to build the k-d tree
            around. If not specified all simplices will be used.
        :type simplices_of_interest: Optional[List[None]]
        """
        try:
            k = len(vertices[0])
        except TypeError:
            k = 1
        KdTreeNode.__init__(self, parent, k, side)
        self.simplices = simplices
        self.vertices = vertices
        # Simplices in the mesh potentially intersecting this k-d tree node
        self.potentially_intersecting_simplices = _get_potentially_intersecting_simplices(self, simplices, vertices,
                                                                                          simplices_of_interest)

    def subdivide(self, plane_point, plane_normal_idx=0):
        """
        Subdivide this node into 2 child nodes by splitting this node into two halves using the given (hyper-)plane.

        :param plane_point: A point in the (hyper-)plane used to split this node.
        :param int plane_normal_idx: Index of the coordinate axis that should be the normal of the hyperplane that
            splits this node (in [0, 1, ..., k - 1]).
        """
        if not self.is_leaf():
            raise RuntimeError("K-d tree node have already been subdivided")
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
        self.children = (KdTreeSMNode(self, 0, self.simplices, self.vertices),
                         KdTreeSMNode(self, 1, self.simplices, self.vertices))


def create_kd_tree(simplices, vertices, fixed_depth=False, max_depth=5, max_simplices_in_leaf=None,
                   simplices_of_interest=None):
    """
    Create a k-d tree which covers a simplicial mesh.

    :param simplices: The simplices in the mesh the k-d tree is built around.
    :type simplices: num simplices by (n + 1) array of integers
    :param vertices: The vertices in the mesh the k-d tree is built around.
    :type vertices: num vertices by n array of floats
    :param bool fixed_depth: Specify how to subdivide the k-d tree when covering the simplicial mesh.
        If fixed depth is True, k-d tree nodes intersecting simplices will be subdivided to the max depth
        specified by the `max_depth` parameter. If fixed_depth is False, the tree will instead be adaptively
        subdivided until the max depth is reached or a leaf node doesn't intersect with more than
        `max_simplices_in_leaf` simplices (if specified).
    :param int max_depth: Maximum depth of the created k-d tree.
    :param Optional[int] max_simplices_in_leaf: If `fixed_depth` is False, the tree will be subdivided
        until no leaf node intersects with more than this number of simplices, or the maximum depth is
        reached.
    :param simplices_of_interest: Optional subset of simplices in the mesh that we want to build the k-d tree around.
        If not specified all simplices will be used.
    :type simplices_of_interest: Optional[List[int]]
    :return: K-d tree build around the line strip.
    :rtype: KdTreeSMNode
    """
    root_node = KdTreeSMNode(None, 0, simplices, vertices, simplices_of_interest)

    # Create function for subdividing the k-d tree
    if fixed_depth:
        def subdivide_fn(kd_tree_node):
            if kd_tree_node.depth() == max_depth:
                # Don't subdivide further
                return
            if not kd_tree_node.potentially_intersecting_simplices:
                # Node don't contain any simplices, no need to subdivide further
                return
            # Subdivide node
            points = [centroid(simplex_vertices(simplices[i], vertices))
                      for i in kd_tree_node.potentially_intersecting_simplices]
            pa = principal_component_axis(points)
            pa = [abs(pa[i]) for i in range(len(pa))]
            axis_idx = pa.index(max(pa))
            kd_tree_node.subdivide(median(points), axis_idx)
    else:
        def subdivide_fn(kd_tree_node):
            if kd_tree_node.depth() == max_depth:
                # Don't subdivide further
                return
            if not kd_tree_node.potentially_intersecting_simplices:
                # Node don't contain any line strip edges, no need to subdivide further
                return
            if len(kd_tree_node.potentially_intersecting_simplices) <= max_simplices_in_leaf:
                # Don't subdivide further
                return
            # Subdivide node
            points = [centroid(simplex_vertices(simplices[i], vertices))
                      for i in kd_tree_node.potentially_intersecting_simplices]
            pa = principal_component_axis(points)
            pa = [abs(pa[i]) for i in range(len(pa))]
            axis_idx = pa.index(max(pa))
            kd_tree_node.subdivide(median(points), axis_idx)

    # Subdivide the k-d tree
    root_node.recurse(subdivide_fn)
    return root_node


def _get_potentially_intersecting_simplices(tree_node, simplices, vertices, simplices_of_interest=None):
    """
    Find which simplices in the given simplicial mesh that potentially intersect the given k-d tree node.

    A simplex is considered potentially intersecting if the AABB for the simplex intersects with the AABB for the k-d
    tree node.

    :param KdTreeSMNode tree_node: K-d tree node which we want to intersect with the line strip.
    :param simplices: The simplices in the simplicial mesh.
    :type simplices: num simplices by (n + 1) array of integers
    :param vertices: The vertices in the simplicial mesh.
    :type vertices: num vertices by n array of floats
    :param simplices_of_interest: Optional subset of simplices in the mesh that we want to build the k-d tree around.
        If not specified all simplices will be used.
    :type simplices_of_interest: Optional[List[int]]
    :return: List of simplices that potentially intersect the k-d tree node.
    :rtype: List[int]
    """
    if tree_node.is_root():
        # The root node covers all the simplices
        if simplices_of_interest is None:
            return [i for i in range(len(simplices))]
        else:
            return simplices_of_interest
    # Only simplices potentially intersecting the parent node could potentially intersect the given node
    aabb = node_aabb(tree_node)
    potentially_intersecting_simplices = []
    for i in tree_node.parent.potentially_intersecting_simplices:
        v = simplex_vertices(simplices[i], vertices)
        edge_aabb = create(v)
        if not is_empty(intersection(aabb, edge_aabb)):
            potentially_intersecting_simplices.append(i)
    return potentially_intersecting_simplices
