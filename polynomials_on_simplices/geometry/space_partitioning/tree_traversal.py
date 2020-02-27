"""Functionality for navigating through a tree."""


def get_descendant_by_address(tree_node, relative_address):
    """
    Get the descendant node with given address relative to the given tree node.

    A relative address with respect to a tree node is a list of successive children to traverse to reach the desired
    descendant. For example:

    [0] is the address of the first child of the given tree node.
    [1, 2] is the address of the third child of the second child of the given tree node.

    :param tree_node: Tree node whose descendant we want to get.
    :param relative_address: Relative address to the descendant we want to get.
    :type relative_address: List[int]
    """
    if len(relative_address) == 0:
        return tree_node
    return get_descendant_by_address(tree_node.children[relative_address[0]], relative_address[1:])


def get_node_by_address(root_node, address):
    """
    Get the node in the given tree with the given address.

    An address in the tree is a list of sibling indices. For example:

    [0] gives the root node.
    [0, 1] gives the second (zero-based indexing) child node of the root node.
    [0, 1, 0] gives the first child node of the second child node of the root node.

    :param root_node: Root node of the tree we want to traverse.
    :param address: Address to the tree node we want to get.
    :type address: List[int]
    """
    assert address[0] == 0
    if len(address) == 1:
        return root_node
    return get_descendant_by_address(root_node, address[1:])
