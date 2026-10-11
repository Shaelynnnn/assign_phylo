"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    distances = distances.copy()

    nodes = [
        Node(
            name=label,
            left=None,
            left_distance=0.0,
            right=None,
            right_distance=0.0,
        ) for label in labels
    ]

    internal_nodes_index = 0

    while len(nodes) > 2:
        row_sum = np.sum(distances, axis=1)

        n = len(nodes)
        Q = np.full(distances.shape, np.inf)

        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = (n - 2) * distances[i, j] - row_sum[i] - row_sum[j]

        index = np.argmin(Q)
        i, j = np.unravel_index(index, Q.shape)

        d_i_j = distances[i, j]

        # caculate the branck len
        distance_i = 0.5 * (d_i_j + (row_sum[i] - row_sum[j]) / (n - 2))
        distance_j = d_i_j - distance_i

        # create a new node
        new_node = Node(
            name=f"internal_node_{internal_nodes_index}",
            left=nodes[i],
            left_distance=distance_i,
            right=nodes[j],
            right_distance=distance_j,
        )

        internal_nodes_index += 1

        left_nodes = [
            k for k in range(n)
            if k != i and k != j
        ]

        new_distances = []
        for k in left_nodes:
            distance_to_k = (distances[i, k] + distances[j, k] - d_i_j) / 2
            new_distances.append(distance_to_k)

        # remove the i and j
        left_distances = distances[np.ix_(left_nodes, left_nodes)]

        # update the new dis matrix
        new_size = n - 1
        update_distances = np.zeros((new_size, new_size))
        # add the left node in distances
        update_distances[:-1, :-1] = left_distances
        # add the new node
        update_distances[-1, :-1] = new_distances
        update_distances[:-1, -1] = new_distances

        distances = update_distances

        nodes = [nodes[k] for k in left_nodes]
        nodes.append(new_node)

    # left two nodes
    final_dis = distances[0, 1] / 2

    root = Node(
        name="root",
        left=nodes[0],
        left_distance=final_dis,
        right=nodes[1],
        right_distance=final_dis
    )

    return root


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    # create an axes
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()

    label_colors = kwargs.get("label_colors", {})

    # recursion
    left_index = [0]

    def insert_node(node: None, x: float) -> float:
        """
        recursively insert the node in the tree
        and draw the subtree
        """

        # recursion base case
        if node.left is None and node.right is None:
            y = left_index[0]
            left_index[0] += 1

            # get the color
            color = label_colors.get(node.name, "black")

            ax.text(
                x + 0.1,
                y,
                node.name,
                va="center",
                color = color,
            )

            return y

        # determine the distance
        left_x = x + node.left_distance
        right_x = x + node.right_distance

        # insert the left and right subtree;
        left_y = insert_node(node.left, left_x)
        right_y = insert_node(node.right, right_x)

        # insert horizontal branches
        ax.plot(
            [x, left_x],
            [left_y, left_y],
            color="black",
        )

        ax.plot(
            [x, right_x],
            [right_y, right_y],
            color="black",
        )

        # vertical branch
        ax.plot(
            [x, x],
            [left_y, right_y],
            color="black",
        )

        y = (left_y + right_y) / 2

        return y

    # insert root
    root_y = insert_node(tree, 0.0)

    ax.plot(
        [-0.2, 0],
        [root_y, root_y],
        color="black"
    )

    # ax.set_yticks([])
    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root

def count_leaves(node):
    if node.left is None and node.right is None:
        return 1

    return count_leaves(node.left) + count_leaves(node.right)

def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    # base basic
    if tree is None:
        return

    if tree.left is None and tree.right is None:
        return

    # Sort both subtrees first
    sort_children_by_leaves(tree.left)
    sort_children_by_leaves(tree.right)

    left_count = count_leaves(tree.left)
    right_count = count_leaves(tree.right)

    # Put subtree with fewer leaves on the left
    if left_count > right_count:
        tree.left, tree.right = (tree.right, tree.left)
        tree.left_distance, tree.right_distance = (
            tree.right_distance,
            tree.left_distance,
        )

def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()