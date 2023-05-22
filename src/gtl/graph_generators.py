"""
Generate synthetic graphs for use in experiments
"""

__all__ = ["add_structural_labels",
           "generate_barbasi",
           "generate_forest_fire"
           ]

import networkx as nx
import igraph as ig

def add_structural_labels(G, k=1, existing_labels=None):
    """
    Group nodes based on the isomorphism of their k-hop ego-graphs. These
    groups become numerical labels, emulating structurally relevant labels.

    The Weisfeiler Lehman test of isomorphism is used to create these groupings.

    Args:
        G: A networkx grapgh
        k: The number of hops to include when looking at a nodes ego-graph.
           Defaults to 2.
        existing_labels: If structural labels have already been generated, use
                         these to label matching nodes. Allows use of labels
                         across multiple graphs.
                         Defaults to None.

    Returns: tuple (G,classes)

        G: A networkx graph with integer structural labels. These are stored in the 'struct'
           attribute of each node.

        classes: A dictionary of { hash : ID }, where each unique hash has a unique ID
    """

    G = G.copy()

    # returns a dictionary of format
    # {node: [0-hop-hash,1-hop-hash,2-hop-hash,..,k-hop-hash] }
    WL_hashes = nx.weisfeiler_lehman_subgraph_hashes(G, iterations=k)

    # {node 1:k-hop-hash, node 2:k-hop-hash, ....}
    WL_hashes = {n: hashes[-1] for (n, hashes) in WL_hashes.items()}

    hash_labels = {}
    next_label = 0
    if existing_labels is not None:
        hash_labels = existing_labels
        next_label = max(hash_labels.values()) + 1

    for node, h in WL_hashes.items():
        if h in hash_labels:
            # use existing label
            G.add_node(node, struct=hash_labels.get(h))

        else:
            # create label for hash
            hash_labels.update({h: next_label})
            G.add_node(node, struct=next_label)
            next_label += 1

    return G, hash_labels


def generate_barbasi(n, m):
    """
    Generate a barabasi-albert graph.

    For more information, see:
        - https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html

    Args:
        n: the number of nodes in the graph.
        m: the number of edges to attach from a new node to existing nodes.

    Returns:
        A networkx graph.
    """
    return nx.barabasi_albert_graph(n, m)


def generate_forest_fire(n, f, b):
    """
    Generate a forest fire graph.

    For more info, see:
        - https://python.igraph.org/en/stable/api/igraph.GraphBase.html#Forest_Fire

        - Graphs over Time: Densification Laws, Shrinking Diameters and Possible Explanations,
          Leskovec et al., 2005

    Args:
        n: the number of nodes in the graph.
        f: The forward burning probability.
        b: The backward burning probability.


    Returns:
        an undirected NetworkX graph with n nodes.
    """
    r = f / b
    return ig.Graph.Forest_Fire(n, f, r).to_networkx()
