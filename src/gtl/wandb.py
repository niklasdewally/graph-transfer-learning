"""
Utilities to log various metadata to Weights and Biases.
"""

import networkx as nx
import wandb


def log_network_properties(graph: nx.Graph, prefix: str = None) -> None:
    """
    Log properties of the given graph to the current wandb run. This includes:
        * Number of nodes
        * Number of edges
        * Average degree assortativity
        * Average clustering coefficient
        * Transitivity

    These will be logged with the prefix specified by prefix, allowing for
    multiple graph's data to be stored within a run.
    """

    # convert (forcibly?) to correct format
    # Graph not MultiDigraph is required
    graph = nx.DiGraph(graph).to_undirected()

    n_nodes = nx.number_of_nodes(graph)
    n_edges = nx.number_of_edges(graph)
    assortativity = nx.degree_assortativity_coefficient(graph)
    avg_clustering_coefficient = nx.average_clustering(graph)
    transitivity = nx.transitivity(graph)

    # TODO: what did peter mean by this?
    # assortativity_in_motifs = None

    if prefix is not None:
        prefix = f"{prefix}-"
    else:
        prefix = ""

    vals = {
        f"{prefix}nodes": n_nodes,
        f"{prefix}edges": n_edges,
        f"{prefix}assortativity": assortativity,
        f"{prefix}avg_clustering_coefficient": avg_clustering_coefficient,
        f"{prefix}transitivity": transitivity,
    }

    wandb.config.update(vals)
