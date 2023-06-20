"""
Create random (poisson) graphs according to a joint degree distribution
of edges and triangles.
"""
__all__ = ["get_clustering_coefficient", "generator"]

from collections.abc import Iterator

import gcmpy
import networkx as nx
from gcmpy import GCMAlgorithmNames, JointDegreeNames


def get_clustering_coefficient(avg_s, avg_t) -> float:
    """
    Given the overall average degree (avg_s) and the average triangles (avg_t)
    FOR POISSON ONLY, return the clustering coefficient of the network.
    """
    return 2 * avg_t / (2 * avg_t + pow(avg_s, 2))


def generator(avg_s: float, avg_t: float, number_of_nodes: int) -> Iterator[nx.Graph]:
    """
    Return a generator that creates random poisson graphs with a given mean
    number of ordinary (avg_s) and triangle (avg_t) edges.
    """
    # can change?
    kmin_s: int = 0  # smallest degree
    kmax_s: int = 50  # largest degree
    kmin_t: int = 0  # smallest degree
    kmax_t: int = 50  # largest degree

    # generalised configuration model generation parameters
    gcm_params = {}
    gcm_params[GCMAlgorithmNames.MOTIF_SIZES] = [2, 3]
    gcm_params[GCMAlgorithmNames.EDGE_NAMES] = ["2-clique", "3-clique"]
    gcm_params[GCMAlgorithmNames.BUILD_FUNCTIONS] = [
        gcmpy.clique_motif,
        gcmpy.clique_motif,
    ]

    # make joint degrees
    params = {}
    params[JointDegreeNames.MOTIF_SIZES] = [2, 3]
    params[JointDegreeNames.ARR_FP] = [
        gcmpy.poisson(avg_s),
        gcmpy.poisson(avg_t),
    ]
    params[JointDegreeNames.LOW_HIGH_DEGREE_BOUND] = [
        (kmin_s, kmax_s),
        (kmin_t, kmax_t),
    ]

    DegreeDistObj = gcmpy.JointDegreeMarginal(params)

    # generate graphs
    while True:
        joint_degrees = DegreeDistObj.sample_jds_from_jdd(number_of_nodes)
        g = gcmpy.GCMAlgorithmNetwork(gcm_params).random_clustered_graph(joint_degrees)
        G: nx.Graph = g.G

        # remove self loops and ensure only keep connected componenet
        G.remove_edges_from(nx.selfloop_edges(G))
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        yield G.subgraph(Gcc[0])
