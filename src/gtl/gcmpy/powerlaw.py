"""
Create scale-free (powerlaw distributed) graphs according to a joint degree 
distribution of edges and triangles.
"""

__all__ = ["generator"]

from collections.abc import Iterator

import gcmpy
import networkx as nx
from gcmpy import GCMAlgorithmNames, JointDegreeNames, power_law


def generator(
    alpha_s: float, alpha_t: float, number_of_nodes: int
) -> Iterator[nx.Graph]:
    """
    Return a generator that creates random scale-free graphs with a given joint
    degree distribution of ordinary and triangle edges.
    """
    # can change?
    # must be atleast 1 for ... reasons
    kmin_s: int = 1  # smallest degree

    kmax_s: int = 100  # largest degree

    # must be atleast 1 for ... reasons
    kmin_t: int = 1  # smallest degree
    kmax_t: int = 100  # largest degree

    # gcm generator paramaters
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
    params[JointDegreeNames.ARR_FP] = [power_law(alpha_s), power_law(alpha_t)]
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

        # remove self loops
        G.remove_edges_from(nx.selfloop_edges(G))

        yield G
