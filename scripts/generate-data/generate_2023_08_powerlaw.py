"""
generate_2023_08_powerlaw.py: generate powerlaw graph dataset for August 2023 experiments.

    python3 generate_2023_08_powerlay.py [--verbose] [--overwrite] [--dry-run] <DATA_DIRECTORY>

Generated graphs are saved to DATA_DIRECTORY. DATA_DIRECTORY must exist and be a directory.


THE GENERATED GRAPHS

Graphs will be generated with the following number of nodes:
    100 250 1,000 10,000 100,000

and the following maximal clique sizes:
    3 4 5

100 graphs are generated for each combination of maximal clique size and node count.


Generated graphs are in gml format and follow the following naming schema:

    powerlaw-{n_nodes}-{max-clique-size}-{i}.gml

This is an identical naming scheme as the poisson graph generator, and the poisson graphs have
poisson in place of powerlaw in the above filename.


Author: Niklas Dewally <nd60@st-andrews.ac.uk>
"""

import sys
from typing import Iterator

import networkx as nx
from gtl import GraphGenerationScript
from gcmpy import (
    JointDegreeMarginal,
    JointDegreeNames,
    GCMAlgorithmNames,
    GCMAlgorithmNetwork,
    power_law,
    clique_motif,
)

SIZES = [100, 250, 10000, 100000]
CLIQUE_SIZES = [3, 4, 5]
N_PER_TYPE = 100


def main() -> int:
    script = GraphGenerationScript(new_generator())
    script.run()
    return 0


def new_generator() -> Iterator[tuple[str, nx.Graph]]:
    for size in SIZES:
        for clique_size in CLIQUE_SIZES:
            for i in range(N_PER_TYPE):
                filename = f"powerlaw-{size}-{clique_size}-{i}.gml"
                graph: nx.Graph = generate_graph(size, clique_size)
                yield (filename, graph)


def generate_graph(size: int, clique_size: int) -> nx.Graph:
    G: nx.Graph = NotImplemented

    # Powerlaw coefficients for edges and motifs.
    #
    # The edge coefficient has been picked such that the network will be ultra small world and
    # scale-free.
    #
    # Networks with c < 2 do not tend to exist, and networks with c >3 become increasingly
    # indistinguishable from random/poisson networks (Barabasi). Therefore, I use the ultra-small
    # world regime to create greater difference between these graphs and the poisson graphs.
    #
    # This range is also found in many real-life networks (Barabasi; Table 4.1).
    #
    #
    # Source: Chapter 4, Network Science by Albert-Laszlo Barabasi.
    # http://www.networksciencebook.com/

    edge_coeff = 2.5

    # The fact that the motif and edge coefficients are identical is arbritrary.
    motif_coeff = edge_coeff

    params = {}
    params[JointDegreeNames.MOTIF_SIZES] = [2, clique_size]
    params[JointDegreeNames.ARR_FP] = [power_law(edge_coeff), power_law(motif_coeff)]
    params[JointDegreeNames.LOW_HIGH_DEGREE_BOUND] = [(0, 20), (0, 10)]

    DegreeDistObj = JointDegreeMarginal(params)
    jds = DegreeDistObj.sample_jds_from_jdd(size)

    params = {}
    params[GCMAlgorithmNames.MOTIF_SIZES] = [2, clique_size]
    params[GCMAlgorithmNames.EDGE_NAMES] = ["2-clique", f"{clique_size}-clique"]
    params[GCMAlgorithmNames.BUILD_FUNCTIONS] = [clique_motif, clique_motif]
    g = GCMAlgorithmNetwork(params).random_clustered_graph(jds)

    G: nx.Graph = g.G

    # ensure no self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # ensure connectivity using greatest connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    # ensure no parallel edges
    G = nx.Graph(G)

    # ensure consecutive node labels after having removed some nodes
    G = nx.convert_node_labels_to_integers(G)

    return G


if __name__ == "__main__":
    sys.exit(main())
