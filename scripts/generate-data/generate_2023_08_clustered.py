"""
Generate clustered graphs for the August 2023 set of experiments (as described in WORDDOC).

See data/README and WORDDOC for more information.
"""


import sys
import argparse
import pathlib
from collections.abc import Iterator

import networkx as nx
from gtl import GraphGenerationScript
from gcmpy import (
    JointDegreeMarginal,
    JointDegreeNames,
    GCMAlgorithmNames,
    GCMAlgorithmNetwork,
    poisson,
    clique_motif,
)

SIZES = [100, 250, 1000, 10000, 100000]
CLIQUE_SIZES = [2, 3, 4, 5]
N_PER_TYPE = 100


def main() -> int:
    generation_script = GraphGenerationScript(_new_generation_strategy())
    generation_script.run()
    return 0


def _new_generation_strategy() -> Iterator[tuple[str, nx.Graph]]:
    for size in SIZES:
        for clique_size in CLIQUE_SIZES:
            for i in range(N_PER_TYPE):
                filename = f"clusteredpoisson-{size}-{clique_size}-{i}.gml"

                yield (filename, _generate(size, clique_size))


def _generate(size: int, clique_size: int) -> nx.Graph:
    mean_degree_2: float = 3  # mean poisson degree of edges
    mean_degree_3: float = 2  # mean poisson degree of triangles / max_clique

    params = {}
    params[JointDegreeNames.MOTIF_SIZES] = [2, clique_size]
    params[JointDegreeNames.ARR_FP] = [poisson(mean_degree_2), poisson(mean_degree_3)]

    params[JointDegreeNames.LOW_HIGH_DEGREE_BOUND] = [(0, 20), (0, 10)]

    DegreeDistObj = JointDegreeMarginal(params)
    jds = DegreeDistObj.sample_jds_from_jdd(size)

    params = {}
    params[GCMAlgorithmNames.MOTIF_SIZES] = [2, clique_size]
    params[GCMAlgorithmNames.EDGE_NAMES] = ["2-clique", f"{clique_size}-clique"]
    params[GCMAlgorithmNames.BUILD_FUNCTIONS] = [clique_motif, clique_motif]
    g = GCMAlgorithmNetwork(params).random_clustered_graph(jds)

    G: nx.Graph = g.G

    # remove self loops and ensure only keep connected componenet
    G.remove_edges_from(nx.selfloop_edges(G))
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)

    # ensure consecutive node labels after having removed some nodes
    g = nx.convert_node_labels_to_integers(G.subgraph(Gcc[0]))
    return g


if __name__ == "__main__":
    sys.exit(main())
