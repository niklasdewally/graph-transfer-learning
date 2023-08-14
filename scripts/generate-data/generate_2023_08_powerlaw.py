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
    pass


if __name__ == "__main__":
    sys.exit(main())
