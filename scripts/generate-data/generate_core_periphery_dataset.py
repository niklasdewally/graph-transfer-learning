#!/usr/bin/env python3
# vim: cc=80: tw=80

"""
Generate core periphery graphs.

TODO

"""

from collections.abc import Iterator, MutableMapping

import networkx as nx
from gtl import GraphGenerationScript
from gtl.gcmpy.poisson import generator as poisson_generator
from gtl.two_part import join_core_periphery, two_part_graph_generator

# Default config for generation
# to be later overwritten by command arguments
config : MutableMapping = {
    "sizes": [(75, 500), (15, 100), (750, 5000)],
    "number_of_repeats": 10,
    "core_mean_degree": 9,
    "core_mean_triangles": 35,
    "periphery_mean_degree": 3,
    "periphery_mean_triangles": 1,
}


def main() -> None:
    GraphGenerationScript(_generator(config))()


def _generator(config: MutableMapping) -> Iterator[tuple[str, nx.Graph]]:
    for core_size, periphery_size in config["sizes"]:
        cores = _core_generator(core_size, config)
        peripherys = _periphery_generator(periphery_size, config)

        output_graphs = two_part_graph_generator(cores, peripherys, join_core_periphery)

        for i in range(config["number_of_repeats"]):
            filename = f"{core_size}-{periphery_size}-{i}.gml"
            g = next(output_graphs)
            yield (filename, g)


def _core_generator(core_size: int, config: MutableMapping) -> Iterator[nx.Graph]:
    return poisson_generator(
        config["core_mean_degree"],
        config["core_mean_triangles"],
        core_size,
    )


def _periphery_generator(periphery_size: int, config: MutableMapping) -> Iterator[nx.Graph]:
    return poisson_generator(
        config["periphery_mean_degree"],
        config["periphery_mean_triangles"],
        periphery_size,
    )


if __name__ == "__main__":
    main()
