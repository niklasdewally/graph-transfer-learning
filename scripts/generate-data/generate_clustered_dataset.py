"""
This file generates clustered and unclustered graphs using poisson and powerlaw distributions.

For more information, see gtl.gcmpy.poisson and gtl.gcmpy.powerlaw.
"""

from collections.abc import Iterator, MutableMapping

import gtl.gcmpy
import networkx as nx
from gtl import GraphGenerationScript

config: MutableMapping = {
    "sizes": [100, 1000],
    "number_of_repeats": 5,
}


def main() -> None:
    GraphGenerationScript(_generator(config))()


def _generator(config: MutableMapping) -> Iterator[tuple[str, nx.Graph]]:
    for x in _poisson_generator(config):
        yield x

    for x in _powerlaw_generator(config):
        yield x


def _poisson_generator(config: MutableMapping) -> Iterator[tuple[str, nx.Graph]]:
    for size in config["sizes"]:
        clustered_generator = gtl.gcmpy.poisson.generator(4.7, 50, size)
        for i in range(config["number_of_repeats"]):
            filename = f"poisson-clustered-{size}-{i}.gml"
            g = next(clustered_generator)

            yield (filename, g)

    for size in config["sizes"]:
        unclustered_generator = gtl.gcmpy.poisson.generator(4.7, 0.1, size)
        for i in range(config["number_of_repeats"]):
            filename = f"poisson-unclustered-{size}-{i}.gml"
            g = next(unclustered_generator)
            yield (filename, g)


def _powerlaw_generator(config: MutableMapping) -> Iterator[tuple[str, nx.Graph]]:
    def gen(p: float) -> Iterator[nx.Graph]:
        while True:
            yield nx.powerlaw_cluster_graph(size, 2, p)

    clustered_generator = gen(0.9)
    for size in config["sizes"]:
        for i in range(config["number_of_repeats"]):
            filename = f"powerlaw-clustered-{size}-{i}.gml"
            g = next(clustered_generator)
            yield (filename, g)

    unclustered_generator = gen(0.01)
    for size in config["sizes"]:
        for i in range(config["number_of_repeats"]):
            filename = f"powerlaw-unclustered-{size}-{i}.gml"
            g = next(unclustered_generator)
            yield (filename, g)


if __name__ == "__main__":
    main()
