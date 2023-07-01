"""
This file generates clustered and unclustered graphs using poisson and powerlaw distributions.

For more information, see gtl.gcmpy.poisson and gtl.gcmpy.powerlaw.
"""

import pathlib
import gtl.gcmpy
import sys
import networkx as nx
from tqdm import tqdm
import argparse

from statistics import mean

from typing import Optional
from collections.abc import Callable, Iterator
from gtl import GraphGenerationScript

config = {
    "sizes": [100, 1000],
    "number_of_repeats": 5,
}


def main() -> None:
    GraphGenerationScript(_generator(config))()


def _generator(config: dict) -> Iterator[str, nx.Graph]:
    for x in _poisson_generator(config):
        yield x

    for x in _powerlaw_generator(config):
        yield x


def _poisson_generator(config: dict) -> Iterator[str, nx.Graph]:
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


def _powerlaw_generator(config: dict) -> Iterator[str, nx.Graph]:
    def gen(p):
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
