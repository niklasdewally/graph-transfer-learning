import argparse
import concurrent.futures
import itertools
import sys
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias
from collections.abc import MutableMapping

import networkx as nx
from gcmpy import NetworkNames
from networkx.readwrite.gml import literal_stringizer

from gtl.cli import standard_generator_parser
from gtl.graph import Graph
import pathlib

GraphGenerationStrategy: TypeAlias = Iterator[(str, nx.Graph)]


class GraphGenerationScript:
    """
    An interactive program to generate graphs according to a given strategy.

    This program takes in command line arguments, then generates graphs and filenames
    from the given strategy, mines features from these graphs, then saves them.

    The graph mining and saving is performed concurrently.

    The GraphGenerationStrategy passed in is expected to return a tuple of (filename,nx graph).
    """

    def __init__(self, generation_strategy: GraphGenerationStrategy) -> None:
        self.generator: GraphGenerationStrategy = generation_strategy
        self.opts: MutableMapping = dict()

        # pyre-ignore[8]:
        self.data_dir: pathlib.Path = None

    def __call__(self) -> None:
        self.run()

    def run(self) -> None:
        self._parse_args()
        self._confirm_overwrite()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(
                GraphGenerationScript._mine_and_save_graph,
                itertools.repeat(self.opts),
                itertools.repeat(self.data_dir),
                self.generator,
            )

            for result in results:
                pass

    @staticmethod
    def _mine_and_save_graph(
        opts: MutableMapping, data_dir: Path, t: tuple[str, nx.Graph]
    ) -> str:
        filename, g = t
        _delete_gcmpy_metadata(g)

        g = GraphGenerationScript._mine_features(g)

        if not opts["dry_run"]:
            nx.write_gml(g, data_dir / filename, stringizer=literal_stringizer)
        print(f"{data_dir / filename}")

        return filename

    @staticmethod
    def _mine_features(g: nx.Graph) -> nx.Graph:
        # pyre-ignore[35]:
        g: Graph = Graph(g)
        g.mine_triangles()
        return g.as_nx_graph()

    def _parse_args(self) -> None | typing.NoReturn:
        parser = argparse.ArgumentParser(
            description=__doc__, parents=[standard_generator_parser()]
        )

        parser.add_argument(
            "data_dir", help="the directory to save generated graphs to."
        )
        self.opts = vars(parser.parse_args())
        self.data_dir = Path(self.opts["data_dir"])

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def _confirm_overwrite(self) -> None | typing.NoReturn:
        # if overwrite is set to be true, do not ask
        # if overwrite is set to be false, quit
        # if overwrite unspecified, ask the user

        if _is_dir_empty(self.data_dir):
            return

        if self.opts["overwrite"] is True:
            _empty_data_directory(self.data_dir)
            return

        if _confirm_choice("The data directory is not empty. Overwrite?"):
            _empty_data_directory(self.data_dir)
            return

        print("Error: data directory is not empty.")
        sys.exit(1)

    def set_generation_strategy(
        self, generation_strategy: GraphGenerationStrategy
    ) -> None:
        self.generator = generation_strategy


def _empty_data_directory(data_dir: Path) -> None:
    for f in data_dir.iterdir():
        if f.is_file():
            f.unlink()


def _confirm_choice(msg: str) -> bool:
    answer = input(f"{msg} [y/n] (default: n) ")
    if answer.lower() in ["y", "yes"]:
        return True
    else:
        return False


def _is_dir_empty(path: Path) -> bool:
    return next(path.iterdir(), None) is None


def _delete_gcmpy_metadata(g: nx.Graph) -> None:
    # nx.write_gml doesnt know what to do for non string attribute keys

    for n in g:
        if NetworkNames.JOINT_DEGREE in g.nodes[n].keys():
            del g.nodes[n][NetworkNames.JOINT_DEGREE]

    for u, v in g.edges:
        if NetworkNames.TOPOLOGY in g.edges[u, v]:
            del g.edges[u, v][NetworkNames.TOPOLOGY]
        if NetworkNames.MOTIF_IDS in g.edges[u, v]:
            del g.edges[u, v][NetworkNames.MOTIF_IDS]
