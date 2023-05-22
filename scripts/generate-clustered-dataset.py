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

# setup directorys to use for storing data
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()
DATA_DIR = PROJECT_DIR / "data" / "generated" / "clustered"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# constants

N_NODES = 100


def generate(overwrite: Optional[bool] = None) -> None:
    n = 5
    if not _is_dir_empty(DATA_DIR):
        if overwrite is None:
            if _confirm_choice("The data directory is not empty. Overwrite?"):
                # empty dir in preparation for data generation
                for f in DATA_DIR.iterdir():
                    if f.is_file():
                        f.unlink()
            else:
                return

        elif not overwrite:
            return

        else:
            # empty dir in preparation for data generation
            for f in DATA_DIR.iterdir():
                if f.is_file():
                    f.unlink()

    # generate poisson graphs
    _generate_poisson(n)

    # generate powerlaw graphs
    _generate_powerlaw(n)


def _generate_poisson(n: int) -> None:
    print("Generating poisson graphs")
    clustered_generator = gtl.gcmpy.poisson.generator(4.7, 50, N_NODES)
    unclustered_generator = gtl.gcmpy.poisson.generator(4.7, 0.1, N_NODES)

    clustered = _generate_n_graphs(
        n, lambda x: DATA_DIR / f"poisson-clustered-{x}.edgelist", clustered_generator
    )

    unclustered = _generate_n_graphs(
        n,
        lambda x: DATA_DIR / f"poisson-unclustered-{x}.edgelist",
        unclustered_generator,
    )

    clustered_mean = mean([nx.average_clustering(g) for g in clustered])
    unclustered_mean = mean([nx.average_clustering(g) for g in unclustered])

    print(f"Clustered poisson graphs have mean clustering of {clustered_mean}")
    print(f"Unclustered poisson graphs have mean clustering of {unclustered_mean}")


def _generate_powerlaw(n: int) -> None:
    # couldnt get good results with gcmpy with powerlaw (after trying a similar method to poisson), so
    # we use networkx instead.

    def gen(p):
        while True:
            yield nx.powerlaw_cluster_graph(N_NODES, 4, p)

    print("Generating powerlaw graphs")

    # unclustered_generator = gtl.gcmpy.powerlaw.generator(2,200,N_NODES)
    # clustered_generator = gtl.gcmpy.powerlaw.generator(2,0.9,N_NODES)

    unclustered_generator = gen(0.01)
    clustered_generator = gen(0.9)

    clustered = _generate_n_graphs(
        n, lambda x: DATA_DIR / f"powerlaw-clustered-{x}.edgelist", clustered_generator
    )

    unclustered = _generate_n_graphs(
        n,
        lambda x: DATA_DIR / f"powerlaw-unclustered-{x}.edgelist",
        unclustered_generator,
    )

    clustered_mean = mean([nx.average_clustering(g) for g in clustered])
    unclustered_mean = mean([nx.average_clustering(g) for g in unclustered])

    print(f"Clustered powerlaw graphs have mean clustering of {clustered_mean}")
    print(f"Unclustered powerlaw graphs have mean clustering of {unclustered_mean}")


def _generate_n_graphs(
    n: int, get_filename: Callable[[int], pathlib.Path], generator: Iterator[nx.Graph]
) -> [nx.Graph]:
    """
    Generate and save to disk n graphs from the given generator.
    """

    gs = []
    for i in tqdm(range(n)):
        filename = get_filename(i)
        g = next(generator)
        nx.write_edgelist(g, filename)
        gs.append(g)
    return gs


def _is_dir_empty(path):
    return next(path.iterdir(), None) is None


def _confirm_choice(msg):
    answer = input(f"{msg} [y/n] (default: n) ")
    if answer.lower() in ["y", "yes"]:
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    generate(overwrite = args.overwrite)
