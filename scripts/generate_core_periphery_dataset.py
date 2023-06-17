#!/usr/bin/env python3
# vim: cc=80: tw=80

"""
Generate core periphery graphs.

The core and periphery are both generated as powerlaw graphs, using preferential
attachment. The triangle count is altered such that the core has a higher
triangle clustering than the periphery.

"""

import argparse
import json
import pathlib
import sys
from collections.abc import Iterator

import networkx as nx
from gtl.two_part import join_core_periphery, two_part_graph_generator
from tqdm import tqdm

# Setup directories
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()
DATA_DIR = PROJECT_DIR / "data" / "generated" / "core_periphery"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Parameters for generation
PARAMS = { "n": 10,
    "core_m": 2,
    "periphery_m": 1,
    "periphery_sizes": [200,50],
    "core_sizes": [1000,250],
    "target_core_clustering": 0.8,
    "target_periphery_clustering": 0.01,
}


def main() -> int:
    options = _parse_args()

    if not _is_dir_empty(DATA_DIR):
        match vars(options):
            case {"overwrite": True}:
                # empty dir in preparation for data generation
                for f in DATA_DIR.iterdir():
                    if f.is_file():
                        f.unlink()

            case {"overwrite": False}:
                return 1

            case _:
                # no overwrite preference
                if _confirm_choice("The data directory is not empty. Overwrite?"):
                    # empty dir in preparation for data generation
                    for f in DATA_DIR.iterdir():
                        if f.is_file():
                            f.unlink()
                else:
                    return 0

    for i in range(len(PARAMS["core_sizes"])):
        _generate_graphs(
            PARAMS["n"], PARAMS["core_sizes"][i], PARAMS["periphery_sizes"][i], options.dry_run
        )
    _write_params_to_file(vars(options), PARAMS)

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _confirm_choice(msg):
    answer = input(f"{msg} [y/n] (default: n) ")
    if answer.lower() in ["y", "yes"]:
        return True
    else:
        return False


def _is_dir_empty(path):
    return next(path.iterdir(), None) is None


def _generate_graphs(n: int, core_size: int, periphery_size: int, dry_run: int):
    cores = _generator(core_size, PARAMS["m"], PARAMS["target_core_clustering"])
    peripherys = _generator(
        periphery_size, PARAMS["m"], PARAMS["target_periphery_clustering"]
    )

    output_graphs = two_part_graph_generator(cores, peripherys, join_core_periphery)

    print(
        f"Generating {n} core-periphery graphs with core size {core_size} "
        f"and periphery size {periphery_size}"
    )
    for i in tqdm(range(n)):
        filename = f"core-periphery-{core_size}-{periphery_size}-{i}.gml"
        g = next(output_graphs)

        if dry_run:
            print(f"{DATA_DIR / filename}")
        else:
         nx.write_gml(g, DATA_DIR / filename)


def _generator(size: int, m: int, target_clustering: int) -> Iterator[nx.Graph]:
    # TODO
    # Vary mean degree
    # 500 nodes periphery, mean degree 3
    # 75 nodes core, mean degree 10
    # Target: 5000,750 (x10)
    raise NotImplementedError()
    return gtl.gcmpy.poisson.generator(NotImplemented,NotImplemented,size)
    #while True:
    #    yield nx.powerlaw_cluster_graph(size, m, target_clustering)


def _write_params_to_file(options: dict, params: dict) -> None:
    """
    Write generation parameters to a json file.
    """
    output_dict = {"command-line-options": options, "generation-parameters": params}

    with open(DATA_DIR / "config.json", "w") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    sys.exit(main())
