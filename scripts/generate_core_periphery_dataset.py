#!/usr/bin/env python3
# vim: cc=80: tw=80

"""
Generate core periphery graphs.

TODO

"""

import argparse
import json
import pathlib
import sys
import typing
from collections.abc import Iterator
from pathlib import Path

import networkx as nx
from gcmpy import NetworkNames
from gtl.cli import standard_generator_parser
from gtl.gcmpy.poisson import generator as poisson_generator
from gtl.two_part import join_core_periphery, two_part_graph_generator
from tqdm import tqdm

# Default config for generation
# to be later overwritten by command arguments
config = {
    "sizes": [(75, 500), (15, 100),(750,5000)],
    "number_of_repeats": 5,
    "core_mean_degree": 9,
    "core_mean_triangles": 35,
    "periphery_mean_degree": 3,
    "periphery_mean_triangles": 1,
}


def main() -> int:
    args = _parse_args()
    config.update(args)

    _confirm_overwrite(config)
    _generate_graphs(config)

    with open(config["data_dir"] / "config.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4, default=str)

    return 0


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description=__doc__, parents=[standard_generator_parser()]
    )

    parser.add_argument("data_dir", help="the directory to save generated graphs to.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    args: dict = vars(args)
    args.update({"data_dir": data_dir})

    return args


def _confirm_overwrite(config: dict) -> None | typing.NoReturn:
    # if overwrite is set to be true, do not ask
    # if overwrite is set to be false, quit
    # if overwrite unspecified, ask the user

    if _is_dir_empty(config["data_dir"]):
        return

    if config["overwrite"] is True:
        _empty_data_directory(config["data_dir"])
        return

    if _confirm_choice("The data directory is not empty. Overwrite?"):
        _empty_data_directory(config["data_dir"])
        return

    print("Error: data directory is not empty.")
    sys.exit(1)


def _empty_data_directory(data_dir):
    for f in data_dir.iterdir():
        if f.is_file():
            f.unlink()


def _confirm_choice(msg):
    answer = input(f"{msg} [y/n] (default: n) ")
    if answer.lower() in ["y", "yes"]:
        return True
    else:
        return False


def _is_dir_empty(path):
    return next(path.iterdir(), None) is None


def _generate_graphs(config: dict):
    for core_size, periphery_size in config["sizes"]:
        cores = _core_generator(core_size, config)
        peripherys = _periphery_generator(periphery_size, config)

        output_graphs = two_part_graph_generator(cores, peripherys, join_core_periphery)

        print(
            f"Generating {config['number_of_repeats']} core-periphery graphs with core size {core_size} "
            f"and periphery size {periphery_size}"
        )

        for i in tqdm(range(config["number_of_repeats"])):
            filename = f"{core_size}-{periphery_size}-{i}.gml"
            g = next(output_graphs)

            if config["dry_run"]:
                print(f"{config['data_dir'] / filename}")
            else:
                _delete_gcmpy_metadata(g)
                nx.write_gml(g, config["data_dir"] / filename)


def _core_generator(core_size: int, config: dict) -> Iterator[nx.Graph]:
    return poisson_generator(
        config["core_mean_degree"], 
        config["core_mean_triangles"],
        core_size, 
    )


def _periphery_generator(periphery_size: int, config: dict) -> Iterator[nx.Graph]:
    return poisson_generator(
        config["periphery_mean_degree"],
        config["periphery_mean_triangles"],
        periphery_size,
    )


def _delete_gcmpy_metadata(g: nx.Graph) -> None:
    # nx.write_gml doesnt know what to do for non string attribute keys

    for n in g:
        del g.nodes()[n][NetworkNames.JOINT_DEGREE]

    for u, v in g.edges:
        if NetworkNames.TOPOLOGY in g.edges[u, v]:
            del g.edges[u, v][NetworkNames.TOPOLOGY]
        if NetworkNames.MOTIF_IDS in g.edges[u, v]:
            del g.edges[u, v][NetworkNames.MOTIF_IDS]


if __name__ == "__main__":
    sys.exit(main())
