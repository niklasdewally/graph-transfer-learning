#!/usr/bin/env python3
# vim: cc=80: tw=80

import argparse
import json
import pathlib
import sys
import random as rng
from collections.abc import Iterator

import networkx as nx
from gtl.two_part import join_core_periphery, two_part_graph_generator
from tqdm import tqdm
from gcmpy import NetworkNames
from IPython import embed
import gtl.gcmpy.powerlaw
import gtl.gcmpy.poisson
from networkx.readwrite.gml import literal_stringizer
# Setup directories
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()
DATA_DIR = PROJECT_DIR / "data" / "generated" / "modular"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Parameters for generation
PARAMS = {
    "n": 10,
    "module_size": [50,100,1000],
    "ba_m":2,
    "poisson_p":0.4,
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


    params = PARAMS.copy()
    params.update(vars(options))

    _generate_graphs(params)
    _write_params_to_file(vars(options),params)

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

def _generate_graphs(params):

    for module_size in params["module_size"]:
        g1s = _ba_generator(module_size,params)
        g2s = _poisson_generator(module_size,params)

        output_graphs = two_part_graph_generator(g1s,g2s,_joiner)
        for i in range(params["n"]):
            g1_type = "ba"
            g1_size = module_size
            g2_type = "poisson"
            g2_size = module_size

            filename = f"{g1_type}-{g1_size}-{g2_type}-{g2_size}-modular-{i}.gml"

            g = next(output_graphs)

            if not params["dry_run"]:
                # delete unneeded metadata from gcmpy
                for n in g:
                    del g.nodes()[n][NetworkNames.JOINT_DEGREE]
                for u,v in g.edges:
                    if NetworkNames.TOPOLOGY in g.edges[u,v]:
                        del g.edges[u,v][NetworkNames.TOPOLOGY]
                    if NetworkNames.MOTIF_IDS in g.edges[u,v]:
                        del g.edges[u,v][NetworkNames.MOTIF_IDS]


                nx.write_gml(g, DATA_DIR / filename,stringizer=literal_stringizer)



def _ba_generator(size,params) -> Iterator[nx.Graph]:
        return gtl.gcmpy.powerlaw.generator(4,3,size)
    #while True:
        #yield nx.binomial_graph(size,params["poisson_p"])

def _poisson_generator(size,params) -> Iterator[nx.Graph]: 
        return gtl.gcmpy.poisson.generator(4,3,size)
    #while True:
    #    yield nx.barabasi_albert_graph(size,params["ba_m"])

def _joiner(g1: nx.Graph, g2: nx.Graph) -> nx.Graph:
    # renumber nodes so that g1 contains nodes 0-n, and g2 contains nodes n-m
    g1 = nx.convert_node_labels_to_integers(g1)
    g2 = nx.convert_node_labels_to_integers(g2,first_label=g1.number_of_nodes())
    g = nx.compose(g1,g2)

    # join the two modules together with a given density
    density = 0.01
    for n in g1.nodes():
        for m in g2.nodes():
            if rng.random() <= density:
                g.add_edge(n, m)
    
    return g


def _write_params_to_file(options: dict, params: dict) -> None:
    """
    Write generation parameters to a json file.
    """
    output_dict = {"command-line-options": options, "generation-parameters": params}

    with open(DATA_DIR / "config.json", "w") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    sys.exit(main())
