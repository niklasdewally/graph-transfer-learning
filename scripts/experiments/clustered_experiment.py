import datetime
import itertools
import pathlib
import sys
from argparse import ArgumentParser, Namespace
from random import shuffle
from collections.abc import MutableMapping

import dgl
import gtl.features
import gtl.training
import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray


# pyre-ignore[21]:
import wandb
from dgl.sampling import global_uniform_negative_sampling
from gtl import Graph
from gtl.cli import add_wandb_options
from gtl.clustered import get_filename
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

PROJECT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "generated" / "clustered"


# Experimental constants
CONFIG: MutableMapping = {
    "batch_size": 50,
    "LR": 0.01,
    "hidden_layers": 32,
    "patience": 10,
    "min_delta": 0.01,
    "epochs": 100,
    "k": {"triangle": 4, "egi": 3},
    "n_runs": 20,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters to sweep
GRAPH_TYPES = ["powerlaw"]
MODELS = ["triangle", "egi"]

# (soruce graph size, target graph size)
# for fewshot learning (train on small, test on large)
SIZES = [(100, 1000), (100, 100), (1000, 1000)]


def _load_edgelist(path: pathlib.Path | str) -> dgl.DGLGraph:
    if not pathlib.Path(path).is_file():
        print(
            f"File {path} does not exist - generate the dataset before running this script!"
        )
        sys.exit(1)

    g: dgl.DGLGraph = dgl.from_networkx(nx.read_edgelist(path)).to(device)

    return g


# DATASETS


def main(opts: Namespace) -> None:
    # parameter sweep
    trials = list(itertools.product(MODELS, GRAPH_TYPES, SIZES))
    shuffle(trials)

    current_date_time = datetime.datetime.now().strftime("%Y%m%dT%H%M")

    for model, graph_type, sizes in trials:
        # sizes of training graphs
        src_size, target_size = sizes

        for src, target in itertools.product([True, False], repeat=2):
            src_name = "clustered" if src else "unclustered"
            target_name = "clustered" if target else "unclustered"

            for i in range(CONFIG["n_runs"]):
                wandb.init(
                    mode=opts.mode,
                    project="Clustered Transfer",
                    name=f"{model}-{graph_type}-{src_name}-{src_size}-{target_name}-{target_size}-{i}",
                    entity="sta-graph-transfer-learning",
                    group=f"Run {current_date_time}",
                    config={
                        "model": model,
                        "graph_type": graph_type,
                        "src": src_name,
                        "target": target_name,
                        "src-size": src_size,
                        "target-size": target_size,
                        "global_config": CONFIG,
                    },
                )

                try:
                    _do_run(model, graph_type, src, target, src_size, target_size)
                except Exception:
                    # report run as failed
                    wandb.finish(exit_code=1)
                    i -= 1
                wandb.finish()


def _do_run(
    model: str,
    graph_type: str,
    src: bool,
    target: bool,
    src_size: int,
    target_size: int,
) -> None:
    src_g: dgl.DGLGraph = _load_edgelist(
        DATA_DIR / get_filename(graph_type, src, src_size, 0)
    )
    target_g: dgl.DGLGraph = _load_edgelist(
        DATA_DIR / get_filename(graph_type, target, target_size, 1)
    )

    encoder = gtl.training.train_egi_encoder(
        Graph.from_dgl_graph(src_g),
        k=CONFIG["k"][model],
        lr=CONFIG["LR"],
        n_hidden_layers=CONFIG["hidden_layers"],
        sampler_type=model,
        save_weights_to="pretrain.pt",
        patience=CONFIG["patience"],
        min_delta=CONFIG["min_delta"],
        n_epochs=CONFIG["epochs"],
    )

    features: torch.Tensor = gtl.features.degree_bucketing(
        src_g, CONFIG["hidden_layers"]
    )
    features = features.to(device)

    embs = encoder(src_g, features)

    # generate negative edges
    negative_us, negative_vs = global_uniform_negative_sampling(
        src_g, (src_g.num_edges())
    )

    # get and shuffle positive edges
    shuffle_mask = torch.randperm(src_g.num_edges())
    us, vs = src_g.edges()
    us = us[shuffle_mask]
    vs = vs[shuffle_mask]

    # convert into node embeddings
    us = embs[us]
    vs = embs[vs]
    negative_us = embs[negative_us]
    negative_vs = embs[negative_vs]

    # convert into edge embeddings
    positive_edges = us * vs
    negative_edges = negative_us * negative_vs

    positive_values = torch.ones(positive_edges.shape[0])
    negative_values = torch.zeros(negative_edges.shape[0])

    # create shuffled edge and value list
    edges = torch.cat((positive_edges, negative_edges), 0)
    values = torch.cat((positive_values, negative_values), 0)

    shuffle_mask = torch.randperm(edges.shape[0])
    edges = edges[shuffle_mask]
    values = values[shuffle_mask]
    # embed()

    # convert to lists for training
    # TODO: train on gpu using pytorch

    train_edges, val_edges, train_classes, val_classes = train_test_split(edges, values)

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(
        train_edges.detach().cpu(), train_classes.detach().cpu()
    )

    score = classifier.score(val_edges.detach().cpu(), val_classes.detach().cpu())

    wandb.summary["source-accuracy"] = score

    #################################
    # Direct transfer of embeddings #
    #################################

    features = gtl.features.degree_bucketing(target_g, CONFIG["hidden_layers"])
    features = features.to(device)

    embs = encoder(target_g, features)

    # generate negative edges
    negative_us, negative_vs = global_uniform_negative_sampling(
        target_g, (target_g.num_edges())
    )

    # get and shuffle positive edges
    shuffle_mask = torch.randperm(target_g.num_edges())
    us, vs = target_g.edges()
    us = us[shuffle_mask]
    vs = vs[shuffle_mask]

    # convert into node embeddings
    us = embs[us]
    vs = embs[vs]
    negative_us = embs[negative_us]
    negative_vs = embs[negative_vs]

    # convert into edge embeddings
    positive_edges = us * vs
    negative_edges = negative_us * negative_vs

    positive_values = torch.ones(positive_edges.shape[0])
    negative_values = torch.zeros(negative_edges.shape[0])

    # create shuffled edge and value list
    edges = torch.cat((positive_edges, negative_edges), 0)
    values = torch.cat((positive_values, negative_values), 0)

    shuffle_mask = torch.randperm(edges.shape[0])
    edges = edges[shuffle_mask]
    values = values[shuffle_mask]

    # convert to lists for training
    # TODO: train on gpu using pytorch

    train_edges, val_edges, train_classes, val_classes = train_test_split(edges, values)

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(
        train_edges.detach().cpu(), train_classes.detach().cpu()
    )

    score = classifier.score(val_edges.detach().cpu(), val_classes.detach().cpu())

    wandb.summary["target-accuracy"] = score


def _get_edge_embedding(emb: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> NDArray:
    return np.multiply(emb[a].detach().cpu(), emb[b].detach().cpu())


if __name__ == "__main__":
    parser: ArgumentParser = add_wandb_options(ArgumentParser())
    opts: Namespace = parser.parse_args()
    if opts.mode is None:
        opts.mode = "online"
    main(opts)
