"""
The aim of this experiment is to learn node labels from a graph of the airports
of one region, and transfer them to another region. This transfer will occur
directly, without finetuning. 

The node labels are the popularity of the airports, as quartiles (1-4).

As described in [1].


[1] Q. Zhu, C. Yang, Y. Xu, H. Wang, C. Zhang, and J. Han, 
‘Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization’, 
in Advances in Neural Information Processing Systems, Curran Associates, Inc., 
2021, pp. 1766–1779. Accessed: Mar. 07, 2023. [Online]. 
Available: 
https://proceedings.neurips.cc/paper/2021/hash/0dd6049f5fa537d41753be6d37859430-Abstract.html
"""


import datetime
import itertools
import pathlib
import tempfile
from argparse import ArgumentParser, Namespace
from collections.abc import MutableMapping
from pathlib import Path
from random import shuffle

import dgl
import gtl
import gtl.models
import gtl.training
import gtl.wandb
import numpy as np
import torch
import torch.nn as nn
from gtl import Graph
from gtl.cli import add_wandb_options
from gtl.features import degree_bucketing
from gtl.typing import PathLike
from numpy.typing import NDArray
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# pyre-ignore[21]
import wandb

# setup directorys to use for airport data
PROJECT_DIR: Path = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR: Path = PROJECT_DIR / "data" / "airports"

# directory to store temporary model weights used while training
TMP_DIR: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()

# some experimental coCONFIG: dict[str, Any] = {
default_config: MutableMapping = {
    "lr": 0.001,
    "hidden_layers": 32,
    "patience": 10,
    "min_delta": 0.01,
    "epochs": 100,
    "n_runs": 1,
    "source-dataset": "europe",
    "target-dataset": "brazil",
    "models": ["graphsage-mean"],
    "k":2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(edgefile: PathLike, labelfile: PathLike) -> tuple[Graph, NDArray]:
    edges = np.loadtxt(edgefile, dtype="int")
    us = torch.from_numpy(edges[:, 0]).to(device)
    vs = torch.from_numpy(edges[:, 1]).to(device)

    dgl_graph: dgl.DGLGraph = dgl.graph((us, vs), device=torch.device("cpu"))
    dgl_graph = dgl.to_bidirected(dgl_graph).to(device)

    graph: Graph = gtl.Graph.from_dgl_graph(dgl_graph)

    labels = np.loadtxt(labelfile, skiprows=1)

    return graph, labels[:, 1]


europe_g: Graph

# pyre-ignore[5]:
europe_labels: NDArray

brazil_g: Graph

# pyre-ignore[5]:
brazil_labels: NDArray

europe_g, europe_labels = load_dataset(
    f"{str(DATA_DIR)}/europe-airports.edgelist",
    f"{str(DATA_DIR)}/labels-europe-airports.txt",
)

# usa_g, usa_labels = load_dataset('data/usa-airports.edgelist',
#                                 'data/labels-usa-airports.txt')

brazil_g, brazil_labels = load_dataset(
    f"{str(DATA_DIR)}/brazil-airports.edgelist",
    f"{str(DATA_DIR)}/labels-brazil-airports.txt",
)


def do_run() -> None:
    model = wandb.config["model"]
    k = wandb.config["k"]

    # node features for encoder
    europe_node_feats = degree_bucketing(
        europe_g.as_dgl_graph(device), wandb.config["hidden_layers"]
    ).to(device)
    brazil_node_feats = degree_bucketing(
        brazil_g.as_dgl_graph(device), wandb.config["hidden_layers"]
    ).to(device)

    # save graph structural properties to wanb for analysis
    gtl.wandb.log_network_properties(europe_g.as_nx_graph(), prefix="source")
    gtl.wandb.log_network_properties(brazil_g.as_nx_graph(), prefix="target")

    ##########################################################################
    #                     TRAIN SOURCE ENCODER (EUROPE)                      #
    ##########################################################################

    # Training encoder for source data (Europe)

    encoder = gtl.training.train(model, europe_g, europe_node_feats, wandb.config)

    embs = (
        encoder(europe_g.as_dgl_graph(device), europe_node_feats)
        .to(torch.device("cpu"))
        .detach()
        .numpy()
    )

    train_embs, val_embs, train_classes, val_classes = train_test_split(
        embs, europe_labels
    )

    classifier = SGDClassifier()
    classifier = classifier.fit(train_embs, train_classes)

    score = classifier.score(val_embs, val_classes)

    wandb.summary["source-classifier-accuracy"] = score

    ##########################################################################
    #                DIRECT TRANSFER TARGET ENCODER (BRAZIL)                 #
    ##########################################################################

    target_embs = (
        encoder(brazil_g.as_dgl_graph(device), brazil_node_feats)
        .to(torch.device("cpu"))
        .detach()
        .numpy()
    )

    train_embs, val_embs, train_classes, val_classes = train_test_split(
        target_embs, brazil_labels
    )

    classifier = SGDClassifier()
    classifier = classifier.fit(train_embs, train_classes)

    score = classifier.score(target_embs, brazil_labels)

    wandb.summary["target-classifier-accuracy"] = score

    print(f"The target classifier has an accuracy score of {score}")

    ##########################################################################
    #                      WRITE RESULTS TO WANDB                            #
    ##########################################################################

    percentage_difference = (
        wandb.summary["target-classifier-accuracy"]
        - wandb.summary["source-classifier-accuracy"]
    ) / wandb.summary["source-classifier-accuracy"]

    wandb.summary["% Difference"] = percentage_difference * 100


if __name__ == "__main__":
    with wandb.init(
        project="Airport debug",
        entity="sta-graph-transfer-learning",
        config={"model": "graphsage-mean"},
    ) as _:
        # add global config
        wandb.config.update(default_config)
        do_run()


TMP_DIR.cleanup()
