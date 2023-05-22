import datetime
import itertools
import pathlib
import sys
from random import shuffle,randint

import dgl
import gtl.features
import gtl.training
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()
DATA_DIR = PROJECT_DIR / "data" / "generated" / "clustered"


# Experimental constants
BATCHSIZE = 50
LR = 0.01
HIDDEN_LAYERS = 32
PATIENCE = 10
MIN_DELTA = 0.01
EPOCHS = 100
K = 3
N_RUNS=5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters to sweep
GRAPH_TYPES = ["scalefree", "poisson"]
MODELS = ["egi", "triangle"]


def _load_edgelist(path: pathlib.Path | str) -> dgl.DGLGraph:
    if not pathlib.Path(path).is_file():
        print(
            f"File {path} does not exist - generate the dataset before running this script!"
        )
        sys.exit(1)

    g: nx.Graph = nx.read_edgelist(path)
    g: dgl.DGLGraph = dgl.from_networkx(g).to(device)

    return g


# DATASETS

GRAPHS = {
    "scalefree": {
        "clustered": {
            "src": _load_edgelist(DATA_DIR / "powerlaw-clustered-0.edgelist"),
            "target": _load_edgelist(DATA_DIR / "powerlaw-clustered-1.edgelist"),
        },
        "unclustered": {
            "src": _load_edgelist(DATA_DIR / "powerlaw-unclustered-0.edgelist"),
            "target": _load_edgelist(DATA_DIR / "powerlaw-unclustered-1.edgelist"),
        },
    },
    "poisson": {
        "clustered": {
            "src": _load_edgelist(DATA_DIR / "poisson-clustered-0.edgelist"),
            "target": _load_edgelist(DATA_DIR / "poisson-clustered-1.edgelist"),
        },
        "unclustered": {
            "src": _load_edgelist(DATA_DIR / "poisson-unclustered-0.edgelist"),
            "target": _load_edgelist(DATA_DIR / "poisson-unclustered-1.edgelist"),
        },
    },
}


def main() -> None:
    # sweep model, graph type
    trials = list(itertools.product(MODELS, GRAPH_TYPES))
    shuffle(trials)

    current_date_time = datetime.datetime.now().strftime("%Y%m%dT%H%M")

    for model, graph_type in trials:
        for src, target in itertools.permutations(["clustered", "unclustered"], r=2):
            for i in range(N_RUNS):
                wandb.init(
                    project="Clustered Transfer",
                    name=f"{model}-{graph_type}-{src}-{target}-{i}",
                    entity="sta-graph-transfer-learning",
                    group=f"Run {current_date_time}",
                    config={
                        "model": model,
                        "graph_type": graph_type,
                        "src": src,
                        "target": target,
                    },
                )

                _do_run(model, graph_type, src, target)
                wandb.finish()


def _do_run(model: str, graph_type: str, src: str, target: str) -> None:
    src_g: dgl.DGLGraph = GRAPHS[graph_type][src]["src"]
    target_g: dgl_DGLGraph = GRAPHS[graph_type][src]["target"]

    encoder: nn.Module = gtl.training.train_egi_encoder(
        src_g,
        k=K,
        lr=LR,
        n_hidden_layers=HIDDEN_LAYERS,
        sampler=model,
        save_weights_to="pretrain.pt",
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        n_epochs=EPOCHS,
    )

    features: torch.Tensor = gtl.features.degree_bucketing(src_g, HIDDEN_LAYERS)
    features = features.to(device)

    embs = encoder(src_g, features)

    src_nx: nx.Graph = dgl.to_networkx(src_g)
    positive_edges = list(src_nx.edges(data=False))
    nodes = list(src_nx.nodes(data=False))
    negative_edges = _generate_negative_edges(positive_edges, nodes, len(positive_edges))

    # create edge embeddings
    edges = []
    values = []

    for u, v in positive_edges:
        edges.append(_get_edge_embedding(embs, u, v))
        values.append(1)

    for u, v in negative_edges:
        edges.append(_get_edge_embedding(embs, u, v))
        values.append(0)

    train_edges, val_edges, train_classes, val_classes = train_test_split(edges, values)
    train_edges = torch.stack(train_edges)  # list of tensors to 3d tensor
    val_edges = torch.stack(val_edges)  # list of tensors to 3d tensor

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(train_edges, train_classes)

    score = classifier.score(val_edges, val_classes)

    wandb.summary["source-accuracy"] = score

    #################################
    # Direct transfer of embeddings #
    #################################

    features = gtl.features.degree_bucketing(target_g, HIDDEN_LAYERS)
    features = features.to(device)
    embs = encoder(target_g, features)

    target_nx: nx.Graph = dgl.to_networkx(target_g)
    positive_edges = list(target_nx.edges(data=False))
    nodes = list(target_nx.nodes(data=False))
    negative_edges = _generate_negative_edges(positive_edges, nodes, len(positive_edges))

    # create edge embeddings
    edges = []
    values = []

    for u, v in positive_edges:
        edges.append(_get_edge_embedding(embs, u, v))
        values.append(1)

    for u, v in negative_edges:
        edges.append(_get_edge_embedding(embs, u, v))
        values.append(0)

    train_edges, val_edges, train_classes, val_classes = train_test_split(edges, values)
    train_edges = torch.stack(train_edges)  # list of tensors to 3d tensor
    val_edges = torch.stack(val_edges)  # list of tensors to 3d tensor

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(train_edges, train_classes)

    score = classifier.score(val_edges, val_classes)

    wandb.summary["target-accuracy"] = score


def _generate_negative_edges(edges, nodes, n):
    negative_edges = []
    for i in range(n):
        u = randint(0, len(nodes))
        v = randint(0, len(nodes))
        while (
            u == v
            or (u, v) in edges
            or (v, u) in edges
            or v not in nodes
            or u not in nodes
        ):
            u = randint(0, n)
            v = randint(0, n)

        negative_edges.append((u, v))

    return negative_edges


def _get_edge_embedding(emb, a, b):
    return np.multiply(emb[a].detach().cpu(), emb[b].detach().cpu())


if __name__ == "__main__":
    main()
