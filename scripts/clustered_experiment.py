import datetime
import itertools
import pathlib
import sys
from random import shuffle, randint, sample

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
from dgl.sampling import global_uniform_negative_sampling
from gtl.clustered import get_filename

from IPython import embed

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
N_RUNS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters to sweep
GRAPH_TYPES = ["powerlaw"]
MODELS = ["triangle","egi"]

# (soruce graph size, target graph size)
# for fewshot learning (train on small, test on large)
SIZES = [(100, 1000), (100, 100),(1000,1000)]


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


def main() -> None:
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

            for i in range(N_RUNS):
                wandb.init(
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
                    },
                )

                try:
                    _do_run(model, graph_type, src, target, src_size, target_size)
                except Exception as e:
                    # report run as failed
                    wandb.finish(exit_code=1)
                    i-= 1
                wandb.finish()


def _do_run(
    model: str, graph_type: str, src: str, target: str, src_size: int, target_size: int
) -> None:
    src_g: dgl.DGLGraph = _load_edgelist(
        DATA_DIR / get_filename(graph_type, src, src_size, 0)
    )
    target_g: dgl.DGLGraph = _load_edgelist(
        DATA_DIR / get_filename(graph_type, target, target_size, 1)
    )

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

    features = gtl.features.degree_bucketing(target_g, HIDDEN_LAYERS)
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


def _get_edge_embedding(emb, a, b):
    return np.multiply(emb[a].detach().cpu(), emb[b].detach().cpu())


if __name__ == "__main__":
    #wandb.init(mode="disabled")
    #_do_run("triangle","powerlaw","clustered","clustered",1000,1000)
    main()
