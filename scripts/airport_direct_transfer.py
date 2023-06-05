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
from argparse import ArgumentParser
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
import wandb
from gtl.argparse import add_wandb_options
from gtl.features import degree_bucketing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from torch.profiler import ProfilerActivity, profile, record_function

# setup directorys to use for airport data
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()
DATA_DIR = PROJECT_DIR / "data" / "airports"

# directory to store temporary model weights used while training
TMP_DIR = tempfile.TemporaryDirectory()

# some experimental constants

BATCHSIZE = 50
LR = 0.01
HIDDEN_LAYERS = 32
PATIENCE = 10
MIN_DELTA = 0.01
EPOCHS = 100
N_RUNS = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opts):
    models = ["egi", "triangle"]
    ks = [1, 2, 3, 4]

    trials = list(itertools.product(models, ks))
    shuffle(trials)

    current_date_time = datetime.datetime.now().strftime("%Y%m%dT%H%M")

    for model, k in trials:
        for i in range(N_RUNS):
            project = "03 Airport Direct Transfer"
            name = f"{model}-k{k}-{i}"
            entity = "sta-graph-transfer-learning"
            group = f"{current_date_time}"
            config = {
                "model": model,
                "k-hops": k,
                "encoder-hidden-layers": HIDDEN_LAYERS,
                "encoder-epochs": EPOCHS,
                "encoder-patience": PATIENCE,
                "encoder-min-delta": MIN_DELTA,
                "encoder-lr": LR,
                "encoder-batchsize": BATCHSIZE,
            }

            with wandb.init(
                project=project,
                name=name,
                entity=entity,
                config=config,
                group=group,
                mode=opts.mode,
            ) as run:
                do_run(k, model)


def load_dataset(edgefile, labelfile):
    edges = np.loadtxt(edgefile, dtype="int")
    us = torch.from_numpy(edges[:, 0]).to(device)
    vs = torch.from_numpy(edges[:, 1]).to(device)

    dgl_graph = dgl.graph((us, vs), device=torch.device("cpu"))
    dgl_graph = dgl.to_bidirected(dgl_graph).to(device)

    labels = np.loadtxt(labelfile, skiprows=1)

    return dgl_graph, labels[:, 1]


def do_run(k, sampler):
    wandb.config.update({"source-dataset": "europe", "target-dataset": "brazil"})

    ##########################################################################
    #                            DATA LOADING                                #
    ##########################################################################

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

    # node features for encoder
    europe_node_feats = degree_bucketing(europe_g, HIDDEN_LAYERS).to(device)
    brazil_node_feats = degree_bucketing(brazil_g, HIDDEN_LAYERS).to(device)

    # save graph structural properties to wanb for analysis
    gtl.wandb.log_network_properties(
        europe_g.cpu().to_simple().to_networkx(), prefix="source"
    )
    gtl.wandb.log_network_properties(
        brazil_g.cpu().to_simple().to_networkx(), prefix="target"
    )

    ##########################################################################
    #                     TRAIN SOURCE ENCODER (EUROPE)                      #
    ##########################################################################

    # Training encoder for source data (Europe)

    encoder = gtl.training.train_egi_encoder(
        europe_g,
        n_epochs=EPOCHS,
        k=k,
        lr=LR,
        n_hidden_layers=HIDDEN_LAYERS,
        batch_size=BATCHSIZE,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        sampler=sampler,
        save_weights_to=Path(TMP_DIR.name, "srcmodel.pt"),
    )

    embs = encoder(europe_g, europe_node_feats).to(torch.device("cpu")).detach().numpy()

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

    target_model = gtl.models.EGI(
        brazil_node_feats.shape[1],
        HIDDEN_LAYERS,
        2,  # see gtl.training.egi
        nn.PReLU(HIDDEN_LAYERS),
    ).to(device)

    target_model.load_state_dict(
        torch.load(Path(TMP_DIR.name, "srcmodel.pt")), strict=False
    )

    target_encoder = target_model.encoder

    target_embs = (
        target_encoder(brazil_g, brazil_node_feats)
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
    parser = add_wandb_options(ArgumentParser())
    opts = parser.parse_args()
    if opts.mode == None:
        opts.mode = "online"
    main(opts)


TMP_DIR.cleanup()
