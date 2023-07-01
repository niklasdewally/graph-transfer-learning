"""
A wandb sweep of a simple airport classification experiment, without transfer learning.
Used to tune hyper-parameters of each model.

For more information on sweeping hyperparameters with wandb, see:
https://docs.wandb.ai/guides/sweep/
"""

import pathlib

import argparse
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
from gtl.cli import add_wandb_options
from gtl.features import degree_bucketing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# setup directorys to use for airport data
SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.resolve()
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "airports"

# directory to store temporary model weights used while training
TMP_DIR: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()

config = {
    "batch_size": 50,
    "lr": 0.01,
    "hidden_layers": 32,
    "patience": 10,
    "min_delta": 0.01,
    "n_runs": 10,
    "k": 2,
}


current_date_time: str = datetime.datetime.now().strftime("%Y-%m-%d (T%H%M)")
sweep_config = {
    "project": "Airport hyperparams sweep",
    "entity": "sta-graph-transfer-learning",
    "metric": {"goal": "maximize", "name": "avg_classification_accuracy"},
    "method": "bayes",
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "hidden_layers": {"min": 16, "max": 128},
        "lr": {"min": 0.0001, "max": 0.1},
        "k": {"min": 1, "max": 5},
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    # read model type from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["egi", "triangle"])

    model = parser.parse_args().model

    # add model and name to sweep
    sweep_config.update({"name": f"{model} ({current_date_time})"}),
    sweep_config["parameters"].update({"model": {"value": model}})

    sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id=sweep_id, function=train)


def train() -> None:
    # start a run, providing defaults that can be over-ridden by a sweep
    run = wandb.init(config=config)

    # to reduce variance, do many runs, and optimise on the average
    results = []
    for i in range(wandb.config["n_runs"]):
        res = do_one_run()
        results.append(res)

    wandb.log({"avg_classification_accuracy": sum(results) / len(results)})
    wandb.finish()


def do_one_run():
    europe_g, europe_labels = load_dataset(
        f"{str(DATA_DIR)}/europe-airports.edgelist",
        f"{str(DATA_DIR)}/labels-europe-airports.txt",
    )

    europe_node_feats = degree_bucketing(europe_g, wandb.config["hidden_layers"]).to(
        device
    )

    encoder = gtl.training.train_egi_encoder(
        europe_g,
        n_epochs=100,
        k=wandb.config.k,
        lr=wandb.config.lr,
        n_hidden_layers=wandb.config["hidden_layers"],
        batch_size=wandb.config["batch_size"],
        patience=wandb.config["patience"],
        min_delta=wandb.config["min_delta"],
        sampler=wandb.config["model"],
    )

    embs = encoder(europe_g, europe_node_feats).to(torch.device("cpu")).detach().numpy()

    train_embs, val_embs, train_classes, val_classes = train_test_split(
        embs, europe_labels
    )

    classifier = SGDClassifier()
    classifier = classifier.fit(train_embs, train_classes)

    score = classifier.score(val_embs, val_classes)

    return score


def load_dataset(edgefile, labelfile):
    edges = np.loadtxt(edgefile, dtype="int")
    us = torch.from_numpy(edges[:, 0]).to(device)
    vs = torch.from_numpy(edges[:, 1]).to(device)

    dgl_graph = dgl.graph((us, vs), device=torch.device("cpu"))
    dgl_graph = dgl.to_bidirected(dgl_graph).to(device)

    labels = np.loadtxt(labelfile, skiprows=1)

    return dgl_graph, labels[:, 1]


if __name__ == "__main__":
    main()
