import datetime
import json
import pathlib
import sys
from collections.abc import MutableMapping
from random import sample
from typing import Any

import dgl
import gtl.features
import gtl.training
import numpy as np
import torch

from torch import Tensor

# pyre-ignore [21]:
import wandb
from dgl.sampling import global_uniform_negative_sampling
from gtl import Graph
from numpy.typing import NDArray
from sklearn.linear_model import SGDClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################
#        PATHS        #
#######################

SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.parent.resolve()
HYPERPARAMS_DIR: pathlib.Path = SCRIPT_DIR / "aug_link_prediction_hyperparams"
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "2023-08-clustered"

##############################
#        WANDB CONFIG        #
##############################

default_config: MutableMapping = {
    "sizes": [250, 1000, 10000, 100000],
    "models": [
        "egi",
        "triangle",
        "graphsage-mean",
        "graphsage-pool",
        "graphsage-lstm",
        "graphsage-gcn",
    ],
    "n_runs": 5,
}

current_date_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
entity = "sta-graph-transfer-learning"
project = "August 2023 01: Triangle detection"
group = f"{current_date_time}"

###############################
#        DATA / SPLITS        #
###############################


def main() -> int:
    for model in default_config["models"]:
        for source_size in default_config["sizes"]:
            for target_size in default_config["sizes"]:
                for i in range(default_config["n_runs"]):
                    print(model)
                    wandb.init(
                        project="August 2023 02 - Link Prediction",
                        entity="sta-graph-transfer-learning",
                        group=f"{current_date_time}",
                        config=default_config,
                        name=f"{model} {source_size}->{target_size} {i}",
                    )

                    wandb.config["source_size"] = source_size
                    wandb.config["target_size"] = target_size
                    wandb.config["model"] = model

                    model_config = gtl.load_model_config(HYPERPARAMS_DIR, model)
                    wandb.config.update(model_config)

                    do_run()

                    wandb.finish()
    return 0


def do_run(eval_mode: str = "test") -> None:
    if eval_mode not in ["test", "validate"]:
        raise ValueError(
            f"Unexpected eval_mode f{eval_mode}, valid evaluation modes are [test,validate]"
        )

    # load data

    _gs: list[Graph] = _load_graphs(wandb.config["source_size"])
    source_graph: Graph = _gs[0]
    val_graph: Graph = _gs[1]
    if eval_mode == "test":
        target_graphs = _load_graphs(wandb.config["target_size"])[2:]

    features = gtl.features.degree_bucketing(
        source_graph.as_dgl_graph(device), wandb.config["hidden_layers"]
    ).to(device)

    encoder = gtl.training.train(
        wandb.config["model"], source_graph, features, wandb.config, device=device
    )

    embs = encoder(source_graph.as_dgl_graph(device), features)

    source_dgl: dgl.DGLGraph = source_graph.as_dgl_graph(device)
    neg_us, neg_vs = global_uniform_negative_sampling(
        source_dgl, source_dgl.num_edges() // 2
    )

    # sample randomly edges from the graph. Only sample the same amount as the amount of negative
    # edges we found

    random_idxs = torch.randperm(source_dgl.num_edges() // 2)[: neg_us.shape[0]]
    pos_us, pos_vs = source_graph.as_dgl_graph(device).edges()
    pos_us = pos_us[random_idxs]
    pos_vs = pos_vs[random_idxs]

    edges = torch.cat(
        (
            _get_edge_embeddings(embs, pos_us, pos_vs),
            _get_edge_embeddings(embs, neg_us, neg_vs),
        )
    )

    classes = torch.cat((torch.ones(pos_us.shape[0]), torch.zeros(neg_us.shape[0])))

    random_idxs = torch.randperm(edges.shape[0])
    edges = edges[random_idxs]
    classes = classes[random_idxs]

    edges_np = edges.detach().cpu().numpy()
    classes_np = classes.detach().cpu().numpy()

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(edges_np, classes_np)

    if eval_mode == "test":
        # Test triangle prediction on all target graphs, and keep the average
        wandb.define_metric("acc", summary="mean")

        for i in range(len(target_graphs)):
            target_graph = target_graphs[i]

            target_dgl: dgl.DGLGraph = target_graph.as_dgl_graph(device)
            features = gtl.features.degree_bucketing(
                target_dgl, wandb.config["hidden_layers"]
            ).to(device)

            embs = encoder(target_dgl, features)

            neg_us, neg_vs = global_uniform_negative_sampling(
                target_dgl, target_dgl.num_edges() // 2
            )

            random_idxs = torch.randperm(target_dgl.num_edges() // 2)[: neg_us.shape[0]]
            pos_us, pos_vs = target_graph.as_dgl_graph(device).edges()
            pos_us = pos_us[random_idxs]
            pos_vs = pos_vs[random_idxs]

            edges = torch.cat(
                (
                    _get_edge_embeddings(embs, pos_us, pos_vs),
                    _get_edge_embeddings(embs, neg_us, neg_vs),
                )
            )

            classes = torch.cat(
                (torch.ones(pos_us.shape[0]), torch.zeros(neg_us.shape[0]))
            )

            random_idxs = torch.randperm(edges.shape[0])
            edges = edges[random_idxs]
            classes = classes[random_idxs]

            edges_np = edges.detach().cpu().numpy()
            classes_np = classes.detach().cpu().numpy()

            wandb.summary["acc"] = classifier.score(edges_np, classes_np)

    elif eval_mode == "validate":
        val_dgl: dgl.DGLGraph = val_graph.as_dgl_graph(device)
        features = gtl.features.degree_bucketing(
            val_dgl, wandb.config["hidden_layers"]
        ).to(device)

        embs = encoder(val_graph.as_dgl_graph(device), features)

        neg_us, neg_vs = global_uniform_negative_sampling(
            val_dgl, val_dgl.num_edges() // 2
        )

        random_idxs = torch.randperm(val_dgl.num_edges() // 2)[: neg_us.shape[0]]
        pos_us, pos_vs = val_graph.as_dgl_graph(device).edges()
        pos_us = pos_us[random_idxs]
        pos_vs = pos_vs[random_idxs]

        edges = torch.cat(
            (
                _get_edge_embeddings(embs, pos_us, pos_vs),
                _get_edge_embeddings(embs, neg_us, neg_vs),
            )
        )

        classes = torch.cat((torch.ones(pos_us.shape[0]), torch.zeros(neg_us.shape[0])))

        random_idxs = torch.randperm(edges.shape[0])
        edges = edges[random_idxs]
        classes = classes[random_idxs]

        edges_np = edges.detach().cpu().numpy()
        classes_np = classes.detach().cpu().numpy()

        wandb.summary["acc"] = classifier.score(edges_np, classes_np)


def _load_graphs(size: int) -> list[Graph]:
    graphs: list[Graph] = []

    def filename(i: int) -> str:
        return f"poisson-{size}-3-{i}.gml"

    for i in range(100):
        graph = Graph.from_gml_file(DATA_DIR / filename(i))
        graphs.append(graph)

    return graphs


def _get_edge_embeddings(embs: Tensor, us: Tensor, vs: Tensor) -> Tensor:
    return embs[us] * embs[vs]


if __name__ == "__main__":
    sys.exit(main())
