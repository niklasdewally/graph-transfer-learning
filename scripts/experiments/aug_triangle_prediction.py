import json
import pathlib
import sys
from collections.abc import MutableMapping
from random import sample
from typing import Any

import datetime
import gtl.features
import gtl.training.egi
import gtl.training.graphsage
import numpy as np
import torch

# pyre-ignore [21]:
import wandb
from gtl import Graph
from numpy.typing import NDArray
from sklearn.linear_model import SGDClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################
#        PATHS        #
#######################

SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.parent.resolve()
HYPERPARAMS_DIR: pathlib.Path = SCRIPT_DIR / "aug_triangle_detection_hyperparams"
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "2023-08-clustered"

##############################
#        WANDB CONFIG        #
##############################

default_config: MutableMapping = {
    "sizes": [250, 1000, 10000, 100000],
    "models": ["graphsage", "egi", "triangle"],
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
                    wandb.init(
                        project="August 2023 01 - Triangle Detection",
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
    # load data

    _a, _b = _load_graphs(wandb.config["source_size"])
    source_graph = _a[0]
    val_graph = _a[1]
    source_neg_triangles = _b[0]
    val_neg_triangles = _b[1]
    del _a, _b

    target_graphs, target_neg_triangles = _load_graphs(wandb.config["target_size"])
    target_graphs = target_graphs[2:]
    target_neg_triangles = target_neg_triangles[2:]

    # Train encoder on a single source graph
    print(wandb.config["source_size"])
    encoder = gtl.training.train(graph=source_graph, **wandb.config)

    # Train triangle prediction

    features = gtl.features.degree_bucketing(
        source_graph.as_dgl_graph(device), wandb.config["hidden_layers"]
    ).to(device)

    embs = encoder(source_graph.as_dgl_graph(device), features).detach().cpu().numpy()

    neg_triangles = np.array(source_neg_triangles)
    all_pos_triangles = source_graph.get_triangles_list()
    pos_triangles = np.array(
        sample(
            all_pos_triangles,
            min(len(all_pos_triangles), neg_triangles.shape[0]),
        )
    )

    triangles = np.concatenate((pos_triangles, neg_triangles))
    triangles = _embed_triangles(triangles, embs)

    classes = np.concatenate(
        (np.ones(pos_triangles.shape[0]), np.zeros(neg_triangles.shape[0]))
    )

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(triangles, classes)

    if eval_mode == "test":
        # Test triangle prediction on all target graphs, and keep the average
        wandb.define_metric("acc", summary="mean")

        for i in range(len(target_graphs)):
            target_graph = target_graphs[i]

            features = gtl.features.degree_bucketing(
                target_graph.as_dgl_graph(device), wandb.config["hidden_layers"]
            ).to(device)

            embs = (
                encoder(target_graph.as_dgl_graph(device), features)
                .detach()
                .cpu()
                .numpy()
            )

            neg_triangles = np.array(target_neg_triangles[i])

            all_pos_triangles = target_graph.get_triangles_list()

            pos_triangles = np.array(
                sample(
                    all_pos_triangles,
                    min(len(all_pos_triangles), neg_triangles.shape[0]),
                )
            )

            triangles = np.concatenate((pos_triangles, neg_triangles))
            triangles = _embed_triangles(triangles, embs)
            classes = np.concatenate(
                (np.ones(pos_triangles.shape[0]), np.zeros(neg_triangles.shape[0]))
            )

            wandb.summary["acc"] = classifier.score(triangles, classes)

    elif eval_mode == "validate":
        features = gtl.features.degree_bucketing(
            val_graph.as_dgl_graph(device), wandb.config["hidden_layers"]
        ).to(device)

        embs = encoder(val_graph.as_dgl_graph(device), features).detach().cpu().numpy()

        neg_triangles = np.array(val_neg_triangles)

        all_pos_triangles = val_graph.get_triangles_list()

        pos_triangles = np.array(
            sample(
                all_pos_triangles,
                min(len(all_pos_triangles), neg_triangles.shape[0]),
            )
        )

        triangles = np.concatenate((pos_triangles, neg_triangles))
        triangles = _embed_triangles(triangles, embs)
        classes = np.concatenate(
            (np.ones(pos_triangles.shape[0]), np.zeros(neg_triangles.shape[0]))
        )

        wandb.summary["acc"] = classifier.score(triangles, classes)


def _load_graphs(size: int) -> tuple[list[Graph], list[list[int]]]:
    graphs: list[Graph] = []
    negative_triangles: list[list[int]] = []

    def filename(i: int) -> str:
        return f"poisson-{size}-3-{i}.gml"

    def negative_triangles_filename(i: int) -> str:
        return f"poisson-{size}-3-{i}-negative-triangles.json"

    for i in range(100):
        graph = Graph.from_gml_file(DATA_DIR / filename(i))
        graphs.append(graph)

        with open(DATA_DIR / negative_triangles_filename(i)) as f:
            negative_triangle = json.load(f)
            negative_triangles.append(negative_triangle)

    return graphs, negative_triangles


# pyre-ignore[2]
# pyre-ignore[3]
def _embed_triangles(triangles: NDArray[Any], embs: NDArray[Any]) -> NDArray[Any]:
    us = triangles[:, 0]
    vs = triangles[:, 1]
    ws = triangles[:, 2]
    return embs[us] * embs[vs] * embs[ws]


if __name__ == "__main__":
    sys.exit(main())
