import argparse
import pathlib
from collections.abc import MutableMapping
from datetime import datetime

import dgl
import gtl
import gtl.features
import gtl.training
import tomllib
import torch

# pyre-ignore[21]
import wandb
from gtl.cli import add_wandb_options
from IPython import embed
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# auto-select device to run pytorch models on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
# PATHS TO RESOURCES #
######################

SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.parent.resolve()
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "generated" / "core-periphery"
HYPERPARAMS_DIR: pathlib.Path = SCRIPT_DIR / "triangle_detection_hyperparams"

################
# WANDB CONFIG #
################

# default, model independent config
# model specific config is loaded from .toml later
default_config: MutableMapping = {
    "repeats_per_trial": 1,
    "models": ["triangle"],
    "source_sizes": [(15, 100), (75, 500), (750, 5000)],
    "target_sizes": [(15, 100), (75, 500), (750, 5000)],
}

current_date_time: str = datetime.now().strftime("%Y-%m-%d %H:%M")
entity = "sta-graph-transfer-learning"
project = "Triangle Prediction Task"
group = f"{current_date_time}"


def main() -> None:
    argument_parser = add_wandb_options(argparse.ArgumentParser(description=__doc__))
    cli_options = argument_parser.parse_args()

    for model in default_config["models"]:
        model_params: MutableMapping = _load_model_config(model)

        for source_sizes in default_config["source_sizes"]:
            for target_sizes in default_config["target_sizes"]:
                source_core, source_periphery = source_sizes
                target_core, target_periphery = target_sizes

                for i in range(default_config["repeats_per_trial"]):
                    wandb.init(
                        project=project,
                        entity=entity,
                        config=default_config,
                        save_code=True,
                        group=group,
                        mode=cli_options.mode,
                    )

                    # add model config and graph details to wandb
                    wandb.config.update(model_params)
                    wandb.config["source_core_size"] = source_core
                    wandb.config["source_periphery_size"] = source_periphery
                    wandb.config["source_size"] = source_periphery + source_core

                    wandb.config["target_core_size"] = target_core
                    wandb.config["target_periphery_size"] = target_periphery
                    wandb.config["target_size"] = target_periphery + target_core

                    do_run()
                    wandb.finish()


def do_run() -> None:
    src_g_name = (
        f"{wandb.config.source_core_size}-{wandb.config.source_periphery_size}-0.gml"
    )
    target_g_name = (
        f"{wandb.config.target_core_size}-{wandb.config.target_periphery_size}-1.gml"
    )

    src_graph: gtl.Graph = gtl.Graph.from_gml_file(DATA_DIR / src_g_name)
    target_graph: gtl.Graph = gtl.Graph.from_gml_file(DATA_DIR / target_g_name)

    src_graph_dgl: dgl.DGLGraph = src_graph.as_dgl_graph(device)
    target_graph_dgl: dgl.DGLGraph = target_graph.as_dgl_graph(device)

    wandb.config["target_graph_filename"] = target_g_name
    wandb.config["source_graph_filename"] = src_g_name

    encoder = gtl.training.train_egi_encoder(
        graph=src_graph,
        k=wandb.config["k"],
        lr=wandb.config["lr"],
        n_hidden_layers=wandb.config["hidden_layers"],
        sampler_type=wandb.config["model"],
        patience=wandb.config["patience"],
        min_delta=wandb.config["min_delta"],
        n_epochs=wandb.config["epochs"],
    )

    src_features: torch.Tensor = gtl.features.degree_bucketing(
        src_graph_dgl, wandb.config["hidden_layers"]
    )
    src_features = src_features.to(device)

    embs: torch.Tensor = encoder(src_graph_dgl, src_features)

    # [[1,2,3],[4,5,6],[7,8,9]]
    pos_triangles = torch.tensor(list(src_graph.get_triangles_list()), device=device)
    neg_triangles = torch.tensor(
        src_graph.sample_negative_triangles(pos_triangles.shape[0]), device=device
    )

    # make same size
    pos_triangles = pos_triangles[:neg_triangles.shape[0]]

    # [ [1,2,3]
    #   [4,5,6]    ===> [1,2,3,4,5,6,7,8,9] ===> embs[1,2,3...9] ==> [10,11,12,13,14,15,16,17,18]
    #   [7,8,9] ]
    #
    #              ===> [[10,11,12],  ==> [132,...,...]
    #                    [13,14,15],
    #                    [16,17,18]]

    pos_triangle_embeddings: torch.Tensor = torch.prod(
        torch.unflatten(embs[torch.flatten(pos_triangles)], 0, (-1, 3)), 1
    )
    neg_triangle_embeddings: torch.Tensor = torch.prod(
        torch.unflatten(embs[torch.flatten(neg_triangles)], 0, (-1, 3)), 1
    )

    pos_labels: torch.Tensor = torch.ones(pos_triangle_embeddings.shape[0])
    neg_labels: torch.Tensor = torch.zeros(neg_triangle_embeddings.shape[0])

    triangle_embs = torch.cat((pos_triangle_embeddings, neg_triangle_embeddings), 0)
    values = torch.cat((pos_labels, neg_labels), 0)

    shuffle_mask = torch.randperm(triangle_embs.shape[0])
    triangle_embs = triangle_embs[shuffle_mask]
    values = values[shuffle_mask]

    train_triangles, val_triangles, train_classes, val_classes = train_test_split(triangle_embs, values)

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(
        train_triangles.detach().cpu(), train_classes.detach().cpu()
    )

    score = classifier.score(val_triangles.detach().cpu(), val_classes.detach().cpu())
    wandb.summary["source-accuracy"] = score


    ## TRANSFER
    target_features: torch.Tensor = gtl.features.degree_bucketing(
        target_graph_dgl, wandb.config["hidden_layers"]
    )
    target_features = target_features.to(device)
    embs: torch.Tensor = encoder(target_graph_dgl, target_features)

    pos_triangles = torch.tensor(list(target_graph.get_triangles_list()), device=device)
    neg_triangles = torch.tensor(
        target_graph.sample_negative_triangles(pos_triangles.shape[0]), device=device
    )

    # make same size
    pos_triangles = pos_triangles[:neg_triangles.shape[0]]
    pos_triangle_embeddings: torch.Tensor = torch.prod(
        torch.unflatten(embs[torch.flatten(pos_triangles)], 0, (-1, 3)), 1
    )
    neg_triangle_embeddings: torch.Tensor = torch.prod(
        torch.unflatten(embs[torch.flatten(neg_triangles)], 0, (-1, 3)), 1
    )

    pos_labels: torch.Tensor = torch.ones(pos_triangle_embeddings.shape[0])
    neg_labels: torch.Tensor = torch.zeros(neg_triangle_embeddings.shape[0])

    triangle_embs = torch.cat((pos_triangle_embeddings, neg_triangle_embeddings), 0)
    values = torch.cat((pos_labels, neg_labels), 0)

    shuffle_mask = torch.randperm(triangle_embs.shape[0])
    triangle_embs = triangle_embs[shuffle_mask]
    values = values[shuffle_mask]

    train_triangles, val_triangles, train_classes, val_classes = train_test_split(triangle_embs, values)

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(
        train_triangles.detach().cpu(), train_classes.detach().cpu()
    )

    score = classifier.score(val_triangles.detach().cpu(), val_classes.detach().cpu())
    wandb.summary["target-accuracy"] = score


def _load_model_config(model: str) -> MutableMapping:
    with open(HYPERPARAMS_DIR / f"{model}.toml", "rb") as f:
        config = tomllib.load(f)
    return config


if __name__ == "__main__":
    main()
