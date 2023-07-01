import argparse
import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pathlib

from collections.abc import MutableMapping
import gtl.features
import gtl.training
from gtl import Graph
import networkx as nx
import torch
from gtl.cli import add_wandb_options
from dgl.sampling import global_uniform_negative_sampling
#pyre-ignore[21]:
import wandb
import tomllib

from pprint import pprint


# auto-select device to run pytorch models on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
# PATHS TO RESOURCES #
######################

SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.resolve()
HYPERPARAMS_DIR: pathlib.Path = SCRIPT_DIR / "core_periphery_hyperparams"
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "generated" / "core-periphery"

################
# WANDB CONFIG #
################

# default, model independent config
# model specific config is loaded from .toml later
default_config: MutableMapping = {
    "repeats_per_trial": 1,
}

current_date_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
entity = "sta-graph-transfer-learning"
project = "Core-Periphery Link Prediction"
group = f"{current_date_time}"


def main() -> None:
    argument_parser = add_wandb_options(argparse.ArgumentParser(description=__doc__))
    cli_options = argument_parser.parse_args()

    for model in ["triangle", "egi"]:
        # load model specific config from .toml file
        model_config: MutableMapping = _load_model_config(model)

        for source_sizes in [(15, 100), (75, 500), (750, 5000)]:
            for target_sizes in [(15, 100), (75, 500), (750, 5000)]:
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
                    wandb.config.update(model_config)
                    wandb.config["source_core_size"] = source_core
                    wandb.config["source_periphery_size"] = source_periphery
                    wandb.config["source_size"] = source_periphery + source_core

                    wandb.config["target_core_size"] = target_core
                    wandb.config["target_periphery_size"] = target_periphery
                    wandb.config["target_size"] = target_periphery + target_core

                    pprint(vars(wandb.config)["_items"], width=1)
                    run()

                    wandb.finish()


def _load_model_config(model: str) -> MutableMapping:
    with open(HYPERPARAMS_DIR / f"{model}.toml", "rb") as f:
        config = tomllib.load(f)
    return config


def run() -> None:
    ###############
    # LOAD GRAPHS #
    ###############

    src_g_name = (
        f"{wandb.config.source_core_size}-{wandb.config.source_periphery_size}-0.gml"
    )
    src_g: Graph = Graph(nx.read_gml(DATA_DIR / src_g_name))

    target_g_name = (
        f"{wandb.config.target_core_size}-{wandb.config.target_periphery_size}-1.gml"
    )

    target_g: Graph = Graph(nx.read_gml(DATA_DIR / target_g_name))

    wandb.config["target_graph_filename"] = target_g_name
    wandb.config["source_graph_filename"] = src_g_name

    ###################
    # TRAIN ON SOURCE #
    ###################

    model_params = wandb.config

    encoder = gtl.training.train_egi_encoder(
        src_g,
        k=model_params["k"],
        lr=model_params["lr"],
        n_hidden_layers=model_params["hidden_layers"],
        sampler_type=wandb.config["model"],
        patience=model_params["patience"],
        min_delta=model_params["min_delta"],
        n_epochs=model_params["epochs"],
    )

    features: torch.Tensor = gtl.features.degree_bucketing(
        src_g.as_dgl_graph(device), model_params["hidden_layers"]
    )
    features = features.to(device)

    node_embeddings: torch.Tensor = encoder(src_g.as_dgl_graph(device), features)

    # generate negative edges
    negative_us, negative_vs = global_uniform_negative_sampling(
        src_g.as_dgl_graph(device), (src_g.as_dgl_graph(device).num_edges())
    )

    # get and shuffle positive edges
    shuffle_mask = torch.randperm(src_g.as_dgl_graph(device).num_edges())
    us, vs = src_g.as_dgl_graph(device).edges()
    us = us[shuffle_mask]
    vs = vs[shuffle_mask]

    # convert into node embeddings
    us = node_embeddings[us]
    vs = node_embeddings[vs]
    negative_us = node_embeddings[negative_us]
    negative_vs = node_embeddings[negative_vs]

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

    # Train link predictor

    train_edges, val_edges, train_classes, val_classes = train_test_split(edges, values)

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(
        train_edges.detach().cpu(), train_classes.detach().cpu()
    )

    score = classifier.score(val_edges.detach().cpu(), val_classes.detach().cpu())

    wandb.summary["source-accuracy"] = score

    #############################
    # DIRECT TRANSFER TO TARGET #
    #############################

    features = gtl.features.degree_bucketing(target_g.as_dgl_graph(device), model_params["hidden_layers"])
    features = features.to(device)

    embs = encoder(target_g.as_dgl_graph(device), features)

    # generate negative edges
    negative_us, negative_vs = global_uniform_negative_sampling(
        target_g, (target_g.as_dgl_graph(device).num_edges())
    )

    # get and shuffle positive edges
    shuffle_mask = torch.randperm(target_g.as_dgl_graph(device).num_edges())
    us, vs = target_g.as_dgl_graph(device).edges()
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


if __name__ == "__main__":
    main()
