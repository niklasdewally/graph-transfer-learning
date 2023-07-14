import argparse
import datetime
import pathlib
import random
from collections.abc import MutableMapping
from pprint import pprint

import gtl.features
import gtl.training
import gtl.training.graphsage
import tomllib
import torch
from dgl.sampling import global_uniform_negative_sampling
from gtl import Graph
from gtl.cli import add_wandb_options
from gtl.splits import LinkPredictionSplit
from sklearn.linear_model import SGDClassifier

# pyre-ignore[21]:
import wandb

# auto-select device to run pytorch models on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
random.seed(0)

######################
# PATHS TO RESOURCES #
######################

SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.parent.resolve()
HYPERPARAMS_DIR: pathlib.Path = SCRIPT_DIR / "core_periphery_hyperparams"
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "generated" / "core-periphery"

################
# WANDB CONFIG #
################

# default, model independent config
# model specific config is loaded from .toml later
default_config: MutableMapping = {
    "repeats_per_trial": 10,
}

current_date_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
entity = "sta-graph-transfer-learning"
project = "Core-Periphery Link Prediction"
group = f"{current_date_time}"


def main() -> None:
    argument_parser = add_wandb_options(argparse.ArgumentParser(description=__doc__))
    cli_options = argument_parser.parse_args()

    for model in ["graphsage"]:
    #for model in ["triangle", "egi","graphsage"]:
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
    src_g: Graph = Graph.from_gml_file(DATA_DIR / src_g_name)
    src_split: LinkPredictionSplit = LinkPredictionSplit(src_g)

    target_g_name = (
        f"{wandb.config.target_core_size}-{wandb.config.target_periphery_size}-1.gml"
    )

    target_g: Graph = Graph.from_gml_file(DATA_DIR / target_g_name)
    target_split: LinkPredictionSplit = LinkPredictionSplit(target_g)

    wandb.config["target_graph_filename"] = target_g_name
    wandb.config["source_graph_filename"] = src_g_name

    ###################
    # TRAIN ON SOURCE #
    ###################

    model_params = wandb.config
    
    model = wandb.config["model"]

    if model in ["egi","triangle"]:
        encoder = gtl.training.train_egi_encoder(
            src_split.mp_graph,
            k=model_params["k"],
            lr=model_params["lr"],
            n_hidden_layers=model_params["hidden_layers"],
            sampler_type=wandb.config["model"],
            patience=model_params["patience"],
            min_delta=model_params["min_delta"],
            n_epochs=model_params["epochs"],
        )
    elif model == "graphsage":
        encoder = gtl.training.graphsage.train_graphsage_encoder(
            src_split.mp_graph,
            k=model_params["k"],
            lr=model_params["lr"],
            n_hidden_layers=model_params["hidden_layers"],
            patience=model_params["patience"],
            min_delta=model_params["min_delta"],
            n_epochs=model_params["epochs"],
                                                                 )
    else:
        raise ValueError(f"Invalid model type {model}")


    features: torch.Tensor = gtl.features.degree_bucketing(
        src_split.full_training_graph.as_dgl_graph(device),
        model_params["hidden_layers"],
    )
    features = features.to(device)

    node_embeddings: torch.Tensor = encoder(
        src_split.full_training_graph.as_dgl_graph(device), features
    )

    # generate negative edges

    # use val_mp_graph as we want to check that the edge neither exists in the message_passing set and the supervision set
    negative_us, negative_vs = global_uniform_negative_sampling(
        src_split.full_training_graph.as_dgl_graph(device), (len(src_split.train_edges))
    )

    # get and shuffle positive edges
    us: torch.Tensor = torch.tensor(
        [u for u, v in src_split.train_edges], device=device
    )
    vs: torch.Tensor = torch.tensor(
        [v for u, v in src_split.train_edges], device=device
    )

    shuffle_mask = torch.randperm(len(src_split.train_edges))
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

    # Train link predictor

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(edges.detach().cpu(), values.detach().cpu())

    # Validate link predictor
    # Same as above, but using val_mp_graph and val_edges / val_graph

    features: torch.Tensor = gtl.features.degree_bucketing(
        src_split.graph.as_dgl_graph(device), model_params["hidden_layers"]
    )
    features = features.to(device)

    node_embeddings: torch.Tensor = encoder(
        src_split.graph.as_dgl_graph(device), features
    )

    # generate negative edges

    # this time, we must ensure that the edge does not exist in the entire graph
    negative_us, negative_vs = global_uniform_negative_sampling(
        src_split.graph.as_dgl_graph(device), (len(src_split.val_edges))
    )

    # get and shuffle positive edges
    us: torch.Tensor = torch.tensor([u for u, v in src_split.val_edges], device=device)
    vs: torch.Tensor = torch.tensor([v for u, v in src_split.val_edges], device=device)

    # convert into node embeddings
    us = node_embeddings[us]
    vs = node_embeddings[vs]

    shuffle_mask = torch.randperm(len(src_split.val_edges))
    us = us[shuffle_mask]
    vs = vs[shuffle_mask]

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

    score = classifier.score(edges.detach().cpu(), values.detach().cpu())

    wandb.summary["source-accuracy"] = score

    #############################
    # DIRECT TRANSFER TO TARGET #
    #############################

    features = gtl.features.degree_bucketing(
        target_split.full_training_graph.as_dgl_graph(device),
        model_params["hidden_layers"],
    )
    features = features.to(device)

    embs = encoder(target_split.full_training_graph.as_dgl_graph(device), features)

    # Train link predictor

    # generate negative edges
    negative_us, negative_vs = global_uniform_negative_sampling(
        target_split.full_training_graph.as_dgl_graph(device),
        (len(target_split.train_edges)),
    )

    # get and shuffle positive edges
    shuffle_mask = torch.randperm(len(target_split.train_edges))
    us: torch.Tensor = torch.tensor(
        [u for u, v in target_split.train_edges], device=device
    )
    vs: torch.Tensor = torch.tensor(
        [v for u, v in target_split.train_edges], device=device
    )
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

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(edges.detach().cpu(), values.detach().cpu())

    # Validate link predictor
    features: torch.Tensor = gtl.features.degree_bucketing(
        target_split.graph.as_dgl_graph(device), model_params["hidden_layers"]
    )
    features = features.to(device)

    node_embeddings: torch.Tensor = encoder(
        target_split.graph.as_dgl_graph(device), features
    )

    # generate negative edges

    # this time, we must ensure that the edge does not exist in the entire graph
    negative_us, negative_vs = global_uniform_negative_sampling(
        target_split.graph.as_dgl_graph(device), (len(target_split.val_edges))
    )

    # get and shuffle positive edges
    us: torch.Tensor = torch.tensor(
        [u for u, v in target_split.val_edges], device=device
    )
    vs: torch.Tensor = torch.tensor(
        [v for u, v in target_split.val_edges], device=device
    )

    shuffle_mask = torch.randperm(len(target_split.val_edges))
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

    score = classifier.score(edges.detach().cpu(), values.detach().cpu())

    wandb.summary["target-accuracy"] = score


if __name__ == "__main__":
    main()
