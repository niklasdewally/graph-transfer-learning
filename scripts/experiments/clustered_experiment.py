import datetime
import itertools
import pathlib
from argparse import ArgumentParser, Namespace
from collections.abc import MutableMapping
from random import shuffle

import gtl.features
import gtl.training
import torch
from dgl.sampling import global_uniform_negative_sampling
from gtl import Graph
from gtl.cli import add_wandb_options
from gtl.splits import LinkPredictionSplit
from sklearn.linear_model import SGDClassifier

# pyre-ignore[21]:
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
# PATHS TO RESOURCES #
######################

PROJECT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "generated" / "clustered"

################
# WANDB CONFIG #
################

# Experimental constants
default_config: MutableMapping = {
    "batch_size": 50,
    "LR": 0.01,
    "hidden-layers": 32,
    "patience": 10,
    "min-delta": 0.01,
    "epochs": 100,
    "k": {"triangle": 4, "egi": 3},
    "n-runs": 1,
}


# Parameters to sweep
GRAPH_TYPES = ["powerlaw"]
MODELS = ["triangle", "egi"]

# (soruce graph size, target graph size)
# for fewshot learning (train on small, test on large)
SIZES = [(100, 1000), (100, 100), (1000, 1000)]


def main(opts: Namespace) -> None:
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

            for i in range(default_config["n-runs"]):
                wandb.init(
                    mode=opts.mode,
                    project="Clustered Transfer",
                    name=f"{model}-{graph_type}-{src_name}-{src_size}-{target_name}-{target_size}-{i}",
                    entity="sta-graph-transfer-learning",
                    group=f"Run {current_date_time}",
                    config=default_config,
                )

                wandb.config.update(
                    {
                        "model": model,
                        "graph-type": graph_type,
                        "src": src_name,
                        "target": target_name,
                        "src-size": src_size,
                        "target-size": target_size,
                    }
                )

                do_run()
                wandb.finish()


def do_run() -> None:
    src_g: Graph = Graph.from_gml_file(
        DATA_DIR
        / f"{wandb.config['graph-type']}-{wandb.config['src']}-{wandb.config['src-size']}-{0}.gml"
    )
    src_split = LinkPredictionSplit(src_g)

    target_g: Graph = Graph.from_gml_file(
        DATA_DIR
        / f"{wandb.config['graph-type']}-{wandb.config['target']}-{wandb.config['target-size']}-{0}.gml"
    )
    target_split = LinkPredictionSplit(target_g)

    encoder = gtl.training.train_egi_encoder(
        graph=src_split.mp_graph,
        k=wandb.config["k"][wandb.config["model"]],
        lr=wandb.config["LR"],
        n_hidden_layers=wandb.config["hidden-layers"],
        sampler_type=wandb.config["model"],
        save_weights_to="pretrain.pt",
        patience=wandb.config["patience"],
        min_delta=wandb.config["min-delta"],
        n_epochs=wandb.config["epochs"],
    )

    # Generate negative and positive edge samples for training.
    # #########################################################

    features: torch.Tensor = gtl.features.degree_bucketing(
        src_split.full_training_graph.as_dgl_graph(device),
        wandb.config["hidden-layers"],
    ).to(device)

    embs = encoder(src_split.full_training_graph.as_dgl_graph(device), features)

    negative_us, negative_vs = global_uniform_negative_sampling(
        src_split.full_training_graph.as_dgl_graph(device), (len(src_split.train_edges))
    )

    shuffle_mask = torch.randperm(len(src_split.train_edges))

    us: torch.Tensor = torch.tensor(
        [u for u, v in src_split.train_edges], device=device
    )[shuffle_mask]
    vs: torch.Tensor = torch.tensor(
        [v for u, v in src_split.train_edges], device=device
    )[shuffle_mask]

    # convert into node embeddings
    us = embs[us]
    vs = embs[vs]

    negative_us = embs[negative_us]
    negative_vs = embs[negative_vs]

    # convert into edge embeddings
    positive_edges = us * vs
    negative_edges = negative_us * negative_vs

    # Create classes for classification
    positive_values = torch.ones(positive_edges.shape[0])
    negative_values = torch.zeros(negative_edges.shape[0])

    # create shuffled edge and class list
    train_edges = torch.cat((positive_edges, negative_edges), 0)
    train_classes = torch.cat((positive_values, negative_values), 0)

    shuffle_mask = torch.randperm(train_edges.shape[0])
    train_edges = train_edges[shuffle_mask]
    train_classes = train_classes[shuffle_mask]

    # Train link predictor!!
    #########################

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(
        train_edges.detach().cpu(), train_classes.detach().cpu()
    )

    # Generate negative and positive edge samples for validation.
    # ###########################################################

    features: torch.Tensor = gtl.features.degree_bucketing(
        src_split.graph.as_dgl_graph(device), wandb.config["hidden-layers"]
    ).to(device)

    embs = encoder(src_split.graph.as_dgl_graph(device), features)

    negative_us, negative_vs = global_uniform_negative_sampling(
        src_split.graph.as_dgl_graph(device), (len(src_split.val_edges))
    )

    shuffle_mask = torch.randperm(len(src_split.val_edges))

    us: torch.Tensor = torch.tensor([u for u, v in src_split.val_edges], device=device)[
        shuffle_mask
    ]
    vs: torch.Tensor = torch.tensor([v for u, v in src_split.val_edges], device=device)[
        shuffle_mask
    ]

    # convert into node embeddings
    us = embs[us]
    vs = embs[vs]

    negative_us = embs[negative_us]
    negative_vs = embs[negative_vs]

    # convert into edge embeddings
    positive_edges = us * vs
    negative_edges = negative_us * negative_vs

    # Create classes for classification
    positive_values = torch.ones(positive_edges.shape[0])
    negative_values = torch.zeros(negative_edges.shape[0])

    # create shuffled edge and class list
    val_edges = torch.cat((positive_edges, negative_edges), 0)
    val_classes = torch.cat((positive_values, negative_values), 0)

    shuffle_mask = torch.randperm(val_edges.shape[0])
    val_edges = val_edges[shuffle_mask]
    val_classes = val_classes[shuffle_mask]

    # Validate
    ###########

    score = classifier.score(val_edges.detach().cpu(), val_classes.detach().cpu())

    wandb.summary["source-accuracy"] = score

    #################################
    # Direct transfer of embeddings #
    #################################

    # Generate negative and positive edge samples for training.
    # #########################################################

    features: torch.Tensor = gtl.features.degree_bucketing(
        target_split.full_training_graph.as_dgl_graph(device),
        wandb.config["hidden-layers"],
    ).to(device)

    embs = encoder(target_split.full_training_graph.as_dgl_graph(device), features)

    negative_us, negative_vs = global_uniform_negative_sampling(
        target_split.full_training_graph.as_dgl_graph(device), (len(target_split.train_edges))
    )

    shuffle_mask = torch.randperm(len(target_split.train_edges))

    us: torch.Tensor = torch.tensor(
        [u for u, v in target_split.train_edges], device=device
    )[shuffle_mask]
    vs: torch.Tensor = torch.tensor(
        [v for u, v in target_split.train_edges], device=device
    )[shuffle_mask]

    # convert into node embeddings
    us = embs[us]
    vs = embs[vs]

    negative_us = embs[negative_us]
    negative_vs = embs[negative_vs]

    # convert into edge embeddings
    positive_edges = us * vs
    negative_edges = negative_us * negative_vs

    # Create classes for classification
    positive_values = torch.ones(positive_edges.shape[0])
    negative_values = torch.zeros(negative_edges.shape[0])

    # create shuffled edge and class list
    train_edges = torch.cat((positive_edges, negative_edges), 0)
    train_classes = torch.cat((positive_values, negative_values), 0)

    shuffle_mask = torch.randperm(train_edges.shape[0])
    train_edges = train_edges[shuffle_mask]
    train_classes = train_classes[shuffle_mask]

    # Train link predictor.
    #########################

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(
        train_edges.detach().cpu(), train_classes.detach().cpu()
    )

    # Generate negative and positive edge samples for validation.
    # ###########################################################

    features: torch.Tensor = gtl.features.degree_bucketing(
        target_split.graph.as_dgl_graph(device), wandb.config["hidden-layers"]
    ).to(device)

    embs = encoder(target_split.graph.as_dgl_graph(device), features)

    negative_us, negative_vs = global_uniform_negative_sampling(
        target_split.graph.as_dgl_graph(device), (len(target_split.val_edges))
    )

    shuffle_mask = torch.randperm(len(target_split.val_edges))

    us: torch.Tensor = torch.tensor(
        [u for u, v in target_split.val_edges], device=device
    )[shuffle_mask]
    vs: torch.Tensor = torch.tensor(
        [v for u, v in target_split.val_edges], device=device
    )[shuffle_mask]

    # convert into node embeddings
    us = embs[us]
    vs = embs[vs]

    negative_us = embs[negative_us]
    negative_vs = embs[negative_vs]

    # convert into edge embeddings
    positive_edges = us * vs
    negative_edges = negative_us * negative_vs

    # Create classes for classification
    positive_values = torch.ones(positive_edges.shape[0])
    negative_values = torch.zeros(negative_edges.shape[0])

    # create shuffled edge and class list
    val_edges = torch.cat((positive_edges, negative_edges), 0)
    val_classes = torch.cat((positive_values, negative_values), 0)

    shuffle_mask = torch.randperm(val_edges.shape[0])
    val_edges = val_edges[shuffle_mask]
    val_classes = val_classes[shuffle_mask]

    # Validate
    ###########

    score = classifier.score(val_edges.detach().cpu(), val_classes.detach().cpu())

    wandb.summary["target-accuracy"] = score


if __name__ == "__main__":
    parser: ArgumentParser = add_wandb_options(ArgumentParser())
    opts: Namespace = parser.parse_args()
    if opts.mode is None:
        opts.mode = "online"
    main(opts)
