"""
Coauthor node classification experiment.


This experiment predicts node labels based on node embeddings learnt from a small-subgraph (~30%).
These node embeddings are trained using a small-subgraph of the original data.

We aim to find out which graph neural network performs the best given limited input data. As shown 
in other experiments, this is not necessarily the same as the best-performing GNN. It seems to be the
case that some GNN architectures converge to a good solution faster than others, despite having a 
lower final performance limit.

In the case of our triangle model, we enforce a predefined structure onto node neighborhoods
(triangular structures). That is, all edges in a neighborhood that are not part of a triangle 
within the neighborhood are removed.

Our model has less expressivity than other models, as it excludes edges. As a consequence, in
experiments with large graphs, we hit a maximum performance lower than other models.

We hope that the structure we empose onto the node neighborhoods helps the neural network learn
useful structural information faster, which will help in scenarios where training data is limited.


The data used is the CoAuthor CS dataset. This dataset represents academic authors, with edges
representing which authors have wrote papers with each-other.

Input features are provided, and are the keywords of each author.

(nd60, 26/07/23)
"""

#########################
#        IMPORTS        #
#########################

import datetime
import pathlib
from collections.abc import MutableMapping

import gtl.features
import gtl.training.egi
import gtl.training.graphsage
import tomllib
import torch
from gtl import Graph
from gtl.coauthor import load_coauthor_npz
from gtl.splits import CoauthorNodeClassificationSplit
from sklearn.linear_model import SGDClassifier

# pyre-ignore [21]:
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################
#        PATHS        #
#######################

SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.parent.resolve()
HYPERPARAMS_DIR: pathlib.Path = SCRIPT_DIR / "coauthor_classification_hyperparams"
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "raw"

##############################
#        WANDB CONFIG        #
##############################

DEFAULT_CONFIG: MutableMapping = {
    "repeats_per_trial": 10,
}

################################
#        DATA / SPLITS         #
################################
cs_graph: Graph
cs_feats: torch.Tensor
cs_labels: torch.Tensor

cs_graph, cs_feats, cs_labels = load_coauthor_npz(DATA_DIR / "coauthor-cs.npz")

cs_feats = cs_feats.to(device)
cs_labels = cs_labels.to(device)

cs_graph.mine_triangles()

cs_split = CoauthorNodeClassificationSplit(cs_graph, cs_labels, device)


def main() -> None:
    current_date_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    group = f"{current_date_time}"

    models = ["graphsage", "triangle", "egi"]

    for model in models:
        model_config: MutableMapping = _load_model_config(model)

        for i in range(DEFAULT_CONFIG["repeats_per_trial"]):
            wandb.init(
                project="Co-Author CS Fewshot Classifier",
                entity="sta-graph-transfer-learning",
                config=DEFAULT_CONFIG,
                save_code=True,
                group=group,
            )
            wandb.config.update(model_config)

            do_run()

            wandb.finish()


def do_run(large_to_large_enabled: bool = True) -> None:
    #################################
    #        SMALL -> LARGE         #
    #################################

    if wandb.config["model"] in ["egi", "triangle"]:
        encoder = gtl.training.train_egi_encoder(
            cs_split.small_g,
            k=wandb.config["k"],
            lr=wandb.config["lr"],
            n_hidden_layers=wandb.config["hidden_layers"],
            sampler_type=wandb.config["model"],
            patience=wandb.config["patience"],
            min_delta=wandb.config["min_delta"],
            n_epochs=wandb.config["epochs"],
            feature_mode="none",
            features=cs_feats[cs_split.small_idxs].to(device),
        )

    elif wandb.config["model"] == "graphsage":
        encoder = gtl.training.graphsage.train_graphsage_encoder(
            cs_split.small_g,
            k=wandb.config["k"],
            lr=wandb.config["lr"],
            n_hidden_layers=wandb.config["hidden_layers"],
            patience=wandb.config["patience"],
            min_delta=wandb.config["min_delta"],
            n_epochs=wandb.config["epochs"],
            feature_mode="none",
            features=cs_feats[cs_split.small_idxs].to(device),
        )
    else:
        raise ValueError(f"Invalid model type {wandb.config['model']}")

    embs: torch.Tensor = encoder(cs_graph.as_dgl_graph(device), cs_feats.to(device))

    train_embs: torch.Tensor = embs[cs_split.train_mask]
    train_labels: torch.Tensor = cs_labels[cs_split.train_mask]

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(train_embs.detach().cpu(), train_labels.detach().cpu())

    val_embs: torch.Tensor = embs[cs_split.val_mask]
    val_labels: torch.Tensor = cs_labels[cs_split.val_mask]

    score = classifier.score(val_embs.detach().cpu(), val_labels.detach().cpu())

    wandb.summary["small-to-large-accuracy"] = score

    if large_to_large_enabled:
        #################################
        #        LARGE -> LARGE         #
        #################################

        if wandb.config["model"] in ["egi", "triangle"]:
            encoder = gtl.training.train_egi_encoder(
                cs_graph,
                k=wandb.config["k"],
                lr=wandb.config["lr"],
                n_hidden_layers=wandb.config["hidden_layers"],
                sampler_type=wandb.config["model"],
                patience=wandb.config["patience"],
                min_delta=wandb.config["min_delta"],
                n_epochs=wandb.config["epochs"],
                feature_mode="none",
                features=cs_feats.to(device),
            )

        elif wandb.config["model"] == "graphsage":
            encoder = gtl.training.graphsage.train_graphsage_encoder(
                cs_graph,
                k=wandb.config["k"],
                lr=wandb.config["lr"],
                n_hidden_layers=wandb.config["hidden_layers"],
                patience=wandb.config["patience"],
                min_delta=wandb.config["min_delta"],
                n_epochs=wandb.config["epochs"],
                feature_mode="none",
                features=cs_feats.to(device),
            )
        else:
            raise ValueError(f"Invalid model type {wandb.config['model']}")

        embs: torch.Tensor = encoder(cs_graph.as_dgl_graph(device), cs_feats.to(device))

        train_embs: torch.Tensor = embs[cs_split.train_mask]
        train_labels: torch.Tensor = cs_labels[cs_split.train_mask]

        classifier = SGDClassifier(max_iter=1000)
        classifier = classifier.fit(
            train_embs.detach().cpu(), train_labels.detach().cpu()
        )

        val_embs: torch.Tensor = embs[cs_split.val_mask]
        val_labels: torch.Tensor = cs_labels[cs_split.val_mask]

        score = classifier.score(val_embs.detach().cpu(), val_labels.detach().cpu())

        wandb.summary["large-to-large-accuracy"] = score


def _load_model_config(model: str) -> MutableMapping:
    with open(HYPERPARAMS_DIR / f"{model}.toml", "rb") as f:
        config = tomllib.load(f)
    return config


if __name__ == "__main__":
    main()
