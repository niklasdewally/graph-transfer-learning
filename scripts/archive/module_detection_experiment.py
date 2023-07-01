import argparse
import datetime
import itertools
import pathlib
import warnings

import dgl
import gtl.features
import gtl.training
import networkx as nx
import torch
import torch.nn as nn
import wandb
from gtl.cli import add_wandb_options
from numpy.typing import NDArray
from sklearn.linear_model import SGDClassifier

warnings.filterwarnings("ignore")

# Setup directories
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.resolve()
DATA_DIR = PROJECT_DIR / "data" / "generated"

# constant settings for experiments
CONFIG = {
    # optimal values of k per model
    "k": {"triangle": 4, "egi": 2},
    "batch_size": 50,
    "LR": 0.01,
    "hidden_layers": 32,
    "patience": 10,
    "min_delta": 0.01,
    "epochs": 100,
    "n_runs": 10,
    "models": ["triangle", "egi"],
    "graph_types": ["modular_ba_poisson", "core_periphery"],
    "sizes": {"core_periphery": [1200], "modular_ba_poisson": [50, 100]},
}


# auto-select device to run pytorch models on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    argument_parser = add_wandb_options(argparse.ArgumentParser(description=__doc__))
    cli_options = argument_parser.parse_args()

    current_date_time = datetime.datetime.now().strftime("%Y%m%dT%H%M")

    trials = list(itertools.product(CONFIG["graph_types"], CONFIG["models"]))

    for graph_type, model in trials:
        for size in CONFIG["sizes"][graph_type]:
            for i in range(CONFIG["n_runs"]):
                with wandb.init(
                    project="Module Detection Experiment",
                    name=f"{graph_type}-{size}-{model}-{i}",
                    entity="sta-graph-transfer-learning",
                    group=f"{current_date_time}",
                    config={
                        "global_config": CONFIG,
                        "model": model,
                        "graph_type": graph_type,
                        "size": size,
                    },
                    mode=cli_options.mode,
                ) as _:
                    do_single_run(model, graph_type, size)


def do_single_run(model: str, graph_type: str, size: int):
    g: nx.Graph = load_data(graph_type, size)
    g: dgl.DGLGraph = dgl.from_networkx(g, node_attrs=["origin"]).to(device)

    encoder: nn.Module = gtl.training.train_egi_encoder(
        g,
        k=CONFIG["k"][model],
        lr=CONFIG["LR"],
        n_hidden_layers=CONFIG["hidden_layers"],
        sampler=model,
        patience=CONFIG["patience"],
        min_delta=CONFIG["min_delta"],
        n_epochs=CONFIG["epochs"],
    )

    features: torch.Tensor = gtl.features.degree_bucketing(g, CONFIG["hidden_layers"])
    features = features.to(device)

    node_embeddings: torch.Tensor = encoder(g, features)
    node_embeddings: NDArray = node_embeddings.detach().cpu().numpy()

    classes: NDArray = g.ndata["origin"].detach().cpu().numpy()

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(node_embeddings, classes)

    # Test!!
    test_g: nx.Graph = load_data(graph_type, size, test=True)
    test_g: dgl.DGLGraph = dgl.from_networkx(test_g, node_attrs=["origin"]).to(device)
    test_features: torch.Tensor = gtl.features.degree_bucketing(
        test_g, CONFIG["hidden_layers"]
    ).to(device)

    test_node_embeddings: torch.Tensor = encoder(test_g, test_features).to(device)
    test_node_embeddings: NDArray = test_node_embeddings.detach().cpu().numpy()

    test_classes: NDArray = test_g.ndata["origin"].detach().cpu().numpy()

    score = classifier.score(test_node_embeddings, test_classes)

    wandb.summary["accuracy"] = score


def load_data(graph_type: str, size: int, test=False):
    if test:
        n = 2
    else:
        n = 3

    match (graph_type, size):
        case (
            "modular_ba_poisson",
            size,
        ):
            path = DATA_DIR / "modular" / f"ba-{size}-poisson-{size}-modular-{n}.gml"

        case ("core_periphery", _):
            path = DATA_DIR / "core_periphery" / f"core-periphery-200-1000-{n}.gml"

        case _:
            raise NotImplementedError()

    return nx.read_gml(path)


if __name__ == "__main__":
    main()
