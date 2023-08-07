__all__ = ["Model", "train"]


"""
Model training loops.
"""

from collections.abc import Callable, Mapping
from functools import partial

import torch
from dgl import DGLGraph
from torch import Tensor
from torch import device as Device

from .. import Graph
from . import _egi, _graphsage, _dgi

Model = Callable[[DGLGraph, Tensor], Tensor]
TrainFunc = Callable[[Graph, Tensor, Device, Mapping], Model]

_model_functions: dict[str, TrainFunc] = {
    "graphsage": partial(_graphsage.train, aggregator="mean"),
    "graphsage-mean": partial(_graphsage.train, aggregator="mean"),
    "graphsage-pool": partial(_graphsage.train, aggregator="pool"),
    "graphsage-gcn": partial(_graphsage.train, aggregator="gcn"),
    "graphsage-lstm": partial(_graphsage.train, aggregator="lstm"),
    "egi": partial(_egi.train, sampler_type="egi"),
    "triangle": partial(_egi.train, sampler_type="triangle"),
    "dgi": _dgi.train
}


def train(
    model: str, graph: Graph, features: Tensor, config: Mapping, device=None
) -> Model:
    """
    Using `graph` and `features`, train an instance of the given model.

    The model is trained in an unsupervised manner, producing a set of
    node-embeddings to be used as input for downstream models such as
    a classifier.

    Args:

        model: The model to use.

            Implemented models are:
                * graphsage-mean
                * graphsage-pool
                * graphsage-gcn
                * graphsage-lstm
                * egi
                * triangle
                * dgi (Deep Graph Infomax)

        graph: The graph to use for training, in gtl format.

        features: A tensor containing features for each node in `graph`.

        config: A dictionary containing hyperparameters and training settings.
            See below for more details.

        device: The pytorch device to use for training and inference.
            If not specified, a device will automatically be selected.


    Returns: a trained graph encoder.

        This takes in the graph in DGL format, and a feature tensor, and returns
        a tensor of node embeddings.


    Configuration values:

        The configuration dictionary takes the following values:

        Hyperparameters
        ===============
            * hidden_layers (REQUIRED)
            * k (REQUIRED)
            * lr (REQUIRED)
            * batch_size (optional)
            * n_epochs (optional)


        Early Stopping
        ==============
            * patience (optional)
                If patience is not specified, early stopping is disabled.

            * min_delta (optional)

        Transfer learning
        =================
            * load_weights_from: a model file, compatible with torch.load() to
              load intial weights from.

              This must be from the same type and size of model as is specified
              in this function call.


            * save_weights_to: a path to save the trained model to.

        Logging
        ========

        wandb is used for logging.

        * wandb_summary_prefix: a prefix to add to reported losses in wandb
            (optional; defaults to "").
    """

    # TODO (niklasdewally): remove save_weights_to

    func = _model_functions.get(model)
    if func is None:
        raise ValueError(f"No such model: {model}")

    # copy config to get rid of junk values.
    new_config = dict()

    for label in ["hidden_layers", "k", "lr"]:
        if label not in config.keys():
            raise ValueError("f{label} is required but not found")
        new_config[label] = config[label]

    new_config["batch_size"] = config.get("batch_size", 50)
    new_config["n_epochs"] = config.get("n_epochs", 100)

    # disable early stopping by making patience > n_epochs
    new_config["patience"] = config.get("patience", new_config["n_epochs"] + 1)
    new_config["min_delta"] = config.get("min_delta", new_config["n_epochs"])

    new_config["load_weights_from"] = config.get("load_weights_from", None)
    new_config["save_weights_to"] = config.get("save_weights_to", None)
    new_config["wandb_summary_prefix"] = config.get("wandb_summary_prefix", "")

    if new_config["k"] < 1:
        raise ValueError(f"k must be 1 or greater, currently is {new_config['k']}")

    if new_config["hidden_layers"] < 1:
        raise ValueError(
            f"hidden_layers must be 1 or greater, currently is {new_config['hidden_layers']}"
        )

    if new_config["lr"] <= 0:
        raise ValueError(f"lr must be above 0, currently is {new_config['lr']}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return func(graph, features, device, new_config)


# pyre-ignore[5]:
