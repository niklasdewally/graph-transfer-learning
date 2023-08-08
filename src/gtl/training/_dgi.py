import tempfile
from collections.abc import Mapping
from pathlib import Path

import dgl
import torch
import torch.nn as nn
import tqdm
import wandb  # pyre-ignore[21]:
from torch import Tensor
from torch import device as Device

from .. import Graph
from ..models.dgi.dgi import DGI

# code adapted from
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/dgi/train.py


# pyre-ignore[3]
def train(graph: Graph, features: Tensor, device: Device, config: Mapping):
    dgl_graph: dgl.DGLGraph = graph.as_dgl_graph(device)
    features = features.to(device)

    model = DGI(
        dgl_graph,
        features.shape[1],
        config["hidden_layers"],
        config["k"] + 1,
        nn.PReLU(config["hidden_layers"]),
        0.0,
    ).to(device)

    wandb.watch(model)

    # setup temporary directory for saving models for early-stopping
    temporary_directory = tempfile.TemporaryDirectory()
    early_stopping_filepath = Path(temporary_directory.name, "stopping.pt")

    if config["load_weights_from"] is not None:
        model.load_state_dict(torch.load(config["load_weights_from"]), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)

    best = 1e9
    best_epoch = -1

    # shuffle nodes before putting into train and validation sets
    indexes = torch.randperm(dgl_graph.nodes().shape[0])
    validation_size = dgl_graph.nodes().shape[0] // 6
    val_nodes = torch.split(dgl_graph.nodes()[indexes], validation_size)[0]
    train_nodes = torch.unique(torch.cat([val_nodes, dgl_graph.nodes()]))

    for epoch in tqdm(range(config["n_epochs"])):
        log = dict()

        model.train()

        loss = 0.0

        model.train()
        optimizer.zero_grad()

        # the sampler returns a list of blocks and involved nodes
        # each block holds a set of edges from a source to destination
        # each block is a hop in the graph
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(config["k"])
        for blocks in dgl.dataloading.DataLoader(
            dgl_graph,
            train_nodes,
            sampler,
            batch_size=config["batch_size"],
            shuffle=True,
            device=device,
        ):
            batch_loss = model(dgl_graph, features, blocks)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss

        log.update({f"{config['wandb_summary_prefix']}-training-loss": loss})

        # VALIDATION

        model.eval()
        loss = 0.0
        blocks = sampler.sample(dgl_graph, val_nodes)
        loss = model(dgl_graph, features, blocks)

        log.update({f"{config['wandb_summary_prefix']}-validation-loss": loss})

        wandb.log(log)

        # early stopping
        if loss <= best + config["min_delta"]:
            best = loss
            best_epoch = epoch
            # save current weights
            torch.save(model.state_dict(), early_stopping_filepath)

        if epoch - best_epoch > config["patience"]:
            print("Early stopping!")
            model.load_state_dict(torch.load(early_stopping_filepath))

            wandb.summary[
                f"{config['wandb_summary_prefix']}-early-stopping-epoch"
            ] = best_epoch

            break

    if config["save_weights_to"] is not None:
        print(f"Saving model parameters to {config['save_weights_to']}")

        torch.save(model.state_dict(), config["save_weights_to"])

    model.eval()
    model.encoder.eval()

    temporary_directory.cleanup()

    return model.encoder
