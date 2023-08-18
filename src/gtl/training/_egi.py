__all__ = ["train"]

import tempfile
from collections.abc import Mapping
from pathlib import Path
from warnings import warn

import dgl
import torch
import torch.nn as nn

# pyre-ignore[21]:
import wandb
from dgl.dataloading import DataLoader
from torch import Tensor
from tqdm import tqdm

from .. import Graph
from ..models import EGI
from ..samplers import KHopTriangleSampler


def train(
    graph: Graph,
    features: Tensor,
    device: torch.device,
    config: Mapping,
    sampler_type: str,
) -> "Model":
    dgl_graph: dgl.DGLGraph = graph.as_dgl_graph(device)

    # setup temporary directory for saving models for early-stopping
    temporary_directory = tempfile.TemporaryDirectory()
    early_stopping_filepath = Path(temporary_directory.name, "stopping.pt")

    if sampler_type == "egi":
        sampler: dgl.dataloading.Sampler = dgl.dataloading.NeighborSampler(
            [10 for i in range(config["k"])]
        )

    elif sampler_type == "triangle":
        if not graph.has_mined_triangles():
            warn("Input graph contains no mined triangles - mining now")
            graph.mine_triangles()
        triangles = graph.get_triangles_dictionary()
        sampler: dgl.dataloading.Sampler = KHopTriangleSampler(
            dgl_graph, [10 for i in range(config["k"])], triangles
        )

    # pyre-ignore[16]:
    features = features.to(device)
    dgl_graph = dgl_graph.to(device)

    in_feats = features.shape[1]

    # in the original code, they set number of layers to equal k +1
    n_layers = config["k"] + 1

    model = EGI(
        in_feats,
        config["hidden_layers"],
        n_layers,
        nn.PReLU(config["hidden_layers"]),
    )

    model = model.to(device)

    wandb.watch(model)

    if config["load_weights_from"] is not None:
        model.load_state_dict(torch.load(config["load_weights_from"]), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)

    # some summary statistics
    best = 1e9
    best_epoch = -1

    # shuffle nodes before putting into train and validation sets
    indexes = torch.randperm(dgl_graph.nodes().shape[0])
    validation_size = dgl_graph.nodes().shape[0] // 6
    val_nodes = torch.split(dgl_graph.nodes()[indexes], validation_size)[0]
    train_nodes = torch.unique(torch.cat([val_nodes, dgl_graph.nodes()]))

    training_dataloader = DataLoader(
        dgl_graph,
        train_nodes,
        sampler,
        batch_size=config["batch_size"],
        shuffle=True,
        device=device,
    )

    for epoch in tqdm(range(config["n_epochs"])):
        log = dict()

        model.train()

        loss = 0.0

        model.train()
        optimizer.zero_grad()

        # the sampler returns a list of blocks and involved nodes
        # each block holds a set of edges from a source to destination
        # each block is a hop in the graph
        for blocks in training_dataloader:
            batch_loss = model(dgl_graph, features, blocks)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.detach()

        log.update({f"{config['wandb_summary_prefix']}-training-loss": loss})

        del batch_loss, loss, blocks

        # VALIDATION

        model.eval()
        with torch.no_grad():
            loss = 0.0
            blocks = sampler.sample(dgl_graph, val_nodes)
            loss = model(dgl_graph, features, blocks)

            log.update(
                {f"{config['wandb_summary_prefix']}-validation-loss": loss.detach()}
            )

            wandb.log(log)

            # early stopping
            if loss <= best - config["min_delta"]:
                best = loss
                best_epoch = epoch
                # save current weights
                torch.save(model.state_dict(), early_stopping_filepath)

            del loss, blocks

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
