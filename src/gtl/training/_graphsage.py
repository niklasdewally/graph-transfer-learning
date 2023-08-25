# pyre-ignore-all-errors
__all__ = ["train"]


# pyre-ignore[21]
import tempfile
from collections.abc import Mapping
from pathlib import Path

import dgl
import torch
import wandb
from dgl.dataloading import DataLoader
from torch import Tensor
from torch import device as Device
from tqdm import tqdm

from .. import Graph
from ..models import graphsage as graphsage


def train(
    graph: Graph,
    features: Tensor,
    device: Device,
    config: Mapping,
    aggregator: str,
):
    dgl_graph: dgl.DGLGraph = graph.as_dgl_graph(device)
    dgl_graph.ndata["feat"] = features.to(device)

    # setup temporary directory for saving models for early-stopping
    temporary_directory = tempfile.TemporaryDirectory()
    early_stopping_filepath = Path(temporary_directory.name, "stopping.pt")

    sampler: dgl.dataloading.Sampler = dgl.dataloading.NeighborSampler(
        [10 for i in range(config["k"])], prefetch_node_feats=["feat"]
    )

    in_feats = features.shape[1]

    model = graphsage.SAGEUnsupervised(
        in_feats,
        config["hidden_layers"],
        n_conv_layers=config["k"] + 1,
        aggregator=aggregator,
    ).to(device)

    wandb.watch(model)

    if config["load_weights_from"] is not None:
        model.load_state_dict(torch.load(config["load_weights_from"]), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)

    # test train split
    val_idxs = torch.randperm(dgl_graph.nodes().shape[0]).to(device)
    validation_size = dgl_graph.nodes().shape[0] // 6
    val_nodes = torch.split(dgl_graph.nodes()[val_idxs], validation_size)[0]
    train_nodes = torch.unique(torch.cat([val_nodes, dgl_graph.nodes()]))

    loss_function = graphsage.SAGEUnsupervisedLoss(graph)

    train_dataloader = DataLoader(
        dgl_graph,
        train_nodes,
        sampler,
        batch_size=config["batch_size"],
        shuffle=True,
        device=device,
    )

    val_dataloader = DataLoader(
        dgl_graph,
        val_nodes,
        sampler,
        batch_size=config["batch_size"],
        shuffle=True,
        device=device,
    )

    # convert entire graph to k-deep message flow graph for creating embeddings.
    (_, _, full_g_blocks) = dgl.dataloading.MultiLayerFullNeighborSampler(
        config["k"]
    ).sample(dgl_graph, dgl_graph.nodes())

    # some summary statistics
    best = 1e9
    best_epoch = -1

    print("Training!")
    for epoch in tqdm(range(config["n_epochs"])):
        epoch_log = dict()

        loss = 0.0

        model.train()
        optimizer.zero_grad()

        # TRAIN
        for i, (batch_inp, batch_out, blocks) in enumerate(train_dataloader):
            # calculate gradients, etc for batch blocks only
            model(blocks, blocks[0].srcdata["feat"])

            # we need to have embeddings for the entire graph for the loss function
            # however, we only back-propagate based on those in our batch
            model.eval()
            embs = model(full_g_blocks, features)
            batch_loss = loss_function(batch_out, embs)
            batch_loss.backward()

            optimizer.step()
            loss += batch_loss.detach().item()

        epoch_log.update(
            {f"{config['wandb_summary_prefix']}-training-loss": loss / (i + 1)}
        )

        # VALIDATE
        model.eval()
        loss = 0.0

        with torch.no_grad():
            for i, (batch_inp, batch_out, blocks) in enumerate(val_dataloader):
                embs = model(full_g_blocks, features)
                batch_loss = loss_function(batch_out, embs)
                loss += batch_loss

            epoch_log.update(
                {f"{config['wandb_summary_prefix']}-validation-loss": loss / (i + 1)}
            )
            wandb.log(epoch_log)

            # early stopping
            if loss <= best - config["min_delta"]:
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

    # save parameters for later fine-tuning if a save path is given
    if config["save_weights_to"] is not None:
        print(f"Saving model parameters to {config['save_weights_to']}")

        torch.save(model.state_dict(), config["save_weights_to"])

    model.eval()

    # Cleanup
    temporary_directory.cleanup()

    def encoder(g, features):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(config["k"])
        (inp, out, blocks) = sampler.sample(g, g.nodes())

        return model(blocks, features)

    return encoder
