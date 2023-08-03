# FIXME (niklasdewally): strict typing
# pyre-unsafe

# pyre-ignore[21]
import wandb
import tempfile
from pathlib import Path
from typing import Callable

from dgl.dataloading import DataLoader
import dgl
import torch
from tqdm import tqdm

from .. import Graph
from .. import typing as gtl_typing
from ..features import degree_bucketing
from ..models import graphsage as graphsage

__all__ = ["train_graphsage_encoder"]

from IPython import embed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_graphsage_encoder(
    graph: Graph,
    k: int = 2,
    lr: float = 0.01,
    n_hidden_layers: int = 32,
    n_epochs: int = 100,
    feature_mode: str = "degree_bucketing",
    features=None,
    pre_train: gtl_typing.PathLike = "",
    save_weights_to: gtl_typing.PathLike = "",
    batch_size: int = 50,
    patience: int = 10,
    min_delta: float = 0.01,
    weight_decay: float = 0.0,
    wandb_summary_prefix: str = "",
    **kwargs,  # pyre-ignore[2]
):
    if k < 1:
        raise ValueError("k must be 1 or greater.")

    if lr <= 0:
        raise ValueError("Learning rate must be above 0.")

    dgl_graph: dgl.DGLGraph = graph.as_dgl_graph(device)

    # setup temporary directory for saving models for early-stopping
    temporary_directory = tempfile.TemporaryDirectory()
    early_stopping_filepath = Path(temporary_directory.name, "stopping.pt")

    # generate features
    match feature_mode:
        case "degree_bucketing":
            features = degree_bucketing(dgl_graph, n_hidden_layers)
        case "none":
            features = features
        case e:
            raise ValueError(f"{e} is not a valid feature generation mode")

    dgl_graph.ndata["feat"] = features

    sampler: dgl.dataloading.Sampler = dgl.dataloading.NeighborSampler(
        [10 for i in range(k)], prefetch_node_feats=["feat"]
    )

    in_feats = features.shape[1]

    model = graphsage.SAGEUnsupervised(
        in_feats, n_hidden_layers, n_conv_layers=k + 1
    ).to(device)

    wandb.watch(model)

    # do transfer learning if we have pretrained weights
    if pre_train != "":
        model.load_state_dict(torch.load(pre_train), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # test train split
    indexes = torch.randperm(dgl_graph.nodes().shape[0]).to(device)
    validation_size = dgl_graph.nodes().shape[0] // 6
    val_nodes = torch.split(dgl_graph.nodes()[indexes], validation_size)[0]
    train_nodes = torch.unique(torch.cat([val_nodes, dgl_graph.nodes()]))

    loss_function = graphsage.SAGEUnsupervisedLoss(
        Graph.from_dgl_graph(dgl_graph), dgl_graph.nodes().detach().cpu().tolist()
    )
    train_dataloader = DataLoader(
        dgl_graph,
        train_nodes,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        device=device,
    )
    val_dataloader = DataLoader(
        dgl_graph,
        val_nodes,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        device=device,
    )

    # some summary statistics
    best = 1e9
    best_epoch = -1

    for epoch in tqdm(range(n_epochs)):
        log = dict()

        loss = 0.0

        model.train()
        optimizer.zero_grad()

        # TRAIN
        for input_nodes, output_nodes, blocks in train_dataloader:
            # we need to have embeddings for the entire graph for the loss function
            # however, we only back propogate based on those in our batch

            loss_in_nodes, loss_out_nodes, loss_blocks = sampler.sample(
                dgl_graph, indexes
            )
            # samplers do not do prefetching of features, only dataloaders do
            # emulate this behaviour ourselves
            # from https://docs.dgl.ai/en/1.0.x/generated/dgl.dataloading.base.LazyFeature.html

            loss_feats = dgl_graph.ndata["feat"][loss_blocks[0].srcdata[dgl.NID]]

            # calculate gradients, etc for batch blocks only
            model.train()
            model(blocks, blocks[0].srcdata["feat"])

            # calculate all the graphs node embeddings for loss function use
            model.eval()
            embs = model(loss_blocks, loss_feats)

            # note that we only use batch output_nodes in our loss calculation
            l = loss_function(output_nodes, embs)
            l.backward()
            optimizer.step()
            loss += l

        log.update({f"{wandb_summary_prefix}-training-loss": loss})

        # VALIDATE
        model.eval()
        loss = 0.0

        # as above, we need embeddings for the entire graph for the loss function
        loss_in_nodes, loss_out_nodes, loss_blocks = sampler.sample(dgl_graph, indexes)
        loss_feats = dgl_graph.ndata["feat"][loss_blocks[0].srcdata[dgl.NID]]

        for input_nodes, output_nodes, blocks in val_dataloader:
            embs = model(loss_blocks, loss_feats)
            l = loss_function(output_nodes, embs)

            loss += l.item()

        log.update({f"{wandb_summary_prefix}-validation-loss": loss})
        wandb.log(log)

        # early stopping
        if loss <= best + min_delta:
            best = loss
            best_epoch = epoch
            # save current weights
            torch.save(model.state_dict(), early_stopping_filepath)

        if epoch - best_epoch > patience:
            print("Early stopping!")
            model.load_state_dict(torch.load(early_stopping_filepath))
            wandb.summary[f"{wandb_summary_prefix}-early-stopping-epoch"] = best_epoch
            break

    # save parameters for later fine-tuning if a save path is given
    if save_weights_to != "":
        print(f"Saving model parameters to {str(save_weights_to)}")

        torch.save(model.state_dict(), save_weights_to)

    model.eval()

    # Cleanup
    temporary_directory.cleanup()

    def encoder(g, features):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(k)
        (inp, out, blocks) = sampler.sample(g, g.nodes())

        return model(blocks, features[blocks[0].srcdata[dgl.NID]])

    return encoder
