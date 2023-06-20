__all__ = ["train_egi_encoder"]

import tempfile
from pathlib import Path
from warnings import warn

import dgl
import torch
import torch.nn as nn
import wandb
from dgl.dataloading import DataLoader
from tqdm import tqdm

from ..features import degree_bucketing
from ..models import EGI
from ..samplers import KHopTriangleSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_egi_encoder(
    dgl_graph,
    gpu=-1,
    k=2,
    lr=0.01,
    n_hidden_layers=32,
    n_epochs=100,
    weight_decay=0.0,
    feature_mode="degree_bucketing",
    optimiser="adam",
    pre_train=None,
    batch_size=50,
    sampler="egi",
    save_weights_to=None,
    patience=10,
    min_delta=0.01,
    wandb_enabled=True,
    wandb_summary_prefix="",
):
    """
    Train an EGI [1] graph encoder.

    An EGI encoder produces node embeddings for a given graph, retaining
    high-level structural features of the training graph to improve
    transferability.

    It does this by considering k-hop ego graphs.

    Args:

        dgl_graph: The input graph, as a DGLGraph.

        k: The number of hops to consider in the ego-graphs.
           Defaults to 2, which was shown to have the best results in the
           original paper.

        lr: Learning rate. Defaults to 0.01. l

        n_hidden_layers: The number of hidden layers in the encoder. Defaults
            to 32.

        n_epochs: The number of epochs to do when training.

        feature_mode: The function to use to generate feature tensors.
            Options are: ['degree_bucketing'].
            Defaults to 'degree_bucketing'.

        optimiser: The optimiser to use.
            Options are: ['adam'].
            Defaults to 'adam'.

        batch_size: The number of nodes to consider in each training batch.

        sampler: The subgraph sampler to use.
            Options are ['egi','triangle']
            Defaults to 'egi'.

        pre_train: Existing model parameters to use for fine tuning in transfer
            learning. This must be a path that points to a file saved by doing:

            torch.save(modelA.state_dict(), PATH)

            Defaults to None.

            For more information, see
            https://pytorch.org/tutorials/beginner/saving_loading_models.html.

        save_weights_to: A file path to save EGI model parameters for use in
            transfer learning.

            Defaults to None.

        patience: The number of epochs to wait before early stopping.

            Defaults to 10.

        min_delta:

        wandb_enabled: Whether the training function is being called within a
            wandb run. If enabled, this will log loss metrics to wandb.

            Defaults to True.

        wandb_summary_prefix: A prefix to be appended to the wandb metrics for this
            model.

            Defaults to "".



    Returns:
        The trained EGI encoder model.


    References:

        [1] Q. Zhu, C. Yang, Y. Xu, H. Wang, C. Zhang, and J. Han,
        ‘Transfer Learning of Graph Neural Networks with Ego-graph Information
         Maximization’.
         arXiv, 2020. doi: 10.48550/ARXIV.2009.05204.

    """

    if gpu != -1:
        warn(
            "Manually specifying a gpu number is deprecated."
            "This is determined automatically by the training function.",
            DeprecationWarning,
        )

    # input validation
    valid_feature_modes = ["degree_bucketing"]
    valid_optimisers = ["adam"]
    valid_samplers = ["egi", "triangle"]

    if feature_mode not in valid_feature_modes:
        raise ValueError(
            f"{feature_mode} is not a valid feature generation "
            "mode. Valid options are {valid_feature_modes}."
        )

    if optimiser not in valid_optimisers:
        raise ValueError(
            f"{optimiser} is not a valid optimiser."
            "Valid options are {valid_optimisers}."
        )

    if sampler not in valid_samplers:
        raise ValueError(
            f"{sampler} is not a valid sampler." "Valid options are {valid_sampler}."
        )

    if k < 1:
        raise ValueError("k must be 1 or greater.")

    if lr <= 0:
        raise ValueError("Learning rate must be above 0.")

    # setup temporary directory for saving models for early-stopping
    temporary_directory = tempfile.TemporaryDirectory()
    early_stopping_filepath = Path(temporary_directory.name, "stopping.pt")

    # generate features

    features = degree_bucketing(dgl_graph, n_hidden_layers)

    if sampler == "egi":
        sampler = dgl.dataloading.NeighborSampler([10 for i in range(k)])
    elif sampler == "triangle":
        sampler = KHopTriangleSampler(dgl_graph, [10 for i in range(k)])

    features = features.to(device)
    dgl_graph = dgl_graph.to(device)

    in_feats = features.shape[1]

    # in the original code, they set number of layers to equal k +1
    n_layers = k + 1

    model = EGI(
        in_feats,
        n_hidden_layers,
        n_layers,
        nn.PReLU(n_hidden_layers),
    )

    model = model.to(device)

    if wandb_enabled:
        wandb.watch(model)

    # do transfer learning if we have pretrained weights
    if pre_train is not None:
        model.load_state_dict(torch.load(pre_train), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # some summary statistics
    best = 1e9
    best_epoch = -1

    # shuffle nodes before putting into train and validation sets
    indexes = torch.randperm(dgl_graph.nodes().shape[0])
    validation_size = dgl_graph.nodes().shape[0] // 6
    val_nodes = torch.split(dgl_graph.nodes()[indexes], validation_size)[0]
    train_nodes = torch.unique(torch.cat([val_nodes, dgl_graph.nodes()]))

    # start training
    for epoch in tqdm(range(n_epochs)):
        log = dict()

        # Enable training mode for model
        model.train()

        loss = 0.0

        # train based on features and ego graphs around specifc egos
        model.train()
        optimizer.zero_grad()

        # the sampler returns a list of blocks and involved nodes
        # each block holds a set of edges from a source to destination
        # each block is a hop in the graph
        for blocks in DataLoader(
            dgl_graph,
            train_nodes,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            device=device,
        ):
            l = model(dgl_graph, features, blocks)
            l.backward()
            optimizer.step()
            loss += l

        log.update({f"{wandb_summary_prefix}-training-loss": loss})

        # validation

        model.eval()
        loss = 0.0
        blocks = sampler.sample(dgl_graph, val_nodes)
        loss = model(dgl_graph, features, blocks)

        log.update({f"{wandb_summary_prefix}-validation-loss": loss})

        if wandb_enabled:
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

            if wandb_enabled:
                wandb.summary[
                    f"{wandb_summary_prefix}-early-stopping-epoch"
                ] = best_epoch

            break

    # save parameters for later fine-tuning if a save path is given
    if save_weights_to is not None:
        print(f"Saving model parameters to {str(save_weights_to)}")

        torch.save(model.state_dict(), save_weights_to)

    model.eval()
    model.encoder.eval()

    # Cleanup
    temporary_directory.cleanup()

    return model.encoder
