"""
Unsupervised GraphSAGE model and loss function.
"""

# Largely taken from:
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/link_pred.py

import random

import dgl
import dgl.nn.pytorch as dglnn
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .. import Graph

__all__ = ["SAGEUnsupervised", "SAGEUnsupervisedLoss"]


class SAGEUnsupervised(nn.Module):
    """
    An unsupervised GraphSAGE model.

    Use this alongside SAGEUnsupervisedLoss (or a custom loss function) to train
    an unsupervised GraphSAGE model that produces node embeddings.

    This is largely based on the DGL GraphSAGE example:
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/link_pred.py
    """

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        n_conv_layers: int = 3,
        aggregator: str = "mean",
    ) -> None:
        """
        Args:
          in_size: The size of the input features.
            That is, for a graph with n nodes, the input features should be a
            tensor of shape (n,in_size).

          hidden_size: The size of the hidden layers.

          n_conv_layers: The number of convolutional (hidden) layers to use.

            This equates to the depth of k-hop neighborhood to consider for
            each node.

            n_conv_layers = n_hops + 1

          aggregator: The graphsage aggregator to use.
            Valid aggregators are: ["mean","pool","lstm","gcn"]
        """

        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, aggregator))
        for _ in range(n_conv_layers - 1):
            self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, aggregator))

        self.hidden_size = hidden_size

    def forward(
        self,
        blocks: list[dgl.DGLGraph],
        feats: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h = feats

        last_layer = len(self.layers) - 1
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != last_layer:
                h = F.relu(h)

        return h


class SAGEUnsupervisedLoss(object):
    """
    The unsupervised loss function from the GraphSAGE paper [1].

    Once initialised, this should be called as a function.

    For example:

    >> loss_fn = SAGEUnsupervisedLoss(g,seed_nodes)
    >> embs = model(all_nodes)
    >> loss = loss_fn(training_nodes,embs)
    >> loss.backward()

    [1] https://arxiv.org/abs/1706.02216
    """

    def __init__(self, g: Graph) -> None:
        """
        Args:
            g: The graph to use while calculating loss.
        """
        self.positive_node_for: dict[int, int] = dict()
        self.negative_nodes_for: dict[int, list[int]] = dict()

        self.graph: nx.Graph = g.as_nx_graph()

        self.WALK_LEN = 1
        self.NEGATIVE_WALK_LEN = 5
        self.Q = 10

    # use via __call__!!
    def _calculate_loss(self, nodes: torch.Tensor, embs: torch.Tensor) -> torch.Tensor:
        positive_node_for = self._run_positive_walks(nodes.tolist())
        negative_nodes_for = self._run_negative_walks(nodes.tolist())

        device = embs.device

        node_losses = torch.empty(0, device=device)

        for node in nodes:
            node = node.item()
            node_loss = 0.0

            assert node in positive_node_for.keys()
            assert node in negative_nodes_for.keys()

            # positive term - node can walk to the new node, so the representations should be similar
            positive_node = positive_node_for[node]
            similarity = self._create_edge_embedding(embs[node], embs[positive_node])
            node_loss += -torch.log(torch.sigmoid(similarity))

            # negative nodes - for each node, sample a node that is not in a random walk.
            # negative nodes should have dissimilar representations
            negative_node_similarities = torch.empty(0, device=device)

            for negative_node in negative_nodes_for[node]:
                similarity = self._create_edge_embedding(
                    embs[node], embs[negative_node]
                )
                negative_node_similarities = torch.cat(
                    (negative_node_similarities, torch.unsqueeze(similarity, 0))
                )

            node_loss += -self.Q * torch.mean(torch.log(torch.sigmoid(-similarity)))

            node_losses = torch.cat((node_losses, torch.unsqueeze(node_loss, 0)))

        return torch.mean(node_losses)

    def _create_edge_embedding(
        self, emb1: torch.Tensor, emb2: torch.Tensor
    ) -> torch.Tensor:
        return F.cosine_similarity(emb1, emb2, dim=0)

    def _run_positive_walks(self, nodes: list[int]) -> dict[int, int]:
        positive_node_for = dict()
        for node in nodes:
            current_node = node
            for i in range(self.WALK_LEN):
                current_node = random.choice(list(self.graph.neighbors(current_node)))

            positive_node_for[node] = current_node

        return positive_node_for

    def _run_negative_walks(self, nodes: list[int]) -> dict[int, list[int]]:
        negative_nodes_for = dict()

        for node in nodes:
            excluded_nodes = set()

            for i in range(self.NEGATIVE_WALK_LEN):
                excluded_nodes = excluded_nodes | set(
                    y for x in excluded_nodes for y in self.graph.neighbors(x)
                )

            negative_nodes = list()

            # FIXME (niklasdewally): moving the graph node iterator to a list is
            # not ideal - do i really need to shuffle this?

            for n in random.sample(list(self.graph), len(list(self.graph))):
                if len(negative_nodes) == self.Q:
                    break

                if node not in excluded_nodes:
                    negative_nodes.append(n)

            negative_nodes_for[node] = negative_nodes

        return negative_nodes_for

    def __call__(self, nodes: torch.Tensor, embs: torch.Tensor) -> torch.Tensor:
        """
        Calculate and return the loss of the given nodes.

        Args:
          nodes: The nodes to calculate the loss for. These nodes must come
            from the graph passed in during initialisation.

          embs: The current node embeddings of the graph passed in during
            initialisation.
        """

        return self._calculate_loss(nodes, embs)
