# example largely from
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/link_pred.py

# TODO (niklasdewally): make strictly typed
# pyre-unsafe

import dgl
import dgl.nn.pytorch as dglnn
import networkx as nx
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from IPython import embed

from .. import Graph


class SAGEUnsupervised(nn.Module):
    """
    An unsupervised GraphSAGE model.

    Designed to be used with the SageLoss loss function class.

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
    The unsupervised loss function from GraphSAGE paper.

    [1] https://arxiv.org/abs/1706.02216
    """

    def __init__(self, g: Graph, train_nodes: list[int]) -> None:
        self.positive_node_for: dict[int, int] = dict()
        self.negative_nodes_for: dict[int, list[int]] = dict()

        self.graph: nx.Graph = g.as_nx_graph()
        self.train_nodes: list[int] = train_nodes

        self.WALK_LEN = 1
        self.NEGATIVE_WALK_LEN = 5
        self.Q = 10

        print("Running random walks for loss function")
        self._run_positive_walks()
        self._run_negative_walks()

    def calculate_loss(
        self, nodes: torch.Tensor, node_embeddings: torch.Tensor
    ) -> torch.Tensor:
        device = node_embeddings.device

        node_losses = torch.empty(0, device=device)

        for node in nodes:
            node = node.item()
            node_loss = 0.0

            assert node in self.positive_node_for.keys()
            assert node in self.negative_nodes_for.keys()

            # positive term - node can walk to the new node, so the representations should be similar
            positive_node = self.positive_node_for[node]
            similarity = self._create_edge_embedding(
                node_embeddings[node], node_embeddings[positive_node]
            )
            node_loss += -torch.log(torch.sigmoid(similarity))

            # negative nodes - for each node, sample a node that is not in a random walk.
            # negative nodes should have dissimilar representations
            negative_node_similarities = torch.empty(0, device=device)

            for negative_node in self.negative_nodes_for[node]:
                similarity = self._create_edge_embedding(
                    node_embeddings[node], node_embeddings[negative_node]
                )
                negative_node_similarities = torch.cat(
                    (negative_node_similarities, torch.unsqueeze(similarity, 0))
                )

            node_loss += -self.Q * torch.mean(torch.log(torch.sigmoid(-similarity)))

            node_losses = torch.cat((node_losses, torch.unsqueeze(node_loss, 0)))

        return torch.mean(node_losses)

    def _create_edge_embedding(self, emb1, emb2) -> torch.Tensor:
        return F.cosine_similarity(emb1, emb2, dim=0)

    def _run_positive_walks(self) -> None:
        for node in self.train_nodes:
            current_node = node
            for i in range(self.WALK_LEN):
                current_node = random.choice(list(self.graph.neighbors(current_node)))

            self.positive_node_for[node] = current_node

    def _run_negative_walks(self) -> None:
        for node in self.train_nodes:
            excluded_nodes = set()

            for i in range(self.NEGATIVE_WALK_LEN):
                excluded_nodes = excluded_nodes | set(
                    y for x in excluded_nodes for y in self.graph.neighbors(x)
                )

            negative_nodes = list()

            # FIXME (niklasdewally): moving the graph node iterator to a list is not ideal - do i really need to shuffle this?
            for n in random.sample(list(self.graph), len(list(self.graph))):
                if len(negative_nodes) == self.Q:
                    break

                if node not in excluded_nodes:
                    negative_nodes.append(n)

            self.negative_nodes_for[node] = negative_nodes

    def __call__(self, *args) -> torch.Tensor:
        return self.calculate_loss(*args)
