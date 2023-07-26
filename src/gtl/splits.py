import math
import random

import dgl
import networkx as nx
import torch

from . import Graph


class LinkPredictionSplit:
    """
    Create a test/validation split for link prediction tasks.

    Edges are first split into message passing edges, used for training the encoder, and supervision
    edges, used to supervise link prediction. This split happens according to the given
    `message_supervision_ratio`.

    Then, the supervision links are further split into training and validation sets according to
    `train_val_ratio`.

    At validation time, both the training message passing and training supervision edges are used
    as input. This is because the model will have learnt about these at this point.

    I refer to this combination as the validation message passing set or the full training graph:
    (`LinkPredictionSplit.full_training_graph`)

    Usage
    =====

    Three graphs can be obtained - the message passing graph (mp_graph), the full training graph,
    consisting of the supervision and message passing training edges (full_training_graph), and the
    full graph.

    Each of these graphs use different and incompatible node-ids.

    The following edge lists are available. Also listed is the graph in which they derive their node
    ids.

    train_edges: supervision training edges - derived from full_training_graph
    val_edges: supervision validation edges - derived from graph


    This is indexed in this way as the link prediction task will use the train_edges over the entire
    training set, and similar for the val_edges.

    References
    ===========

    http://snap.stanford.edu/class/cs224w-2020/slides/08-GNN-application.pdf [p58]
    """

    def __init__(
        self,
        graph: Graph,
        message_supervision_ratio: float = 0.6,
        train_val_ratio: float = 0.7,
    ) -> None:
        self.graph = graph

        self.message_supervision_ratio = message_supervision_ratio
        self.train_val_ratio = train_val_ratio

        n_mp_edges: int = math.floor(
            self.message_supervision_ratio * self.graph._G.number_of_edges()
        )

        # shuffle to get representative samples
        # use undirected graph so we dont treat a->b and b->a as seperate edges
        edges = list(self.graph._G.to_undirected().edges())
        random.shuffle(edges)

        # First, split edges into supervision and message-passing
        mp_edges = edges[:n_mp_edges]
        supervision_edges = edges[n_mp_edges:]

        # Then, split supervision edges into train val
        n_train_edges: int = math.floor(self.train_val_ratio * len(supervision_edges))

        random.shuffle(supervision_edges)
        train_edges: list[tuple[int, int]] = supervision_edges[:n_train_edges]
        val_edges: list[tuple[int, int]] = supervision_edges[n_train_edges:]

        full_training_edges: list[tuple[int, int]] = train_edges + mp_edges

        # Turn these into useful subgraphs
        self.mp_graph: Graph = self.graph.edge_subgraph(mp_edges)
        """
        A graph containing only the message passing split.

        This should be used to train node embedding.
        """

        self.full_training_graph: Graph = self.graph.edge_subgraph(full_training_edges)
        """
        A graph containing both the message-passing and training supervision splits.

        This represents the full training set of the graph, and should be used alongside train_edges
        for training link prediction.

        It can also be used for validating an encoder, as it is assumed that an encoder, once complete
        would have learnt these supervision nodes.
        """

        # Index train edges relative to full_training_graph
        # nx gives us new -> old - flip direction for old -> new
        mapping = {
            v: k
            for k, v in nx.get_node_attributes(
                self.full_training_graph._G, "old_id"
            ).items()
        }

        self.train_edges: list[tuple[int, int]] = []
        """
        The training supervision edges.

        These use node ids from the full training graph
        """

        for u, v in train_edges:
            self.train_edges.append((mapping[u], mapping[v]))

        # Index val edges relative to fullgraph
        # (no work needs to be done)
        self.val_edges: list[tuple[int, int]] = val_edges
        """
        The validation supervision edges.

        These use node ids from the full graph.
        """


# TODO (niklasdewally): Write doc string


class TrianglePredictionSplit:
    """ """

    def __init__(
        self,
        graph: Graph,
        message_supervision_ratio: float = 0.6,
        train_val_ratio: float = 0.7,
    ) -> None:
        self.graph = graph
        """
        The graph
        """

        self.message_supervision_ratio = message_supervision_ratio
        self.train_val_ratio = train_val_ratio

        triangles: list[list[int]] = self.graph.get_triangles_list()
        n_triangles: int = len(triangles)

        # shuffle to get representative samples
        random.shuffle(triangles)

        n_mp_triangles: int = math.floor(self.message_supervision_ratio * n_triangles)

        # First, split edges into supervision and message-passing
        # mp_triangles = triangles[:n_mp_triangles]
        supervision_triangles = triangles[n_mp_triangles:]

        self.mp_graph: Graph = self.graph.copy()
        """
        A graph containing only the message passing split.

        This should be used to train node embedding.
        """

        for node1, node2, node3 in supervision_triangles:
            self.mp_graph.remove_triangle(node1, node2, node3)

        # Then, split supervision edges into train val
        n_train_triangles: int = math.floor(
            self.train_val_ratio * len(supervision_triangles)
        )

        random.shuffle(supervision_triangles)
        train_triangles: list[list[int]] = supervision_triangles[:n_train_triangles]
        val_triangles: list[list[int]] = supervision_triangles[n_train_triangles:]

        self.full_training_graph: Graph = self.graph.copy()
        """
        A graph containing both the message-passing and training supervision splits.

        This represents the full training set of the graph, and should be used alongside train_triangles
        for training link prediction.

        It can also be used for validating an encoder, as it is assumed that an encoder, once complete
        would have learnt these supervision nodes.
        """

        for node1, node2, node3 in val_triangles:
            self.full_training_graph.remove_triangle(node1, node2, node3)

        self.train_triangles: list[list[int]] = train_triangles
        """
        The training supervision triangles.
        """

        self.val_triangles: list[list[int]] = val_triangles
        """
        The validation supervision triangles.
        """


class CoauthorNodeClassificationSplit:
    """
    Node splits for the CoAuthor classification experiment.

    This experiment trys to learn node classes based on node embeddings trainied from a small
    subgraph.

        Train node embeddings (small graph) --> Train classifier (full graph + label splits)

    This split creates train/test/val splits for training the classification, as well as the small
    graph used to train node embeddings.

    For the train/test/val sets, nodes with each label is added randomly according to the train/test/val
    proportions. Therefore, these splits have the same distribution of labels as the full graph.

    The small graph's nodes are a strict subset of the training nodes.

    The small pretrain graph takes the first half of training nodes per label. This results in a
    graph that is approximately ~30% the size of the full graph that has similar feature/label
    distribution.

    This can be inspected further using `scripts/vis/visualise_coauthor_feats.py`.

    See also:
        gtl.coauthor.load_coauthor_npz()
    """

    def __init__(
        self,
        graph: Graph,
        label_tensor: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.train_idxs: torch.Tensor
        """
        A tensor containing the indicies of all nodes used as train labels.
        """

        self.train_mask: torch.Tensor
        """
        Mask tensor for the training label set.
        """

        self.val_idxs: torch.Tensor
        """
        A tensor containing the indicies of all nodes used as validation labels.
        """

        self.val_mask: torch.Tensor
        """
        Mask tensor for the validation label set.
        """

        self.test_idxs: torch.Tensor
        """
        A tensor containing the indicies of all nodes used as test labels.
        """

        self.test_mask: torch.Tensor
        """
        Mask tensor for the test label set.
        """

        self.small_idxs: torch.Tensor
        """
        A tensor containing the indicies of all nodes used in the small pre-training graph. 
        """

        self.small_mask: torch.Tensor
        """
        Mask tensor for the small pre-training graph.
        """

        self.small_g: Graph
        """
        The small pre-training graph.
        """

        torch.manual_seed(0)

        label_tensor = label_tensor.to(device)

        # Split nodes into train,val,test
        train_idxs = torch.empty([0], dtype=torch.int64, device=device)
        test_idxs = torch.empty([0], dtype=torch.int64, device=device)
        val_idxs = torch.empty([0], dtype=torch.int64, device=device)
        small_idxs = torch.empty([0], dtype=torch.int64, device=device)

        for label in label_tensor.unique():
            label = label.item()

            idxs_with_label = torch.where(label_tensor == label)[0]
            n_nodes_with_label = idxs_with_label.shape[0]
            shuffled_idxs = idxs_with_label[
                torch.randperm(n_nodes_with_label, device=device)
            ]

            n_train = math.floor(n_nodes_with_label * 0.6)
            n_val = math.floor(n_nodes_with_label * 0.2)

            train_idxs = torch.cat((train_idxs, shuffled_idxs[:n_train]))
            small_idxs = torch.cat((small_idxs, shuffled_idxs[: n_train // 2]))
            val_idxs = torch.cat((val_idxs, shuffled_idxs[n_train : n_train + n_val]))
            test_idxs = torch.cat((test_idxs, shuffled_idxs[n_train + n_val :]))

        self.train_idxs = train_idxs.to(device)

        train_mask = torch.zeros(label_tensor.shape, dtype=torch.bool, device=device)
        self.train_mask = train_mask.scatter_(0, self.train_idxs, 1).to(device)

        self.val_idxs = val_idxs.to(device)

        val_mask = torch.zeros(label_tensor.shape, dtype=torch.bool, device=device)
        self.val_mask = val_mask.scatter_(0, self.val_idxs, 1).to(device)

        self.test_idxs: torch.Tensor = test_idxs.to(device)

        test_mask: torch.Tensor = torch.zeros(
            label_tensor.shape, dtype=torch.bool, device=device
        )
        self.test_mask = test_mask.scatter_(0, self.test_idxs, 1).to(device)

        self.small_g: Graph = graph.node_subgraph(small_idxs.tolist())

        # the subgraph sampler removes some isolated nodes, so update small_idxs to reflect this
        self.small_idxs = torch.tensor(
            list(nx.get_node_attributes(self.small_g.as_nx_graph(), dgl.NID).values()),
            device=device,
        )

        small_mask: torch.Tensor = torch.zeros(
            label_tensor.shape, dtype=torch.bool, device=device
        )
        self.small_mask = small_mask.scatter_(0, self.small_idxs, 1).to(device)
