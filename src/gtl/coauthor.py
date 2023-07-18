import numpy as np
import scipy.sparse as sp
from gtl.typing import PathLike
from gtl import Graph

import networkx as nx
import torch


"""
Load the coauthor citation datasets from the original npz files into Graph objects.

The original data can be downloaded from https://github.com/shchur/gnn-benchmark/
"""


# Some attributes are saved as sparse scipy arrays, others as np arrays
def load_coauthor_npz(location: PathLike) -> tuple[Graph,torch.Tensor]:
    file_data = np.load(location, allow_pickle=True)

    # adjaceny matrix is sparse
    adj_data = file_data.get("adj_data")
    adj_indices = file_data.get("adj_indices")
    adj_indptr = file_data.get("adj_indptr")
    adj_shape = file_data.get("adj_shape")

    adj: sp.csr_array = sp.csr_array((adj_data, adj_indices, adj_indptr), adj_shape)

    # attribute data is sparse
    attr_data = file_data.get("attr_data")
    attr_indices = file_data.get("attr_indices")
    attr_indptr = file_data.get("attr_indptr")
    attr_shape = file_data.get("attr_shape")

    attr = sp.csr_array(
        (attr_data, attr_indices, attr_indptr), attr_shape
    )

    attr_tensor : torch.Tensor = torch.tensor(attr.toarray())

    labels: np.typing.NDArray = file_data["labels"]
    node_names: np.typing.NDArray = file_data["node_names"]
    attr_names: np.typing.NDArray = file_data["attr_names"]
    class_names: np.typing.NDArray = file_data["class_names"]

    nx_graph: nx.Graph = nx.Graph(adj)
    nx.set_node_attributes(nx_graph, attr, name="feats")

    return Graph(nx_graph),attr_tensor
