
from .. import Graph
from torch import Tensor
from torch import device as Device
from collections.abc import Mapping

def train(graph: Graph, features: Tensor, device: Device, config: Mapping):
    raise NotImplementedError("DGI training is not yet implemented")

