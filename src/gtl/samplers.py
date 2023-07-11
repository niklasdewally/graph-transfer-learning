import dgl
import torch
from dgl.dataloading import Sampler

from IPython import embed
from random import sample

__all__ = ["KHopTriangleSampler"]


class KHopTriangleSampler(Sampler):
    def __init__(self, g: dgl.DGLGraph, fanouts:list[int], triangles: dict[int,list[list[int]]]) -> None:
        super().__init__()
        self.fanouts = fanouts
        self.g = g
        self.triangles = triangles

    #pyre-ignore[3]
    def sample(self, _, seed_nodes: torch.Tensor):
        # loosely inspired by
        # https://github.com/dmlc/dgl/blob/master/python/dgl/dataloading/neighbor_sampler.py
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = self._sample_triangle_neighbors(seed_nodes, fanout)
            eids = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier)
            block.edata[dgl.EID] = eids
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks

    def _sample_triangle_neighbors(self, seed_nodes: torch.Tensor, fanout: int) -> dgl.DGLGraph:
        """

        Sample triangles that contain the given nodes, and return the induced subgraph.

        The original IDs of the sampled nodes are stored as the `dgl.NID` feature
        in the returned graph.

        This algorithm currently only works for undirected graphs.

        Args:
            g: DGLGraph
                The graph.

            seed_nodes: tensor
                Node IDs to sample triangles from.

            fanout : int
                The number of edges to be sampled.
        """
        edges = torch.empty(0, device=self.g.device, dtype=torch.int64)

        for nid_tensor in seed_nodes:
            nid : int = nid_tensor.item()
            sampled_triangles = torch.tensor(
                sample(self.triangles[nid], min(fanout,len(self.triangles[nid]))),
                device=self.g.device,
                dtype=torch.int64,
            )

            # [1,2,3,4,...]
            sampled_nodes = torch.flatten(sampled_triangles).unique()
            new_edges = self.g.edge_ids(
                torch.tensor([nid],device=self.g.device).repeat(sampled_nodes.shape[0]), sampled_nodes,return_uv=True
            )[2].to(self.g.device)

            edges = torch.cat([edges, new_edges])

        subg = dgl.edge_subgraph(self.g, edges.unique()).to(self.g.device)
        subg.edata[dgl.EID] = edges.clone().detach()
        return subg
