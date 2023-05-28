import dgl
import torch

from dgl.dataloading import Sampler
from IPython import embed

__all__ = ["KHopTriangleSampler"]


class KHopTriangleSampler(Sampler):
    def __init__(self, fanouts):
        super().__init__()
        self.fanouts = fanouts

    def sample(self, g, seed_nodes):
        # loosely inspired by https://github.com/dmlc/dgl/blob/master/python/dgl/dataloading/neighbor_sampler.py
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = sample_triangle_neighbors(g,seed_nodes,fanout)
            eids = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier)
            block.edata[dgl.EID] = eids
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks


def sample_triangle_neighbors(g, seed_nodes, fanout):
    """

    Sample triangles that contain the given nodes, and return the induced node subgraph.

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

    edges = torch.empty(0,dtype=torch.int64).to(g.device)

    for nid in seed_nodes:
        neighbors = g.successors(nid)
        if neighbors.shape[0] < 2:
            continue
        triangles = []
        count = 0
        i = 0

        triads = torch.combinations(neighbors)
        while count < min(fanout,len(seed_nodes)) and i < len(triads):

            neighbor = triads[i][0]
            neighbor2 = triads[i][1]
            if torch.all(g.has_edges_between([neighbor],[neighbor2])):
                newedges = g.edge_ids([nid,nid,neighbor],[neighbor,neighbor2,neighbor2],return_uv=False).to(g.device)
                edges = torch.cat((edges,newedges)).to(g.device)
                count += 1
                i += 1
                continue
            # no triangle
            i += 1

    # return node induced subgraph
    subg = dgl.edge_subgraph(g, edges).to(g.device)

    # set node features
    subg.edata[dgl.EID] = edges.clone().detach()

    return subg


