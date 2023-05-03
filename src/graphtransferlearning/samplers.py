import dgl
import torch

from dgl.dataloading import Sampler
from IPython import embed


class KHopTriangleSampler(Sampler):
    def __init__(self, fanouts):
        super().__init__()
        self.fanouts = fanouts

    def sample(self, g, seed_nodes):
        # loosely inspired by https://github.com/dmlc/dgl/blob/master/python/dgl/dataloading/neighbor_sampler.py
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(seed_nodes, fanout)

            eids = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[dgl.EID] = eids
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks


def _sample_triangle_neighbors(g, seed_nodes, fanout):
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

    edges = []

    for nid in seed_nodes:
        neighbors = nid.successors()
        for i in range(min(fanout, len(seed_nodes))):
            neighbor = neighbors[i]
            triangle_edges = get_triangle(g, nid, neighbor)

            if triangle_edges is not None:
                edges.append(triangle_edges)

    # return node induced subgraph
    subg = dgl.edge_subgraph(g, edges)

    # set node features
    subg.edata[dgl.EID] = torch.tensor(edges)

    return subg


def _in_triangle(g, node1, node2):
    # these form a triangle if the two nodes have a common neighbor
    node1_neighbors = g.successors(node2)
    node2_neighbors = g.successors(node2)

    # stick the two tensors together, and see if any nodes are eliminated when calling unique.
    cat = torch.cat(node1_neighbors, node2_neighbors)
    uniques = torch.unique(cat)

    if cat.shape[0] == uniques.shape[0]:
        print(f"{node1} -> {node2} does not form a triangle")
        return None

    # generate list of edge ids

    us = torch.tensor([node1, node2])
    vs = cat[cat != unique]
    vs = vs[vs != us]
    eids = g.find_edges(us, vs)
    embed()

    return eids
