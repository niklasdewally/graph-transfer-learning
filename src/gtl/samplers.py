import dgl
import torch

from dgl.dataloading import Sampler
from IPython import embed

__all__ = ["KHopTriangleSampler"]


class KHopTriangleSampler(Sampler):
    def __init__(self, g,fanouts):
        super().__init__() 
        self.fanouts = fanouts
        self.triangles = None
        self.g = g

    def sample(self,_,seed_nodes):
        # loosely inspired by https://github.com/dmlc/dgl/blob/master/python/dgl/dataloading/neighbor_sampler.py
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = self._sample_triangle_neighbors(seed_nodes,fanout)
            eids = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier)
            block.edata[dgl.EID] = eids
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks




    def _populate_triangles(self):
        self.triangles = dict()
        for nid in self.g.nodes():
            self.triangles[nid.item()] = self._get_triangles(nid.item())


    def _get_triangles(self,nid):
            edges= torch.empty(0,device=self.g.device,dtype=torch.int64)
            neighbors = self.g.successors(nid)

            triads = torch.combinations(neighbors)
            for i in range(triads.shape[0]):
                neighbor = triads[i][0]
                neighbor2 = triads[i][1]
                if torch.all(self.g.has_edges_between([neighbor],[neighbor2])):
                    newedges = self.g.edge_ids([nid,nid,neighbor],[neighbor,neighbor2,neighbor2],return_uv=False).to(self.g.device)
                    edges = torch.cat((edges,newedges)).to(self.g.device)

            return edges

    def _sample_triangle_neighbors(self,seed_nodes, fanout):
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
        if self.triangles is None:
            self._populate_triangles()
        
        edges= torch.empty(0,device=self.g.device,dtype=torch.int64)

        for nid in seed_nodes:
            nid = nid.item()
            new_edges = self.triangles[nid]
            random_mask = torch.randperm(new_edges.shape[0])
            sampled_edges = new_edges[random_mask][:min(fanout*2,new_edges.shape[0])]
            edges = torch.cat([edges,sampled_edges])

        # return node induced subgraph
        subg = dgl.edge_subgraph(self.g, edges).to(self.g.device)
        subg.edata[dgl.EID] = edges.clone().detach()
        return subg



