import itertools

from copy import deepcopy
from random import sample, choice
from typing import Optional, List
import numpy as np

import dgl
import networkx as nx
import torch
from networkx.readwrite.gml import literal_destringizer, literal_stringizer

from .typing import PathLike

TRIANGLES = "gtl_triangles"


class Graph:
    """

    A Graph wraps a networkx graph, providing an easier interface for adding / removing
    graph features (triangles, etc.).

    These features are stored in node, edge, and graph attributes.
    """

    def __init__(self, g: nx.Graph) -> None:
        self._G: nx.Graph = deepcopy(g)
        self.triads_by_type: dict[str, list[nx.Graph]] | None = None

        # pyre-ignore[8]
        self._dgl_g: dgl.DGLGraph = None

    @staticmethod
    def from_gml_file(path: PathLike) -> "Graph":
        # implicitly remove multiedges by passing to nx.Graph()
        g = nx.Graph(nx.read_gml(path, destringizer=literal_destringizer))
        return Graph(g)

    @staticmethod
    def from_dgl_graph(
        g: dgl.DGLGraph,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
    ) -> "Graph":
        nx_g: nx.Graph = dgl.to_networkx(
            g.cpu(), node_attrs, edge_attrs
        ).to_undirected()
        return Graph(nx_g)

    def as_nx_graph(self) -> nx.Graph:
        """
        Return the graph, in networkx format.
        """
        return self._G.copy().to_undirected()

    def as_dgl_graph(self, device: torch.device) -> dgl.DGLGraph:
        if self._dgl_g is not None:
            return self._dgl_g.to(device)

        edges: list[tuple[int, int]] = self._G.edges()

        us: torch.Tensor = torch.empty([0], dtype=torch.int64, device=device)
        vs: torch.Tensor = torch.empty([0], dtype=torch.int64, device=device)

        for u, v in edges:
            us = torch.cat((us, torch.tensor([u], device=device)))
            vs = torch.cat((vs, torch.tensor([v], device=device)))

        # Add reverse edges
        # DGL models undirected graphs as graphs where edge has an associated reverse edge
        undirectional_us = torch.cat((us,vs))
        undirectional_vs = torch.cat((vs,us))

        self._dgl_g = dgl.graph((undirectional_us, undirectional_vs), device=device)

        return self._dgl_g

        # this is too slow for large graphs
        # return dgl.from_networkx(self._G, device=device, **kwargs)

    def to_gml_file(self, path: PathLike) -> None:
        nx.write_gml(self._G, path, literal_stringizer)

    def add_triangle(self, node1: int, node2: int, node3: int) -> None:
        nodes = [node1, node2, node3]
        # FIXME (niklasdewally): check that edge exists and TRIANGLES is set to 0 forall edges.

        for i, current_node in enumerate(nodes):
            if TRIANGLES not in self._G.nodes[current_node].keys():
                self._G.nodes[current_node][TRIANGLES] = list()

            elif not isinstance(self._G.nodes[current_node][TRIANGLES], list):
                raise TypeError(
                    f"{TRIANGLES} attribute of {current_node} should be a list"
                )

            # save nodes that the current node forms a triangle with
            # this can easily be conveerted into edges downstream by doing
            # current_node -> n1, current_node -> n2
            # use a sorted ordering so that order does not matter.
            # this will make removal easier!
            triangle_nodes = sorted(x for j, x in enumerate(nodes) if j != i)
            if self._G.nodes[current_node][TRIANGLES].__contains__(triangle_nodes):
                self._on_change()

                # triangle already exists
                # FIXME (niklasdewally): more robust solution???
                return

            self._G.nodes[current_node][TRIANGLES].append(triangle_nodes)

        for u, v in itertools.combinations(nodes, 2):
            # FIXME (niklasdewally): this assumes the graph is undirected
            self._G.edges[u, v][TRIANGLES] += 1

        self._on_change()

    def copy(self) -> "Graph":
        new_g: nx.Graph = self._G.copy()
        return Graph(new_g)

    def remove_triangle(self, node1: int, node2: int, node3: int) -> None:
        # FIXME (niklasdewally): ALSO REMOVE FROM EDGES!!
        nodes = [node1, node2, node3]
        triangle_exists = False
        for i, current_node in enumerate(nodes):
            try:
                triangle_nodes = sorted(x for j, x in enumerate(nodes) if j != i)
                # remove from metadata
                self._G.nodes[current_node][TRIANGLES].remove(triangle_nodes)

                triangle_exists = True
            except KeyError:
                pass

        if not triangle_exists:
            # triangle never existed, so dont remove the edges
            return

        for u, v in itertools.permutations(nodes, 2):
            try:
                self._G.remove_edge(u, v)
            except nx.NetworkXError:
                pass

        self._on_change()

    def get_triangles(self, node: int) -> list[list[int]]:
        try:
            if not isinstance(self._G.nodes[node][TRIANGLES], list):
                raise TypeError(f"{TRIANGLES} attribute of {node} should be a list")

            return self._G.nodes[node][TRIANGLES]
        except KeyError:
            return list()

    def get_triangles_dictionary(self) -> dict[int, list[list[int]]]:
        if not self.has_mined_triangles():
            raise ValueError(
                "This graph has no mined triangles - generate them using mine_triangles()"
            )
        return nx.get_node_attributes(self._G, TRIANGLES)

    def get_triangles_list(self) -> list[list[int]]:
        triangles = []
        for k, v in self.get_triangles_dictionary().items():
            for triangle in v:
                triangle.append(k)
                triangles.append(triangle)

        return triangles

    def get_edge_triangle_counts(self) -> np.ndarray:
        # TODO (niklasdewally): docstring
        triangle_counts = nx.get_edge_attributes(self._G, TRIANGLES)

        lst = []
        for (u, v), count in triangle_counts.items():
            lst.append([u, v, count])

        return np.array(lst)

    def edge_subgraph(self, edges: list[tuple[int, int]]) -> "Graph":
        """
        Create a subgraph based on edges.

        Nodes are reindexed from 0 - old ids are stored in the "old_id" node attribute
        of the underlying networkx graph.
        """
        new_g: nx.Graph = (
            self._G.edge_subgraph(edges).to_undirected().to_directed().copy()
        )

        # delete triangles that no longer exist
        for nid in new_g:
            triangles = new_g.nodes[nid][TRIANGLES]
            for i, triangle in enumerate(list(triangles)):
                if not all(x in new_g.nodes for x in triangle):
                    triangles.remove(triangle)
            new_g.nodes[nid][TRIANGLES] = triangles

        # reindex nodes and triangles to be sequential
        new_g = nx.convert_node_labels_to_integers(new_g, label_attribute="old_id")

        # make map of old ids -> new ids
        # then, use this to relabel ids of triangles

        mapping = {v: k for k, v in nx.get_node_attributes(new_g, "old_id").items()}
        for nid in new_g:
            triangles = new_g.nodes[nid][TRIANGLES]
            new_triangles = []
            for triangle in triangles:
                new_triangles.append([mapping[x] for x in triangle])

            new_g.nodes[nid][TRIANGLES] = new_triangles

        return Graph(new_g)

    # pyre-ignore[2]
    def node_subgraph(self, nodes: list[int], device="cpu") -> "Graph":
        """
        Create a subgraph based on nodes.

        Nodes are reindexed from 0 - old ids are stored in the "old_nid" node attribute
        of the underlying networkx graph.

        Isolate nodes are also removed.
        """
        new_dgl_g: dgl.DGLGraph = dgl.node_subgraph(self.as_dgl_graph(device), nodes)
        isolated_nodes = (
            ((new_dgl_g.in_degrees() == 0) & (new_dgl_g.out_degrees() == 0))
            .nonzero()
            .squeeze(1)
        )
        new_dgl_g.remove_nodes(isolated_nodes)

        # make map of old ids -> new ids
        # then, use this to relabel ids of triangles

        old_nids: torch.Tensor = new_dgl_g.ndata[dgl.NID]
        old_to_new_nid = {old_nids[i].item(): i for i in range(old_nids.shape[0])}

        # convert old ids to new ids in triangles, and remove any that do not exist anymore
        new_triangles = dict()
        for new_nid in range(new_dgl_g.num_nodes()):
            # pyre-ignore[9]:
            old_nid: int = old_nids[new_nid].item()

            new_triangles[new_nid] = list()
            triangles = self.get_triangles(old_nid)

            for triangle in triangles:
                new_triangle = [old_to_new_nid.get(x) for x in triangle]
                triangle_exists_in_new_g = all(x is not None for x in new_triangle)

                if triangle_exists_in_new_g:
                    new_triangles[new_nid].append(new_triangle)

        new_g: "Graph" = Graph.from_dgl_graph(new_dgl_g, node_attrs=[dgl.NID])

        # add triangles to new_g
        for new_nid, triangles in new_triangles.items():
            for triangle in triangles:
                new_g.add_triangle(new_nid, triangle[0], triangle[1])

        return new_g

    def mine_triangles(self) -> None:
        self._init_triangles_store()
        # https://stackoverflow.com/questions/1705824/finding-cycle-of-3-nodes-or-triangles-in-a-graph
        all_cliques: list[int] = nx.enumerate_all_cliques(self._G)

        # pyre-ignore[6]:
        triangles: list[int] = [x for x in all_cliques if len(x) == 3]

        for triangle in triangles:
            # pyre-ignore[23]:
            n1, n2, n3 = triangle
            self.add_triangle(n1, n2, n3)

        self._on_change()

    def has_mined_triangles(self) -> bool:
        for node in self._G.nodes:
            if TRIANGLES not in self._G.nodes[node].keys():
                return False
            if not isinstance(self._G.nodes[node][TRIANGLES], list):
                return False

        return True

    def sample_triangles(self, n: int) -> list[list[int]]:
        """
        Sample n triangles from the graph.
        If the graph contains less than n triangles, this returns all the triangles.
        """
        triangles = self.get_triangles_list()

        return sample(triangles, min(len(triangles), n))

    def sample_negative_triangles(self, n: int) -> list[list[int]]:
        """
        Sample n triangles that do not exist in the graph.

        If there are less than n negative triangles, all the negative triangles found will be returned.


        """

        MAX_K = 4

        if not self.has_mined_triangles():
            raise ValueError(
                "This graph has no mined triangles - generate them using mine_triangles()"
            )

        triangle_counts = self.get_edge_triangle_counts()
        triangle_counts = triangle_counts[np.argsort(triangle_counts[:, 2])]

        negative_triangles = []

        for edge_idx in range(triangle_counts.shape[0]):
            if n == 0:
                return negative_triangles

            u = triangle_counts[edge_idx, 0]
            v = triangle_counts[edge_idx, 1]

            # aim to find close nodes to u that are not directly connected
            k_hop_neighbors = list()
            for distance, neighbors in enumerate(nx.bfs_layers(self._G, u)):
                if distance > 1:
                    k_hop_neighbors.extend(neighbors)

                if distance == MAX_K:
                    break

            if len(k_hop_neighbors) > 0:
                w = choice(k_hop_neighbors)
                negative_triangles.append([int(u), int(v), int(w)])
                n -= 1

        return negative_triangles

    def _generate_triads_by_type(self) -> None:
        self.triads_by_type = nx.triads_by_type(self._G.to_directed())

    def _init_triangles_store(self) -> None:
        for node in self._G.nodes:
            self._G.nodes[node][TRIANGLES] = list()

        for u, v in self._G.edges:
            self._G.edges[u, v][TRIANGLES] = 0

        self._on_change()

    def _on_change(self) -> None:
        # invalidate caches
        self.triads_by_type = None
        # pyre-ignore[8]
        self._dgl_g = None
