import itertools
from random import sample
from copy import deepcopy
from .typing import PathLike
from typing import Optional
from collections.abc import Mapping
import torch

from IPython import embed
import dgl
import networkx as nx
from networkx.readwrite.gml import literal_destringizer, literal_stringizer

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

    @staticmethod
    def from_gml_file(path: PathLike) -> "Graph":
        g = nx.read_gml(path, destringizer=literal_destringizer)
        return Graph(g)

    @staticmethod
    def from_dgl_graph(
        g: dgl.DGLGraph,
        node_attrs: Optional[Mapping] = None,
        edge_attrs: Optional[Mapping] = None,
    ) -> "Graph":
        nx_g: nx.Graph = dgl.to_networkx(
            g.cpu(), node_attrs, edge_attrs
        ).to_undirected()
        return Graph(nx_g)

    def as_nx_graph(self) -> nx.Graph:
        """
        Return the graph, in networkx format.
        """
        return self._G

    def as_dgl_graph(
        self, device: torch.device, **kwargs: Optional[Mapping]
    ) -> dgl.DGLGraph:
        return dgl.from_networkx(self._G, device=device, **kwargs)

    def to_gml_file(self, path: PathLike) -> None:
        nx.write_gml(self._G, path, literal_stringizer)

    def add_triangle(self, node1: int, node2: int, node3: int) -> None:
        nodes = [node1, node2, node3]

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
            self._G.nodes[current_node][TRIANGLES].append(triangle_nodes)

            self._on_change()

    def copy(self) -> "Graph":
        new_g: nx.Graph = self._G.copy()
        return Graph(new_g)

    def remove_triangle(self, node1: int, node2: int, node3: int) -> None:
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
            #triangle never existed, so dont remove the edges
            return 

        for u,v in itertools.permutations(nodes,2):
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

    def edge_subgraph(self,edges: list[tuple[int,int]]) -> "Graph":
        """
        Create a subgraph based on edges.

        Nodes are reindexed from 0 - old ids are stored in the "old_id" node attribute
        of the underlying networkx graph.
        """
        new_g : nx.Graph = self._G.edge_subgraph(edges).to_undirected().to_directed().copy()
        
        # delete triangles that no longer exist
        for nid in new_g:
            triangles = new_g.nodes[nid][TRIANGLES]
            for i,triangle in enumerate(list(triangles)):
                if not all(x in new_g.nodes for x in triangle):
                    triangles.remove(triangle)
            new_g.nodes[nid][TRIANGLES] = triangles


        # reindex nodes and triangles to be sequential
        new_g = nx.convert_node_labels_to_integers(new_g,label_attribute="old_id")

        # make map of old ids -> new ids
        # then, use this to relabel ids of triangles

        mapping = {v:k for k,v in nx.get_node_attributes(new_g,"old_id").items()}
        for nid in new_g:
            triangles = new_g.nodes[nid][TRIANGLES]
            new_triangles = []
            for triangle in triangles:
                new_triangles.append([mapping[x] for x in triangle])

            new_g.nodes[nid][TRIANGLES] = new_triangles

        return Graph(new_g)


    def node_subgraph(self, nodes:list[int]) -> "Graph":
        """
        Create a subgraph based on nodes.

        Nodes are reindexed from 0 - old ids are stored in the "old_id" node attribute
        of the underlying networkx graph.
        """
        new_g : nx.Graph = self._G.subgraph(nodes).to_undirected().to_directed().copy()
        
        # delete triangles that no longer exist
        for nid in new_g:
            triangles = new_g.nodes[nid][TRIANGLES]
            for i,triangle in enumerate(list(triangles)):
                if not all(x in new_g.nodes for x in triangle):
                    triangles.remove(triangle)
            new_g.nodes[nid][TRIANGLES] = triangles


        # reindex nodes and triangles to be sequential
        new_g = nx.convert_node_labels_to_integers(new_g,label_attribute="old_id")

        # make map of old ids -> new ids
        # then, use this to relabel ids of triangles

        mapping = {v:k for k,v in nx.get_node_attributes(new_g,"old_id").items()}
        for nid in new_g:
            triangles = new_g.nodes[nid][TRIANGLES]
            new_triangles = []
            for triangle in triangles:
                new_triangles.append([mapping[x] for x in triangle])

            new_g.nodes[nid][TRIANGLES] = new_triangles

        return Graph(new_g)

    def mine_triangles(self) -> None:
        self._reset_triangles()
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

        For any three nodes A B C, these include the triads:
            * A, B, C
            * A, B <-> C
            * A <-> B <-> C

        If there are less than n negative triangles, all the negative triangles found will be returned.


        This function caches triad information once generated, so may be slow on first run, but faster subsequent times.
        """

        if self.triads_by_type is None:
            self._generate_triads_by_type()

        # pyre-ignore[16]
        # A B C
        disconnected_triads = [list(g) for g in self.triads_by_type["033"]]

        # A -> B -> C
        open_triangles = [list(g) for g in self.triads_by_type["201"]]

        # A -> B , C
        pair_and_node = [list(g) for g in self.triads_by_type["102"]]

        all_non_triangles: list[list[int]] = list(
            itertools.chain.from_iterable(
                [disconnected_triads, open_triangles, pair_and_node]
            )
        )

        return sample(all_non_triangles, min(len(all_non_triangles), n))

    def _generate_triads_by_type(self) -> None:
        self.triads_by_type = nx.triads_by_type(self._G.to_directed())

    def _reset_triangles(self) -> None:
        for node in self._G.nodes:
            self._G.nodes[node][TRIANGLES] = list()

        self._on_change()

    def _on_change(self) -> None:
        # invalidate caches
        self.triads_by_type = None
