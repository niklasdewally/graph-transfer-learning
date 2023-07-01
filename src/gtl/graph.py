from copy import deepcopy
from os import PathLike
from typing import NoReturn

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

    def __init__(self, g: nx.Graph):
        self._G = deepcopy(g)

    @staticmethod
    def from_gml_file(path: PathLike):
        g = nx.read_gml(path, destringizer=literal_destringizer)
        return Graph(g)

    @staticmethod
    def from_dgl_graph(g: dgl.DGLGraph, node_attrs=None, edge_attrs=None):
        nx_g: nx.Graph = dgl.to_networkx(g.cpu(), node_attrs, edge_attrs).to_undirected()
        return Graph(nx_g)

    def as_nx_graph(self) -> nx.Graph:
        """
        Return the graph, in networkx format.
        """
        return self._G

    def as_dgl_graph(self, device, **kwargs) -> dgl.DGLGraph:
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

    def remove_triangle(self, node1, node2, node3) -> None:
        nodes = [node1, node2, node3]
        for i, current_node in enumerate(nodes):
            try:
                triangle_nodes = sorted(x for j, x in enumerate(nodes) if j != i)
                self._G.nodes[current_node][TRIANGLES].remove(triangle_nodes)
            except KeyError:
                pass

    def get_triangles(self, node) -> list[list]:
        try:
            if not isinstance(self._G.nodes[node][TRIANGLES], list):
                raise TypeError(f"{TRIANGLES} attribute of {node} should be a list")

            return self._G.nodes[node][TRIANGLES]
        except KeyError:
            return list()

    def get_triangles_dictionary(self) -> dict[int,list[list]]:
        if not self.has_mined_triangles():
            raise ValueError("This graph has no mined triangles - generate them using mine_triangles()")
        return nx.get_node_attributes(self._G, TRIANGLES)

    def mine_triangles(self) -> None:
        self._reset_triangles()
        # https://stackoverflow.com/questions/1705824/finding-cycle-of-3-nodes-or-triangles-in-a-graph
        all_cliques: list[int] = nx.enumerate_all_cliques(self._G)
        triangles: list[int] = [x for x in all_cliques if len(x) == 3]

        for triangle in triangles:
            n1, n2, n3 = triangle
            self.add_triangle(n1, n2, n3)

    def has_mined_triangles(self) -> bool:
        for node in self._G.nodes:
            if TRIANGLES not in self._G.nodes[node].keys():
                return False
            if not isinstance(self._G.nodes[node][TRIANGLES], list):
                return False

        return True

    def _reset_triangles(self) -> None:
        for node in self._G.nodes:
            self._G.nodes[node][TRIANGLES] = list()
