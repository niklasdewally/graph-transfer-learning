# coding: utf-8
from gtl.coauthor import load_coauthor_npz
import networkx as nx
import matplotlib.pyplot as plt

coauthor_data = load_coauthor_npz("data/raw/coauthor-cs.npz")
graph, feats, labels = coauthor_data
graph.mine_triangles()
a = graph.sample_negative_triangles(100000)

plt.ion()

plt.figure()

b = zip(range(1, 51), a[0:50])

nodes = []

for label, ns in b:
    for n in ns:
        graph._G.nodes[n]["col"] = label
    nodes.extend(ns)

g = graph.as_nx_graph().subgraph(nodes)

nx.draw_spring(
    g,
    node_color=list(nx.get_node_attributes(g, "col").values()),
    labels=nx.get_node_attributes(g, "col"),
)


plt.show(block=True)
