"""
Functions for interactive use (in Jupyter notebooks / Ipython)
"""

import networkx as nx
import matplotlib.pyplot as plt
from statistics import *


def draw_twopart(G):
    fig = plt.figure()
    pos = nx.spring_layout(G)
    colour_map = [
        "green" if (origin == 1 or origin == "core") else "red"
        for origin in nx.get_node_attributes(G, "origin").values()
    ]

    nx.draw_networkx_nodes(G, pos=pos, node_size=50, node_color=colour_map)
    nx.draw_networkx_edges(G, pos=pos, alpha=0.3)
