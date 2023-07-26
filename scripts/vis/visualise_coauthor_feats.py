# pyre-ignore-all-errors

"""
Produce 2d plots of TSNE embeddings for the coauthor datasets, based on input features.
Produces the following files in the results/vis folder:
    cs_small_split.png,
    cs_full.png,
    phys_small_split.png,
    phys_full.png
"""
import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gtl.coauthor import load_coauthor_npz
from gtl.splits import CoauthorNodeClassificationSplit
from sklearn.manifold import TSNE
import seaborn as sns


SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.parent.resolve()
OUT_DIR = PROJECT_DIR / "results" / "vis"


def main(interactive=False):
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    ####################
    #        CS        #
    ####################
    if interactive:
        plt.ion()

    print("Loading CS Graph")
    coauthor_data = load_coauthor_npz(PROJECT_DIR / "data" / "raw" / "coauthor-cs.npz")
    graph, feats, labels = coauthor_data
    splits = CoauthorNodeClassificationSplit(graph, labels, "cpu")

    labels = labels.numpy()
    feats = feats.numpy()
    small_labels = labels[splits.small_idxs.tolist()]

    print("Writing summary")
    save_summary_table("cs", splits.small_g._G, small_labels, graph._G, labels)

    print("Plotting mixing matrix")

    plot_mixing(
        "cs_mixing_full.png", graph._G, labels, "Mixing matrix for full CS graph"
    )

    plot_mixing(
        "cs_mixing_small.png",
        splits.small_g._G,
        small_labels,
        "Mixing matrix for sampled CS subgraph",
    )

    print("Embedding CS Features")
    embeds = TSNE().fit_transform(feats)

    # explicitly pass in max value for consistant colours
    vmax = np.max(labels)
    print("Plotting CS Embeddings")
    plot_embs(
        "cs_small_split.png",
        embeds[splits.small_idxs.tolist()],
        small_labels,
        "CS Co-Author input features (small split)",
        vmax,
    )

    plot_embs("cs_full.png", embeds, labels, "CS Co-Author input features", vmax)

    print("Plotting CS spectral embeddings")

    plot_spectral(
        "cs_spectral.png", graph._G, labels, "Coauthor CS (Spectral Embedding)", vmax
    )
    plot_spectral(
        "cs_spectral_small.png",
        splits.small_g._G,
        small_labels,
        "Coauthor CS Small Subgraph (Spectral Embedding)",
        vmax,
    )
    if interactive:
        plt.show(block=True)


def plot_embs(filename, embeds, labels, title, vmax):
    plt.figure()

    plt.scatter(x=embeds[:, 0], y=embeds[:, 1], vmin=0, vmax=vmax, c=labels)
    plt.title(title)

    plt.savefig(OUT_DIR / filename)


def plot_spectral(filename, g, labels, title, vmax):
    plt.figure()
    pos = nx.spectral_layout(g, scale=0.4)
    nx.draw_networkx(
        g,
        pos=pos,
        vmin=0,
        vmax=vmax,
        with_labels=False,
        node_color=labels,
        width=0,
        node_size=10,
    )
    plt.title(title)
    plt.savefig(OUT_DIR / filename)


def plot_mixing(
    filename,
    g,
    labels,
    title,
):
    plt.figure()
    nx.set_node_attributes(
        g, {i: labels[i] for i in range(0, labels.shape[0])}, "labels"
    )

    mixing = nx.attribute_mixing_matrix(
        g, "labels", mapping={i: i for i in range(0, np.max(labels))}, normalized=True
    )

    sns.heatmap(
        mixing,
        annot=True,
        linewidths=0.5,
        annot_kws={"fontsize": "xx-small"},
        robust=True,
    )

    plt.title(title)
    plt.savefig(OUT_DIR / filename)


def save_summary_table(graph_name, small_g, small_labels, full_g, full_labels):
    # Metrics as used in https://arxiv.org/abs/1506.02449

    small_g_data = dict()
    full_g_data = dict()

    small_g = nx.Graph(small_g).to_undirected()
    full_g = nx.Graph(full_g).to_undirected()
    small_g_data["nodes"] = nx.number_of_nodes(small_g)
    small_g_data["edges"] = nx.number_of_edges(small_g)
    small_g_data["density"] = nx.density(small_g)
    small_g_data["clustering"] = nx.average_clustering(small_g)
    small_g_data["avg_degree"] = (
        sum([x for _, x in small_g.degree()]) / small_g.number_of_nodes()
    )

    full_g_data["nodes"] = nx.number_of_nodes(full_g)
    full_g_data["edges"] = nx.number_of_edges(full_g)
    full_g_data["density"] = nx.density(full_g)
    full_g_data["clustering"] = nx.average_clustering(full_g)
    full_g_data["avg_degree"] = (
        sum([x for _, x in full_g.degree()]) / full_g.number_of_nodes()
    )

    with open(OUT_DIR / f"{graph_name}-summary.txt", "w") as f:
        write_title(f)

        for key in small_g_data.keys():
            write_fixed_row(f, key, small_g_data[key], full_g_data[key])

        for label in range(0, np.max(full_labels)):
            small_n_labels = np.count_nonzero(small_labels == label)
            full_n_labels = np.count_nonzero(full_labels == label)

            write_fixed_row(f, f"# label {label}", small_n_labels, full_n_labels)

        for label in range(0, np.max(full_labels)):
            small_n_labels = np.count_nonzero(small_labels == label)
            small_percent_is_label = 100 * small_n_labels // small_g_data["nodes"]

            full_n_labels = np.count_nonzero(full_labels == label)
            full_percent_is_label = 100 * full_n_labels // full_g_data["nodes"]

            write_fixed_row(
                f, f"% label {label}", small_percent_is_label, full_percent_is_label
            )

        for label in range(0, np.max(full_labels)):
            full_mod = -1
            small_mod = -1

            full_nodes_with_label = set((full_labels == label).nonzero()[0].tolist())
            full_nodes_without_label = set((full_labels != label).nonzero()[0].tolist())

            partitions = [full_nodes_with_label, full_nodes_without_label]

            if len(full_nodes_with_label) != 0:
                full_mod = nx.community.modularity(full_g, partitions)

            small_nodes_with_label = set((small_labels == label).nonzero()[0].tolist())
            small_nodes_without_label = set(
                (small_labels != label).nonzero()[0].tolist()
            )

            partitions = [small_nodes_with_label, small_nodes_without_label]

            if len(small_nodes_with_label) != 0:
                small_mod = nx.community.modularity(small_g, partitions)

            write_fixed_row(f, f"modularity : {label}", small_mod, full_mod)


def write_title(f):
    col0 = ""
    col1 = "Small"
    col2 = "Big"
    f.write("+----------------------+---------+---------+\n")
    f.write(f"| {col0:^20} | {col1:^7} | {col2: ^7} |\n")
    f.write("+----------------------+---------+---------+\n")


def write_fixed_row(f, name, val1, val2):
    f.write(f"| {name:^20} | {val1:^7,.2g} | {val2:^7,.2g} |\n")


if __name__ == "__main__":
    main()
