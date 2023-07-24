# pyre-ignore-all-errors

"""
Produce 2d plots of TSNE embeddings for the coauthor datasets, based on input features.
Produces the following files in the results/vis folder:
    cs_small_split.png,
    cs_full.png,
    phys_small_split.png,
    phys_full.png
"""
from gtl.coauthor import load_coauthor_npz
from gtl.splits import CoauthorNodeClassificationSplit
import networkx as nx

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import pathlib

SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.parent.resolve()
OUT_DIR = PROJECT_DIR / "results" / "vis"

def main():

    OUT_DIR.mkdir(exist_ok =True,parents=True)

    ####################
    #        CS        #
    ####################

    coauthor_data = load_coauthor_npz(PROJECT_DIR / "data" / "raw" / "coauthor-cs.npz")
    graph,feats,labels = coauthor_data
    splits = CoauthorNodeClassificationSplit(graph,labels,"cpu")

    embeds = TSNE().fit_transform(np.array(feats))

    plot("cs_small_split.png",embeds[splits.small_idxs.tolist()],labels[splits.small_idxs.tolist()],"CS Co-Author input features (small split)")
    plot("cs_full.png",embeds,labels,"CS Co-Author input features")

    #embeds = np.array(nx.spectral_layout(graph._G).values())
    #embeds_small = embeds[splits.small_idxs.tolist()]

    #plot("cs_small_split_spectral.png",embeds_small,labels[splits.small_idxs.tolist()],"CS Co-Author spectral embedding (small split)")
    #plot("cs_full.png_spectral",embeds,labels,"CS Co-Author spectral embedding")

    save_summary_table("cs", splits.small_g._G,graph._G)

    ######################
    #        PHYS        #
    ######################

    coauthor_data = load_coauthor_npz(PROJECT_DIR / "data" / "raw" / "coauthor-phy.npz")
    graph,feats,labels = coauthor_data
    splits = CoauthorNodeClassificationSplit(graph,labels,"cpu")

    embeds = TSNE().fit_transform(np.array(feats))

    plot("phys_small_split.png",embeds[splits.small_idxs.tolist()],labels[splits.small_idxs.tolist()],"Phys Co-Author input features (small split)")
    plot("phys_full.png",embeds,labels,"Phys Co-Author input features")


    #embeds = np.array(nx.spectral_layout(graph._G).values())
    #embeds_small = embeds[splits.small_idxs.tolist()]

    #plot("phys_small_split_spectral.png",embeds_small,labels[splits.small_idxs.tolist()],"Phys Co-Author spectral embedding (small split)")
    #plot("phys_full.png_spectral",embeds,labels,"Phys Co-Author spectral embedding")


    save_summary_table("phys", splits.small_g._G,graph._G)


def plot(filename, embeds,labels,title):

    plt.figure()


    plt.scatter(x=embeds[:,0],y=embeds[:,1],c=labels)
    plt.title(title)

    plt.savefig(OUT_DIR / filename)


def save_summary_table(graph_name,small_g,full_g):
    # Metrics as used in https://arxiv.org/abs/1506.02449

    small_g_data =  dict()
    full_g_data = dict()

    small_g = nx.Graph(small_g).to_undirected()
    full_g = nx.Graph(full_g).to_undirected()
    small_g_data['nodes'] =  nx.number_of_nodes(small_g)
    small_g_data['edges'] =  nx.number_of_edges(small_g)
    small_g_data['density'] =  nx.density(small_g)
    small_g_data['clustering'] =  nx.average_clustering(small_g)
    small_g_data['avg_degree'] =  sum([ x for _,x in small_g.degree()])/small_g.number_of_nodes()

    full_g_data['nodes'] =  nx.number_of_nodes(full_g)
    full_g_data['edges'] =  nx.number_of_edges(full_g)
    full_g_data['density'] =  nx.density(full_g)
    full_g_data['clustering'] =  nx.average_clustering(full_g)
    full_g_data['avg_degree'] =  sum([ x for _,x in full_g.degree()])/full_g.number_of_nodes()
    
    with open(OUT_DIR / f"{graph_name}-summary.txt","w") as f:
        for key in small_g_data.keys():
            write_fixed_row(f,key,small_g_data[key],full_g_data[key])


def write_fixed_row(f,name,val1,val2):
    f.write(f"| {name:^20} | {val1:5.2g} | {val2:5.2g} |\n")

if __name__ == '__main__':
    main()
