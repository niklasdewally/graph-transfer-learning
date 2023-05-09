import datetime
import itertools
from random import randint, sample, shuffle

import dgl
import graphtransferlearning as gtl
import networkx as nx
import numpy as np
import torch
import wandb

from dgl.data import CoraGraphDataset, PubmedGraphDataset
from graphtransferlearning.features import degree_bucketing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

BATCHSIZE = 50
LR = 0.01 
HIDDEN_LAYERS = 128
PATIENCE = 10
MIN_DELTA = 0.01
EPOCHS = 100
N_RUNS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    models = ["egi","triangle"]
    ks = [1,2,3,4]

    trials = list(itertools.product(models,ks))
    shuffle(trials)

    current_date_time = datetime.datetime.now().strftime("%Y%m%dT%H%M")

    for model,k in trials:
        for i in range(N_RUNS):
            project = "01 CORA -> Pubmed Link prediction"
            name = f"{model}-k{k}-{i}"
            entity = "sta-graph-transfer-learning"
            group = f"Run {current_date_time}"
            config = {
                "model": model,
                "k-hops": k,
                "encoder-hidden-layers": HIDDEN_LAYERS,
                "encoder-epochs": EPOCHS,
                "encoder-patience": PATIENCE,
                "encoder-min-delta": MIN_DELTA,
                "encoder-lr":LR,
                "encoder-batchsize":BATCHSIZE
            }

            with wandb.init(
                project=project, name=name, entity=entity, config=config, group=group
                ) as run:
                    do_run(k, model)


def get_edge_embedding(emb, a, b):
    return np.multiply(emb[a].detach().cpu(), emb[b].detach().cpu())


def generate_negative_edges(edges, nodes, n):
    negative_edges = []
    for i in range(n):
        u = randint(0, n)
        v = randint(0, n)
        while (
            u == v
            or (u, v) in edges
            or (v, u) in edges
            or v not in nodes
            or u not in nodes
        ):
            u = randint(0, n)
            v = randint(0, n)

        negative_edges.append((u, v))

    return negative_edges


def do_run(k, sampler):

    ##########################################################################
    #                            DATA LOADING                                #
    ##########################################################################

    # Generate small pubmed graph for few-shot learning
    pubmed = PubmedGraphDataset()[0].to_networkx()

    # only consider a small subgraph for transfer

    pubmed_nx = nx.edge_subgraph(
        pubmed, sample(list(pubmed.edges), 1000)
    ).to_undirected(reciprocal=False)
    pubmed_nx = nx.convert_node_labels_to_integers(
        pubmed_nx
    )  # renumber nodes to be sequential integers

    pubmed = dgl.from_networkx(pubmed_nx)
    pubmed = pubmed.to(device)


    # track graph properties in configuration
    
    gtl.wandb.log_network_properties(pubmed_nx,prefix="pubmed-target")

    ##########################################################################
    #            Base Case : Train directly on small pubmed graph            #
    ##########################################################################

    encoder = gtl.training.train_egi_encoder(
        pubmed,
        n_epochs=EPOCHS,
        k=k,
        lr=LR,
        n_hidden_layers=HIDDEN_LAYERS,
        batch_size=BATCHSIZE,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        sampler=sampler,
        wandb_summary_prefix="no-pretraining"
    )

    encoder = encoder.to(device)

    # the maximum degree must be the same as used in training.
    features = degree_bucketing(pubmed, HIDDEN_LAYERS)  
    features = features.to(device)

    embs = encoder(pubmed, features)

    positive_edges = list(pubmed_nx.edges(data=False))
    nodes = list(pubmed_nx.nodes(data=False))
    negative_edges = generate_negative_edges(positive_edges, nodes, len(positive_edges))

    edges = []
    values = []

    for u, v in positive_edges:
        edges.append(get_edge_embedding(embs, u, v))
        values.append(1)

    for u, v in negative_edges:
        edges.append(get_edge_embedding(embs, u, v))
        values.append(0)

    train_edges, val_edges, train_classes, val_classes = train_test_split(
        edges, values
    )
    train_edges = torch.stack(train_edges)  # list of tensors to 3d tensor
    val_edges = torch.stack(val_edges)  # list of tensors to 3d tensor

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(train_edges, train_classes)

    score = classifier.score(val_edges, val_classes)

    wandb.summary["no-pretraining-accuracy"] = score

    ##########################################################################
    #                Transfer Case : Pretrain on CORA Graph                  #
    ##########################################################################

    cora = CoraGraphDataset()[0].to(device)
    gtl.wandb.log_network_properties(cora.cpu().to_simple().to_networkx(),prefix="cora-source")

    tmp_file = "tmp_pretrain.pt"

    # train encoder for cora
    encoder = gtl.training.train_egi_encoder(
        cora,
        n_epochs=EPOCHS,
        k=k,
        lr=LR,
        n_hidden_layers=HIDDEN_LAYERS,
        batch_size=BATCHSIZE,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        sampler=sampler,
        wandb_summary_prefix="pretrain-cora",
        save_weights_to=tmp_file,
    )
   
    # CORA node features
    features = degree_bucketing(cora, HIDDEN_LAYERS)

    # the maximum degree must be the same as used in training.
    # this is usually equal to n_hidden

    features = features.to(device)

    # node embeddings for CORA
    embs = encoder(cora, features)
    embs = embs.to(device)

    # fine-tune embedder for link predictor
    cora_nx = cora.cpu().to_networkx()
    positive_edges = list(cora_nx.edges(data=False))
    nodes = list(cora_nx.nodes(data=False))
    negative_edges = generate_negative_edges(positive_edges, nodes, len(positive_edges))

    edges = []
    values = []

    for u, v in positive_edges:
        edges.append(get_edge_embedding(embs, u, v))
        values.append(1)

    for u, v in negative_edges:
        edges.append(get_edge_embedding(embs, u, v))
        values.append(0)

    train_edges, val_edges, train_classes, val_classes = train_test_split(
        edges, values
    )
    train_edges = torch.stack(train_edges)  # list of tensors to 3d tensor
    val_edges = torch.stack(val_edges)  # list of tensors to 3d tensor

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(train_edges, train_classes)

    # perform transfer learning - finetune on pubmed

    transfer_encoder = gtl.training.train_egi_encoder(
        pubmed,
        n_epochs=EPOCHS,
        k=k,
        lr=LR,
        n_hidden_layers=HIDDEN_LAYERS,
        batch_size=BATCHSIZE,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        sampler=sampler,
        pre_train=tmp_file,
        wandb_summary_prefix="finetune",
    )

    features = degree_bucketing(pubmed, HIDDEN_LAYERS).to(device)
    embs = transfer_encoder(pubmed, features).to(device)

    # fine-tune embedder for link predictor
    positive_edges = list(pubmed_nx.edges(data=False))
    nodes = list(pubmed_nx.nodes(data=False))
    negative_edges = generate_negative_edges(positive_edges, nodes, len(positive_edges))

    edges = []
    values = []

    for u, v in positive_edges:
        edges.append(get_edge_embedding(embs, u, v))
        values.append(1)

    for u, v in negative_edges:
        edges.append(get_edge_embedding(embs, u, v))
        values.append(0)

    train_edges, val_edges, train_classes, val_classes = train_test_split(
        edges, values
    )

    train_edges = torch.stack(train_edges)  # list of tensors to 3d tensor
    val_edges = torch.stack(val_edges)  # list of tensors to 3d tensor

    classifier = classifier.partial_fit(train_edges, train_classes)

    score = classifier.score(val_edges, val_classes)
    wandb.summary["pretrain-accuracy"] = score

    percentage_difference = (
        wandb.summary["pretrain-accuracy"]
        - wandb.summary["no-pretraining-accuracy"]
    ) / wandb.summary["no-pretraining-accuracy"]

    wandb.summary["% Difference"] = percentage_difference * 100


if __name__ == "__main__":
    main()
