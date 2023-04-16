import argparse
import tempfile
from random import randint,sample
import warnings

import graphtransferlearning as gtl
from graphtransferlearning.features import degree_bucketing

import dgl
from dgl.data import CoraGraphDataset,PubmedGraphDataset

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

import networkx as nx
import numpy as np

def get_edge_embedding(emb,a,b):
    return np.multiply(emb[a].detach().cpu(),emb[b].detach().cpu())

def generate_negative_edges(edges,nodes,n):
    negative_edges = []
    for i in range(n):
        u = randint(0,n)
        v = randint(0,n)
        while u == v or (u,v) in edges or (v,u) in edges or v not in nodes or u not in nodes:
            u = randint(0,n)
            v = randint(0,n)

        negative_edges.append((u,v))
    
    return negative_edges



def main(opts):

    cora = CoraGraphDataset()[0]

    if (torch.cuda.is_available()):
        cora = cora.to('cuda:0')
            
    tmp_file = 'model.pickle'

    # train encoder for cora
    print("Training CORA encoder")
    encoder = gtl.training.train_egi_encoder(cora,gpu=0,save_weights_to=tmp_file)
        

    print("Training CORA link predictor")
    # CORA node features
    features = degree_bucketing(cora,32) # the maximum degree must be the same as used in training.
                                              # this is usually equal to n_hidden

    torch.cuda.set_device(torch.device('cuda:0'))

    features = features.cuda()

    # node embeddings for CORA
    embs = encoder(features)

    embs = embs.cuda()

    # training link prediction classifiers

    cora_nx = cora.cpu().to_networkx()

    positive_edges = list(cora_nx.edges(data=False))
    nodes = list(cora_nx.nodes(data=False))

    negative_edges = generate_negative_edges(positive_edges,nodes,len(positive_edges))

    edges = []
    values = []

    for u,v in positive_edges:
        edges.append(get_edge_embedding(embs,u,v))
        values.append(1)
        
    for u,v in negative_edges:
        edges.append(get_edge_embedding(embs,u,v))
        values.append(0)

    train_edges,test_edges,train_classes,test_classes = train_test_split(edges,values)

    train_edges =torch.stack(train_edges) # list of tensors to 3d tensor
    test_edges =torch.stack(test_edges) # list of tensors to 3d tensor

    classifier = SGDClassifier(max_iter=1000).fit(train_edges,train_classes)

    print(f"The Cora link predictor has an accuracy score of {classifier.score(test_edges,test_classes)}")


    # perform transfer learning
    transfer_g = PubmedGraphDataset()[0].to_networkx()


    # only consider a small subgraph for transfer
    transfer_g_nx = nx.edge_subgraph(transfer_g,sample(list(transfer_g.edges),1000)).to_undirected(reciprocal=False)
    transfer_g_nx = nx.convert_node_labels_to_integers(transfer_g_nx) # renumber nodes to be sequential integers

    transfer_g = dgl.from_networkx(transfer_g_nx)
    if (torch.cuda.is_available()):
        transfer_g = transfer_g.to('cuda:0')

    print("Fine-tuning the encoder on PubMed")
    transfer_encoder = gtl.training.train_egi_encoder(transfer_g,gpu=0,pre_train=tmp_file)

    features = degree_bucketing(transfer_g,32) # the maximum degree must be the same as used in training.
    if (torch.cuda.is_available()):
        features = features.cuda()

    embs = transfer_encoder(features)

    # fine-tune link predictor
    positive_edges = list(transfer_g_nx.edges(data=False))
    nodes = list(transfer_g_nx.nodes(data=False))
    negative_edges = generate_negative_edges(positive_edges,nodes,len(positive_edges)) 

    edges = []
    values = []

    for u,v in positive_edges:
        edges.append(get_edge_embedding(embs,u,v))
        values.append(1)
        
    for u,v in negative_edges:
        edges.append(get_edge_embedding(embs,u,v))
        values.append(0)

    train_edges,test_edges,train_classes,test_classes = train_test_split(edges,values)
    train_edges =torch.stack(train_edges) # list of tensors to 3d tensor
    test_edges =torch.stack(test_edges) # list of tensors to 3d tensor

    classifier2 = classifier.partial_fit(train_edges,train_classes)
    print(f"The transferred link predictor has an accuracy score of \
    {classifier2.score(test_edges,test_classes)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='EGI link prediction and finetuning between CORA and Pubmed datasets')

    #parser.add_argument("--gpu",dest="gpu",required=true,
    #   help="Which gpu to run the model on. To run on cpu, set this to -1.")

    # TODO: more arguments

    main(parser.parse_args())

