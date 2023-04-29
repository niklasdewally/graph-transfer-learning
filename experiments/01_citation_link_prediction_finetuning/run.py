import argparse
import tempfile
from random import randint,sample
import warnings

import graphtransferlearning as gtl
from graphtransferlearning.features import degree_bucketing

import datetime

import dgl
from dgl.data import CoraGraphDataset,PubmedGraphDataset

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

import networkx as nx
import numpy as np


HIDDEN_LAYERS = 128

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



def main(log_dir,sampler):
    writer = SummaryWriter(log_dir)
    layout = {'':{
                'Base encoder-loss':['Multiline',['base/training-loss','base/validation-loss']],
                'Transfer encoder-loss':['Multiline',['transfer/training-loss','transfer/validation-loss']]
                }}
                    
    writer.add_custom_scalars(layout)
    results = dict()
    hparams = { "encoder-sampler":sampler }

    # Generate small pubmed graph for few-shot learning
    pubmed = PubmedGraphDataset()[0].to_networkx()

    # only consider a small subgraph for transfer

    pubmed_nx = nx.edge_subgraph(pubmed,sample(list(pubmed.edges),1000)).to_undirected(reciprocal=False)
    pubmed_nx = nx.convert_node_labels_to_integers(pubmed_nx) # renumber nodes to be sequential integers

    pubmed = dgl.from_networkx(pubmed_nx)
    if (torch.cuda.is_available()):
        pubmed = pubmed.to('cuda:0')



    # Base case: pubmed direct training
    print("Contro case: train model on pubmed without transfer")
    encoder = gtl.training.train_egi_encoder(pubmed,gpu=0,kfolds=5,sampler=sampler,
                                             n_hidden_layers=HIDDEN_LAYERS,
                                             writer=writer,tb_prefix='base')

    features = degree_bucketing(pubmed,HIDDEN_LAYERS) # the maximum degree must be the same as used in training.

    if (torch.cuda.is_available()):
        features = features.cuda()

    embs = encoder(pubmed,features)

    # fine-tune link predictor
    positive_edges = list(pubmed_nx.edges(data=False))
    nodes = list(pubmed_nx.nodes(data=False))
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

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(train_edges,train_classes)

    print(f"The link-predictor has an accuracy score of \
    {classifier.score(test_edges,test_classes)}")

    score = classifier.score(test_edges,test_classes)
    results.update({"hp/base-accuracy":score})


    # Transfer case: cora pre-train, pubmed fine-tuning
    print("Train on cora, finetune on pubmed")
    cora = CoraGraphDataset()[0]

    if (torch.cuda.is_available()):
        cora = cora.to('cuda:0')
            
    tmp_file = 'model.pickle'


    # train encoder for cora
    print("Training CORA encoder")
    encoder = gtl.training.train_egi_encoder(cora,gpu=0,kfolds=10,save_weights_to=tmp_file,
                                             sampler=sampler,
                                             n_hidden_layers=HIDDEN_LAYERS,
                                             writer=writer,
                                             tb_prefix='transfer')
        

    print("Training CORA link predictor")
    # CORA node features
    features = degree_bucketing(cora,HIDDEN_LAYERS) # the maximum degree must be the same as used in training.
                                              # this is usually equal to n_hidden

    torch.cuda.set_device(torch.device('cuda:0'))

    features = features.cuda()

    # node embeddings for CORA
    embs = encoder(cora,features)

    embs = embs.cuda()





    # perform transfer learning

    print("Fine-tuning the encoder on PubMed")
    transfer_encoder = gtl.training.train_egi_encoder(pubmed,gpu=0,pre_train=tmp_file,sampler=sampler,n_hidden_layers=HIDDEN_LAYERS)

    features = degree_bucketing(pubmed,HIDDEN_LAYERS) # the maximum degree must be the same as used in training.
    if (torch.cuda.is_available()):
        features = features.cuda()

    #
    embs = transfer_encoder(pubmed,features)

    # fine-tune embedder for link predictor
    positive_edges = list(pubmed_nx.edges(data=False))
    nodes = list(pubmed_nx.nodes(data=False))
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

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(train_edges,train_classes)
    print(f"The transferred link predictor has an accuracy score of \
    {classifier.score(test_edges,test_classes)}")


    score = classifier.score(test_edges,test_classes)
    results.update({"hp/transfer-accuracy":score})
    difference = results["hp/transfer-accuracy"] - results["hp/base-accuracy"]
    results.update({"hp/difference":difference})
    writer.add_hparams(hparams,results)



    return results


if __name__ == "__main__":
    n = 10

    current_date_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    SAMPLERS = ["egi","triangle"]
    for sampler in SAMPLERS:
        for i in range(n):
           log_dir = f"./runs/{current_date_time}/{sampler}/{i}"
           print(f"Running experiment for {sampler} sampler.")
           print(f"Saving results in {log_dir}")

           main(log_dir,sampler)

