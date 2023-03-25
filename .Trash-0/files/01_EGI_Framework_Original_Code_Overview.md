---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region toc=true -->
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-0"><span class="toc-item-num">0&nbsp;&nbsp;</span>Introduction</a></span><ul class="toc-item"><li><span><a href="#Some-Setup" data-toc-modified-id="Some-Setup-0.1"><span class="toc-item-num">0.1&nbsp;&nbsp;</span>Some Setup</a></span></li></ul></li><li><span><a href="#The-Airport-Dataset" data-toc-modified-id="The-Airport-Dataset-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>The Airport Dataset</a></span></li><li><span><a href="#Data-Preparation" data-toc-modified-id="Data-Preparation-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Preparation</a></span><ul class="toc-item"><li><span><a href="#Conversion-to-DGL" data-toc-modified-id="Conversion-to-DGL-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Conversion to DGL</a></span></li><li><span><a href="#Creation-of-training-and-test-sets" data-toc-modified-id="Creation-of-training-and-test-sets-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Creation of training and test sets</a></span></li></ul></li><li><span><a href="#Generation-of-k-hop-ego-graphs" data-toc-modified-id="Generation-of-k-hop-ego-graphs-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Generation of k-hop-ego-graphs</a></span><ul class="toc-item"><li><span><a href="#A-look-at-some-specific-ego-graphs" data-toc-modified-id="A-look-at-some-specific-ego-graphs-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>A look at some specific ego-graphs</a></span></li></ul></li><li><span><a href="#The-Encoder-and-Discriminator" data-toc-modified-id="The-Encoder-and-Discriminator-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>The Encoder and Discriminator</a></span></li><li><span><a href="#Training" data-toc-modified-id="Training-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Training</a></span></li><li><span><a href="#Transfer-learning" data-toc-modified-id="Transfer-learning-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Transfer learning</a></span></li></ul></div>
<!-- #endregion -->

```python
# Change this to disable plotting - speeds up execution time!
plot_graphs = True
```

<!-- #region -->
# Introduction 

The aim of this notebook is to give an overview of the EGI framework.

This notebook primarily runs through the `run_airport.py` experiment.


![Figure 2, taken from (Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization, Zhu et al, 2021)](figures/fig2.png)


The code in this notebook is taken largely from the original code, however modifications have been made for readability and compatability.

The code is archived at https://github.com/niklasdewally/EGI.
<!-- #endregion -->

## Some Setup 


Check if the GPU works.

This code should work on any CUDA 11 compatible GPU, but has been written and tested on a 3060 only.

```python
!nvcc --version
```

```python
import sys
sys.path.append("../../egi")
```

```python
# Imports
import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraphStale as DGLGraph
from dgl.data import register_data_args, load_data

from models.dgi import DGI, MultiClassifier
from  models.subgi import GNNDiscLayer,GIN
from models.utils import *

from IPython import embed
import scipy.sparse as sp
from collections import defaultdict
from torch.autograd import Variable
from tqdm.notebook import tqdm
import pickle
from collections import defaultdict
from sklearn.manifold import SpectralEmbedding
from types import SimpleNamespace
import plotly.graph_objects as go
import ipywidgets as widgets

```

# The Airport Dataset

This notebook uses the provided airport data set that links airports that fly to eachother with edges, and labels airports based on their relative popularity.

The USA is used for the training and test sets, and Brazil for validation.

<!-- #region -->
The source of this data is the paper `struc2vec: Learning Node Representations from Structural Identity (Ribeiro et al)`
    
    
> Airports will be assigned a label corresponding to their level of activity, measured in flights or people (discussed below). 
    > We consider the following datasets (collected for this study):
    >
    >Brazilian air-traffc network: Data collected from the National Civil Aviation Agency (ANAC)1 from January to December 2016.
    >   The network has 131 nodes, 1,038 edges (diameter is 5). 
    > Airport activity is measured by the total number of landings plus takeoffs in the corresponding year.
    >   
    > American air-traffic network: Data collected from the Bureau of Transportation Statistics2 from January to October, 2016.
    >   The e network has 1,190 nodes, 13,599 edges (diameter is 8). 
    >   Airport activity is measured by the total number of people that passed (arrived plus departed) the airport in the corresponding period. 
    >    
    > European air-traffic network: Data collected from the Statistical Office of the European Union (Eurostat)3 from January to November 2016. 
    > The e network has 399 nodes, 5,995 edges (diameter is 5).
    > Airport activity is measured by the total number of landings plus takeoffs in the corresponding period.
    >
    > For each airport, we assign one of four possible labels corresponding to their activity.
    > In particular, for each dataset, we use the quartiles obtained from the empirical activity distribution to split the dataset in four groups, assigning a different label for each group. 
    > Thus, label 1 is given to the 25% less active airports, and so on. 
    >Note that all classes (labels) have the same size (number of airports).
    > Moreover, classes are related more to the role played by the airport.
<!-- #endregion -->

**First, set some options for the model:**

```python
opts = SimpleNamespace(
    edge_path = "../../egi/data/usa-airports.edgelist",
    label_path="../../egi/data/labels-usa-airports.txt",
    data_src="",
    data_id="",
    gpu=0,
    model_id=2,
    dropout=0.0,
    dgi_lr=0.001,
    classifier_lr=1e-2,
    n_dgi_epochs=100,
    n_classifier_epochs=100,
    n_hidden=32,
    n_layers=2,
    weight_decay=0.,
    patience=20,
    model=True,
    self_loop=True,
    model_type=2,
    graph_type="DD"
)
```

* * * 
**A quick look at the input data:**


The edge list:

```python
!head -n 5 "../../egi/data/usa-airports.edgelist"
```

The node labels:

```python
!head -n 5 "../../egi/data/labels-usa-airports.txt"
```

---


**Now, read in the dataset as a NetworkX graph**

```python code_folding=[22]
"""
Read in a graph from a given edge list and node label list.


edge_path: A file containing edges. This must be in the form:
    <int> <int>
    <int> <int>

    where each line contains the integer IDs of the nodes on each edge.
    
    
label_path: A file containing node labels. This must be in the form:
    <int> <string>
    <int> <string>

    where each line contains the integer ID of a node, followed by its label.
    
    
    
Returns: a networkx graph, and a dictionary of labels.

"""
def read_graph(edge_path,label_path):
    g = nx.Graph()
    
    with open(edge_path) as IN:
        for line in IN:
            tmp = line.strip().split()
            g.add_edge(int(tmp[0]), int(tmp[1]))
    labels = dict()
    with open(label_path) as IN:
        IN.readline()
        for line in IN:
            tmp = line.strip().split(' ')
            labels[int(tmp[0])] = int(tmp[1])
    return g, labels
```

```python
g,labels = read_graph(opts.edge_path,opts.label_path)
print(g)
```

```python code_folding=[10]
"""
Plot and show a given airport graph. 

Colours the graph according to airport popularity.

G : The airport graph
labels: a map of node ids to labels

Returns: None
"""
def plot_airport_graph(G,labels):
    if not plot_graphs:
        return
    # adapted from https://plotly.com/python/network-graphs/
    
    # give the nodes positions
    positions = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Airport popularity',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    node_popularity = []
    for node in G.nodes():
        node_popularity.append(labels.get(node,0))
        
    node_trace.marker.color = node_popularity
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="The airports network",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.show()
```

```python
plot_airport_graph(g,labels)
```

**The aim is to direct-transfer the node labels from this graph onto another with a different topology.**

In this case, this will be the Brazil dataset.

```python code_folding=[]
validation_g,validation_labels = read_graph("../../egi/data/brazil-airports.edgelist",
                                            "../../egi/data/brazil-airports.edgelist")
plot_airport_graph(validation_g,{})
```

# Data Preparation


Remove self-loops from the graph:

```python
g.remove_edges_from(nx.selfloop_edges(g))
```

## Conversion to DGL


The graph needs to be converted to a `DGL` graph, and the labels to a `Tensor`.

```python
"""
Convert the graph from a NetworkX graph into a DGL graph.

graph: a networkx graph
labels: a dictionary mapping node IDs to labels


Returns: a tuple of:
    graph: the graph, as a DGL graph.
    labels: the labels, as a LongTensor .
    
"""
def construct_DGL(graph, labels):
    node_mapping = defaultdict(int)
    
    relabels = []
    for node in sorted(list(graph.nodes())):
        node_mapping[node] = len(node_mapping)
        relabels.append(labels[node])

    assert len(node_mapping) == len(labels)
    
    new_g = DGLGraph()
    new_g.add_nodes(len(node_mapping))
    
    for i in range(len(node_mapping)):
        new_g.add_edge(i, i)
    for edge in graph.edges():
        new_g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
        new_g.add_edge(node_mapping[edge[1]], node_mapping[edge[0]])
        
    # convert labels to tensor
    relabels = torch.LongTensor(relabels)
    return new_g, relabels


```

```python
g, labels = construct_DGL(g, labels)
```

## Creation of training and test sets

```python
"""
Partition the labels into training and test sets.

labels: a LongTensor of labels to partition into training and test sets.

valid_mask: ???

train_ratio: the proportion of data to use as training data.
    Default: 0.8
    
Returns: a tuple containing a training mask and a test mask. These are both BoolTensors.
"""
def createTraining(labels, valid_mask = None, train_ratio=0.8):
 
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)
    test_mask = torch.ones(labels.shape, dtype=torch.bool)
    
    num_train = int(labels.shape[0] * train_ratio)
    all_node_index = list(range(labels.shape[0]))
    np.random.shuffle(all_node_index)
    #for i in range(len(idx) * train_ratio):
    # embed()
    train_mask[all_node_index[:num_train]] = 1
    test_mask[all_node_index[:num_train]] = 0
    if valid_mask is not None:
        train_mask *= valid_mask
        test_mask *= valid_mask
        
    return torch.BoolTensor(train_mask), torch.BoolTensor(test_mask)

```

```python
train_mask, test_mask = createTraining(labels)
```

# Generation of k-hop-ego-graphs


The set of all sampled k-hop-ego-graphs is represented as a `NodeFlow`.

In a `NodeFlow`, the set of edges between layers is known as a `block`.

Layers contain the nodes reachable from the seed nodes after a certain number of hops.

What `opt` calls `n_layers` is the `k` number used in the model.


![The NodeFlow object (source https://github.com/dmlc/dgl/issues/368)](figures/nodeflow.png)


* * *


The following code can be used to sample k-hop ego-graphs from the graph.
For visualisation purposes, consider a small k.

```python
k = opt.n_layers
```

```python
g.readonly() # A readonly DGL graph is required for sampling.


# https://docs.dgl.ai/en/0.2.x/api/python/sampler.html
sampler = dgl.contrib.sampling.NeighborSampler(g, 256, 5,
                                               neighbor_type='in', num_workers=1,
                                               num_hops=k, shuffle=True)    
```

## A look at some specific ego-graphs


First, get a list of all edges in the ego graph for some `start` node, alongside their block number:

```python
def get_edges_from_flow(node_flow,start,k):
    edges = []
    if k==0:
        return None
    
    for next_node in node_flow.successors(start):
        next_node = next_node.item()
        edges += [[k,start,next_node]]
        a = get_edges_from_flow(node_flow,next_node,k-1)
        if a is not None:
            edges += a
    
    return edges

```

```python
node_flow = sampler.fetch(1)[0]
```

```python
get_edges_from_flow(node_flow,1,k)[0:5]
```

Now, visualise the ego-graphs:

```python
"""
Plot and show a given ego-graph. 

Colours the graph according to n-hops from the centre.

nf: The nodeflow representing all possible ego-graphs
start: the node to visualise the ego_graph of
k: 

Returns: None
"""
def plot_ego_graph(nf,start,k):
    if not plot_graphs:
        return

    # adapted from https://plotly.com/python/network-graphs/
    edges = get_edges_from_flow(nf,start,k)
    
    # First, convert to networkX
    G = nx.Graph()
    
    # Add the start node in the centre
    G.add_node(start,pos=[0,0]) 
    colours = [0]
    
    # Add nodes, and store colours
    # For now, just colour the centre
    # Do this before edges so colour and nodes are the same ordering
    for colour,src,dest in edges:
        colours += [5]
        G.add_node(dest)
    
    
    for _,src,dest in edges:
        G.add_edge(src,dest)
        
    
    # give the nodes positions
    positions = nx.spring_layout(G,center=[0,0])
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Ego Graph',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

        
    node_trace.marker.color = colours
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="An egograph",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.show()
```

```python
plot_ego_graph(node_flow,1,k)
```

```python
plot_ego_graph(node_flow,2,k)
```

```python
plot_ego_graph(node_flow,20,k)
```

# The Encoder and Discriminator


*The below code has been adapted from `models/dgi.py`.*


The encoder is trained as a GAN. It produces both real (positive) and fake (negative) node embeddings for a discriminator which then tries to guess which is which.

```python
class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        
        self.g = g
        self.conv = GIN(g, n_layers + 1, 1, in_feats, n_hidden, n_hidden, dropout, True, 'sum', 'sum')

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features

```

The corrupt flag of `forward()` is used to generate the fake output (the *negative ego graph*).

This is the EGI discriminator:

```python
class SubGDiscriminator(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers = 2):
        super(SubGDiscriminator, self).__init__()
        self.g = g
        
        self.dc_layers = nn.ModuleList()
        
        for i in range(n_layers):
            self.dc_layers.append(GNNDiscLayer(in_feats, n_hidden))
        
        self.linear = nn.Linear(in_feats + 2 * n_hidden, n_hidden, bias = True)
        self.in_feats = in_feats
        self.U_s = nn.Linear(n_hidden, 1)

        
    def edge_output(self, edges):
        return {'h': torch.cat([edges.src['root'], edges.src['m'], edges.dst['x']], dim=1)}

    def find_common(self, layer_nid, nf):
        reverse_nodes = set()
        for i in range(nf.num_blocks):
            u, v = self.g.find_edges(nf.block_parent_eid(i))
            reverse_nodes.update(u.tolist())
            reverse_nodes.update(v.tolist())
            
        layer_nid = set(layer_nid.tolist())
        
        return torch.tensor(list(layer_nid.intersection(reverse_nodes)))

    def forward(self, nf, emb, features):
        reverse_edges = []
        for i in range(nf.num_blocks):

            u,v = self.g.find_edges(nf.block_parent_eid(i))
            reverse_edges += self.g.edge_ids(v,u).numpy().tolist()
            
            
        small_g = self.g.edge_subgraph( reverse_edges)
        small_g.ndata['root'] = emb[small_g.ndata['_ID']]
        small_g.ndata['x'] = features[small_g.ndata['_ID']]
        small_g.ndata['m']= torch.zeros_like(emb[small_g.ndata['_ID']])

        edge_embs = []
        for i in range(nf.num_blocks)[::-1]:

            v = small_g.map_to_subgraph_nid(nf.layer_parent_nid(i+1))

            uid = small_g.out_edges(v, 'eid')

            if i+1 == nf.num_blocks:
                h = self.dc_layers[0](small_g, v, uid, 1)
            else:
                h = self.dc_layers[0](small_g, v, uid, 2)

            edge_embs.append(self.U_s(F.relu(self.linear(h))))
        return edge_embs


```

These are trained together as a GAN like so:
    
This example considers the `SubGI` version of the model, but other versions are also given:

```python
class SubGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, pretrain=None):
        super(SubGI, self).__init__()
        
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
       
        self.g = g

        self.subg_disc = SubGDiscriminator(g, in_feats, n_hidden) # Discriminator
        
        self.loss = nn.BCEWithLogitsLoss()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        if pretrain is not None:
            print("Loaded pre-train model: {}".format(pretrain) )
            self.load_state_dict(torch.load(pretrain))
    
    def reset_parameters(self):
        self.encoder = Encoder(self.g, self.in_feats, self.n_hidden, self.n_layers, self.activation, self.dropout)
        self.encoder.conv.g = self.g
        self.subg_disc = SubGDiscriminator(self.g, self.in_feats, self.n_hidden, self.model_id)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features, nf):
        positive = self.encoder(features, corrupt=False)
        
        perm = torch.randperm(self.g.number_of_nodes())
        negative = positive[perm]


        positive_batch = self.subg_disc(nf, positive, features)

        negative_batch = self.subg_disc(nf, negative, features)

        E_pos, E_neg, l = 0.0, 0.0, 0.0
        pos_num, neg_num = 0, 0
        
        for positive_edge, negative_edge in zip(positive_batch, negative_batch):

            E_pos += get_positive_expectation(positive_edge, 'JSD', average=False).sum()
            pos_num += positive_edge.shape[0]

            E_neg += get_negative_expectation(negative_edge, 'JSD', average=False).sum()
            neg_num += negative_edge.shape[0]

            l += E_neg - E_pos

        return E_neg / neg_num - E_pos / pos_num
    
    # TODO: this was never actually fully implemented?
    def train_model(self):
        self.train()
        cur_loss = []
        
        for nf in self.train_sampler:

            self.optimizer.zero_grad()
            l = self.forward(self.features, nf)
            l.backward()
            cur_loss.append(l.item())

            self.optimizer.step()

        return np.mean(cur_loss)

```

# Training

<!-- #region -->
Training occurs in two parts. First a model is trained to encode the graph in such a way that it's structural info is retained. Then, a classifier is trained on this encoding and the node labels.

These two parts of the model are entirely seperate - the encoder, once trained, could then be used as part of link prediction or other graph learning tasks.

* * * 

The features given as input are node degrees, but can be other node specific features such as PageRank scores, spectral-embeddings, etc. These should be a function of the graph structure - i.e. sensitive to changes in the graph structures.

The encoder is also given the ego-graphs during training, but is only ran on node features at evaluation-time.


After embedding the graph, the node classifications (1-4) are used to train the classifier.
<!-- #endregion -->

```python
"""
For a given graph, create a tensor of nodes to node degrees.

graph: A DGL graph
opts: The model options

Return: a Tensor with the shape (number_of_nodes,max_degree). 

    For a node n with degree d, this tensor contains a 1 
    in position feature[n][d], and a 0 otherwise.
....
"""
def degree_bucketing(graph, opts, degree_emb=None, max_degree = 10):
    
    max_degree = opts.n_hidden
    features = torch.zeros([graph.number_of_nodes(), max_degree])

    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features
```

Train the encoder:

```python
degree_emb = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, [100, opts.n_hidden])), requires_grad=False)

# the features are the node degrees
features = degree_bucketing(g, opts, degree_emb)
in_feats = features.shape[1]

n_classes = labels.max().item() + 1
n_edges = g.number_of_edges()


# Is this going to be ran on the GPU?
if opts.gpu < 0:
    cuda = False

else:
    cuda = True
    torch.cuda.set_device(opts.gpu)
    features = features.cuda()
    in_feats
    labels = labels.cuda()

    
# initialise encoder discriminator duo
encoder = SubGI(g,
            in_feats,
            opts.n_hidden,
            opts.n_layers,
            nn.PReLU(opts.n_hidden),
            opts.dropout)

if cuda:
    encoder = encoder.cuda()
    
encoder_optimizer = torch.optim.Adam(encoder.parameters(),
                              lr=opts.dgi_lr,
                              weight_decay=opts.weight_decay)

# some summary statistics
cnt_wait = 0
best = 1e9
best_t = 0
dur = []

# hacky hack to make DGL happy 
g.ndata['features'] = features.to(torch.device('cpu')) 

# start training
for epoch in tqdm(range(opts.n_dgi_epochs)):
    
    # initialise ego-graph sampler
    train_sampler = dgl.contrib.sampling.NeighborSampler(g, 256, 5,
                                            neighbor_type='in', num_workers=1,
                                            num_hops=opts.n_layers + 1, shuffle=True)
    
    # Enable training mode for model
    encoder.train()
    
    
    
    if epoch >= 3:
        t0 = time.time()

    
    loss = 0.0
    
    # train based on features and ego-graphs
    for nf in train_sampler:
        encoder_optimizer.zero_grad()
        l = encoder(features,nf) # forward propogate
        l.backward()
        loss += l
        encoder_optimizer.step()


    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(encoder_optimizer.state_dict(), 'best_classification_{}.pkl'.format(opts.model_type))
    else:
      cnt_wait += 1

    if cnt_wait == opts.patience:
      print('Early stopping!')
      break

    if epoch >= 3:
      dur.append(time.time() - t0)

```

Train the classifier:

```python
# How good is the model doing?
def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


```

```python
classifier = MultiClassifier(opts.n_hidden, n_classes)
classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                        lr=opts.classifier_lr,
                                        weight_decay=opts.weight_decay)

# now that training is done, the discriminator is no longer needed
encoder = encoder.encoder

if cuda:
    classifier.cuda()

embeds = encoder(features, corrupt=False)
    
embeds = embeds.detach()
    
dur = []

classifier.train() # enable training mode

for epoch in tqdm(range(opts.n_classifier_epochs)):
    
    if epoch >= 3:
        t0 = time.time()

    classifier_optimizer.zero_grad() # reset gradient
    
    preds = classifier(embeds)
    
    loss = F.nll_loss(preds[train_mask], labels[train_mask])
    
    loss.backward()
    classifier_optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)
        
    accuracy = evaluate(classifier, embeds, labels, test_mask)

    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
          "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                        accuracy, n_edges / np.mean(dur) / 1000))

    

```

# Transfer learning


Transfer the trained models to the Brazil dataset:

```python
# prepare validation data
validation_g,validation_labels = construct_DGL(validation_g,validation_labels)
validation_features = degree_bucketing(validation_g,opts)

# use encoder to create node embeddings
embeddings = encoder(validation_features)
embeddings = embeddings.detach()

# generate predictions
predictions = classifier(embeddings)

validation_accuracy = evaluate(classifier,embeddings,validation_labels,torch.ones(validation_labels.shape, dtype=torch.bool))

print(f"The model has accuracy {accuracy}, and accuracy {validation_accuracy} when transferred")
```

```python

```
