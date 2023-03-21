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

# Introduction 

The aim of this notebook is to give a high level overview of the EGI framework code.

This notebook primarily runs through the run_airport.py experiment, annotating and refactoring code.

![Figure 2, taken from (Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization, Zhu et al, 2021)](figures/fig2.png)






## Some setup 


Does our GPU work?

This works on any CUDA 11 compatible GPU, but has been written and tested on a 3060 only

```python
!nvcc --version
```

```python
# Import egi code here
# TODO: make this a pip library
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
from models.subgi import SubGI
#from models.vgae import VGAE
from IPython import embed
import scipy.sparse as sp
from collections import defaultdict
from torch.autograd import Variable
from tqdm.notebook import tqdm
import pickle
from collections import defaultdict
from sklearn.manifold import SpectralEmbedding
from types import SimpleNamespace
```

# Generation of ego-graphs

> Sample M ego-graphs {(g1, x1), ..., (gM , xM )} from empirical distribution P without replacement

```python
torch.BoolTensor
```

Manually set some options for the experiment

```python
opts = SimpleNamespace(
    file_path = "../../egi/data/usa-airports.edgelist",
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
    n_layers=1,
    weight_decay=0.,
    patience=20,
    model=True,
    self_loop=True,
    model_type=2,
    graph_type="DD"

)
```

```python
def read_struct_net(args):

    g = nx.Graph()

    with open(args.file_path) as IN:
        for line in IN:
            tmp = line.strip().split()
            # print(tmp[0], tmp[1])
            g.add_edge(int(tmp[0]), int(tmp[1]))
    labels = dict()
    
    with open(args.label_path) as IN:
        IN.readline()
        for line in IN:
            tmp = line.strip().split(' ')
            labels[int(tmp[0])] = int(tmp[1])
    return g, labels

```

```python
def degree_bucketing(graph, args, degree_emb=None, max_degree = 10):
    max_degree = args.n_hidden
    features = torch.zeros([graph.number_of_nodes(), max_degree])
    #return features
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features

```

```python
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
    return train_mask, test_mask

```

```python
def constructDGL(graph, labels):
    node_mapping = defaultdict(int)
    relabels = []
    for node in sorted(list(graph.nodes())):
        node_mapping[node] = len(node_mapping)
        relabels.append(labels[node])
    # embed()
    assert len(node_mapping) == len(labels)
    new_g = DGLGraph()
    new_g.add_nodes(len(node_mapping))
    for i in range(len(node_mapping)):
        new_g.add_edge(i, i)
    for edge in graph.edges():
        new_g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
        new_g.add_edge(node_mapping[edge[1]], node_mapping[edge[0]])
        
    
    return new_g, relabels


```

```python
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
"""
Run the model.

- args: A namespace containing the experiment options.

Returns: the success-rate of the model.
"""

def run_model(args):
      g,labels = read_struct_net(args)

      valid_mask = None

      g.remove_edges_from(nx.selfloop_edges(g))

      g, labels = constructDGL(g, labels)

      labels = torch.LongTensor(labels)

      degree_emb = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, [100, args.n_hidden])), requires_grad=False)

      features = degree_bucketing(g, args, degree_emb)

      train_mask, test_mask = createTraining(labels, valid_mask)


      train_mask = torch.BoolTensor(train_mask)
      test_mask = torch.BoolTensor(test_mask)

      in_feats = features.shape[1]
      n_classes = labels.max().item() + 1
      n_edges = g.number_of_edges()

      if args.gpu < 0:
          cuda = False
      else:
          cuda = True
          torch.cuda.set_device(args.gpu)
          features = features.cuda()
          labels = labels.cuda()

      g.readonly()
      n_edges = g.number_of_edges()

      # create DGI model
      # TODO: broken imports
  #
  #   if args.model_type == 1:
  #       dgi = VGAE(g,
  #           in_feats,
  #           args.n_hidden,
  #           args.n_hidden,
  #           args.dropout)


  #       dgi.prepare()
  #       dgi.adj_train = sp.csr_matrix(output_adj(g))

      if args.model_type == 0:
          dgi = DGI(g,
                  in_feats,
                  args.n_hidden,
                  args.n_layers,
                  nn.PReLU(args.n_hidden),
                  args.dropout)

      elif args.model_type == 2:
          dgi = SubGI(g,
                  in_feats,
                  args.n_hidden,
                  args.n_layers,
                  nn.PReLU(args.n_hidden),
                  args.dropout,
                  args.model_id)


      if cuda:
          dgi.cuda()

      dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                                      lr=args.dgi_lr,
                                      weight_decay=args.weight_decay)


      cnt_wait = 0
      best = 1e9
      best_t = 0
      dur = []
      g.ndata['features'] = features.to(torch.device('cpu'))
      for epoch in range(args.n_dgi_epochs):
          train_sampler = dgl.contrib.sampling.NeighborSampler(g, 256, 5,
                                                               neighbor_type='in', num_workers=1,
                                                               num_hops=args.n_layers + 1, shuffle=True)

          dgi.train()
          if epoch >= 3:
              t0 = time.time()

          loss = 0.0

          # VGAE mode
          #if args.model_type == 1:
          #   dgi.optimizer = dgi_optimizer
          #   dgi.train_sampler = train_sampler
          #   dgi.features = features
          #   loss = dgi.train_model()

          # EGI mode
          if args.model_type == 2:
              #if True:
              for nf in train_sampler:
                  dgi_optimizer.zero_grad()
                  l = dgi(features, nf)
                  l.backward()
                  loss += l
                  dgi_optimizer.step()

          # DGI mode
          elif args.model_type == 0:
              dgi_optimizer.zero_grad()
              loss = dgi(features)
              loss.backward()
              dgi_optimizer.step()


          if loss < best:
              best = loss
              best_t = epoch
              cnt_wait = 0
              torch.save(dgi.state_dict(), 'best_classification_{}.pkl'.format(args.model_type))
          else:
              cnt_wait += 1

          if cnt_wait == args.patience:
              print('Early stopping!')
              break

          if epoch >= 3:
              dur.append(time.time() - t0)



          # create classifier model
          classifier = MultiClassifier(args.n_hidden, n_classes)
          if cuda:
              classifier.cuda()

          classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                                  lr=args.classifier_lr,
                                                  weight_decay=args.weight_decay)



          # flags used for transfer learning
          if args.data_src != args.data_id:
              pass
          else:
              dgi.load_state_dict(torch.load('best_classification_{}.pkl'.format(args.model_type)))

          with torch.no_grad():
              if args.model_type == 1:
                  _, embeds, _ = dgi.forward(features)
              elif args.model_type == 2:
                  embeds = dgi.encoder(features, corrupt=False)
              elif args.model_type == 0:
                  embeds = dgi.encoder(features)
              else:
                  dgi.eval()
                  test_sampler = dgl.contrib.sampling.NeighborSampler(g, g.number_of_nodes(), -1,
                                                                          neighbor_type='in', num_workers=1,
                                                                          add_self_loop=False,
                                                                          num_hops=args.n_layers + 1, shuffle=False)
                  for nf in test_sampler:
                      nf.copy_from_parent()
                      embeds = dgi.encoder(nf, False)
                      print("test flow")

          embeds = embeds.detach()

          dur = []
          for epoch in range(args.n_classifier_epochs):
              classifier.train()
              if epoch >= 3:
                  t0 = time.time()

              classifier_optimizer.zero_grad()
              preds = classifier(embeds)
              loss = F.nll_loss(preds[train_mask], labels[train_mask])
              # embed()
              loss.backward()
              classifier_optimizer.step()

              if epoch >= 3:
                  dur.append(time.time() - t0)
              #acc = evaluate(classifier, embeds, labels, train_mask)
              #acc = evaluate(classifier, embeds, labels, val_mask)
              #print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              #      "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
              #                                    acc, n_edges / np.mean(dur) / 1000))

          # print()
          acc = evaluate(classifier, embeds, labels, test_mask)

          return acc;


```

```python
# Run the model multiple times, and print the average and standard deviation of the results
test_results = []

for runs in tqdm(range(10)):
  test_results.append(run_model(opts));

print("Test Accuracy {:.4f}, std {:.4f}".format(np.mean(test_results), np.std(test_results)))
```

# The encoder


# Center Node embedding


# Discriminator 

```python

```

```python

```
