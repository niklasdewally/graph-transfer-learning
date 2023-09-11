======================
Unsupervised learning
======================

The GNN models considered in this research are **unsupervised**.

In the context of graphs, this means that the aim for learning is to produce
a set of **node embeddings that capture structural graph information as much as
possible**. 

To test the accuracy of these node embeddings, we pass these to **downstream**
models for specific tasks.

These downstream models can be simple classifiers, such as `SGDClassifier
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_
from `sklearn`, or implemented using PyTorch.

Note that these are two seperate models - this differs from many of the graph
tutorials online, which use a single PyTorch model to create embeddings and
perform the downstream task.



