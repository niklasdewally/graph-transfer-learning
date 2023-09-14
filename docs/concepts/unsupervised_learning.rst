======================
Unsupervised Learning
======================

The GNN models considered in this research are unsupervised.

The learning aim is to produce a set of node embeddings that capture structural
graph information as much as possible. 

**The gtl.train module produces unsupervised embeddings for this aim.**

These embeddings are then generated for the entire graph and are passed to
further models trained for downstream tasks such as link prediction or node
classification.

These downstream models can be simple classifiers, such as `SGDClassifier
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_
from ``sklearn``, or a *Multilayer Perceptron* implemented using ``pytorch``.

This differs from supervised learning approaches, in which a single
model is created for a task by using `GNN` layers then some `Linear` layers.

--------------
Classification
--------------

.. graphviz :: classification.dot

---------------
Link Prediction 
---------------

.. graphviz:: linkpred.dot


Creating Edge Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~

We consider an edge to be a combination of the nodes that it consists of.
Hence, we use a binary operator to turn node embeddings into edge embeddings.

Possible operators include:

* Hadamard product
* dot product
* concatenation
* average
* ``abs(a-b)``
  
