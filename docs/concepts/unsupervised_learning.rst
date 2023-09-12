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
from `sklearn`, or Multilayer perceptrons using PyTorch.

This differs from supervised learning approaches, in which a single
model is created for a task by adding some `Linear` layers to the end of a
model using `GNN` layers. 

--------------
Classification
--------------

.. image:: unsupervised_learning_1.png

---------------
Link Prediction 
---------------

.. image:: unsupervised_learning_2.png


Any binary operator that combines two vectors into one can be used to turn node
embeddings into edge embeddings.

These include (non exhaustively):

* haddamard product
* dot product
* concatenation
* abs(A-B)
* average
  
