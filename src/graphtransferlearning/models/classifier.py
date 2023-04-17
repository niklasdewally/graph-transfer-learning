"""
This file contains classifiers for use with a separately trained unsupervised
graph encoder (e.g. DGI, EGI, Node2Vec,...).
"""


import torch.nn as nn

class LogisticRegression(nn.Module):
    """
    A logistic regression classifier.

    Args:
        input_dim: The dimension of the input.
        n_classes: The number of classes. This forms the output dimension of
                   the classifier.
    """

    def __init__(self,input_dim,n_classes):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(input_dim,n_classes)

    def forward(self,x):
        return self.linear(x)

