# Code adapted from https://github.com/GentleZhu/EGI/

import torch
import numpy as np
from sklearn.manifold import SpectralEmbedding

def spectral_embedding(graph,n_components):
    """
    Use spectral embedding to create a feature tensor representing the given graph.

    This function wraps sklearn - for more information on spectral embedding,
    refer to its user manual:
    https://scikit-learn.org/stable/modules/manifold.html#spectral-embedding
    
    Args:
        graph (DGLGraphStale): A DGL graph.

        n_components (int): The dimension of
            In the original EGI code, this was set to be the number of hidden
            layers in the model.


    Returns:
        a feature Tensor of type int and shape
        (graph.number_of_nodes(),graph.number_of_nodes()).

    """

    A = np.zeros([graph.number_of_nodes(), graph.number_of_nodes()])
    a,b = graph.all_edges()
    
    # add edges to a numpy array
    for id_a, id_b in zip(a.numpy().tolist(), b.numpy().tolist()):
        A[id_a, id_b] = 1

    embedding = SpectralEmbedding(n_components=args.n_hidden)

    features = torch.FloatTensor(embedding.fit_transform(A))

    return features

