# Code adapted from original EGI project: https://github.com/GentleZhu/EGI

import torch


def degree_bucketing(graph, max_degree=10):
    """
    Create a feature tensor for a graph's nodes using degree bucketing.

    Args:
        graph (DGLGraph): A DGL graph

        max_degree (int): The maximum degree of a node. Nodes with degrees
            greater than this will have their degree value truncated to max_degree.

            This should be equal to the number of hidden layers in the network.

    Returns:
        An feature Tensor with type int and shape
        (graph.number_of_nodes(),max_degree).

        For a node n with degree d, this tensor contains a 1
        in position feature[n][d], and a 0 otherwise.
    """

    features = torch.zeros([graph.number_of_nodes(), max_degree])

    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degrees(i), max_degree - 1)] = 1

        except:
            features[i][0] = 1
    return features
