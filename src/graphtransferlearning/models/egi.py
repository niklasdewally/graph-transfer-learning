import torch.nn
import torch.nn as nn
import math
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import numpy as np

class EGI(nn.Module):

    """ 
    An implementation of EGI based on a SubGI model.

    The node encoder for this EGI model can be retrieved with .encoder.

    This code is primarily adapted from the reference implementation found at
    https://github.com/GentleZhu/EGI.

    Args:
        g: The input graph, as a DGLGraph.
        in_feats: The input feature size (as a tensor).
        n_hidden: The number of hidden dimensions in the model.
        n_layers: The number of layers in the model.
            In the original implementation, this is equal to the k
            hyper-parameter.

        activation: The activation function for the model.
        dropout: The dropout rate of the underlying layers. 
            Defaults to 0.0.
    """

    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout=0.0):
        super(EGI, self).__init__()
        
        self.encoder = _Encoder(g, in_feats, n_hidden, n_layers, activation,dropout)
       
        self.g = g

        self.subg_disc = _SubGDiscriminator(g, in_feats, n_hidden) # Discriminator

        self.loss = nn.BCEWithLogitsLoss()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
    
    def reset_parameters(self):
        self.encoder = Encoder(self.g, self.in_feats, self.n_hidden, self.n_layers, self.activation)
        self.encoder.conv.g = self.g
        self.subg_disc = SubGDiscriminator(self.g, self.in_feats, self.n_hidden, self.model_id)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features, ego_node):
        """
        Returns the loss of the model.

        For encoding, use .encoder.
        """

        positive = self.encoder(features, corrupt=False)
        
        # generate negative edges through random permutation
        perm = torch.randperm(self.g.number_of_nodes())
        negative = positive[perm]

        positive_batch = self.subg_disc(ego_node, positive, features)

        negative_batch = self.subg_disc(ego_node, negative, features)

        E_pos, E_neg, l = 0.0, 0.0, 0.0
        pos_num, neg_num = 0, 0

        
        for positive_edge, negative_edge in zip(positive_batch, negative_batch):

            E_pos += get_positive_expectation(positive_edge, 'JSD', average=False).sum()
            pos_num += positive_edge.shape[0]

            E_neg += get_negative_expectation(negative_edge, 'JSD', average=False).sum()
            neg_num += negative_edge.shape[0]

            l += E_neg - E_pos

        assert(pos_num != 0)
        assert(neg_num != 0)

        return E_neg / neg_num - E_pos / pos_num
    

class _Encoder(nn.Module):
    """ 

    The EGI encoder.

    Produces node embeddings for a given graph based on structural node features.

    """
    def __init__(self, g, in_feats, n_hidden, n_layers, activation,dropout):
        super(_Encoder, self).__init__()
        
        self.g = g
        self.conv = GIN(g, n_layers, 1, in_feats, n_hidden, n_hidden, dropout, True, 'sum', 'sum')

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features

class _SubGDiscriminator(nn.Module):
    """
    The EGI discriminator.

    TODO: what does this do in detail?

    """
    def __init__(self, g, in_feats, n_hidden, n_layers = 2):
        super(_SubGDiscriminator, self).__init__()

        self.g = g
        self.k = n_layers

        self.in_feats = in_feats

        # discriminator convolutional layers
        # used to encode neighbor embeddings
        self.dc_layers = nn.ModuleList()
        
        for i in range(n_layers):
            self.dc_layers.append(GNNDiscLayer(in_feats, n_hidden))
       
        # these layers consist of the scoring function
        self.linear = nn.Linear(in_feats + 2 * n_hidden, n_hidden, bias = True)
        self.U_s = nn.Linear(n_hidden, 1)

    def forward(self, ego_node, emb, features):

        # k-hop ego graph.
        ego_graph,n = dgl.khop_in_subgraph(self.g,ego_node,self.k,store_ids=True)

        if torch.cuda.is_available():
            ego_graph = ego_graph.to('cuda:0')

        ego_eg = n.item() # ego-node relative to ego graph

        # idk tbh - used by discriminator
        ego_graph.ndata['root'] = emb[ego_graph.ndata['_ID']]
        ego_graph.ndata['x'] = features[ego_graph.ndata['_ID']]
        ego_graph.ndata['m']= torch.zeros_like(emb[ego_graph.ndata['_ID']])

        edge_scores = []

        # Sample edges using breadth-first-search, starting from the ego.
        # Returns a list of "edge frontiers". Each edge frontier will be
        # another hop in the ego graph.
        frontiers = dgl.bfs_edges_generator(ego_graph,ego_eg,reverse=True)

        max_hop = len(frontiers)

        # go over hops in ego-graph, starting from outside
        for i in range(max_hop)[::-1]:
            edges = frontiers[i]

            if torch.cuda.is_available():
                edges = edges.to('cuda:0')
            
            # get nodes in the edges
            us,vs = ego_graph.find_edges(edges)

            # TODO: not sure if ive reversed the ego-graph right.
            # Get edge score
            if i+1 == max_hop:
                h = self.dc_layers[0](ego_graph,vs,us, 2)
                edge_scores.append(self.U_s(F.relu(self.linear(h))))
            else:
                h = self.dc_layers[0](ego_graph,vs,us, 1)
                edge_scores.append(self.U_s(F.relu(self.linear(h))))


        # return total scores
        return edge_scores


       # # for every hop in the ego-graph
       # for i in range(nf.num_blocks):

       #     # pick nodes from an edge
       #     u,v = self.g.find_edges(nf.block_parent_eid(i))

       #     # add the reverse edge (WHY reverse?) to a list
       #     reverse_edges += self.g.edge_ids(v,u).numpy().tolist()
       #     
       # # induce a subgraph based on these edges
       # small_g = self.g.edge_subgraph( reverse_edges)

       # # ???
       # small_g.ndata['root'] = emb[small_g.ndata['_ID']]
       # small_g.ndata['x'] = features[small_g.ndata['_ID']]
       # small_g.ndata['m']= torch.zeros_like(emb[small_g.ndata['_ID']])

       # edge_embs = []
       # 
       # go through ego-graph hop edges in reverse
       # for i in range(nf.num_blocks)[::-1]:

            # get edges on given layer ids' in ego_graph
       #     v = small_g.map_to_subgraph_nid(nf.layer_parent_nid(i+1))
       #     uid = small_g.out_edges(v, 'eid')

       #     if i+1 == nf.num_blocks:
       #         h = self.dc_layers[0](small_g, v, uid, 1)
       #     else:
       #         h = self.dc_layers[0](small_g, v, uid, 2)

    
       #     edge_embs.append(self.U_s(F.relu(self.linear(h))))

       # return edge_embs


# Functions below copied verbatim from original egi code, for use in this model

class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, g, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.g = g

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers):
            h = self.ginlayers[i](self.g, h)
            # print('batch norm')
            # 
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        # only need node embedding
        return h

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h

class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq


class GNNDiscLayer(nn.Module):
    def __init__(self, in_feats, n_hidden):
        super(GNNDiscLayer, self).__init__()
        self.fc = nn.Linear(in_feats, n_hidden)
        self.layer_1 = True

    def reduce(self, nodes):
        return {'m': F.relu(self.fc(nodes.data['x']) + nodes.mailbox['m'].mean(dim=1) )
               ,'root':nodes.mailbox['root'].mean(dim=1)}

    def msg(self, edges):
        if self.layer_1:
            return {'m': self.fc(edges.src['x'])
                   ,'root': edges.src['root']}
        else:
            return {'m': self.fc(edges.src['m'])
                   ,'root': edges.src['root']}
    
    def edges(self, edges):
        return {'output':torch.cat([edges.src['root'], edges.src['m'], edges.dst['x']], dim=1)}

    def forward(self, g, v, edges, depth=1):

        if depth == 1:
            self.layer_1 = True
        else:
            self.layer_1 = False

        g.apply_edges(self.edges, edges)

        g.push(v, self.msg, self.reduce)
        
        return g.edata.pop('output')[edges]
