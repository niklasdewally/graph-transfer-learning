"""
Consider a BA and forest fire graph. Give these structural labels by checking
k-hop neighbor similarilty (using WL).

Aim is to direct transfer labels from one to another.

"""
import dgl
import torch
from torch.utils.tensorboard import SummaryWriter

import graphtransferlearning as gtl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def do_run(k=2):
    """
    Perform a specific run of the model for a given set of hyperparameters.
    """

    writer = SummaryWriter() # outputs to ./runs by default

    # synthetic graph generation parameters
    n_nodes = 500
    ba_attached= 2
    p_forward = 0.4
    p_backward = 0.3


    # Hyper parameters
    lr = 0.01
    n_hidden_layers=256
    n_epochs=10
    weight_decay = 0.
    feature_mode='degree_bucketing'
    optimiser='adam'

    # Generate synthetic graphs

    barbasi_nx,labels = gtl.add_structural_labels(gtl.generate_barbasi(n_nodes,ba_attached),
                                  k=k)
    forest_fire_nx,labels = gtl.add_structural_labels(gtl.generate_forest_fire(n_nodes,p_forward,p_backward),
                                  k=k,
                                  existing_labels=labels)



    # Train source graph (barbasi)
    barbasi_dgl = dgl.from_networkx(barbasi_nx).to(device)

    encoder = gtl.training.train_egi_encoder(
                  barbasi_dgl,
                  (0 if torch.cuda.is_available() else -1),
                  k,
                  lr,
                  n_hidden_layers,
                  n_epochs,
                  weight_decay,
                  feature_mode,
                  optimiser,
                  save_weights_to='model.pickle',
                  writer=writer
                  )

    writer.add_hparams(
            {'k': k, 
             'lr': lr, 
             '# Hidden Layers': n_hidden_layers, 
             '# Epochs':n_epochs,
             'Feature mode':feature_mode,
             'Optimiser':optimiser,
             '# Nodes': n_nodes,},
            {})

if __name__ == '__main__':
    do_run()
