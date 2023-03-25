from graphtransferlearning.features import degree_bucketing
from graphtransferlearning.models import EGI
import torch
import torch.nn as nn
import dgl
import time
from tqdm import tqdm

def train_egi_encoder(dgl_graph,
                      gpu=-1,
                      k=2,
                      lr=0.01,
                      n_hidden_layers=32,
                      n_epochs=100,
                      weight_decay = 0.,
                      feature_mode='degree_bucketing',
                      optimiser='adam',
                      pre_train=None,
                      save_weights_to=None):


    """
    Train an EGI encoder.

    An EGI encoder produces node embeddings for a given graph, retaining
    high-level strucutral features of the training graph to improve
    transferability.

    It does this by considering k-hop ego graphs.

    Args:
        dgl_graph: The input graph, as a DGLGraphStale.

        gpu: The gpu device to use. Defaults to -1, which trains on the CPU.

        k: The number of hops to consider in the ego-graphs.
           Defaults to 2, which was shown to have the best results in the
           original paper.

        lr: Learning rate. Defaults to 0.01.

        n_hidden_layers: The number of hidden layers in the encoder. Defaults
            to 32.

        n_epochs: The number of epochs to do when training.

        feature_mode: The function to use to generate feature tensors.
            Options are: ['degree_bucketing'].
            Defaults to 'degree_bucketing'.

        optimiser: The optimiser to use.
            Options are: ['adam'].
            Defaults to 'adam'.

        pre_train: Existing model parameters to use for fine tuning in transfer
            learning. This must be a path that points to a file saved by doing:

            torch.save(modelA.state_dict(), PATH)

            Defaults to None.

            For more information, see
            https://pytorch.org/tutorials/beginner/saving_loading_models.html.

        save_weights_to: A file path to save EGI model parameters for use in
            transfer learning.

            Defaults to None.

    Returns:
        The trained EGI encoder model.

    """

    # input validation
    valid_feature_modes = ['degree_bucketing']
    valid_optimisers = ['adam']

    if feature_mode not in valid_feature_modes:
        raise ValueError(f"{feature_mode} is not a valid feature generation "\
                           "mode. Valid options are {valid_feature_modes}.")

    if optimiser not in valid_optimisers:
        raise ValueError(f"{optimiser} is not a valid optimiser."\
                           "Valid options are {valid_optimisers}.")

    if k < 1: 
        raise ValueError("k must be 1 or greater.")

    if lr <= 0:
        raise ValueError("Learning rate must be above 0.")


    # generate features
    
    features = degree_bucketing(dgl_graph,n_hidden_layers)

    # are we running on a gpu?
    if gpu < 0:
        cuda = False

    else:
        cuda = True
        torch.cuda.set_device(gpu)
        features = features.cuda()


    in_feats = features.shape[1]

    # in the original code, they set number of layers to equal k +1
    n_layers = k + 1

    model = EGI(dgl_graph,
                in_feats,
                n_hidden_layers,
                n_layers,
                nn.PReLU(n_hidden_layers),
                )
    if cuda:
        model = model.cuda()

    # do transfer learning if we have pretrained weights
    if pre_train is not None:
        model.load_state_dict(torch.load(pre_train),strict=False)

    optimizer= torch.optim.Adam(model.parameters(),
                                lr = lr,
                                weight_decay = weight_decay)



    # some summary statistics
    best = 1e9
    best_t = 0
    dur = []

    # start training
    for epoch in tqdm(range(n_epochs)):
        
        # initialise ego-graph sampler
        ego_graph_sampler = dgl.contrib.sampling.NeighborSampler(dgl_graph, 256, 5,
                                                neighbor_type='in', num_workers=1,
                                                num_hops=k, shuffle=True)
        
        # Enable training mode for model
        model.train()
        
        if epoch >= 3:
            t0 = time.time()

        
        loss = 0.0
        
        # train based on features and ego-graphs
        for nf in ego_graph_sampler:
            optimizer.zero_grad()
            l = model(features,nf) # forward propagate to find loss
            l.backward()
            loss += l
            optimizer.step()


        if loss < best:
            best = loss
            best_t = epoch


        if epoch >= 3:
          dur.append(time.time() - t0)

    # save parameters for later fine-tuning if a save path is given
    if save_weights_to is not None:
        print(f"Saving model parameters to {save_weights_to}")

        torch.save(model.state_dict(), save_weights_to)


    model.eval()
    model.encoder.eval()

    return model.encoder
