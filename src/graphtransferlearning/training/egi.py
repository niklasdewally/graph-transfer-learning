from graphtransferlearning.features import degree_bucketing

from dgl.dataloading import DataLoader

from graphtransferlearning.models import EGI
import graphtransferlearning as gtl
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import dgl
import time
from tqdm import tqdm
from random import sample
from IPython import embed

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
                      batch_size=50,
                      kfolds = 10,
                      sampler="egi",
                      save_weights_to=None,
                      patience=25,
                      min_delta=0.01,
                      writer=None,
                      tb_prefix=""):


    """
    Train an EGI encoder.

    An EGI encoder produces node embeddings for a given graph, retaining
    high-level strucutral features of the training graph to improve
    transferability.

    It does this by considering k-hop ego graphs.

    Args:
        dgl_graph: The input graph, as a DGLGraph.

        gpu: The gpu device to use. Defaults to -1, which trains on the CPU.

        k: The number of hops to consider in the ego-graphs.
           Defaults to 2, which was shown to have the best results in the
           original paper.

        lr: Learning rate. Defaults to 0.01. l

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

        batch_size: The number of nodes to consider in each training batch.

        kfolds: The number of partitions to use in the data for cross-validation.

        sampler: The subgraph sampler to use.
            Options are ['egi','triangle']
            Defaults to 'egi'.

        save_weights_to: A file path to save EGI model parameters for use in
            transfer learning.

            Defaults to None.

        writer: A torch.utils.tensorboard.SummaryWriter. Used to write loss to 
                a tensor board.

                the metrics {tb_prefix}/training-loss and
                {tb_prefix}/validation-loss are saved.

                Defaults to None.

        tb_prefix: A prefix to attach to variables on tensor board.
            Useful if a given model has multiple encoders.

    Returns:
        The trained EGI encoder model.

    """


    # input validation
    valid_feature_modes = ['degree_bucketing']
    valid_optimisers = ['adam']
    valid_samplers = ['egi','triangle']

    if feature_mode not in valid_feature_modes:
        raise ValueError(f"{feature_mode} is not a valid feature generation "\
                           "mode. Valid options are {valid_feature_modes}.")

    if optimiser not in valid_optimisers:
        raise ValueError(f"{optimiser} is not a valid optimiser."\
                           "Valid options are {valid_optimisers}.")

    if sampler not in valid_samplers:
        raise ValueError(f"{sampler} is not a valid sampler."\
                           "Valid options are {valid_sampler}.")

    if k < 1: 
        raise ValueError("k must be 1 or greater.")

    if lr <= 0:
        raise ValueError("Learning rate must be above 0.")


    # generate features
    
    features = degree_bucketing(dgl_graph,n_hidden_layers)

    if sampler == 'egi':
        sampler = dgl.dataloading.NeighborSampler([10 for i in range(k)])
    elif sampler == 'triangle':
        sampler = gtl.KHopTriangleSampler([10 for i in range(k)])

    # are we running on a gpu?
    device = 'cpu' if gpu < 0 else f"cuda:{gpu}"

    features = features.to(device)
    dgl_graph = dgl_graph.to(device)


    in_feats = features.shape[1]

    # in the original code, they set number of layers to equal k +1
    n_layers = k + 1

    model = EGI(in_feats,
                n_hidden_layers,
                n_layers,
                nn.PReLU(n_hidden_layers),
                )
    
    model = model.to(device)

    # do transfer learning if we have pretrained weights
    if pre_train is not None:
        model.load_state_dict(torch.load(pre_train),strict=False)

    optimizer= torch.optim.Adam(model.parameters(),
                                lr = lr,
                                weight_decay = weight_decay)



    # some summary statistics
    best = 1e9
    best_epoch=-1

    # setup cross-validation
    fold_size = dgl_graph.num_nodes() // kfolds
    assert(fold_size >= 1)
        
    # shuffle nodes before putting into folds
    indexes = torch.randperm(dgl_graph.nodes().shape[0])
    folds = torch.split(dgl_graph.nodes()[indexes],fold_size)

    # start training
    for epoch in tqdm(range(n_epochs)):
        current_fold = epoch % kfolds
        val_nodes = folds[current_fold]
        train_nodes = torch.unique(torch.cat([val_nodes,dgl_graph.nodes()]))
            
        # Enable training mode for model
        model.train()
        
        if epoch >= 3:
            t0 = time.time()

        
        loss = 0.0
        
        # train based on features and ego graphs around specifc egos
        model.train()
        optimizer.zero_grad()

        # the sampler returns a list of blocks and involved nodes
        # each block holds a set of edges from a source to destination
        # each block is a hop in the graph
        for blocks in DataLoader(dgl_graph,train_nodes,sampler,batch_size=batch_size,shuffle=True):
            l = model(dgl_graph,features,blocks)
            l.backward()
            optimizer.step()
            loss += l         

        if writer:
            writer.add_scalar(f'{tb_prefix}/training-loss',loss,global_step=epoch)

        # validation

        model.eval()
        loss = 0.0
        blocks = sampler.sample(dgl_graph,val_nodes) 
        loss = model(dgl_graph,features,blocks)

        # early stopping
        if loss <= best + min_delta:
            # https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/
            best = loss
            best_epoch = epoch
            # save current weights
            torch.save(model.state_dict(), 'stopping')


        if (epoch - best_epoch > patience):
            print("Early stopping!")
            model.load_state_dict(torch.load('stopping'))
            break

        if writer:
            writer.add_scalar(f'{tb_prefix}/validation-loss',loss,global_step=epoch)

        if epoch >= 3 and writer is not None:
            if writer:
                writer.add_scalar(f'{tb_prefix}/time-per-epoch',time.time() - t0 ,global_step=epoch)


    # save parameters for later fine-tuning if a save path is given
    if save_weights_to is not None:
        print(f"Saving model parameters to {str(save_weights_to)}")

        torch.save(model.state_dict(), save_weights_to)


    model.eval()
    model.encoder.eval()

    return model.encoder
