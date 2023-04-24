"""
Consider a BA and forest fire graph. Give these structural labels by checking
k-hop neighbor similarilty (using WL).

Aim is to direct transfer labels from one to another.
"""
import dgl
import graphtransferlearning as gtl
from graphtransferlearning.features import degree_bucketing
from graphtransferlearning.models import EGI
import numpy as np
import torch
import torch.nn as nn
import time

from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

from torchmetrics import Accuracy
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from random import shuffle,sample

from IPython import embed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def do_run(k=2,src_lr=0.1):
    """
    Perform a specific run of the model for a given set of hyperparameters.
    """

    writer = SummaryWriter() # outputs to ./runs by default

    # synthetic graph generation parameters

    n_nodes = 100
    n_graphs = 60
    ba_attached= 2
    p_forward = 0.4
    p_backward = 0.3


    # Encoder hyper parameters
    
    lr = 0.01
    n_hidden_layers=10
    n_epochs=100
    max_degree_in_feat = n_hidden_layers
    weight_decay = 0.
    feature_mode='degree_bucketing'
    optimiser='adam'

    ###########################################################################

    # Generate synthetic graphs

    g = gtl.generate_forest_fire(n_nodes,p_forward,p_backward)
    g,classes = gtl.add_structural_labels(g,k)

    # store graphs in two formats for easy use
    forest_fire_nx = [g]
    forest_fire_dgl = [dgl.from_networkx(g,node_attrs=['struct']).to(device)]

    print(f"Generating {n_graphs} forest-fire graphs with {n_nodes} nodes each")
    for i in tqdm(range(1,n_graphs)):
        g = gtl.generate_forest_fire(n_nodes,p_forward,p_backward)
        g,classes = gtl.add_structural_labels(g,k,existing_labels=classes)
        forest_fire_nx.append(g)
        forest_fire_dgl.append(dgl.from_networkx(g,node_attrs=['struct']).to(device))

    print(f"Generating {n_graphs} barbasi graphs with {n_nodes} nodes each")
    barbasi_nx = []
    barbasi_dgl = []
    for i in tqdm(range(0,n_graphs)):
        g = gtl.generate_barbasi(n_nodes,ba_attached)
        g,classes = gtl.add_structural_labels(g,k,existing_labels=classes)
        barbasi_nx.append(g)
        barbasi_dgl.append(dgl.from_networkx(g,node_attrs=['struct']).to(device))


    classes = torch.as_tensor(list(classes.values()))
    n_classes = len(classes)

    # Test-validate split
    shuffle(barbasi_dgl)
    barbasi_dgl_val = barbasi_dgl[:20]
    barbasi_dgl_train = barbasi_dgl[20:]

    ###########################################################################

    print(f"Training EGI encoder on barbasi graphs")

    barbasi_train_feats = [degree_bucketing(g,max_degree_in_feat).to(device) for g in barbasi_dgl_train]
    barbasi_val_feats = [degree_bucketing(g,max_degree_in_feat).to(device) for g in barbasi_dgl_val]

    # see training/egi.py for more info 
    model = EGI(barbasi_dgl_train[0],
                barbasi_train_feats[0].shape[1],
                n_hidden_layers,
                k+1,
                nn.PReLU(n_hidden_layers),
                ).to(device)

    optimizer= torch.optim.Adam(model.parameters(),lr = lr,weight_decay = weight_decay)
    
    # sample k hop ego-graphs with max 10 neighbors each hop
    sampler = dgl.dataloading.NeighborSampler([10 for i in range(k)])
    for epoch in tqdm(range(n_epochs)):
        model.train()

        t0 = time.time()

        loss = 0.0
        
            
        # train based on features and ego graphs around specific egos
        for i,g in enumerate(barbasi_dgl_train):
            optimizer.zero_grad()

            features = barbasi_train_feats[i]
            

            # the sampler returns a list of blocks and involved nodes
            # each block holds a set of edges from a source to destination
            # each block is a hop in the greaph
            blocks = sampler.sample(g,g.nodes())
            l = model(features,blocks)

            l.backward()
            loss += l

                
            optimizer.step()

        writer.add_scalar(f"encoder/training-loss",loss/len(barbasi_dgl_train),global_step=epoch)


        # calculate validation loss
        model.eval()
        loss = 0.0
        for i,g in enumerate(barbasi_dgl_val):
            blocks = sampler.sample(g,g.nodes())
            features = barbasi_val_feats[i]
            l = model(features,blocks)
            loss += l
                

        writer.add_scalar(f"encoder/validation-loss",loss/len(barbasi_dgl_val),global_step=epoch)


    # source classifier preparation
    encoder = model.encoder

    barbasi_train_embeddings = [encoder(x).to(device)for x in barbasi_train_feats]
    barbasi_val_embeddings = [encoder(x).to(device)for x in barbasi_val_feats]
    # classes defined earlier

    ###########################################################################

    # Source classifier hyperparameters
    src_n_epochs = 100

    src_input_dim = max_degree_in_feat


    # Train source classifier
    print("Training source classifier")

    classifier = gtl.models.LogisticRegression(src_input_dim,n_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(classifier.parameters(),lr=src_lr)
    accuracy = Accuracy(task='multiclass',num_classes=n_classes).to(device)


    # cross-validate using a sliding window of the training data
    validation_set_size = 5
    validation_start = 0
    validation_end = validation_start + validation_set_size
    validation_range = range(validation_start,validation_end+1)

    for epoch in tqdm(range(src_n_epochs)):

        classifier.train()

        # train on each training graph
        total_loss = 0
        total_accuracy = 0 

        for i,emb in enumerate(barbasi_train_embeddings):

            # this graph is in the validation set - skip
            if i in validation_range:
                continue

            optimizer.zero_grad()

            preds = classifier(emb)
            targets = barbasi_dgl_train[i].ndata['struct']

            loss = criterion(preds,targets)
            total_loss += loss
            total_accuracy += accuracy(preds,targets)

            loss.backward(retain_graph=True)
            optimizer.step()

        avg_loss = total_loss / len(barbasi_dgl_train)
        avg_accuracy = total_accuracy / len(barbasi_dgl_train)

        writer.add_scalar("source-classifier/training-loss",avg_loss,global_step=epoch)
        writer.add_scalar("source-classifier/training-accuracy",avg_accuracy,global_step=epoch)

        # Compute validation metrics
        classifier.eval()

        total_loss = 0
        total_accuracy = 0 
        
        for i in validation_range:
            emb = barbasi_train_embeddings[i]
            preds = classifier(emb)
            targets = barbasi_dgl_train[i].ndata['struct']

            loss = criterion(preds,targets)

            total_loss += loss
            total_accuracy += accuracy(preds,targets)

        avg_loss = total_loss / validation_set_size
        avg_accuracy = total_accuracy / validation_set_size

        writer.add_scalar("source-classifier/validation-loss",avg_loss,global_step=epoch)
        writer.add_scalar("source-classifier/validation-accuracy",avg_accuracy,global_step=epoch)


        # shift cross validation range
        validation_start += validation_set_size
        validation_end += validation_set_size

        if (validation_start >= len(barbasi_dgl_train) or validation_end >= len(barbasi_dgl_train)):
            validation_start = 0
            validation_end = validation_set_size

        validation_range = range(validation_start,validation_end+1)



    ###########################################################################
    # Direct transfer embeddings, but fine tune classifier.

    target_accuracy = 0.0

    forest_fire_feats = [degree_bucketing(g,max_degree_in_feat).to(device) for g in forest_fire_dgl]
    forest_fire_embeddings = [encoder(x).to(device) for x in forest_fire_feats]

    print("Training target classifier")

    classifier = gtl.models.LogisticRegression(src_input_dim,n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(),lr=src_lr)
    accuracy = Accuracy(task='multiclass',num_classes=n_classes).to(device)

    # cross-validate using sliding window of the training data
    validation_set_size = 5
    validation_start = 0
    validation_end = validation_start + validation_set_size
    validation_range = range(validation_start,validation_end+1)

    for epoch in tqdm(range(src_n_epochs)):
        classifier.train()

        total_loss = 0
        total_accuracy = 0 

        for i,emb in enumerate(forest_fire_embeddings):
            if i in validation_range:
                continue

            optimizer.zero_grad()

            preds = classifier(emb)

            targets = forest_fire_dgl[i].ndata['struct']

            loss = criterion(preds,targets)
            total_loss += loss
            total_accuracy += accuracy(preds,targets)

            loss.backward(retain_graph=True)
            optimizer.step()

        target_avg_loss = total_loss / len(forest_fire_dgl)
        target_avg_accuracy = total_accuracy / len(forest_fire_dgl)

        writer.add_scalar("target-classifier/training-loss",target_avg_loss,global_step=epoch)
        writer.add_scalar("target-classifier/training-accuracy",target_avg_accuracy,global_step=epoch)

        # Compute validation metrics
        classifier.eval()

        total_loss = 0
        total_accuracy = 0 
        
        for i in validation_range:
            emb = forest_fire_embeddings[i]
            preds = classifier(emb)
            targets = forest_fire_dgl[i].ndata['struct']

            loss = criterion(preds,targets)

            total_loss += loss
            total_accuracy += accuracy(preds,targets)

        target_avg_loss = total_loss / validation_set_size
        target_avg_accuracy = total_accuracy / validation_set_size

        writer.add_scalar("target-classifier/validation-loss",target_avg_loss,global_step=epoch)
        writer.add_scalar("target-classifier/validation-accuracy",target_avg_accuracy,global_step=epoch)


        # shift cross validation range
        validation_start += validation_set_size
        validation_end += validation_set_size

        if (validation_start >= len(forest_fire_dgl) or validation_end >= len(forest_fire_dgl)):
            validation_start = 0
            validation_end = validation_set_size

        validation_range = range(validation_start,validation_end+1)



    ###########################################################################
    # Write hyperparameters and results to tensorboard
    difference = avg_accuracy - target_avg_accuracy
    percent_difference =(avg_accuracy - target_avg_accuracy) / target_avg_accuracy

    writer.add_hparams(
            {'k': k, 
             'max-degree-in-feature': max_degree_in_feat,
             'encoder/lr': lr, 
             'encoder/hidden-layers ': n_hidden_layers, 
             'encoder/epochs':n_epochs,
             'encoder/type':'EGI',
             'source-classifier/epochs':src_n_epochs,
             'source-classifier/lr': src_lr,
             },
            {'source-classifier/validation-accuracy':avg_accuracy,
             'target/validation-accuracy':target_avg_accuracy,
             'difference':difference,
              '% difference':percent_difference})


if __name__ == '__main__':
    do_run()
