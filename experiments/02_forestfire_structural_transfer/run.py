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

def do_run(k=2):
    """
    Perform a specific run of the model for a given set of hyperparameters.
    """

    writer = SummaryWriter() # outputs to ./runs by default

    # synthetic graph generation parameters
    n_nodes = 100
    n_graphs = 40
    ba_attached= 2
    p_forward = 0.4
    p_backward = 0.3


    # Encoder hyper parameters
    lr = 0.01
    n_hidden_layers=10
    n_epochs=40
    max_degree_in_feat = n_hidden_layers
    weight_decay = 0.
    feature_mode='degree_bucketing'
    optimiser='adam'


    # Generate synthetic graphs



    g = gtl.generate_forest_fire(n_nodes,p_forward,p_backward)
    g,classes = gtl.add_structural_labels(g,k)

    # store graphs in two formats for easy use
    forest_fire_nx = [g]
    forest_fire_dgl = [dgl.from_networkx(g).to(device)]

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


    barbassi_classes = classes.values()

    # Test-validate split
    shuffle(barbasi_dgl)
    barbasi_dgl_val = barbasi_dgl[:n_graphs//10]
    barbasi_dgl_train = barbasi_dgl[n_graphs//10:]

    print(f"Training EGI encoder on barbasi graphs")


    # Train model
    
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
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
        for epoch in tqdm(range(n_epochs)):
            model.train()

            t0 = time.time()

            loss = 0.0
            
            # train based on features and ego graphs around specific egos
            for i,g in enumerate(barbasi_dgl_train):
                for ego in sample(list(g.nodes()),n_nodes):
                    optimizer.zero_grad()

                    l = model(barbasi_train_feats[i],ego)
                    l.backward()

                    loss += l
                    
                optimizer.step()

            writer.add_scalar(f"encoder/training-loss",loss,global_step=epoch)


            # calculate validation loss
            model.eval()
            loss = 0.0
            for i,g in enumerate(barbasi_dgl_val):
                for ego in sample(list(g.nodes()),n_nodes):

                    l = model(barbasi_train_feats[i],ego)
                    loss += l
                    

            writer.add_scalar(f"encoder/validation-loss",loss,global_step=epoch)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    ###############################


    # source classifier preparation
    encoder = model.encoder

    barbasi_train_embeddings = encoder(barbasi_train_features).to(device)
    barbasi_val_embeddings = encoder(barbasi_train_features).to(device)
    # barbassi_classes defined earlier

    # Source classifier hyperparameters
    src_n_epochs = 50
    src_lr = 0.1

    src_input_dim = max_degree_in_feat
    src_n_classes = len(barbassi_classes.keys())


    # Train source classifier
    print("Training source classifier")

    classifier = gtl.models.LogisticRegression(src_input_dim,src_n_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(classifier.parameters(),lr=src_lr)
    accuracy = Accuracy(task='multiclass',num_classes=src_n_classes).to(device)


    for epoch in tqdm(range(src_n_epochs)):
        classifier.train()

        # train on each training graph
        total_loss = 0
        total_accuracy = 0 

        for i,emb in enumerate(barbasi_train_embeddings):
            optimizer.zero_grad()
            preds = classifier(emb)
            loss = criterion(preds,barbasi_train_feats[i])
            total_loss += loss
            total_accuracy += accuracy(preds,barbasi_train_feats[i])

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
        
        for i,emb in enumerate(barbasi_val_embeddings):
            preds = classifier(emb)
            loss = criterion(preds,barbasi_val_feats[i])

            total_loss += loss
            total_accuracy += accuracy(preds,barbasi_val_feats[i])

        avg_loss = total_loss / len(barbasi_dgl_val)
        avg_accuracy = total_accuracy / len(barbasi_dgl_val)

        writer.add_scalar("source-classifier/validation-loss",avg_loss,global_step=epoch)
        writer.add_scalar("source-classifier/validation-accuracy",avg_accuracy,global_step=epoch)


    
    # Write hyperparameters to tensorboard
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
            {'source-classifier/validation-accuracy':avg_accuracy})

    # Transfer to target graph

    # Create target encoder (no finetuning)

    # Create target classifier (no finetuning)
    target_accuracy = 0;

if __name__ == '__main__':
    do_run()
