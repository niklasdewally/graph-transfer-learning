import datetime

import dgl
import graphtransferlearning as gtl
import numpy as np
import torch
import torch.nn as nn
from graphtransferlearning.features import degree_bucketing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# some experimental constants
HIDDEN_LAYERS = 32
PATIENCE = 10
EPOCHS = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    n = 10

    current_date_time = datetime.datetime.now().strftime("%Y%m%dT%H:%M")

    samplers = ["egi", "triangle"]

    for sampler in samplers:
        for i in range(n):
            log_dir = f"./runs/{current_date_time}/{sampler}/{i}"

            print(f"Running experiment for {sampler} : {i+1}/{n}")

            do_run(log_dir, sampler)


def load_dataset(edgefile, labelfile):
    edges = np.loadtxt(edgefile, dtype="int")
    us = torch.from_numpy(edges[:, 0]).to(device)
    vs = torch.from_numpy(edges[:, 1]).to(device)

    dgl_graph = dgl.graph((us, vs), device=torch.device("cpu"))
    dgl_graph = dgl.to_bidirected(dgl_graph).to(device)

    labels = np.loadtxt(labelfile, skiprows=1)

    return dgl_graph, labels[:, 1]


def do_run(log_dir, sampler):
    # setup tensorboard logging and visualisation.
    writer = SummaryWriter(log_dir)
    layout = {
        "Experiment": {
            "Source Encoder Loss": [
                "Multiline",
                ["src/training-loss", "src/validation-loss"],
            ],
        }
    }

    writer.add_custom_scalars(layout)

    # keep track of summary results and experimental parameters for tensorboard
    # to store later on.
    results = dict()
    hparams = {"model": sampler}

    ##########################################################################
    #                            DATA LOADING                                #
    ##########################################################################

    europe_g, europe_labels = load_dataset(
        "data/europe-airports.edgelist", "data/labels-europe-airports.txt"
    )

    # usa_g, usa_labels = load_dataset('data/usa-airports.edgelist',
    #                                 'data/labels-usa-airports.txt')

    brazil_g, brazil_labels = load_dataset(
        "data/brazil-airports.edgelist", "data/labels-brazil-airports.txt"
    )

    # node features for encoder
    europe_node_feats = degree_bucketing(europe_g, HIDDEN_LAYERS).to(device)
    brazil_node_feats = degree_bucketing(brazil_g, HIDDEN_LAYERS).to(device)

    ##########################################################################
    #                     TRAIN SOURCE ENCODER (EUROPE)                      #
    ##########################################################################

    print("Training encoder for source data (Europe).")

    encoder = gtl.training.train_egi_encoder(
        europe_g,
        gpu=0,
        kfolds=10,
        sampler=sampler,
        n_hidden_layers=HIDDEN_LAYERS,
        patience=PATIENCE,
        save_weights_to="srcmodel.pickle",
        writer=writer,
        tb_prefix="src",
    )

    embs = encoder(europe_g, europe_node_feats).to(torch.device("cpu")).detach().numpy()

    train_embs, val_embs, train_classes, val_classes = train_test_split(
        embs, europe_labels
    )

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(train_embs, train_classes)

    score = classifier.score(val_embs, val_classes)
    results.update({"hp/src-accuracy": score})

    print(f"The source classifier has an accuracy score of {score}")

    ##########################################################################
    #                DIRECT TRANSFER TARGET ENCODER (BRAZIL)                 #
    ##########################################################################

    target_encoder = gtl.models.EGI(
        brazil_node_feats.shape[1],
        HIDDEN_LAYERS,
        2,  # see gtl.training.egi
        nn.PReLU(HIDDEN_LAYERS),
    ).to(device)

    target_encoder.load_state_dict(torch.load("srcmodel.pickle"), strict=False)

    target_encoder = target_encoder.encoder
    target_embs = (
        target_encoder(brazil_g, brazil_node_feats)
        .to(torch.device("cpu"))
        .detach()
        .numpy()
    )

    train_embs, val_embs, train_classes, val_classes = train_test_split(
        target_embs, brazil_labels
    )

    classifier = SGDClassifier(max_iter=1000)
    classifier = classifier.fit(train_embs, train_classes)

    score = classifier.score(target_embs, brazil_labels)

    results.update({"hp/target-accuracy": score})

    print(f"The target classifier has an accuracy score of {score}")

    ##########################################################################
    #                      WRITE RESULTS TO TENSORBOARD                      #
    ##########################################################################

    difference = results["hp/target-accuracy"] - results["hp/src-accuracy"]
    results.update({"hp/difference": difference})
    writer.add_hparams(hparams, results)

    return results


if __name__ == "__main__":
    main()
