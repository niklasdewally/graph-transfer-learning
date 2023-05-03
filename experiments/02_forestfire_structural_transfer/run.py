"""
Consider a BA and forest fire graph. Give these structural labels by checking
k-hop neighbor similarilty (using WL).

Aim is to direct transfer labels from one to another.
"""

import datetime
import time
from random import shuffle

import dgl
from dgl.dataloading import DataLoader

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy

from tqdm import tqdm


import graphtransferlearning as gtl
from graphtransferlearning.features import degree_bucketing
from graphtransferlearning.models import EGI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_synthetic_datasets(
    n_nodes=100, n_graphs=60, ba_attached=2, p_forward=0.4, p_backward=0.3, k=2
):
    """
    Generate the synthetic graphs used for the experiment, alongside their structural labels.

    Returns:
        ba_graphs,ff_graphs,classes

        ba_graphs: A list of barbasi-albert graphs, as DGLGraph format.
        ff_graphs: A list of forest-fire graphs, as DGLGraph format.
        classes: A tensor containing all possible classes.

        The structural labels can be accessed from these graphs by doing:
            ba_graphs[i].ndata['struct']
    """

    g = gtl.generate_forest_fire(n_nodes, p_forward, p_backward)

    g, classes = gtl.add_structural_labels(g, k)

    ff_graphs = [dgl.from_networkx(g, node_attrs=["struct"]).to(device)]

    print(f"Generating {n_graphs} forest-fire source graphs with {n_nodes} nodes each")
    for i in tqdm(range(1, n_graphs)):
        g = gtl.generate_forest_fire(n_nodes, p_forward, p_backward)
        g, classes = gtl.add_structural_labels(g, k, existing_labels=classes)

        ff_graphs.append(dgl.from_networkx(g, node_attrs=["struct"]).to(device))

    print(f"Generating {n_graphs} barbasi source graphs with {n_nodes} nodes each")
    ba_graphs = []
    for i in tqdm(range(0, n_graphs)):
        g = gtl.generate_barbasi(n_nodes, ba_attached)
        g, classes = gtl.add_structural_labels(g, k, existing_labels=classes)

        ba_graphs.append(dgl.from_networkx(g, node_attrs=["struct"]).to(device))

    return ba_graphs, ff_graphs, torch.tensor(list(classes.values())).to(device)


def do_run(
    k=2,
    encoder_lr=0.01,
    encoder_hidden_layers=32,
    encoder_epochs=80,
    weight_decay=0.0,
    sampler_type="egi",
    classifier_lr=0.1,
    classifier_epochs=100,
    run_name=None,
):
    """
    Perform a specific run of the model for a given set of hyperparameters.
    """

    results = dict()
    hparams = dict()  # used for tensorboard
    if run_name is None:
        writer = SummaryWriter()  # outputs to ./runs by default
    else:
        writer = SummaryWriter(log_dir=run_name)

    layout = {
        "Encoders": {
            "base-loss": [
                "Multiline",
                [
                    "base-encoder/training-loss",
                    "base-encoder/validation-loss",
                ],
            ],
            "transfer-loss": [
                "Multiline",
                ["transfer-encoder/training-loss", "transfer-encoder/validation-loss"],
            ],
        },
        "Classifiers": {
            "base-loss": [
                "Multiline",
                ["base-classifier/training-loss", "base-clasifier/validation-loss"],
            ],
            "transfer-loss": [
                "Multiline",
                [
                    "transfer-classifier/training-loss",
                    "transfer-clasifier/validation-loss",
                ],
            ],
            "transfer-accuracy": [
                "Multiline",
                [
                    "transfer-classifier/training-accuracy",
                    "transfer-classifier/validation-accuracy",
                ],
            ],
            "base-accuracy": [
                "Multiline",
                [
                    "base-classifier/training-accuracy",
                    "base-classifier/validation-accuracy",
                ],
            ],
        },
    }

    writer.add_custom_scalars(layout)

    # Encoder hyper parameters
    max_degree_in_feat = encoder_hidden_layers
    feature_mode = "degree_bucketing"

    hparams.update(
        {
            "encoder-lr": encoder_lr,
            "encoder-hidden-layers": encoder_hidden_layers,
            "encoder-epochs": encoder_epochs,
            "encoder-max-feature-degree": max_degree_in_feat,
            "encoder-weight-decay": weight_decay,
            "encoder-feature-mode": feature_mode,
            "model-type": sampler_type,
            "k": k,
            "classifier-lr": classifier_lr,
            "classifier-epochs": classifier_epochs,
        }
    )

    ###########################################################################

    # Generate synthetic graphs

    ff_graphs, ba_graphs, classes = generate_synthetic_datasets(k=k)

    n_classes = len(classes)

    # Create test and validation sets, and generate node features based on degrees.

    shuffle(ba_graphs)
    ba_val = ba_graphs[:20]
    ba_train = ba_graphs[20:]

    ba_train_feats = [
        degree_bucketing(g, max_degree_in_feat).to(device) for g in ba_train
    ]
    ba_val_feats = [degree_bucketing(g, max_degree_in_feat).to(device) for g in ba_val]

    shuffle(ff_graphs)
    ff_val = ff_graphs[:20]
    ff_train = ff_graphs[20:]

    ff_train_feats = [
        degree_bucketing(g, max_degree_in_feat).to(device) for g in ff_train
    ]
    ff_val_feats = [degree_bucketing(g, max_degree_in_feat).to(device) for g in ff_val]

    ###########################################################################
    print("==== 1: Base case - forest - no transfer.")

    # see training/egi.py for more info
    model = EGI(
        ff_train_feats[0].shape[1],
        encoder_hidden_layers,
        k + 1,
        nn.PReLU(encoder_hidden_layers),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=encoder_lr, weight_decay=weight_decay
    )

    # sample k hop ego-graphs with max 10 neighbors each hop
    if sampler_type == "egi":
        sampler = dgl.dataloading.NeighborSampler([10 for i in range(k)])
    elif sampler_type == "triangle":
        sampler = gtl.KHopTriangleSampler([10 for i in range(k)])
    else:
        raise NotImplementedError(f"Sampler {sampler_type} is not implemented!")

    print("Training encoder")
    for epoch in tqdm(range(encoder_epochs)):
        model.train()

        time.time()

        loss = 0.0

        # train based on features and ego graphs around specific egos
        for i, g in enumerate(ff_train):
            # the sampler returns a list of blocks and involved nodes
            # each block holds a set of edges from a source to destination
            # each block is a hop in the graph
            # perform mini-batch training
            features = ff_train_feats[i]
            optimizer.zero_grad()
            for blocks in DataLoader(
                g, g.nodes(), sampler, device=device, batch_size=20, shuffle=True
            ):
                l = model(g, features, blocks)
                l.backward()
                optimizer.step()
                loss += l

        writer.add_scalar(
            "base-encoder/training-loss", loss / len(ff_train), global_step=epoch
        )

        # Calculate validation metrics.

        model.eval()
        loss = 0.0

        for i, g in enumerate(ff_val):
            features = ff_val_feats[i]
            for blocks in DataLoader(
                g, g.nodes(), sampler, device=device, batch_size=20, shuffle=True
            ):
                l = model(g, features, blocks)
                loss += l

        writer.add_scalar(
            "base-encoder/validation-loss", loss / len(ff_val), global_step=epoch
        )

    results.update({"hp/base-encoder-loss": loss})

    # classifier preparation
    encoder = model.encoder

    # we dont use validation set here - instead, we cross validate on the training set only.
    ff_train_embeddings = [
        encoder(g, x).to(device) for g, x in zip(ff_train, ff_train_feats)
    ]

    cl_input_dim = max_degree_in_feat

    # Train source classifier
    print("Training classifier")

    classifier = gtl.models.LogisticRegression(cl_input_dim, n_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr)

    accuracy = Accuracy(task="multiclass", num_classes=n_classes).to(device)

    # cross-validate using a sliding window of the training data
    validation_set_size = 5
    validation_start = 0
    validation_end = validation_start + validation_set_size
    validation_range = range(validation_start, validation_end + 1)

    for epoch in tqdm(range(classifier_epochs)):
        classifier.train()

        # train on each training graph
        total_loss = 0
        total_accuracy = 0

        for i, emb in enumerate(ff_train_embeddings):
            # this graph is in the validation set - skip
            if i in validation_range:
                continue

            optimizer.zero_grad()

            preds = classifier(emb)
            targets = ff_train[i].ndata["struct"]

            loss = criterion(preds, targets)
            total_loss += loss
            total_accuracy += accuracy(preds, targets)

            loss.backward(retain_graph=True)
            optimizer.step()

        avg_loss = total_loss / len(ff_train)
        avg_accuracy = total_accuracy / len(ff_train)

        writer.add_scalar("base-classifier/training-loss", avg_loss, global_step=epoch)
        writer.add_scalar(
            "base-classifier/training-accuracy", avg_accuracy, global_step=epoch
        )

        # Compute validation metrics.
        classifier.eval()

        total_loss = 0
        total_accuracy = 0

        for i in validation_range:
            emb = ff_train_embeddings[i]
            preds = classifier(emb)
            targets = ff_train[i].ndata["struct"]

            loss = criterion(preds, targets)

            total_loss += loss
            total_accuracy += accuracy(preds, targets)

        val_loss = total_loss / validation_set_size
        val_accuracy = total_accuracy / validation_set_size

        writer.add_scalar(
            "base-classifier/validation-loss", val_loss, global_step=epoch
        )
        writer.add_scalar(
            "base-classifier/validation-accuracy", val_accuracy, global_step=epoch
        )

        # shift cross validation range
        validation_start += validation_set_size
        validation_end += validation_set_size

        if validation_start >= len(ff_train) or validation_end >= len(ff_train):
            validation_start = 0
            validation_end = validation_set_size

        validation_range = range(validation_start, validation_end + 1)

    results.update(
        {"hp/base-accuracy": val_accuracy, "hp/base-classifier-loss": val_loss}
    )
    ###########################################################################
    print(
        "==== 2: Transfer case - directly transfer barbasi encoder. Use this to train a forest-fire classifier."
    )

    # see training/egi.py for more info
    model = EGI(
        ba_train_feats[0].shape[1],
        encoder_hidden_layers,
        k + 1,
        nn.PReLU(encoder_hidden_layers),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=encoder_lr, weight_decay=weight_decay
    )

    # sample k hop ego-graphs with max 10 neighbors each hop
    sampler = dgl.dataloading.NeighborSampler([10 for i in range(k)])

    print("Training encoder")
    for epoch in tqdm(range(encoder_epochs)):
        model.train()

        time.time()

        loss = 0.0

        # train based on features and ego graphs around specific egos
        for i, g in enumerate(ff_train):
            optimizer.zero_grad()
            features = ba_train_feats[i]

            # the sampler returns a list of blocks and involved nodes
            # each block holds a set of edges from a source to destination
            # each block is a hop in the graph
            # perform mini-batch training
            for blocks in DataLoader(
                g, g.nodes(), sampler, device=device, batch_size=20, shuffle=True
            ):
                l = model(g, features, blocks)
                l.backward()
                optimizer.step()
                loss += l

        writer.add_scalar(
            "transfer-encoder/training-loss", loss / len(ba_train), global_step=epoch
        )

        # Calculate validation metrics.

        model.eval()
        loss = 0.0

        for i, g in enumerate(ba_val):
            features = ba_val_feats[i]
            for blocks in DataLoader(
                g, g.nodes(), sampler, device=device, batch_size=20, shuffle=True
            ):
                l = model(g, features, blocks)
                loss += l

        writer.add_scalar(
            "transfer-encoder/validation-loss", loss / len(ba_val), global_step=epoch
        )

    results.update({"hp/transfer-encoder-loss": loss})

    # classifier preparation
    encoder = model.encoder

    # we dont use validation set here - instead, we cross validate on the training set only.
    ff_train_embeddings = [
        encoder(g, x).to(device) for g, x in zip(ff_train, ff_train_feats)
    ]

    # Source classifier hyperparameters
    cl_input_dim = max_degree_in_feat

    # Train source classifier
    print("Training classifier")

    classifier = gtl.models.LogisticRegression(cl_input_dim, n_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr)

    accuracy = Accuracy(task="multiclass", num_classes=n_classes).to(device)

    # cross-validate using a sliding wturindow of the training data
    validation_set_size = 5
    validation_start = 0
    validation_end = validation_start + validation_set_size
    validation_range = range(validation_start, validation_end + 1)

    for epoch in tqdm(range(classifier_epochs)):
        classifier.train()

        # train on each training graph
        total_loss = 0
        total_accuracy = 0

        for i, emb in enumerate(ff_train_embeddings):
            # this graph is in the validation set - skip
            if i in validation_range:
                continue

            optimizer.zero_grad()

            preds = classifier(emb)
            targets = ff_train[i].ndata["struct"]

            loss = criterion(preds, targets)
            total_loss += loss
            total_accuracy += accuracy(preds, targets)

            loss.backward(retain_graph=True)
            optimizer.step()

        avg_loss = total_loss / len(ff_train)
        avg_accuracy = total_accuracy / len(ff_train)

        writer.add_scalar(
            "transfer-classifier/training-loss", avg_loss, global_step=epoch
        )
        writer.add_scalar(
            "transfer-classifier/training-accuracy", avg_accuracy, global_step=epoch
        )

        # Compute validation metrics.
        classifier.eval()

        total_loss = 0
        total_accuracy = 0

        for i in validation_range:
            emb = ff_train_embeddings[i]
            preds = classifier(emb)
            targets = ff_train[i].ndata["struct"]

            loss = criterion(preds, targets)

            total_loss += loss
            total_accuracy += accuracy(preds, targets)

        val_loss = total_loss / validation_set_size
        val_accuracy = total_accuracy / validation_set_size

        writer.add_scalar(
            "transfer-classifier/validation-loss", val_loss, global_step=epoch
        )
        writer.add_scalar(
            "transfer-classifier/validation-accuracy", val_accuracy, global_step=epoch
        )

        # shift cross validation range
        validation_start += validation_set_size
        validation_end += validation_set_size

        if validation_start >= len(ff_train) or validation_end >= len(ff_train):
            validation_start = 0
            validation_end = validation_set_size

        validation_range = range(validation_start, validation_end + 1)

    results.update(
        {"hp/transfer-accuracy": val_accuracy, "hp/transfer-classifier-loss": val_loss}
    )

    ###########################################################################
    # Write hyperparameters and results to tensorboard

    difference = results["hp/transfer-accuracy"] - results["hp/base-accuracy"]
    results.update({"hp/difference": difference})
    writer.add_hparams(hparams, results)

    return results


if __name__ == "__main__":
    n = 10

    time_now = datetime.datetime.now().strftime("%y%m%dT%H%M")
    SAMPLERS = ["egi", "triangle"]
    for sampler in SAMPLERS:
        for i in range(n):
            print(f"Running experiment for {sampler} sampler.")
            results = do_run(
                sampler_type=sampler, run_name=f"./runs/{time_now}/{sampler}/{i}"
            )
