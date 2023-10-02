import networkx as nx
import wandb
import sys
import dgl
import gtl.gcmpy
import gtl.training
from gtl import Graph
import gtl.features
from dgl.sampling import global_uniform_negative_sampling
import torch
from torch import Tensor
import pathlib
import datetime
from sklearn.linear_model import SGDClassifier
from gcmpy import (
    JointDegreeMarginal,
    JointDegreeNames,
    GCMAlgorithmNames,
    GCMAlgorithmNetwork,
    poisson,
    clique_motif,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


source_graph: Graph
target_graph: Graph

#######################
#        PATHS        #
#######################

SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.parent.resolve()
HYPERPARAMS_DIR: pathlib.Path = SCRIPT_DIR / "aug_link_prediction_hyperparams"
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "2023-08-poisson"

current_date_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
group = f"{current_date_time}"


def generate_graph(avg_k: float, avg_t: float) -> Graph:
    clique_size = 3

    params = {}
    params[JointDegreeNames.MOTIF_SIZES] = [2, clique_size]
    params[JointDegreeNames.ARR_FP] = [poisson(avg_k), poisson(avg_t)]

    params[JointDegreeNames.LOW_HIGH_DEGREE_BOUND] = [(0, 30), (0, 30)]

    DegreeDistObj = JointDegreeMarginal(params)
    jds = DegreeDistObj.sample_jds_from_jdd(5000)

    params = {}
    params[GCMAlgorithmNames.MOTIF_SIZES] = [2, clique_size]
    params[GCMAlgorithmNames.EDGE_NAMES] = ["2-clique", f"{clique_size}-clique"]
    params[GCMAlgorithmNames.BUILD_FUNCTIONS] = [clique_motif, clique_motif]
    g = GCMAlgorithmNetwork(params).random_clustered_graph(jds)

    G: nx.Graph = g.G

    # no self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # ensure connectivity using greatest connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    # ensure no parallel edges
    G = nx.Graph(G)

    # ensure consecutive node labels after having removed some nodes
    G = nx.convert_node_labels_to_integers(G)

    g = Graph(G)
    return g


def main() -> int:
    for avg_k in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]:
        global source_graph, target_graph
        source_graph = generate_graph(avg_k, avg_k)
        target_graph = generate_graph(avg_k, avg_k)
        for i in range(4):
            wandb.init(
                project="Phase Change",
                entity="sta-graph-transfer-learning",
                group=f"{current_date_time}",
                name=f"{avg_k}-{i}",
            )

            model_config = gtl.load_model_config(HYPERPARAMS_DIR, "graphsage-mean")
            wandb.config.update(model_config)
            wandb.config.update({"model": "graphsage-mean"})
            wandb.config.update({"avg_k": avg_k})

            do_run()
            wandb.finish()

    return 0


def do_run() -> None:
    features = gtl.features.degree_bucketing(
        source_graph.as_dgl_graph(device), wandb.config["hidden_layers"]
    ).to(device)

    encoder = gtl.training.train(
        wandb.config["model"], source_graph, features, wandb.config, device=device
    )

    embs = encoder(source_graph.as_dgl_graph(device), features)

    source_dgl: dgl.DGLGraph = source_graph.as_dgl_graph(device)
    neg_us, neg_vs = global_uniform_negative_sampling(
        source_dgl, source_dgl.num_edges() // 2
    )

    # sample randomly edges from the graph. Only sample the same amount as the amount of negative
    # edges we found

    random_idxs = torch.randperm(source_dgl.num_edges() // 2)[: neg_us.shape[0]]
    pos_us, pos_vs = source_graph.as_dgl_graph(device).edges()
    pos_us = pos_us[random_idxs]
    pos_vs = pos_vs[random_idxs]

    edges = torch.cat(
        (
            _get_edge_embeddings(embs, pos_us, pos_vs),
            _get_edge_embeddings(embs, neg_us, neg_vs),
        )
    )

    classes = torch.cat((torch.ones(pos_us.shape[0]), torch.zeros(neg_us.shape[0])))

    random_idxs = torch.randperm(edges.shape[0])
    edges = edges[random_idxs]
    classes = classes[random_idxs]

    edges_np = edges.detach().cpu().numpy()
    classes_np = classes.detach().cpu().numpy()

    classifier = SGDClassifier(max_iter=1000, loss="log_loss")
    classifier = classifier.fit(edges_np, classes_np)
    wandb.summary["training-acc"] = classifier.score(edges_np, classes_np)

    target_dgl: dgl.DGLGraph = target_graph.as_dgl_graph(device)
    features = gtl.features.degree_bucketing(
        target_dgl, wandb.config["hidden_layers"]
    ).to(device)

    embs = encoder(target_dgl, features)

    neg_us, neg_vs = global_uniform_negative_sampling(
        target_dgl, target_dgl.num_edges() // 2
    )

    random_idxs = torch.randperm(target_dgl.num_edges() // 2)[: neg_us.shape[0]]
    pos_us, pos_vs = target_graph.as_dgl_graph(device).edges()
    pos_us = pos_us[random_idxs]
    pos_vs = pos_vs[random_idxs]

    edges = torch.cat(
        (
            _get_edge_embeddings(embs, pos_us, pos_vs),
            _get_edge_embeddings(embs, neg_us, neg_vs),
        )
    )

    classes = torch.cat((torch.ones(pos_us.shape[0]), torch.zeros(neg_us.shape[0])))

    random_idxs = torch.randperm(edges.shape[0])
    edges = edges[random_idxs]
    classes = classes[random_idxs]

    edges_np = edges.detach().cpu().numpy()
    classes_np = classes.detach().cpu().numpy()

    wandb.summary["acc"] = classifier.score(edges_np, classes_np)


def _load_graphs(size: int) -> list[Graph]:
    graphs: list[Graph] = []

    def filename(i: int) -> str:
        return f"poisson-{size}-3-{i}.gml"

    for i in range(100):
        graph = Graph.from_gml_file(DATA_DIR / filename(i))
        graphs.append(graph)

    return graphs


def _get_edge_embeddings(embs: Tensor, us: Tensor, vs: Tensor) -> Tensor:
    # Based on testing, concat is the only method that gets >0.5 accuracy on GraphSAGE.
    # (I exclude l1,l2 as these are binary only, so will not extend to triangle prediction)

    # HADAMARD
    # return embs[us] * embs[vs]

    # return F.cosine_similarity(embs[us],embs[vs],dim=1)

    # INNER
    # out = torch.empty((us.shape[0],1),device=embs.get_device())
    # for i in range(us.shape[0]):
    # out[i] = torch.inner(embs[us[i]],embs[vs[i]])

    # CONCAT
    out = torch.empty((us.shape[0], embs.shape[1] * 2), device=embs.get_device())
    for i in range(us.shape[0]):
        out[i] = torch.cat((embs[us[i]], embs[vs[i]]))
    return out

    # AVG
    # out = torch.empty((us.shape[0],embs.shape[1]),device=embs.get_device())
    # for i in range(us.shape[0]):
    #    out[i] = embs[us[i]] + embs[vs[i]] / 2
    # return out

    # ABS Sub
    # out = torch.empty((us.shape[0],embs.shape[1]),device=embs.get_device())
    # for i in range(us.shape[0]):
    #    out[i] = torch.abs(embs[us[i]] - embs[vs[i]])
    # return out


if __name__ == "__main__":
    sys.exit(main())
