import networkx as nx
from dgl import DGLGraphStale

class GraphDataset():
    """
    An immutable wrapper class for an input dataset.

    A GraphDataset holds different representations of the graph and node labels
    together for easy use.

    Graphs can be accessed as:
        - NetworkX Graph objects.
        - DGLGraphStale objects.

    Labels can be accessed as:
        - A dictionary from nodeID to value.
        - A LongTensor.
    """

    def __init__(self,edges_path,labels_path):
        """
        Load a dataset from disk.

        Args:

            edges_path: A file containing edges. This must be in the form:

                <int> <int>
                <int> <int>

                where each line contains the integer IDs of the nodes on each edge.

        
            labels_path: A file containing node labels. This must be in the form:
                SOME  HEADER
                <int> <int>
                <int> <int>

                where each line contains the integer ID of a node, followed by its label.

        Returns:
            a new GraphDataset
    
        """

        _read_files(edges_path,label_path)


    def _read_files(self,edges_path,label_path):

        nx_graph  = nx.Graph()

        # somewhat adapted from original EGI code

        label_dict = dict()

        with open(label_path) as inp:
            inp.readline() # skip headers
            for line in inp:
                tmp = line.strip().split(' ')
                label_dict[int(tmp[0])] = int(tmp[1])


        # initialise networkx
        with open(edges_path) as inp:
            for i,line in enumerate(inp):
                tmp = line.strip().split()

                a,b = int(tmp[0]),int(tmp[1])
                nx_graph.add_edge(a,b,id=i) # give edges ids so dgl retains ordering

        dgl_graph = DGLGraphStale()
        dgl_graph = dgl_graph.from_networkx(nx_graph)

        # label tensor
        label_tensor = []
        for node in sorted(list(nx_graph.nodes())):
            label_tensor.append(label_dict[node])
        
        label_tensor = torch.LongTensor(label_tensor)

        self.label_tensor = label_tensor
        self.nx_graph = nx_graph
        self.label_dict = label_dict
        self.dgl_graph = dgl_graph

    def nx_graph(self):
        return self.nx_graph.copy()
    
    def dgl_graph(self):
        return self.dgl_graph

    def label_dict(self):
        return self.label_dict

    def label_tensor(self):
        return self.label_tensor
    



