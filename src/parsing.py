import pickle
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx import is_connected, draw_kamada_kawai, connected_components
from ot import unif
from numpy.random import choice
import numpy as np

from .transform import transform
from .utils import make_p_dist

def parse(name="graphs_for_P34972", percent=1):
    with open(f"./data/{name}.pkl", "rb") as file:
        df = pickle.load(file)
    
    classes = [[]]
    i_class = 0
    for i, l in enumerate(df.values):
        data, label = l[0], l[1]
        G = to_networkx(
                data, 
                node_attrs=['x'], 
                edge_attrs=['edge_attr'], 
                graph_attrs=['smiles'],
                to_undirected=True
            )

        if not is_connected(G):
            print(f"Graph {df.index[i]} is not connected")
            largest_cc = max(connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        while label > i_class:
            classes.append([])
            i_class += 1
        classes[label].append(G)
    
    sampled_classes = [choice(np.array(c, dtype=type(classes[0][0])), (int(len(c)*percent),), replace=False) for c in classes]
    return sampled_classes

def parse_and_transform(name="graphs_for_P34972", pdv=2, pde=2, beta=0.5, Nmax=-1, percent=1):
    """
    Does the same as parse but pretransforms the graphs.

    Args :
        - name (str) : name
        - pdv (float, must be > 1, optionnal (default : 2)) : arg p for the p-distance used for labels.
        - pde (float, must be > 1, optionnal (default : 2)) : arg p for the p-distance used for edges.
        - beta (float, must be between 0 and 1) : arg for the transformation
        - Nmax (int) : max number of elements to put in a same class. -1 means every element.
    
    Output :
        - new_graphs (list of lists of nx.Graph) : list of graphs grouped by classe
        - new_histos (list of list of histograms) : list of histograms grouped by classes. new_histos[c][i] correspond to new_graph[c][i]
        - d : new distance function
    """
    classes = parse(name, percent)
    new_graphs = []
    new_histos = []
    for c in classes:
        graphs = []
        histo = []
        for G in c:
            dv = make_p_dist(pdv)
            de = make_p_dist(pde)
            hv = unif(G.number_of_nodes())
            he = unif(G.number_of_edges())
            G2, d, h = transform(G, dv, de, hv, he, beta)
            graphs.append(G2)
            histo.append(h)
        if Nmax == -1:
            new_graphs.append(graphs)
            new_histos.append(histo)
        else:
            new_graphs.append(graphs[:Nmax])
            new_histos.append(histo[:Nmax])
    
    return new_graphs, new_histos, d
