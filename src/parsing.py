import pickle
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx import is_connected, draw_kamada_kawai, connected_components
from ot import unif
import matplotlib.pyplot as plt

from .transform import transform
from .utils import make_p_dist

def parse(name="graphs_for_P34972"):
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
            """ draw_kamada_kawai(G)
            plt.show() """

        while label > i_class:
            classes.append([])
            i_class += 1
        classes[label].append(G)
    
    return classes

def parse_and_transform(name="graphs_for_P34972", pdv=2, pde=2, beta=0.5, Nmax=-1):
    classes = parse(name)
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
