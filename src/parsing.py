import pickle
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx import is_connected, connected_components
from ot import unif
from numpy.random import choice
import numpy as np

from .transform import transform
from .utils import make_p_dist

def parse(name="graphs_for_P34972", attention=None, percent=1, connected=True, smiles=True):
    """
    Parse a dataset.
    The dataset must be a pandas dataset with torch_geometric Data for graph representation

    Args :
    - name (string, default : "graphs_for_P34972") : name of the dataset. f"{name}.pkl" must be found in ./data
        Labels must be >= 0 integers
    - attention (string, default : None) : name of the attention file. If None, no attention will be used.
        Attention data must be a list of couples (index, attention matrix). It is only used on a subset of graphs.
    - percent (float, default : 1) : percentage of graphs to use per class
    - connected (bool, default : True) : if True, only the largest connected component will be kept.
    - smiles (bool, default : True) : if True, will use smiles stored in the dataset. Else, will not use smiles data.
    """
    with open(f"./data/{name}.pkl", "rb") as file:
        df = pickle.load(file)
    
    if not (attention is None):
        with open(f"./data/{attention}.pkl", "rb") as f:
            att = pickle.load(f)
    
    classes = [[]]
    i_class = 0
    attention_used = not (attention is None)

    if attention_used:
        #
        attention_index = att[0][0]
        attention_i = 0
        
    for i, l in enumerate(df.values):
        data, label = l[0], l[1]
        if smiles:
            G = to_networkx(
                    data, 
                    node_attrs=['x'], 
                    edge_attrs=['edge_attr'], 
                    graph_attrs=['smiles'],
                    to_undirected=True
                )
        else:
            G = to_networkx(
                    data, 
                    node_attrs=['x'], 
                    edge_attrs=['edge_attr'], 
                    to_undirected=True
                )
        
        G.graph['index'] = df.index[i]

        for e in G.edges:
            #Conversion into arrays
            G.edges[e]['edge_attr'] = np.array(G.edges[e]['edge_attr'])

        if attention_used and df.index[i] == attention_index:
            attention_g = att[attention_i][1]
            for n, u in enumerate(G.nodes):
                au = attention_g[0,:,n,:].flatten()
                G.nodes[u]['x'] = np.concatenate((G.nodes[u]['x'], au))

            attention_i += 1
            if attention_i < len(att):
                attention_index = att[attention_i][0]
            else:
                attention_index = -1
        
        if connected and not is_connected(G):
            print(f"Graph {df.index[i]} is not connected")
            largest_cc = max(connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()


        while label > i_class:
            classes.append([])
            i_class += 1
        classes[label].append(G)
    
    #Takes a percentage of all classes
    sampled_classes = [choice(np.array(c, dtype=type(classes[0][0])), (int(len(c)*percent),), replace=False) for c in classes]

    #Old code when a single index was forgotten
    """ for k, c in enumerate(sampled_classes):
        for i, G in enumerate(c):
            if G.graph['index'] == 52353:
                sampled_classes[k] = np.delete(c, i) """
    
    #Get the list of indexes of used graphs
    indexes = []
    for c in sampled_classes:
        indexes += [G.graph['index'] for G in c]
    indexes.sort()

    return sampled_classes, indexes

def parse_and_transform(name="graphs_for_P34972", attention=None, pdv=2, pde=2, beta=0.5, percent=1, connected=True, smiles=True):
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
    classes, indexes = parse(name, attention, percent, connected=connected, smiles=smiles)
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
        new_graphs.append(graphs)
        new_histos.append(histo)
    
    return new_graphs, new_histos, d, indexes

def get_espam(classes, indexes, espam="ESPAMS", choice_fun=(lambda x : x > 0), histo_fun=(lambda v : v/np.sum(v))):
    """
    Returns the subgraphs from all classes using ESPAM and the histograms used with ESPAM values

    Args :
    - classes : list of list of graphs
    - indexes : list of used indexes. Used to make a link between the index of a graph and the position in the espam values list
    - espam (string, default : "ESPAMS") : name of the ESPAM data. f"{espam}.pkl" must be stored in ./data
    - choice_fun (np.array -> bool. Defaut : lambda x : x > 0) : choice function used to select a sub graph. Called like this :
        `np.where(choice_fun(array_espam))[0]`
    - histo_fun (np.array -> float. Default : lambda v : v/np.sum(v)) : function used to make an histogram with ESPAM values. Used like this :
        `histo_fun(array_of_indexes_of_vertexes_to_keep)`

    Output :
    - new_classes : list of list of new graphs (one list per class)
    - histos : list of list of histograms for graphs. histos[c][i] correspond to new_classes[c][i]
    """
    with open(f"data/{espam}.pkl", "rb") as f:
        espam = pickle.load(f)
    
    new_classes = []
    histos = []

    for c in classes:
        new_c = []
        h = []
        for G in c:
            array_espam = np.array(espam[indexes.index(G.graph['index'])].tolist())
            espam_score = np.where(choice_fun(array_espam))[0]
            h.append(histo_fun(array_espam[espam_score]))
            G.graph['espam'] = sum(array_espam[espam_score])

            new_c.append(G.subgraph(espam_score))
        histos.append(h)
        new_classes.append(new_c)
    return new_classes, histos

def parse_transform_espam(name="graphs_for_P34972", espam="ESPAMS", attention=None, dv=make_p_dist(2), de=make_p_dist(2), beta=0.5, percent=1, connected=True, smiles=True, espam_histo=False, choice_fun=(lambda x : x > 0), histo_fun=(lambda v : v/np.sum(v))):
    """
    Combine parse, transform and ESPAM values.

    Args :
    - name (string, default : "graphs_for_P34972") : name of the dataset
    - espam (string, default : ESPAMS) : name of ESPAM data
    - attention (string, default : None) : name of attention data. If None, no attention will be used
    - dv (distance function, default : euclidian) : distance between node labels
    - de (distance function, default : euclidian) : distance between edge labels
    - beta (float, default : 0.5) : beta coefficient for transformation
    - percent (float, default : 1) : percentage of graphs to use
    - espam_histos (bool, default : False) : if True, then the histograms on nodes will be computed with ESPAM values. Else, histograms are uniform on nodes and on edges
    - choice_fun, histo_fun : same documentation as get_espam function
    """
    graphs, indexes = parse(name, attention=attention, percent=percent, connected=connected, smiles=smiles)
    espam_graphs, histos = get_espam(graphs, indexes, espam, choice_fun, histo_fun)

    new_graphs = []
    new_histos = []
    for i, c in enumerate(espam_graphs):
        gs = []
        hs = []
        for j, G in enumerate(c):
            hv = histos[i][j] if espam_histo else unif(G.number_of_nodes())
            he = unif(G.number_of_edges())
            G2, d, h = transform(G, dv, de, hv, he, beta)
            gs.append(G2)
            hs.append(h)
        new_graphs.append(gs)
        new_histos.append(hs)
    
    return new_graphs, new_histos, d, indexes