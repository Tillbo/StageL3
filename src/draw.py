import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import MolFromSmiles
from os import mkdir

def plot(G, mapv, mape=None, title="", n=1, m=1, i=1):
    """
    Plot the graph with the kamanda_kawai layout

    Args :
        - G (networkx.Graph()) : Graph to plot
        - mapv (dict, list) : map between labels of nodes and colors
        - mape (dict, list, optionnal (default : None)) : map between labels of edges and width. If None, then edge with is 3
        - title (string) : title
        - n, m, i (ints) : parameters for subplot
    """
    plt.subplot(n, m, i)
    plt.title(title)

    node_labels = {}
    edge_width = []
    node_colors = []

    for u in G:
        node_labels[u] = mapv[G.nodes[u]['x']][0]
        node_colors.append(mapv[G.nodes[u]['x']][1])
    
    if mape is None:
        edge_width = 3
    else:
        for (u, v) in G.edges:
            edge_width.append(mape[G[u][v]['edge_attr']])
    
    nx.draw_kamada_kawai(G, node_color=node_colors, labels=node_labels, width=edge_width)

def save_mol_folder(Gs, folder, names):
    try:
        mkdir(f"save/{folder}")
    except:
        pass
    for G, name in zip(Gs, names):
        MolToFile(MolFromSmiles(G.graph['smiles']), f"save/{folder}/{name}.png")