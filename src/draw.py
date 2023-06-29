import networkx as nx
import matplotlib.pyplot as plt

def plot(G, mapv, mape=None, title="", n=1, m=1, i=1) -> None:
    """
    Plot the graph with the kamanda_kawai layout
    Labels must be integers
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