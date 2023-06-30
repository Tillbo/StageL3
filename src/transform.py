import networkx as nx
import numpy as np
from .const import *

def transform(G : nx.Graph(), dv, de, hv, he, beta) :
    """
    Takes a graph G with labels on nodes and edges.
    Labels are respectively denoted by 'v' and 'e'

    Args :
        - G (nx.Graph) : input graph
        - dv : distance function on node labels
        - de : distance function on edge labels
        - hv : histogram on vertex labels
        - he : histogram on edge labels
        - beta : beta coefficient for the histogram : h(u) = beta*hv(u) ; h(e) = (1-beta)*he(e)

    Output :
        - G2 (nx.Graph) : transformed graph
        - d : new distance
        - h : histogram on the new nodes of the transformed graph

        CAUTION : The character '*' cannot be a label
    """

    G2 = nx.Graph()

    def d(a, b):
        if a[0] == RESERVED and b[0] == RESERVED:
            return de(a[1], b[1])
        if a[1] == RESERVED and b[1] == RESERVED:
            return dv(a[0], b[0])
        return INF
 

    nodes = []
    edges = []
    node_index = {}
    edge_index = {}
    for i, u in enumerate(G.nodes):
        nodes.append((u, {'x': (G.nodes[u]['x'], RESERVED)}))
        node_index[u] = i
    for i, (u, v) in enumerate(G.edges):
        nodes.append(((u, v), {'x': (RESERVED, G[u][v]['edge_attr'])}))
        edges.append(((u, v), u, {'edge_attr':0}))
        edges.append((v, (u, v), {'edge_attr':0}))
        edge_index[(u, v)] = i
    
    G2.add_nodes_from(nodes)
    G2.add_edges_from(edges)

    h = []
    for u in G2.nodes:
        if G2.nodes[u]['x'][0] == RESERVED:
            h.append((1-beta)*he[edge_index[u]])
        else:
            h.append(beta*hv[node_index[u]])

    return G2, d, np.array(h)

def inverse_transform(G : nx.Graph(), d, h):
    """
    Process the inverse tranformation, from a node-only labeled graph to a node+edge labeled graph

    Args :
        - G (nx.Graph()) : transformed graph
        - d : distance function between nodes in the graph
        - h : histogram of the graph
    
    Output :
        - H (nx.Graph()) : untransformed graph
        - dv : distance on nodes
        - de : distance on edges
        - hv : histogram on nodes
        - he : histogram on edges
        - beta : beta coefficient used for the transformation
    
    WARNING : 
        Nothing is testes to know if the graph is really a transformed graph. The behaviour of this function on non transformed graphs is not guaranted.
    """
    def dv(a, b):
        return d((a, RESERVED), (b, RESERVED))
    def de(a, b):
        return d((RESERVED, a), (RESERVED, b))
    
    
    G2 = nx.Graph()
    nodes = []
    edges = []

    hv = {}
    he = {}

    beta = 0
    for i, u in enumerate(G.nodes):
        if G.nodes[u]['x'][0] == RESERVED:
            #Edge
            edges.append((u[0], u[1], {'edge_attr': G.nodes[u]['x'][1]}))
            he[(u[0], u[1])] = h[i]
        else:
            nodes.append((u, {'x':G.nodes[u]['x'][0]}))
            hv[u] = h[i]
            beta += h[i]
    
    for i in range(len(hv)):
        hv[i] /= beta
    
    for e in he.keys():
        he[e] /= (1-beta)

    
    G2.add_nodes_from(nodes)
    G2.add_edges_from(edges)

    return G2, dv, de, hv, he, beta