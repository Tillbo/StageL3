import numpy as np
import networkx as nx

from .const import INF

def all_to_all(G : nx.Graph()):
    """
    Compute all_to_all distance matrix

    Args :
        - G (nx.Graph())
    
    Output :
        - C (np.array) : distance array
    """
    d = dict(nx.shortest_path_length(G))
    n = G.number_of_nodes()
    C = np.zeros((n, n), dtype=np.float64)

    for i, u in enumerate(G.nodes):
        for j, v in enumerate(G.nodes):
            try:
                C[i, j] = d[u][v]
            except KeyError:
                C[i, j] = INF

    return C

def mean_structure(G : nx.Graph()):
    """
    Returns the "mean structure" of graph G.

    It is a block matrix :

     CV | CVE
    --------
    CEV | CE 

    Where CV is the distance matrix of G
    CVE is a distance between node and edge, computed with the mean distance of the extremities of the edge to the node
    Idem for CEV
    CE is a distance between edges, computed with the mean distance between the extremities of the edges
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    CV = all_to_all(G)

    CEV = np.zeros((m, n), dtype=float)
    for i, (v1, v2) in enumerate(G.edges):
        for u in G.nodes:
            CEV[i, u] = (CV[u, v1] + CV[u, v2])/2
            
    CVE = np.zeros((n, m), dtype=float)
    for u in G.nodes:
        for i, (v1, v2) in enumerate(G.edges):
            CVE[u, i] = (CV[u, v1] + CV[u, v2])/2

    CE = np.zeros((m, m), dtype=float)
    for i, (u1, v1) in enumerate(G.edges):
        for j, (u2, v2) in enumerate(G.edges):
            CE[i,j] = (CV[u1, u2] + CV[u1, v2] + CV[v1, u2] + CV[v1, v2])/4

    return np.block([[CV, CVE], [CEV, CE]])