try:
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    #sklearn is only used in clustering functions that are no longer used
except (ImportError, ModuleNotFoundError):
    pass

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from .utils import condense


def clusters_from_labels(c, labels):
    """
    Depreciated
    """
    clusters = [[]]
    i_clust = 0

    for i, G in enumerate(c):
        while i_clust < labels[i]:
            i_clust += 1
            clusters.append([])
        clusters[labels[i]].append(G)
    
    return clusters

def cluster_num(c, D, N=2):
    """
    Use Agglomerative Clustering (hierarchical) to compute a given number of clusters
    Depreciated
    Args :
        - c (array like, size n) : set of items to cluster
        - D (array like, size (n, n)) : distance matrix between elements of c
        - N (int, optionnal (default : 2)) : number of clusters to make
    
    Output :
        - clusters (list of lists, size N) : list of different clusters
    """
    AggCluster = AgglomerativeClustering(N, metric='precomputed', linkage='average')
    cluster_labels = AggCluster.fit(D).labels_
    clusters = [[] for _ in range(N)]

    for i, G in enumerate(c):
        clusters[cluster_labels[i]].append(G)
    
    return clusters

def cluster_dist(c, D, d):
    """
    Use Agglomerative Clustering (hierarchical) to compute a given number of clusters
    Depreciated
    Args :
        - c (array like, size n) : set of items to cluster
        - D (array like, size (n, n)) : distance matrix between elements of c
        - d (float) : maximum distance within a cluster
    
    Output :
        - clusters (list of lists) : list of different clusters
    """
    AggCluster = AgglomerativeClustering(None, metric='precomputed', linkage='average', distance_threshold=d, compute_full_tree=True)
    cluster_labels = AggCluster.fit(D).labels_

    return clusters_from_labels(c, cluster_labels)

def cluster_dbscan(c, D, eps, min_samples):
    """
    Deperciated
    """
    DBSCANCluster = DBSCAN(eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = DBSCANCluster.fit(D).labels_
    return clusters_from_labels(c, cluster_labels)

def dendo(D, Nclusters=2):
    """
    Computes a dendogram from distance matrix D and returns clusters

    Args :
    - D (np.array) : distance matrix
    - Nclusters (int, default : 2) : number of clusters to make at the end.
    """
    print("Condensing...", end=" ", flush=True)
    y = condense(D)
    print("OK")
    print("Linkage...", end=" ", flush=True)
    Z = linkage(y, method='weighted')
    print("OK")
    print("Plotting dendogram...", end=" ", flush=True)
    dendrogram(Z)#, truncate_mode='lastp', orientation='bottom')
    print("OK")
    return fcluster(Z, Nclusters, criterion='maxclust')