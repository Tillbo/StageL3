from sklearn.cluster import AgglomerativeClustering

def cluster_num(c, D, N=2):
    """
    Use Agglomerative Clustering (hierarchical) to compute a given number of clusters

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

    Args :
        - c (array like, size n) : set of items to cluster
        - D (array like, size (n, n)) : distance matrix between elements of c
        - d (float) : maximum distance within a cluster
    
    Output :
        - clusters (list of lists) : list of different clusters
    """
    AggCluster = AgglomerativeClustering(None, metric='precomputed', linkage='average', distance_threshold=d, compute_full_tree=True)
    cluster_labels = AggCluster.fit(D).labels_

    clusters = [[]]
    i_clust = 0

    for i, G in enumerate(c):
        while i_clust < cluster_labels[i]:
            i_clust += 1
            clusters.append([])
        clusters[cluster_labels[i]].append(G)
    
    return clusters

