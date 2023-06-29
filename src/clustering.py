from sklearn.cluster import AgglomerativeClustering

def cluster_num(c, D, N=2):
    AggCluster = AgglomerativeClustering(N, metric='precomputed', linkage='average')
    cluster_labels = AggCluster.fit(D).labels_
    clusters = [[] for _ in range(N)]

    for i, G in enumerate(c):
        clusters[cluster_labels[i]].append(G)
    
    return clusters

def cluster_dist(c, D, d):
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

