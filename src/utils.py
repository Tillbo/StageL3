import numpy as np

def make_p_dist(p=2):
    """
    Returns the p-distance between vectors.

    For the infinite-distance, p must be equal to np.inf
    """
    def d(v1, v2):
        if type(v1) is list:
            v1 = np.array(v1)
        if type(v2) is list:
            v2 = np.array(v2)
        return np.linalg.norm(v1-v2, ord=p)
    return d

def unif_dist(d=1):
    """
    Returns a distance function such as d(x, y) = 0 if x == y, else d
    """
    def f(x, y):
        if type(x) is np.ndarray and type(y) is np.ndarray:
            return 0 if np.all(x == y) else d
        return 0 if x == y else d
    return f

def condense(D):
    """
    Condense a distance matrix within a set into a vector. Usefull for scipy's clustering
    """
    n = D.shape[0]
    c = np.array(list(range(int(n*(n-1)/2))))
    for i in range(n):
        for j in range(i+1, n):
            c[n * i + j - ((i + 2) * (i + 1)) // 2] = D[i, j]
    return c