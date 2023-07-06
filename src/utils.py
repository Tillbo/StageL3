import numpy as np

def make_p_dist(p=2):
    """
    Returns the p-distance between vectors.

    For the infinite-distance, p must be in :
    [
        numpy.inf
    ]
    """
    def d(v1, v2):
        return np.linalg.norm(v1-v2, ord=p)
    return d