from numpy import inf

def make_p_dist(p=2):
    """
    Returns the p-distance between vectors.

    For the infinite-distance, p must be in :
    [
        'inf',
        numpy.inf
        -1
        float("inf")
    ]
    """
    if p in ['inf', inf, -1, float("inf")]:
        def d(v1, v2):
            v = abs(v1-v2)
            return max(v)
    
        return d
    if p < 1:
        print("Not a distance")
        return
    def d(v1, v2):
        s = 0
        for u1, u2 in zip(v1, v2):
            s += (u1-u2)**p
        return s**(1/p)
    return d