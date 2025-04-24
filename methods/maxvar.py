import numpy as np
from numba import jit, cuda

@jit(target_backend='cuda')
def maxVar(dataset, n_selected_features):
    n_samples, n_features = dataset.shape
    features = []

    for it in range(n_features):
        variance = np.var(dataset[:, it])
        features.append([it, variance])

    features = sorted(features, key=lambda x: x[1], reverse=True)
    features = [f[0] for f in features]
    
    return features[n_selected_features:]