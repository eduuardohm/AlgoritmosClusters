import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist

def SRCFS(fea, para_K=5, para_s=20, para_m=10):
    """
    fea:      n x d data matrix, each row is a sample
    para_K:   number of nearest neighbors
    para_s:   number of random subspaces in each basic feature partition
    para_m:   number of basic feature partitions

    Returns:
    rankings: indices of features ranked by importance (descending)
    feaWeights: computed feature weights
    """
    n, d = fea.shape

    if para_s > 1:
        feas, feaIdxs = genRandomFeaSets(fea, para_s, para_m)
        fWs = np.zeros((d, len(feas)))

        for iFS, fea_sub in enumerate(feas):
            knnW = buildSimGraphKNN(fea_sub, para_K)
            tmpWs = LaplacianScore(fea_sub, knnW)
            fWs[feaIdxs[iFS], iFS] = tmpWs
        feaWeights = np.sum(fWs, axis=1) / para_m
    else:
        knnW = buildSimGraphKNN(fea, para_K)
        feaWeights = LaplacianScore(fea, knnW)

    rankings = np.argsort(-feaWeights)
    return rankings, feaWeights

def buildSimGraphKNN(data, K):
    n, d = data.shape
    D = EuDist2(data, data)

    np.fill_diagonal(D, 1e100)

    dump = np.zeros((n, K))
    idx = np.zeros((n, K), dtype=int)

    D_copy = D.copy()
    for i in range(K):
        dump[:, i] = np.min(D_copy, axis=1)
        idx[:, i] = np.argmin(D_copy, axis=1)
        temp = idx[:, i] * n + np.arange(n)
        D_copy.flat[temp] = 1e100  # mark as visited

    sigma = np.mean(dump)

    dump = np.exp(-(dump ** 2) / (2 * sigma ** 2))

    Gidx = np.tile(np.arange(n).reshape(-1, 1), (1, K))
    Gjdx = idx

    W = sp.coo_matrix((dump.flatten(), (Gidx.flatten(), Gjdx.flatten())), shape=(n, n))
    W = W.maximum(W.T)
    return W

def genRandomFeaSets(data, para_s, para_m):
    n, d = data.shape
    d_each = max(int(np.ceil(d / para_s)), 1)

    datas = []
    dataIdxs = []

    for _ in range(para_m):
        np.random.seed()  # reseed to get different random
        rDidxs = np.random.permutation(d)
        for iF in range(para_s):
            start = iF * d_each
            end = min((iF + 1) * d_each, d)
            tmpIdx = rDidxs[start:end]
            if len(tmpIdx) == 0:
                break
            dataIdxs.append(tmpIdx)
            datas.append(data[:, tmpIdx])

    return datas, dataIdxs

def EuDist2(fea_a, fea_b=None, bSqrt=True):
    if fea_b is None:
        aa = np.sum(fea_a ** 2, axis=1)
        ab = np.dot(fea_a, fea_a.T)
        D = np.add.outer(aa, aa) - 2 * ab
        D = np.maximum(D, 0)
        if bSqrt:
            D = np.sqrt(D)
        D = np.maximum(D, D.T)
    else:
        aa = np.sum(fea_a ** 2, axis=1)
        bb = np.sum(fea_b ** 2, axis=1)
        ab = np.dot(fea_a, fea_b.T)
        D = np.add.outer(aa, bb) - 2 * ab
        D = np.maximum(D, 0)
        if bSqrt:
            D = np.sqrt(D)
    return D

def LaplacianScore(X, W):
    nSmp, nFea = X.shape
    if W.shape[0] != nSmp:
        raise ValueError('W has wrong shape')

    D = np.array(W.sum(axis=1)).flatten()
    L = W

    allone = np.ones((nSmp, 1))
    tmp1 = np.dot(D, X)

    D_diag = sp.diags(D)

    DPrime = np.sum((X.T @ D_diag).T * X, axis=0) - (tmp1 ** 2) / np.sum(D)
    LPrime = np.sum((X.T @ L).T * X, axis=0) - (tmp1 ** 2) / np.sum(D)

    DPrime[DPrime < 1e-12] = 10000

    Y = LPrime / DPrime
    Y = np.asarray(Y).flatten()
    return Y
