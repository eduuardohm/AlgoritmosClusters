import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans

from methods.dash2002 import Dash2002
from methods.mitra2002 import feature_selection
from methods.maxvar import maxVar
from methods.laplacian_score import laplacian_score
from methods.VCSDFS import VCSDFS
from methods.FMIUFS import ufs_FMI

from timeit import default_timer as timer
from datasets.datasets import selectDataset
from utility.util import calculate_accuracy, run_filter
from utility.exec_mfcm import exec_mfcm

warnings.simplefilter(action='ignore', category=FutureWarning)

def run_feature_selection(method, X, y, nclusters, p, n_features, result=None):
    numVar = int(p * n_features)

    if method == 'maxvar':
        features = maxVar(X, numVar)
        return X[:, np.array(features, dtype='int32')]

    elif method == 'ls':
        features = laplacian_score(X, numVar)
        return X[:, features]

    elif method == 'mitra2002':
        threshold = numVar
        features = feature_selection(X, k=int(threshold))
        return X[:, features]

    elif method == 'dash2002':
        threshold = numVar
        model = Dash2002(X, threshold)
        model.execute(X)
        features = model.forward_selection()
        return X[:, features]

    elif method == 'vcsdfs':
        XX = X.T @ X
        XXX = XX @ XX
        rho = 0.1
        W, index, obj = VCSDFS(X, XX, XXX, rho, numVar, NITER=100)
        features = index[:numVar]
        return X[:, features]

    elif method == 'fmiufs':
        lammda = 0.5
        features = ufs_FMI(X, lammda)
        features = features[:numVar]
        return X[:, features]

    elif method == 'varfilter':
        return run_filter('mean', X, result, numVar, nclusters)

    elif method == 'sumfilter':
        return run_filter('var', X, result, numVar, nclusters)

    else:
        raise ValueError(f"Unknown method: {method}")


def run_kmeans_and_log(method, X_selected, y, seed, log):
    KMeansModel = KMeans(n_clusters=len(np.unique(y)), random_state=seed, n_init=10)
    KMeansModel.fit(X_selected)
    y_pred = KMeansModel.labels_

    results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred, y, None, X_selected)))
    log += f"{results[0]},{results[1]},{results[2]},{results[3]}\n"
    return log


def evaluate(indexData, pVar, mc, nRep, seed, selected_method):
    log = ''

    X, y, nclusters, dataset_name = selectDataset(indexData)
    n_samples, n_features = X.shape

    print(f'*{"-"*30}* {dataset_name} - {selected_method.upper()} *{"-"*30}*\n') 
    print(f'Seed: {seed} | Samples: {n_samples} | Features: {n_features} | Clusters: {nclusters}\n\n')

    print('ari,nmi,sillhouette,db')

    for p in pVar:
        if method == 'varfilter' or method == 'sumfilter':
            result, mfcm_time, centers = exec_mfcm(indexData, mc, nRep)
        else: result = None

        try:
            X_selected = run_feature_selection(selected_method, X, y, nclusters, p, n_features, result=result)
            log = run_kmeans_and_log(selected_method, X_selected, y, seed, log)
        except Exception as e:
            log += f"Erro ao rodar {selected_method.upper()} com p={p}: {e}\n"

    return log


if __name__ == '__main__':
    # log_file = open('logs/evaluation_classic_methods.txt', 'a', newline='\n')

    SEED = 42
    nRep = 100
    datasets = [4]
    pVars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    methods = ['maxvar', 'ls', 'mitra2002', 'dash2002', 'vcsdfs', 'fmiufs', 'varfilter', 'sumfilter']

    for d in datasets:
        for method in methods:
            log = evaluate(d, pVars, 1, nRep, SEED, method)
            print(log)
            # log_file.write(log)

    print(f"\n{'-'*30}> Done <{'-'*30}")
    # log_file.close()