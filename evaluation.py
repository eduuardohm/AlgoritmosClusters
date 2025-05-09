import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans

from methods.FSEM import Dash2002
from methods.FSFS import feature_selection
# from methods.MAXVAR import maxVar
from methods.LS import laplacian_score
from methods.VCSDFS import VCSDFS
from methods.FMIUFS import ufs_FMI
from methods.SRCFS import SRCFS
# from methods.DUFS import Model, DataSet

from timeit import default_timer as timer
from datasets.datasets import selectDataset
from utility.util import calculate_accuracy, run_filter
from utility.exec_mfcm import exec_mfcm

warnings.simplefilter(action='ignore', category=FutureWarning)

def run_dufs(params, X, y):
    model = Model(**params)
    dataset = DataSet(**{'_data': np.asarray(X), '_labels': np.asarray(y)}, labeled=True)
    
    model.train(dataset, learning_rate=1, batch_size=X.shape[0], display_step=100, num_epoch=3000, labeled=True)
    return model

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
    
    elif method == 'srcfs':
        para_K = 5
        para_s = 20
        para_m = 10

        rankings, feaWeights = SRCFS(X, para_K=para_K, para_s=para_s, para_m=para_m)
        rankings = rankings[:numVar]
        return X[:, rankings]
    
    elif method == 'dufs':
        prob_alpha = result.get_prob_alpha()
        rankings = np.argsort(-prob_alpha)
        
        return X[:, rankings[:numVar]]

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

    ## Treinar modelos antes da execução do K-Means
    if method == 'varfilter' or method == 'sumfilter':
            result, mfcm_time, centers = exec_mfcm(indexData, mc, nRep, seed)
    elif method == 'dufs':
        params = {
        'lam': 1e-4,
        'input_dim': X.shape[1],
        'is_param_free_loss': True,
        'knn': 2,
        'fac': 5
        }
        result = run_dufs(params, X, y)
    else: result = None

    print(f'*{"-"*30}* {dataset_name} - {selected_method.upper()} *{"-"*30}*\n') 
    print(f'Seed: {seed} | Samples: {n_samples} | Features: {n_features} | Clusters: {nclusters}\n\n')

    print('ari,nmi,sillhouette,db')

    for p in pVar:
        try:
            X_selected = run_feature_selection(selected_method, X, y, nclusters, p, n_features, result=result)
            log = run_kmeans_and_log(selected_method, X_selected, y, seed, log)
        except Exception as e:
            log += f"Erro ao rodar {selected_method.upper()} com p={p}: {e}\n"

    return log, dataset_name

if __name__ == '__main__':
    # log_file = open('logs/evaluation_classic_methods.txt', 'a', newline='\n')
    # log_file = open('logs/COIL20.txt', 'a', newline='\n')

    SEED = 42
    nRep = 100
    datasets = [10]
    pVars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # methods = ['maxvar', 'ls', 'mitra2002', 'dash2002', 'vcsdfs', 'fmiufs', 'srcfs', 'varfilter', 'sumfilter']
    # methods = ['dufs']
    methods = ['varfilter', 'sumfilter']

    for d in datasets:
        for method in methods:
            log, dataset_name = evaluate(d, pVars, 1, nRep, SEED, method)
            print(log)
            metrics = 'ari,nmi,sillhouette,db'

            data_execucao = "08_05_2025"
            if not os.path.exists(f'logs/{data_execucao}'):
                os.makedirs(f'logs/{data_execucao}', exist_ok=True)
    
            log_file = open(f'logs/{data_execucao}/{dataset_name}.txt', 'a', newline='\n')
            log_file.write(f"{method.upper()} - {dataset_name} - {metrics}\n")
            log_file.write(log)
            log_file.write('\n')
            log_file.write(f"{'='*30}\n")
            log_file.close()

    print(f"\n{'-'*30}> Done <{'-'*30}")