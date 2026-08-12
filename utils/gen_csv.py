import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as sklearnKMeans
from literature_methods.dash2002 import Dash2002
from literature_methods.mitra2002 import feature_selection
from literature_methods.maxvar import maxVar
from literature_methods.laplacian_score import laplacian_score
from experiment import *
from datasets import selectDataset
from timeit import default_timer as timer

def evaluate(indexData, pVar, mc, nRep, seed):
    ## Loading Data

    log = ''
    
    dataset, ref, nclusters, dataset_name = selectDataset(indexData)
    n_samples, n_features = dataset.shape

    # print(f'Dataset selecionado: {dataset_name}\n')

    log += f'*{"-"*30}* {dataset_name} *{"-"*30}*\n\n'
    log += f'Seed: {seed}\n'
    log += f'Dataset: {dataset_name} | n_samples: {n_samples} | n_features: {n_features} | n_clusters: {nclusters}\n'
    log += 'Methods: MaxVar, LS, Mitra2002'

    log += '\nRESULTS:\n'

    maxvar_s = ''
    ls_s = ''
    mitra_s = ''
    dash_s = ''

    for p in pVar:
        numVar = int(p * n_features)

        if numVar < 1:
            numVar = 1
        # elif numVar == n_features:
        #     numVar -= 1

        # print(f'numVar: {numVar}')

        ## Métodos propostos

        # start = timer()
        # dataset_varfilter = run_filter(1, dataset, result, ref, numVar, nclusters)
        # end = timer()
        # filvar_time = round(end - start, 4)

        # start = timer()
        # dataset_sumfilter = run_filter(2, dataset, result, ref, numVar, nclusters)
        # end = timer()
        # filsum_time = round(end - start, 4)

        ## MaxVar

        start = timer()
        maxVar_features = maxVar(dataset, numVar)
        end = timer()
        maxvar_time = end - start

        ## Laplacian Score:

        start = timer()
        LS_features = laplacian_score(dataset, numVar)
        end = timer()
        ls_time = end - start

        ## Dash2002

        # threshold = 0.5 * n_features
        threshold = numVar
        # start = timer()

        # model = Dash2002(dataset, threshold)
        # entropy = model.execute(dataset)
        # dash_features = model.forward_selection()
        # print("concluido.")

        # end = timer()
        # dash_time = end - start
        # print('Dash2002 executado.')

        # log += f"Entropy: {entropy}\n"
        # log += (f"Variáveis selecionadas: {dash_features}")

        ## Mitra2002

        start = timer()
        mitra_features = feature_selection(dataset, k=int(threshold))
        # log += (f"Variáveis selecionadas: {mitra_features}")
        end = timer()
        mitra_time = end - start

        ## Feature selection
        X_maxvar = dataset[:, np.array(maxVar_features, dtype='int32')]
        X_LS = dataset[:, LS_features]
        # X_dash = dataset[:, dash_features]
        X_mitra = dataset[:, mitra_features]

        ## K Means
        K = nclusters
        
        # KMeansresultVar = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        # KMeansresultVar.fit(dataset_varfilter)
        # KMeansresultVar = KMeansresultVar.labels_

        # KMeansresultSum = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        # KMeansresultSum.fit(dataset_sumfilter)
        # KMeansresultSum = KMeansresultSum.labels_

        maxvar_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        maxvar_Kmeans.fit(X_maxvar)
        y_pred0 = maxvar_Kmeans.labels_

        LS_KMeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        LS_KMeans.fit(X_LS)
        y_pred1 = LS_KMeans.labels_

        mitra_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        mitra_Kmeans.fit(X_mitra)
        y_pred2 = mitra_Kmeans.labels_

        # dash_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        # dash_Kmeans.fit(X_dash)
        # y_pred3 = dash_Kmeans.labels_

        ## Metrics

        results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred0, ref, None, X_maxvar)))
        maxvar_s += f'{results[0]},{results[1]},{results[2]},{results[3]}\n'

        results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred1, ref, None, X_LS)))
        ls_s += f'{results[0]},{results[1]},{results[2]},{results[3]}\n'

        results = list(map(lambda x: round(x, 8),calculate_accuracy(y_pred2, ref, None, X_mitra)))
        mitra_s += f'{results[0]},{results[1]},{results[2]},{results[3]}\n'

        # results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred3, ref, None, X_dash)))
        # dash_s += f'{results[0]},{results[1]},{results[2]},{results[3]}\n'

    log += 'MaxVar:\n'
    log += maxvar_s
    log += 'LS:\n'
    log += ls_s
    log += 'Mitra2002:\n'
    log += mitra_s
    # log += 'Dash2002:\n'
    # log += dash_s
    log += '\n'
    
    return log


if __name__ == '__main__':

    SEED = 42
    nRep = 10
    
    datasets = [21]
    pVars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for d in datasets:
        log = evaluate(d, pVars, 1, nRep, SEED)
        print(log + '\n')

    print(f"\n{'-'*30}> Done <{'-'*30}")