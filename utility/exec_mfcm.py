import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as sklearnKMeans
from experiment import *
from datasets import selectDataset
from timeit import default_timer as timer


def evaluate(indexData, pVar, mc, nRep, seed):
    ## Loading Data

    log = ''
    
    dataset, ref, nclusters, dataset_name = selectDataset(indexData)
    n_samples, n_features = dataset.shape

    log += f'*{"-"*30}* {dataset_name} *{"-"*30}*\n\n'
    log += f'Seed: {seed}\n'
    log += f'Dataset: {dataset_name} | n_samples: {n_samples} | n_features: {n_features} | n_clusters: {nclusters}\n'
    log += '\nParameters:\n'
    
    log += '\nMFCM\n'
    log += f'MC: {mc} | nRep: {nRep}\n'

    print(log)

    ## MFCM
    result, mfcm_time, centers = exec_mfcm(indexData, mc, nRep)
    print('\nMFCM executado.\n')

    log += '\nRESULTS:\n\n'

    MF_M_s = ''
    MF_V_s = ''

    for p in pVar:
        print(f'Pvar: {p}')
        numVar = int(p * n_features)

        start = timer()
        dataset_varfilter = run_filter('var', dataset, result, numVar, nclusters)
        end = timer()
        filvar_time = round(end - start, 4)

        start = timer()
        dataset_sumfilter = run_filter('mean', dataset, result, numVar, nclusters)
        end = timer()
        filsum_time = round(end - start, 4)

        ## K Means
        K = nclusters
        
        KMeansresultVar = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        KMeansresultVar.fit(dataset_varfilter)
        KMeansresultVar = KMeansresultVar.labels_

        KMeansresultSum = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        KMeansresultSum.fit(dataset_sumfilter)
        KMeansresultSum = KMeansresultSum.labels_

        print('KMeans executado.\n')

        ## Metrics

        # log += f'\n{p*100}% OF FEATURES:\n\n'

        # log += '\nFiltro por Variância:\n'
        results = list(map(lambda x: round(x, 8), calculate_accuracy(KMeansresultVar, ref, None, dataset_varfilter)))
        # MF_M_s += f'{results[0]},{results[1]},{results[2]},{results[3]}, Time: {round(mfcm_time, 4)}s (MFCM), {filvar_time}s\n'
        MF_M_s += f'{results[0]},{results[1]},{results[2]},{results[3]}\n'

        # log += '\nFiltro por Somatório:\n'
        results = list(map(lambda x: round(x, 8), calculate_accuracy(KMeansresultSum, ref, None, dataset_sumfilter)))
        # MF_V_s += f'{results[0]},{results[1]},{results[2]},{results[3]}, Time: {round(mfcm_time, 4)}s (MFCM), {filsum_time}s\n'
        MF_V_s += f'{results[0]},{results[1]},{results[2]},{results[3]}\n'

    log += "MF_M:\nari,nmi,sillhouette,db\n"
    log += MF_M_s
    log += "\nMF_V:\nari,nmi,sillhouette,db\n"
    log += MF_V_s
    log += '\n'
    
    return log


if __name__ == '__main__':
    log_file = open('logs/mfcm.txt', 'a', newline='\n')

    SEED = 42
    nRep = 100
    
    datasets = [21]
    pVars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for d in datasets:
        log = evaluate(d, pVars, 1, nRep, SEED)
        print(log + '\n')
        log_file.write(log)

    print(f"\n{'-'*30}> Done <{'-'*30}")
    log_file.close()