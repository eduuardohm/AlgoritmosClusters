import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from methods.dash2002 import Dash2002
from methods.mitra2002 import feature_selection
from methods.maxvar import maxVar
from methods.laplacian_score import laplacian_score
from methods.VCSDFS import VCSDFS
from methods.FMIUFS import ufs_FMI

from timeit import default_timer as timer
from datasets.datasets import selectDataset
from utility.calculate_acc import calculate_accuracy

def evaluate(indexData, pVar, mc, nRep, seed):
    ## Loading Data
    log = ''
    
    X, y, nclusters, dataset_name = selectDataset(indexData)
    n_samples, n_features = X.shape

    print(f'Dataset selecionado: {dataset_name}\n')

    log += f'*{"-"*30}* {dataset_name} *{"-"*30}*\n\n'
    log += f'Seed: {seed}\n'
    log += f'Dataset: {dataset_name} | n_samples: {n_samples} | n_features: {n_features} | n_clusters: {nclusters}\n'
    log += 'Methods: MaxVar, LS, Mitra2002'

    # log += '\n\nParameters:\n'
    
    # log += '\nMFCM\n'
    # log += f'MC: {mc} | nRep: {nRep}\n'

    # ## MFCM
    # result, mfcm_time, centers = exec_mfcm(indexData, mc, nRep)
    # print('MFCM executado.\n')

    log += '\n\nRESULTS:\n\n'
    log += "ari,nmi,sillhouette,db\n"

    for p in pVar:
        # print(f'Pvar: {p}\n')
        numVar = int(p * n_features)

        ## Métodos propostos

        # start = timer()
        # dataset_varfilter = run_filter('mean', X, result, y, numVar, nclusters)
        # end = timer()
        # filvar_time = round(end - start, 4)

        # start = timer()
        # dataset_sumfilter = run_filter('var', X, result, y, numVar, nclusters)
        # end = timer()
        # filsum_time = round(end - start, 4)

        ## MaxVar

        # start = timer()
        # maxVar_features = maxVar(X, p)
        # end = timer()
        # maxvar_time = end - start
        # print('MaxVar executado.')

        ## Laplacian Score:

        # start = timer()
        # LS_features = laplacian_score(X, p)
        # end = timer()
        # ls_time = end - start
        # print('LS executado.')

        ## Dash2002

        # threshold = 0.5 * n_features
        # threshold = numVar
        # start = timer()

        # model = Dash2002(X, threshold)
        # entropy = model.execute(X)
        # dash_features = model.forward_selection()

        # end = timer()
        # dash_time = end - start
        # print('Dash2002 executado.')

        # log += f"Entropy: {entropy}\n"
        # log += (f"Variáveis selecionadas: {dash_features}")

        ## FSFS

        # start = timer()
        # mitra_features = feature_selection(X, k=int(threshold))
        # # log += (f"Variáveis selecionadas: {mitra_features}")
        # end = timer()
        # mitra_time = end - start

        ## VCSDFS

        XX = X.T @ X
        XXX = XX @ XX
        rho = 0.1   # Regularização (ajuste conforme necessário)

        W, index, obj = VCSDFS(X, XX, XXX, rho, numVar, NITER=100)
        vcsdfs_features = index[:numVar]

        ## FMIUFS
        
        lammda = 0.5 # Definir o valor de lambda
        fmiufs_features = ufs_FMI(X, lammda)
        fmiufs_features = fmiufs_features[:numVar]

        ## Feature selection
        
        # X_maxvar = X[:, np.array(maxVar_features, dtype='int32')]
        # X_LS = X[:, LS_features]
        # #X_dash = X[:, dash_features]
        # X_mitra = X[:, mitra_features]
        X_vcsdfs = X[:, vcsdfs_features]
        X_fmiufs = X[:, fmiufs_features]

        ## K Means
        K = nclusters

        og_Kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
        og_Kmeans.fit(X)
        y_pred_og = og_Kmeans.labels_
        
        # KMeansresultVar = KMeans(n_clusters=K, random_state=seed, n_init=10)
        # KMeansresultVar.fit(dataset_varfilter)
        # KMeansresultVar = KMeansresultVar.labels_

        # KMeansresultSum = KMeans(n_clusters=K, random_state=seed, n_init=10)
        # KMeansresultSum.fit(dataset_sumfilter)
        # KMeansresultSum = KMeansresultSum.labels_

        # maxvar_Kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
        # maxvar_Kmeans.fit(X_maxvar)
        # y_pred0 = maxvar_Kmeans.labels_

        # LS_KMeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
        # LS_KMeans.fit(X_LS)
        # y_pred1 = LS_KMeans.labels_

        # mitra_Kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
        # mitra_Kmeans.fit(X_mitra)
        # y_pred2 = mitra_Kmeans.labels_

        # dash_Kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
        # dash_Kmeans.fit(X_dash)
        # y_pred3 = dash_Kmeans.labels_

        vcsdfs_Kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
        vcsdfs_Kmeans.fit(X_vcsdfs)
        y_pred4 = vcsdfs_Kmeans.labels_

        fmiufs_Kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
        fmiufs_Kmeans.fit(X_fmiufs)
        y_pred5 = fmiufs_Kmeans.labels_

        # print('KMeans executado.')

        ## Metrics

        # log += 'KMeans sem filtro:\n'
        # results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred_og, y, None, X)))
        # log += (f'{results[0]},{results[1]},{results[2]},{results[3]}\n')

        # log += '\nFiltro por Variância:\n'
        # results = list(map(lambda x: round(x, 8), calculate_accuracy(KMeansresultVar, y, None, dataset_varfilter)))
        # log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(mfcm_time, 4)}s (MFCM), {filvar_time}s\n'

        # log += '\nFiltro por Somatório:\n'
        # results = list(map(lambda x: round(x, 8), calculate_accuracy(KMeansresultSum, y, None, dataset_sumfilter)))
        # log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(mfcm_time, 4)}s (MFCM), {filsum_time}s\n'

        # log += '\nMaxVar:\n'
        # results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred0, y, None, X_maxvar)))
        # log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(maxvar_time, 4)}s\n'

        # log += '\nLS:\n'
        # results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred1, y, None, X_LS)))
        # log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(ls_time, 4)}s\n'

        # log += '\nMitra2002:\n'
        # results = list(map(lambda x: round(x, 8),calculate_accuracy(y_pred2, y, None, X_mitra)))
        # log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(mitra_time, 4)}s\n'

        # log += '\nDash2002:\n'
        # results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred3, y, None, X_dash)))
        # log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(dash_time, 4)}s\n'

        # log += '\nVCSDFS:\n'
        # results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred4, y, None, X_vcsdfs)))
        # log += f'{results[0]},{results[1]},{results[2]},{results[3]}\n'

        # log += '\FMIUFS:\n'
        results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred5, y, None, X_fmiufs)))
        log += f'{results[0]},{results[1]},{results[2]},{results[3]}\n'

    log += '\n'
    return log


if __name__ == '__main__':
    log_file = open('logs/evaluation_classic_methods.txt', 'a', newline='\n')

    SEED = 42
    nRep = 10
    
    datasets = [14]
    pVars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for d in datasets:
        print("ari,nmi,sillhouette,db")
        log = evaluate(d, pVars, 1, nRep, SEED)
        print(log)
        # log_file.write(log)

    print(f"\n{'-'*30}> Done <{'-'*30}")
    log_file.close()