import numpy as np
import matplotlib.pyplot as plt
from experiment import *
from filters import *
from datasets import selectDataset

def skewness_coefficient(data):
    """Calcula a assimetria dos dados."""
    n = len(data)
    mean = np.mean(data)
    numerator = np.sum((data - mean) ** 3) / n
    denominator = (np.sum((data - mean) ** 2) / n) ** (3 / 2)
    return numerator / denominator

def sigma_skewness(n):
    """Calcula o desvio padrão da assimetria."""
    return np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))

def doane_bins(data):
    """Calcula o número de bins ideal usando a fórmula de Doane."""
    n = len(data)
    if n < 2:
        raise ValueError("A quantidade de dados deve ser maior que 1 para calcular o número de bins.")
    
    skewness = skewness_coefficient(data)
    sigma = sigma_skewness(n)
    bins = int(np.log2(n) + 1 + np.log2(1 + abs(skewness) / sigma))
    
    return bins

def evaluate(indexData, mc, nRep, seed):
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

    result, mfcm_time, centers = exec_mfcm(indexData, mc, nRep)
    
    r1, fn1 = sum_filter(dataset, result['bestM'], nclusters)
    r1 = [v[0] for v in r1]
    # max_r1 = max(r1)
    # r1 = [x / max_r1 for x in r1]  # Dividindo cada valor pelo máximo
    r1 = np.sqrt(np.array(r1))
    # r1 = np.log(np.array(r1) + 1e-6)

    print(f'MF-M: {r1}')
    
    r2, fn2 = variance_filter(dataset, result['bestM'], nclusters)
    r2 = [v[0] for v in r2]
    # max_r2 = max(r2)
    # r2 = [x / max_r2 for x in r2]  # Dividindo cada valor pelo máximo
    r2 = np.sqrt(np.array(r2))
    # r2 = np.log(np.array(r2) + 1e-6)

    print(f'MF-V: {r2}')

    # Definindo o número de bins (Regra de Doane)
    num_bins_m = 10#doane_bins(r1)
    num_bins_v = 10#doane_bins(r2)
    print(num_bins_m)
    print(num_bins_v)

    # Gerando histogramas
    plt.figure(figsize=(10, 6))

    # Histograma para r1
    plt.subplot(1, 2, 1)
    plt.hist(r1, bins=num_bins_m, range = (min(r1), max(r1)), color='blue', alpha=0.7)
    plt.xlabel('Relevância')
    plt.ylabel('Frequência')
    plt.title(f'{fn1}')
    # x_ticks = np.arange(min(r1), max(r1), 0.001)
    # plt.xticks(x_ticks)

    # Histograma para r2
    plt.subplot(1, 2, 2)
    plt.hist(r2, bins=num_bins_v, range=(min(r2), max(r2)), color='green', alpha=0.7)
    plt.xlabel('Relevância')
    plt.ylabel('Frequência')
    plt.title(f'{fn2}')

    # Exibir os gráficos
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    SEED = 42
    nRep = 100
    mc = 1
    
    datasets = [4,6,7,9,10,11,13,15]

    for d in datasets:
        evaluate(d, mc, nRep, SEED)
