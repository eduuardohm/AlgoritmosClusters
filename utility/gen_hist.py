import numpy as np
import matplotlib.pyplot as plt
import ast  # Para converter as strings em listas de números
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

def evaluate(indexData, mc, nRep, seed, mf_m_line, mf_v_line):
    mf_m_line = np.sqrt(np.array(mf_m_line))
    mf_v_line = np.sqrt(np.array(mf_v_line))

    ## Plotagem do histograma
    num_bins_m = 10
    num_bins_v = 10
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.hist(mf_m_line, bins=num_bins_m, color='blue', alpha=0.7)
    plt.xlabel('Relevância')
    plt.ylabel('Frequência')
    plt.title(f'MF_M')
    # x_ticks = np.linspace(min(mf_m_line), 0.1, num=7)
    x_ticks = np.arange(min(mf_m_line), max(mf_m_line), 0.005)
    plt.xticks(x_ticks)

    plt.subplot(1, 2, 2)
    plt.hist(mf_v_line, bins=num_bins_v, color='green', alpha=0.7)
    plt.xlabel('Relevância')
    plt.ylabel('Frequência')
    plt.title(f'MF_V')
    x_ticks = np.arange(min(mf_v_line), max(mf_v_line), 0.005)
    plt.xticks(x_ticks)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    SEED = 42
    nRep = 100
    mc = 1
    
    datasets = [21]

    # Abrir arquivos mf_m e mf_v e ler as linhas
    with open('logs/mf_m.txt', 'r') as file_m, open('logs/mf_v.txt', 'r') as file_v:
        mf_m_lines = file_m.readlines()  # Lê todas as linhas do arquivo mf_m
        mf_v_lines = file_v.readlines()  # Lê todas as linhas do arquivo mf_v
    
    # O loop deve rodar 4 vezes, já que os arquivos possuem 4 linhas cada
    for i in range(4):
        # Pegando a linha correspondente de cada arquivo e convertendo para lista de números
        mf_m_line = ast.literal_eval(mf_m_lines[i].strip())
        mf_v_line = ast.literal_eval(mf_v_lines[i].strip())
        
        # Imprimir as listas convertidas para garantir que são numéricas
        print(mf_m_line)
        print(mf_v_line)

        # Executando a função evaluate com as listas convertidas
        evaluate(datasets[0], mc, nRep, SEED, mf_m_line, mf_v_line)
