import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

#mpl.rcParams['text.usetex'] = False
#plt.style.use(['science'])  # Removendo 'ieee' style

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y_labels = ['Adjusted Rand Index (ARI)', 'Normalized Mutual Information (NMI)', 'Silhouette Score', 'Davies-Bouldin Score (DB)']
dataset_names = ['Hiva']
datasets = ['Hiva']

for j, dataset in enumerate(datasets):
    path = 'csv/' + dataset
    mf_m = pd.read_csv(path + '/MF_M.csv')
    mf_v = pd.read_csv(path + '/MF_V.csv')
    maxvar = pd.read_csv(path + '/MAXVAR.csv')
    ls = pd.read_csv(path + '/LS.csv')
    fsfs = pd.read_csv(path + '/FSFS.csv')
    mcfs = pd.read_csv(path + '/MCFS.csv')
    udfs = pd.read_csv(path + '/UDFS.csv')
    baseline = pd.read_csv(path + '/BASELINE.csv')

    fig, axs = plt.subplots(1, 4, figsize=(20, 10))

    for i in range(4):
        axs[i].plot(x, mf_m.iloc[:, i], label='MF_M')
        axs[i].plot(x, mf_v.iloc[:, i], label='MF_V')
        axs[i].plot(x, maxvar.iloc[:, i], label='MaxVar')
        axs[i].plot(x, ls.iloc[:, i], label='LS')
        axs[i].plot(x, fsfs.iloc[:, i], label='FSFS')
        axs[i].plot(x, mcfs.iloc[:, i], label='MCFS')
        axs[i].plot(x, udfs.iloc[:, i], label='UDFS')
        axs[i].plot(x, baseline.iloc[:, i], label='Baseline')

        axs[i].set_xlabel('Percentage (%) of features', weight='bold')
        axs[i].set_ylabel(y_labels[i], weight='bold')
        axs[i].legend()

        axs[i].set_xticks(x)  # Definindo os ticks do eixo x
        axs[i].grid(True)  # Ativando o grid para todos os subplots

    fig.suptitle(f'{dataset_names[j]} Dataset', weight='bold', fontsize=20)
    
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=10)  # Ajustando o tamanho da fonte dos rótulos dos eixos

    plt.show()
