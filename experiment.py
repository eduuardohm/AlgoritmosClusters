import numpy as np
import matplotlib.pyplot as plt
from utility.metrics import FR_Index
from sklearn.metrics import f1_score, davies_bouldin_score, silhouette_score, normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import random
import time

from datasets.datasets import selectDataset
from algorithms.FCM import FCM
from algorithms.MFCM import MFCM
from algorithms.KMeans import KMeans
from methods.filters import *

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def execute(nRep, dataset, centersAll, exec_time):
	Jmin = 2147483647	# int_max do R
	bestL = 0			# melhor valor de L_resp
	bestM = 0			# melhor valor de M_resp
	best_centers = 0

	for r in range(nRep):
		print(f'Rep: {r+1}/{nRep}')
		centers = list(map(int, centersAll[r,].tolist()))

		resp = MFCM(dataset, centers, 2)

		J = resp[0]
		L_resp = resp[1]
		M_resp = resp[2]

		if (Jmin > J):
			Jmin = J
			bestL = L_resp
			bestM = M_resp
			best_centers = centers

		exec_time += resp[4]
		
	dict = {'Jmin': Jmin, 'bestL': bestL, 'bestM': bestM, 'exec_time': exec_time, 'best_centers': centersAll}
	# Retorna os centers de todas as iterações para o KMeans (mudar para criar uma nova lista exclusiva para o KMeans)
	
	return dict

def exec_mfcm(indexData, mc, nRep, seed):

	## Inicializando variáveis
	exec_time = 0
	result = {}
	Jmin = 2147483647
	centers = 0

	## Monte Carlo
	for i in range(mc):
		np.random.seed(seed)
		random.seed(seed)

		synthetic = selectDataset(indexData)
		dataset = synthetic[0]
		ref = synthetic[1]
		nClusters = synthetic[2]
		dataName = synthetic[3]

		centersMC = np.zeros((nRep, nClusters))

		for c in range(nRep):
			centersMC[c] = random.sample(range(1, len(dataset)), nClusters)

		clustering = execute(nRep, dataset, centersMC, exec_time)
		exec_time += clustering['exec_time']

		if clustering['Jmin'] < Jmin:
			Jmin = clustering['Jmin']
			result = clustering
			centers = clustering['best_centers']

	return (result, exec_time, centers)

def exec_kmeans(K, nRep, dataset, centers):

	bestResult = 0
	Jmin = 2147483647

	for r in range(nRep):
		centersRep = list(map(int, centers[r,].tolist()))
		k = KMeans(K, nRep,)
		result, time, J = k.predict(dataset, SEED, centersRep)

		if J < Jmin:
			Jmin = J
			bestResult = result

	return [bestResult, time]

def run_filter(method, dataset, result, numVar, numClusters):
		
	if method == 'mean':
		resultado_filtro = sum_filter(dataset, result['bestM'], numClusters)
	elif method == 'var':
		resultado_filtro = variance_filter(dataset, result['bestM'], numClusters)
	elif method == 'mean_image':
		resultado_filtro = sum_filter(dataset, result['bestM'], numClusters)
		dataset = apply_image_filter(dataset, resultado_filtro, numVar, method)
		return dataset
	elif method == 'var_image':
		resultado_filtro = variance_filter(dataset, result['bestM'], numClusters)
		dataset = apply_image_filter(dataset, resultado_filtro, numVar, method)
		return dataset
	dataset = apply_filter(dataset, resultado_filtro, numVar, method)

	return dataset

def calculate_accuracy(L, ref, U, dataset):
	ari = adjusted_rand_score(L, ref)
	nmi = normalized_mutual_info_score(ref, L)
	silhouette = silhouette_score(dataset, L)
	db = davies_bouldin_score(dataset, L)

	return [ari, nmi, silhouette, db]

def experiment(indexData, mc, nRep, nVar, method):
	
	exec_time = 0
	centers = 0

	ari_scores = []
	nmi_scores = []
	silhouette_scores = []
	db_scores = []

	for i in range(mc):
		np.random.seed(i + int(time.time()))
		random.seed(i + int(time.time()))

		synthetic = selectDataset(indexData)
		dataset = synthetic[0]
		ref = synthetic[1]
		nClusters = synthetic[2]
		dataName = synthetic[3]
		# parameters = synthetic[4]

		i == 0 and print(f'Dataset escolhido: {dataName}')

		centersMC = np.zeros((nRep, nClusters))

		for c in range(nRep):
			centersMC[c] = random.sample(range(1, len(dataset)), nClusters)

		mfcm_result = execute(nRep, dataset, centersMC, exec_time)
		centers = mfcm_result['best_centers']
		exec_time += mfcm_result['exec_time']
		
		filtered_data = run_filter(method, dataset, mfcm_result, nVar, nClusters)

		filtered_result, filtered_time = exec_kmeans(nClusters, nRep, filtered_data, centers)

		metrics = calculate_accuracy(filtered_result, ref, None, filtered_data)

		ari_scores.append(metrics[0])
		nmi_scores.append(metrics[1])
		silhouette_scores.append(metrics[2])
		db_scores.append(metrics[3])

		print(f'MC: {i+1}/{mc}')

	print(f'Labels previstos: {filtered_result}\nLabels reais: {ref}')
		
	mean_ari = np.mean(ari_scores)
	mean_nmi = np.mean(nmi_scores)
	mean_silhouette = np.mean(silhouette_scores)
	mean_db = np.mean(db_scores)

	print(f'Média ARI: {mean_ari}')
	print(f'Média NMI: {mean_nmi}')
	print(f'Média Silhouette: {mean_silhouette}')
	print(f'Média Davies-Bouldin: {mean_db}')

	data_info = (f'\nDataset: {dataName} | N_samples: {len(dataset)} | N_variaveis: {len(dataset[0])} | N_clusters: {nClusters}\nMetodo: {method} | MC: {mc} | MFCM_Rep: {nRep} | Variaveis cortadas: {nVar}\n')
	metrics_info = (f'Resultados do filtro:\nARI: {mean_ari}\nNMI: {mean_nmi}\nSilhoutte: {mean_silhouette}\nDB: {mean_db}\n')
	# parameters_info = (f'Parametros de distribuicao do dataset:\n{parameters}\n')

	atualizaTxt(f'logs/{dataName}.txt', data_info)
	atualizaTxt(f'logs/{dataName}.txt', metrics_info)
	# atualizaTxt(f'logs/{dataName}.txt', parameters_info)
	atualizaTxt(f'logs/{dataName}.txt', '#################################################################')

	# PLOT A SEGUIR:
	
	return (mfcm_result, filtered_time, centers)

def plot_results(plot_info, ref, dataset_name, exec_time, U, dataset, dataset_antigo, datasetName):
	fig, axes = plt.subplots(2, 2, figsize=(10, 8))

	# fig.subplots_adjust(bottom=0.25)
	
	# Define as posições do plot em variáveis
	ax_top_left = axes[0, 0]
	ax_top_right = axes[0, 1]
	ax_bottom = axes[1, 0]
	ax_info = axes[1, 1]

	# Plota resultados
	x_axis = dataset_antigo[:, plot_info[0]]
	y_axis = dataset_antigo[:, plot_info[1]]

	ax_top_left.scatter(x_axis, y_axis, c=ref, label='Referência')
	ax_top_right.scatter(x_axis, y_axis, c=plot_info[4], label=f'{dataset_name} - Sem Filtro')
	
	x_axis = dataset[:, 0]
	y_axis = dataset[:, 1]
	ax_bottom.scatter(x_axis, y_axis, c=plot_info[5], label=f'{dataset_name} - Com Filtro')

	# Adiciona títulos aos subplots
	ax_top_left.set_title('Referência', weight='bold')
	ax_top_right.set_title(f'{dataset_name} - Sem Filtro', weight='bold')
	ax_bottom.set_title(f'{dataset_name} - Com Filtro', weight='bold')

	# Remove o gráfico no canto inferior direito e adiciona métricas
	acc1 = calculate_accuracy(plot_info[4], ref, U, dataset_antigo)
	acc2 = calculate_accuracy(plot_info[5], ref, U, dataset)
	
	metrics_info1 = ("Resultado sem filtro:\nARI: {:.2f}\nNMI: {:.2f}\nSilhoutte: {:.2f}\nDB: {:.2f}\nTempo de Execucao: {:.2f}s".format(acc1[0], acc1[1], acc1[2], acc1[3], exec_time[0]))
	metrics_info2 = ("Resultado com filtro:\nARI: {:.2f}\nNMI: {:.2f}\nSilhoutte: {:.2f}\nDB: {:.2f}\nTempo de Execucao: {:.2f}s".format(acc2[0], acc2[1], acc2[2], acc2[3], exec_time[1]))

	metrics_txt1 = [acc1[0], acc1[1], acc1[2], acc1[3], exec_time[0], datasetName]
	metrics_txt2 = [acc2[0], acc2[1], acc2[2], acc2[3], exec_time[1], datasetName]

	for i in range(len(metrics_txt1) - 1):
		metrics_txt1[i] = round(metrics_txt1[i], 2)
		metrics_txt2[i] = round(metrics_txt2[i], 2)

	atualizaTxt(f'logs/{dataset_name}.txt', metrics_txt1)
	atualizaTxt(f'logs/{dataset_name}.txt', metrics_txt2)
	atualizaTxt(f'logs/{dataset_name}.txt', '')
	
	# Ajusta os espaçamentos entre os subplots
	ax_info.text(0, 0.75, metrics_info1, va='center', fontsize=11)
	ax_info.text(0, 0.25, metrics_info2, va='center', fontsize=11)
	
	ax_info.axis('off')

	plt.tight_layout()

	plt.show()

def atualizaTxt(nome, lista):
	arquivo = open(nome, 'a')
	for i in range(len(lista)):
		arquivo.write(str(lista[i]))
	arquivo.write('\n')
	arquivo.close()

if __name__ == "__main__":
	mc = 5
	nRep = 5
	indexData = 6
	numVar = 1

	# result, ref, centers = experiment(indexData, mc, nRep, numVar, 'mean')
	result, ref, centers = experiment(indexData, mc, nRep, numVar, 'mean')