from sklearn.metrics import davies_bouldin_score, silhouette_score, normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from methods.filters import *

def calculate_accuracy(L, ref, U, dataset):
	ari = adjusted_rand_score(L, ref)
	nmi = normalized_mutual_info_score(ref, L)
	silhouette = silhouette_score(dataset, L)
	db = davies_bouldin_score(dataset, L)

	return [ari, nmi, silhouette, db]

def run_filter(method, dataset, result, numVar, numClusters):
		
	if method == 'mean':
		resultado_filtro = sum_filter(dataset, result['bestM'], numClusters)
	elif method == 'var':
		resultado_filtro = variance_filter(dataset, result['bestM'], numClusters)
	dataset = apply_filter(dataset, resultado_filtro, numVar, method)

	return dataset