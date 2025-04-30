import sys
import numpy as np
from sklearn.datasets import load_wine, load_iris

def pho(x, y):
  covariance_matrix = np.cov(x, y)
  correlation_coefficient = covariance_matrix[0, 1] / (np.sqrt(np.var(x) * np.var(y)))

  return correlation_coefficient

def lambda2(x, y):
  correlation = pho(x, y)
  discriminant = np.square(np.var(x) + np.var(y)) - 4 * np.var(x) * np.var(y) * (1 - np.square(correlation))

  return max((np.var(x) + np.var(y) - np.sqrt(discriminant)) / 2, 0)

def find_k_nearest_features(data, feature, k=3):
  n_samples, n_features = data.shape
  selected_features = [[sys.maxsize, 0] for _ in range(k)]

  for i in range(n_features):
    if i != feature:
      value = lambda2(data[:,feature], data[:,i]) # aqui devo calcular o valor de lambda2
      max_value = max(selected_features, key=lambda x: x[0])

      if value < max_value[0]:
        selected_features.remove(max_value)
        selected_features.append([value, i])

  return selected_features

def feature_selection(data, k=3):
  n_samples, n_features = data.shape

  R = [i for i in range(n_features)]

  selected_feature = [sys.maxsize, 0]

  for i in range(n_features):
      result = find_k_nearest_features(data, i, k)
      furthest_neighbor = max(result)

      if furthest_neighbor[0] < selected_feature[0]:
        selected_feature[0] = furthest_neighbor[0]
        selected_feature[1] = furthest_neighbor[1]

  k_neighbors = find_k_nearest_features(data, selected_feature[1], k)

  for f, i in k_neighbors:
    if i in R:
      R.remove(i)

  e = max(k_neighbors)[0]

  while True:
    if k > len(R) - 1:
      k = len(R) - 1

    if k <= 1:
      break

    k -= 1
    selected_feature = [sys.maxsize, 0]

    for i in R:
      result = find_k_nearest_features(data, i, k)
      furthest_neighbor = max(result)

      if furthest_neighbor[0] < selected_feature[0]:
        selected_feature[0] = furthest_neighbor[0]
        selected_feature[1] = furthest_neighbor[1]

    k_neighbors = find_k_nearest_features(data, selected_feature[1], k)
    for f, i in k_neighbors:
      if i in R:
        R.remove(i)

    if max(k_neighbors)[0] <= e:
      break

  return R


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    R = feature_selection(X, k=2)

    print('Dataset utilizado: Iris\n')
    print(f'Número original de variáveis: {X.shape[1]}')
    print('Variáveis selecionadas:')
    print(R)