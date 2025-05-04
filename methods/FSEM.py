import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

'''
Feature Selection Based on Entropy Measure (FSEM)
'''

def euclidean_distance(xi, xj):
    return np.sqrt(np.sum(np.square(xj-xi)))

def normalize(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

def normalized_euclidean_distance(data):
    n_samples = data.shape[0]

    distances = []
    for i in range(n_samples-1):
        for j in range(i+1, n_samples):
            if data.ndim == 1:
                distances.append(euclidean_distance(data[i], data[j]))
            else:
                distances.append(euclidean_distance(data[i, :], data[j, :]))

    distances = np.sort(distances)
    xmin = np.min(distances)
    xmax = np.max(distances)

    norm_distances = []
    for dist in distances:
        norm_distances.append(normalize(dist, xmin, xmax))

    return norm_distances

class Dash2002:

    def __init__(self, data, threshold, beta=10, b=100):
        self.data = data
        self.threshold = threshold
        self.beta = beta
        self.b = b

    def execute(self, data):
        distances = normalized_euclidean_distance(data)
        buckets = [0 for _ in range(self.b+1)]

        for dist in distances:
            index = int(dist*self.b)
            buckets[index] += 1

        calculated_entropy = self.entropy(distances)

        return round(calculated_entropy, 4)

    def calc_mu(self, distances):
        Qmin = 0.05 * len(distances)

        Bi = 0
        dist = 0
        for i in range(15):
            if distances[i] > Qmin and distances[i] > distances[Bi]:
                Bi = i
                dist = distances[i]

        Et = 0.02
        mu = (1/self.beta) * np.log((np.exp(self.beta * dist) + np.exp(0)) / Et - 1)

        return mu

    def compute_eij(self, distances, dist):
        mu = self.calc_mu(distances)

        if mu >= dist:
            eij = ( np.exp(self.beta*dist) - np.exp(0) ) / ( np.exp(self.beta*mu) - np.exp(0) )
        else:
            eij = ( np.exp(self.beta*(1-dist)) - np.exp(0) ) / ( np.exp(self.beta*(1-mu)) - np.exp(0) )

        return eij

    def entropy(self, distances):
        e = 0

        for dist in distances:
            e += self.compute_eij(distances, dist)

        return e

    def forward_selection(self):
        knn = KNeighborsClassifier(n_neighbors=1)
        sfs = SequentialFeatureSelector(knn, n_features_to_select=self.threshold, direction='forward')
        sfs.fit(self.data, np.zeros(self.data.shape[0]))  # Aqui estou usando np.zeros apenas como um rótulo fictício

        selected_features = sfs.get_support(indices=True)

        return selected_features