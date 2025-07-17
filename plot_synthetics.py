import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})
colors = plt.get_cmap('tab10').colors

n = 300
nClusters = 3

mu = np.array([
    [0, 0],
    [0, 15],
    [0, 30]
])
sigma = np.array([
    [10, 1],
    [10, 1],
    [10, 1]
])
sigma = [np.diag(sigma[i]) for i in range(3)]


X_class1 = np.random.multivariate_normal(mu[0], sigma[0], size=int(n/nClusters))
X_class2 = np.random.multivariate_normal(mu[1], sigma[1], size=int(n/nClusters))
X_class3 = np.random.multivariate_normal(mu[2], sigma[2], size=int(n/nClusters))


plt.figure(figsize=(8, 6), dpi=100)
# plt.scatter(X_class1[:, 0], X_class1[:, 1], color='r', label='Class 1')
# plt.scatter(X_class2[:, 0], X_class2[:, 1], color='g', label='Class 2')
# plt.scatter(X_class3[:, 0], X_class3[:, 1], color='b', label='Class 3')
plt.scatter(X_class1[:, 0], X_class1[:, 1], color=colors[0], label='Class 1')
plt.scatter(X_class2[:, 0], X_class2[:, 1], color=colors[1], label='Class 2')
plt.scatter(X_class3[:, 0], X_class3[:, 1], color=colors[2], label='Class 3')
plt.xlabel('Variable $v_{1}$')
plt.ylabel('Variable $v_{2}$')
plt.title('Visualization of synthetic dataset DS-1a')
plt.legend()
plt.grid(True)

plt.xlim(-15, 15)
plt.ylim(-10, 40)

plt.show()