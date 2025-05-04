from scipy.io import loadmat

# Carregar o arquivo .mat
path = "datasets/lymphoma.mat"
mat = loadmat(path)

# Ver as chaves do arquivo .mat
print(mat.keys())