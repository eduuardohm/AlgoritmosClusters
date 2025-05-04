import numpy as np
# from numba import jit, cuda
from timeit import default_timer as timer

# @jit(target_backend='cuda', forceobj=True)
def MFCM(data, centers, parM):

  start = timer()

  maxIteration = 100
  J = np.iinfo(np.int32).max

  count = 0

  P = initializePrototypes(data,centers)

  Ubefore = None
  Jbefore = J + 1.0

  while (Jbefore - J) > 0.0001 and count < maxIteration:

    count += 1
    D = updateDistances(data, P)
    U = updateMembership(D, parM)
    P = updatePrototypes(data, U, parM)
    Jbefore = J
    J = updateCriterion(U, D, parM)
    Ubefore = U	

  M = np.ones((len(centers), data.shape[1]))
  memb = aggregateMatrix(Ubefore, M)
  L = getPartition(memb)

  end = timer()

  resp = [J, L, Ubefore, count, end - start, memb]

  return resp
	

def initializePrototypes(data,centers):
  return np.array([data[c] for c in centers])

# @jit(target_backend='cuda')
def updatePrototypes(data, memberships, parM):
  
  U = np.stack(memberships)               # → (nVar, nObj, nProt)
  W = U ** parM                           # → pesos elevados a m

  data_T = data.T[:, :, None]             # → (nVar, nObj, 1)

  numer = np.sum(W * data_T, axis=1)      # → (nVar, nProt)
  denom = np.sum(W, axis=1)               # → (nVar, nProt)
  P = (numer / denom).T                   # → (nProt, nVar)

  return P

#@jit(target_backend='cuda')
def updateDistances(data, prototypes):
  
  diff = data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]  # (nObj, nProt, nVar)
  D_full = diff ** 2  

  D = D_full.transpose(2, 0, 1)     

  return D

# @jit(target_backend='cuda')
def updateMembership(distances, parM):
  nObj = distances[0].shape[0]
  nProt = distances[0].shape[1]
  nVar = len(distances)
  U = []
  
  # Stack all distances into a single 3D array for efficient computation
  distances_stacked = np.stack(distances)  # Shape: (nVar, nObj, nProt)
  
  # Precompute the exponent term
  exponent = 1.0 / (parM - 1.0)
  
  for v in range(nVar):
      Uvar = np.zeros((nObj, nProt))
      for i in range(nObj):
          # Extract the distance for the current object i and variable v
          d = distances_stacked[v, i, :]  # Shape: (nProt,)
          
          # Reshape d to (nProt, 1, 1) for broadcasting
          d_reshaped = d[:, None, None]  # Shape: (nProt, 1, 1)
          
          # Reshape distances_stacked[:, i, :] to (1, nVar, nProt) for broadcasting
          dd_reshaped = distances_stacked[:, i, :][None, :, :]  # Shape: (1, nVar, nProt)
          
          # Compute the ratio (d + 1e-7) / (dd + 1e-7) for all vv and kk
          ratio = (d_reshaped + 1e-7) / (dd_reshaped + 1e-7)  # Shape: (nProt, nVar, nProt)
          
          # Raise the ratio to the power of the exponent
          ratio_pow = ratio ** exponent  # Shape: (nProt, nVar, nProt)
          
          # Sum over vv and kk
          soma = np.sum(ratio_pow, axis=(1, 2))  # Shape: (nProt,)
          
          # Compute the final membership value
          Uvar[i, :] = soma ** (-1.0)
      
      U.append(Uvar)
  
  return U

# @jit(target_backend='cuda')
def updateCriterion(memberships,distances,parM):
  U = np.stack(memberships)               # → (nVar, nObj, nProt)
  return np.sum((U ** parM) * distances)


def aggregateMatrix(memberships, M):
    
  nObj, nProt = memberships[0].shape
  nVar = len(memberships)
  memb = np.zeros((nObj, nProt))
  
  for j in range(nObj):
      soma0 = sum(sum(M[k, i] * memberships[i][j, k] for i in range(nVar)) for k in range(nProt))
      for k in range(nProt):
          memb[j, k] = sum(M[k, i] * memberships[i][j, k] for i in range(nVar)) / soma0
  
  return memb

def computeAij(memberships):
  memberships = np.stack(memberships)  # Stack list of arrays into a 3D array
  soma = np.sum(memberships, axis=(0, 1))  # Sum over objects and variables for each prototype
  M = np.sum(memberships, axis=1) / soma  # Normalize
  return M.T  # Transpose to match the original shape


def getPartition(memberships):
  
  # L = []

  # for object in memberships:
  #   L.append(np.argmax(object) + 1)

  L = [np.argmax(object) + 1 for object in memberships]
  
  return L