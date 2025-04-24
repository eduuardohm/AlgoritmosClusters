import numpy as np
from numba import jit, cuda
from timeit import default_timer as timer

@jit(target_backend='cuda', forceobj=True)
def MFCM(data, centers, parM):

  start = timer()

  maxIteration = 100
  J = 1000

  count = 0

  P = initializePrototypes(data,centers)

  Ubefore = 0
  Jbefore = J + 1.0

  while (Jbefore - J) > 0.0001 and count < maxIteration:

    count += 1
    
    D = updateDistances(data, P)
    # print('Distancia atualizadas')

    U = updateMembership(D, parM)
    # print('Membership atualizadas')

    P = updatePrototypes(data, U, parM)
    # print('Prototipos atualizados')

    Jbefore = J
    J = updateCriterion(U, D, parM)
    # print('Criterio atualizados')

    Ubefore = U	

  M = np.arange(len(centers) * data.shape[1])
  M = M.reshape((len(centers), data.shape[1]))

  M = (np.ones_like(M)).astype('float64')

  memb = aggregateMatrix(Ubefore,M)
  L = getPartition(memb)

  end = timer()

  resp = [J, L, Ubefore, count, end - start, memb]

  return resp
	

def initializePrototypes(data,centers):
 
  nVar = data.shape[1]
  nProt = len(centers)
  
  P = np.arange(nProt * nVar)
  P = P.reshape((nProt, nVar))
  
  P = (np.zeros_like(P)).astype('float64')
  
  for k in range(0, nProt):
    P[k,] = data[centers[k],]
  
  return P

@jit(target_backend='cuda')
def updatePrototypes(data, memberships, parM):
  
  nObj = data.shape[0]
  nVar = data.shape[1]
  nProt = memberships[0].shape[1]
  
  P = np.arange(nProt * nVar)
  P = P.reshape((nProt, nVar))
  P = (np.zeros_like(P)).astype('float64')

  for i in range(0, nVar):
  
    for k in range(0, nProt):
        
      s = 0
      ss = 0
      
      for j in range(0, nObj):
          
        delta = pow(memberships[i][j, k], parM)
        obj = data[j,i]
        
        s += delta*obj
        ss += delta
      
      s = s/ss
      P[k,i] = s
  
  return P

#@jit(target_backend='cuda')
def updateDistances(data, prototypes):
  
  nObj = data.shape[0]
  nProt = prototypes.shape[0]
  nVar = data.shape[1]
  
  D = np.arange(nVar * nObj * nProt)
  D = D.reshape((nVar, nObj, nProt))
  
  D = (np.zeros_like(D)).astype('float64')
  
  Dvar = np.arange(nObj * nProt)
  Dvar = Dvar.reshape((nObj, nProt))

  for i in range(0, nVar):
    
    Dvar = (np.zeros_like(Dvar)).astype('float64')
  
    for j in range(0, nObj):
        
      obj = data[j, i]
      
      for k in range(0, nProt):
        prot = prototypes[k,i]
        distance = pow(obj-prot, 2)
        # distance = np.sum(np.square(np.subtract(obj, prot)))
        Dvar[j,k] = distance


    D[i] = Dvar.copy()
      
  # print(f'D dim: {D.ndim}')
  # print(D)

  return D

@jit(target_backend='cuda')
def updateMembership(distances, parM):
  
  nObj = distances.shape[1]
  nProt = distances[0].shape[1]
  nVar = len(distances)
  
  U = np.arange(nVar * nObj * nProt)
  U = U.reshape((nVar, nObj, nProt))
  
  U = (np.zeros_like(U)).astype('float64')

  for v in range(0, nVar):

    Uvar = np.arange(nObj * nProt)
    Uvar = Uvar.reshape((nObj, nProt))
    
    Uvar = (np.zeros_like(Uvar)).astype('float64')
  
    for i in range(0, nObj):
        
      for k in range(0, nProt):
      
        d = distances[v][i,k]
        # print(f'd: {d}')
        soma = 0

        for vv in range(0, nVar):
          for kk in range(0, nProt):
            dd = distances[vv][i,kk]
            
            aux1 = ((int(d)+0.0000001)/(int(dd)+0.0000001))
            aux2 = (1.0/(parM-1.0))

            soma += pow(aux1, aux2)
    
        soma = pow(soma, -1.0)
    
        Uvar[i,k] = soma
    
    U[v] = Uvar.copy()

  # print(f'U: {U}')
  
  return U

@jit(target_backend='cuda')
def updateCriterion(memberships,distances,parM):
  
  J = 0
  
  nObj = distances.shape[1]
  nProt = distances[0].shape[1]
  nVar = len(distances)

  # print('Criterio')
  # print(memberships.size)
  # print(memberships)

  # print(memberships)

  for i in range(0, nVar):
  
    for j in range(0, nObj):
        
      for k in range(0, nProt):
        delta = pow(memberships[i][j,k], parM)
        distance = distances[i][j,k]
        
        J += delta*distance
    
  return J


def aggregateMatrix(memberships, M):
    
  nObj = memberships.shape[1]
  nProt = memberships[0].shape[1]
  nVar = len(memberships)

  memb = np.arange(nObj * nProt)
  memb = memb.reshape((nObj, nProt))
  
  memb = (np.zeros_like(memb)).astype('float64')	

  for j in range(0, nObj):
    soma0 = 0
    
    for k in range(0,nProt):
      soma = 0
      
      for i in range(0, nVar):
          soma += M[k,i]*memberships[i][j,k]
          soma0 += M[k,i]*memberships[i][j,k]
      
      memb[j,k] = soma
    
    for k in range(0,nProt):
      memb[j,k] = memb[j,k]/soma0

  return memb


def computeAij(memberships):
    
  nProt = memberships.shape[1]
  nVar = len(memberships)

  M = np.arrange(nProt * nVar)
  M = M.reshape((nProt, nVar))

  M = (np.ones_like(M)).astype('float64')

  for j in range(0, nProt):
    for k in range(0,nVar):
      M[j,k] = np.sum(memberships[k][:,j])
      soma = 0
      
      for kk in range(0, nVar):
        soma =  soma + sum(memberships[kk][:,j])
      
      M[j,k] = M[j,k]/soma

  return M


def getPartition(memberships):
  
  # L = []

  # for object in memberships:
  #   L.append(np.argmax(object) + 1)

  L = [np.argmax(object) + 1 for object in memberships]
  
  return L