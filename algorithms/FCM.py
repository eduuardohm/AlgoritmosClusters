import numpy as np
from timeit import default_timer as timer

def FCM(data, centers, parM):
  
  start = timer()
  
  maxIteration = 100
  J = 1000
 
  count = 0
  
  P = initializePrototypes(data,centers)
  
  Ubefore = 0
  Jbefore = J + 1.0
  
  while abs(Jbefore - J) > 0.000001 and count < maxIteration:
    
    count += 1
    
    D = updateDistances(data,P)
    # print("Matrix D: ")
    # print(D)
    # print("\n\n")
 
    U = updateMembership(D,parM)
    # print("Matrix U: ")
    # print(U)
    # print("\n\n")
 
    P = updatePrototypes(data,U,parM)
    # print("Matrix P: ")
    # print(P)
    # print("\n\n")
    
    Jbefore = J
    J = updateCriterion(U,D,parM)
    # print('J: ', J)
    
    Ubefore = U 
  
  L = getPartition(Ubefore)

  end = timer()
  
  result = [J, L, Ubefore, count, end - start]
  return result
 
def initializePrototypes(data,centers):
 
  nVar = data.shape[1]
  nProt = len(centers)
  # print(f'Numero de var: {nVar}\nNumero de prot: {nProt}')
  
  P = np.arange(nProt * nVar)
  P = P.reshape((nProt, nVar))
  
  P = (np.zeros_like(P)).astype('float64')
  
  for k in range(0, nProt):
    P[k,] = data[centers[k],]
    # print(P[k,])

  # print(P)    # Não tá inicializando corretamente a matriz P.
  
  return P
 
def updatePrototypes(data, memberships, parM):
  
  nObj = data.shape[0]
  nVar = data.shape[1]
  nProt = memberships.shape[1]
  
  P = np.arange(nProt * nVar)
  P = P.reshape((nProt, nVar))
  P = (np.zeros_like(P)).astype('float64')
  
  for k in range(0, nProt):
    
    s = 0
    ss = 0
    
    for j in range(0, nObj):
        
        delta = pow(memberships[j, k], parM)
        obj = data[j,]
        
        s += delta*obj
        ss += delta
    
    s = s/ss
    P[k,] = s
  
  return P
  
def updateDistances(data, prototypes):
  
  nObj = data.shape[0]
  nProt = prototypes.shape[0]
  
  D = np.arange(nObj * nProt)
  D = D.reshape((nObj, nProt))
  
  D = (np.zeros_like(D)).astype('float64')
  
  for j in range(0, nObj):
    
    obj = data[j,]
    
    for k in range(0, nProt):
        prot = prototypes[k,]
        
        # distance = sum((obj - prot)^2)
 
        distance = np.sum(np.square(np.subtract(obj, prot)))
        D[j,k] = distance
      
  return D
 
def updateMembership(distances, parM):
  
  nObj = distances.shape[0]
  nProt = distances.shape[1]
  
  U = np.arange(nObj * nProt)
  U = U.reshape((nObj, nProt))
  
  U = (np.zeros_like(U)).astype('float64')
  
  for i in range(0, nObj):
    
    for k in range(0, nProt):
      
      d = distances[i,k]
      soma = 0
      
      for kk in range(0, nProt):
        dd = distances[i,kk];       
        
        aux1 = ((int(d)+0.0000001)/(int(dd)+0.0000001))
        aux2 = (1.0/(parM-1.0))
        # soma += ((int(d)+0.0000001)/(int(dd)+0.0000001))^(1.0/(parM-1.0))
        soma += pow(aux1, aux2)
    
      soma = pow(soma, -1.0)
      
      U[i,k] = soma
  
  return U
 
def updateCriterion(memberships,distances,parM):
  
  J = 0
  
  nObj = distances.shape[0]
  nProt = distances.shape[1]
  
  for j in range(0, nObj):
    
    for k in range(0, nProt):
      
      delta = pow(memberships[j,k], parM)
      # print("Memberships:")
      # print(memberships[j,k])
      # print("Delta:")
      # print(delta)
      distance = distances[j,k]
      # print("Distancia:")
      # print(distance)
      
      J += delta*distance
  
  return J

def getPartition(memberships):
  
  L = []

  for object in memberships:
    L.append(np.argmax(object) + 1)
  
  return L