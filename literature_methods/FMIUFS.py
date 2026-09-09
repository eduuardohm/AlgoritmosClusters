import numpy as np

'''
Yuan Zhong, Chen Hongmei, Zhang Pengfei, Wan Jihong, Li Tianrui. A novel unsupervised approach to 
heterogeneous feature selection based on fuzzy mutual information, IEEE Transactions on Fuzzy Systems. 
2021, DOI: 10.1109/TFUZZ.2021.3114734
'''

def fmiufs_kersim(a, x, e):
    if abs(a - x) > e:
        return 0
    if e == 0:
        return 1 if a == x else 0
    return 1 - abs(a - x)

def entropy(M):
    a, _ = M.shape
    K = 0
    for i in range(a):
        Si = -(1 / a) * np.log2(np.sum(M[i, :]) / a)
        K += Si
    return K

def ufs_FMI(data, lammda):
    row, attrinu = data.shape
    delta = np.zeros(attrinu)
    
    for j in range(attrinu):
        if np.min(data[:, j]) == 0 and np.max(data[:, j]) == 1:
            delta[j] = np.std(data[:, j], ddof=1) / lammda
    
    # Compute the relation matrix
    ssr = {}
    for i in range(attrinu):
        col = i
        r = np.zeros((row, row))
        
        for j in range(row):
            a = data[j, col]
            x = data[:, col]
            
            for m in range(len(x)):
                r[j, m] = fmiufs_kersim(a, x[m], delta[i])
        
        ssr[col] = r
    
    # UFS based on fuzzy mutual information
    unSelect_Fea = []
    Select_Fea = []
    sig = []
    base = np.ones(row)
    
    E = np.zeros(attrinu)
    Joint_E = np.zeros((attrinu, attrinu))
    MI = np.zeros((attrinu, attrinu))
    
    for j in range(attrinu):
        r = ssr[j]
        E[j] = entropy(r)
    
    for i in range(attrinu):
        ri = ssr[i]
        for j in range(i + 1):
            rj = ssr[j]
            Joint_E[i, j] = entropy(np.minimum(ri, rj))
            Joint_E[j, i] = Joint_E[i, j]
            MI[i, j] = E[i] + E[j] - Joint_E[i, j]
            MI[j, i] = MI[i, j]
    
    Ave_MI = np.mean(MI, axis=1)
    sorted_indices = np.argsort(Ave_MI)[::-1]
    sig.append(Ave_MI[sorted_indices[0]])
    Select_Fea.append(sorted_indices[0])
    unSelect_Fea = list(sorted_indices[1:])
    
    while unSelect_Fea:
        Red = np.zeros((len(unSelect_Fea), len(Select_Fea)))
        
        for i in range(len(unSelect_Fea)):
            for j in range(len(Select_Fea)):
                Red[i, j] = Ave_MI[Select_Fea[j]] - (
                    (Joint_E[Select_Fea[j], unSelect_Fea[i]] - E[unSelect_Fea[i]])
                    / E[Select_Fea[j]] * Ave_MI[Select_Fea[j]]
                )
        
        max_sig_idx = np.argmax(Ave_MI[unSelect_Fea] - np.mean(Red, axis=1))
        max_sig = Ave_MI[unSelect_Fea[max_sig_idx]]
        sig.append(max_sig)
        Select_Fea.append(unSelect_Fea[max_sig_idx])
        unSelect_Fea.remove(unSelect_Fea[max_sig_idx])
    
    return Select_Fea
