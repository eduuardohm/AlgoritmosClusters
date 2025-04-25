import numpy as np

def sum_filter(data, U, nClusters):
    V = []
    nVar = data.shape[1]

    for i in range(0, nVar):
        aTotal = [] # Relevância total
        nObj = data.shape[0]

        for j in range(0, nClusters):  
            a = 0
            for k in range(0, nObj):
                u = U[i][k][j] # Acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
                a += u
        
            a /= nObj # Relevância em relação ao cluster j
            # print(f'a: {a}| cluster: {j} | variável: {i}')
            aTotal.append(a)

        aTotal = np.mean(aTotal)
        aTotal = round(aTotal, 5)
        V.append((aTotal, i))
    
    return (V, 'Filtro por Somatório')

def variance_filter(data, U, nClusters):
    V = []
    nVar = data.shape[1]

    for i in range(0, nVar):
        # print(f'i: {i}')
        aTotal = [] # Relevância total
        nObj = data.shape[0]

        for j in range(0, nClusters):  
            a = []
            for k in range(0, nObj):
                u = U[i][k][j] # Acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
                a.append(u)
        
            a = np.asarray(a)
            var = np.var(a)
            aTotal.append(var)

        aTotal = np.asarray(aTotal)
        # print(aTotal)
        media = np.mean(aTotal)

        V.append((round(media, 5), i))
    
    return (V, 'Filtro por Variância')


def apply_filter(dataset, result, n, method=1):

    # print(f'Resultado sem ordenação: {result}')
    if method == 'var': # Variância
        result[0].sort(key=lambda k : k[0], reverse=True)
    else: # Somatório
        result[0].sort(key=lambda k : k[0])

    # print(f'\n{result[1]}: ')
    # for item in range(len(result[0])):
        # print(f'Variável {result[0][item][1]}: {result[0][item][0]}')

    listaCorte = [result[0][i][1] for i in range(n)]
    dataset = np.delete(dataset, listaCorte, axis = 1)

    # print("\nVariáveis deletadas:")
    # for var in listaCorte:
    #    print(f"Variável {var}")
        
    return dataset