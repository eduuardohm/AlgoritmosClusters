import numpy as np

def FR_Index(U, ref):

    nObj = U.shape[0]
    nClass = U.shape[1]

    soma = 0

    for i in range (0, nObj):
        vp1 = U[i,]

        vq1 = np.repeat(0, nClass, axis=0)
        vq1[ref[i]-1] = 1.0

        if i > 1:
            for j in range(0, i-1):
                vp2 = U[j,]

                distp = np.absolute(pow(vp1-vp2, 2))
                distp = np.sum(distp)/nClass
                Ep = 1.0 - distp

                vq2 = np.repeat(0, nClass, axis=0)
                vq2[ref[j]-1] = 1.0

                distq = np.sum(pow(vq1-vq2, 2))/nClass
                Eq = 1.0 - distq

                soma = soma + abs(Ep - Eq)

    fr = 1.0 - soma/(nObj*(nObj-1.0)/2.0)
    return fr