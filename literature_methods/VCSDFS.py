import numpy as np

'''
S. Karami, F. Saberi-Movahed, P. Tiwari, P. Marttinen, and S. Vahdati, 
“Unsupervised feature selection based on variance–covariance subspace distance,” 
Neural Netw., vol. 166, pp. 188–203, 2023
'''

def VCSDFS(X, XX, XXX, rho, k, NITER):
    """
    Unsupervised Feature Selection Based on Variance-Covariance Subspace Distance
    
    Parameters:
    X : ndarray
        Data matrix of shape (n, d), where n is the number of samples and d is the number of features.
    XX : ndarray
        Precomputed (X.T @ X).
    XXX : ndarray
        Precomputed (XX @ XX).
    rho : float
        Regularization parameter (needs to be tuned).
    k : int
        Number of selected features.
    NITER : int
        Maximum number of iterations.

    Returns:
    W : ndarray
        Feature weight matrix of shape (d, k).
    index : ndarray
        Indices of selected features sorted by importance.
    obj : list
        Objective function values over iterations.
    """
    d = X.shape[1]
    W = np.random.rand(d, k)
    obj = []
    
    for _ in range(NITER):
        # Update W
        numW = XXX @ W + rho * W
        A = XX @ W
        denW = A @ W.T @ A + rho * np.ones((d, d)) @ W
        fracW = np.power(numW / denW, 1/4)
        W *= fracW
        
        # Compute the objective function
        XW = X @ W
        WW = W @ W.T
        obj_val = 0.5 * (np.linalg.norm(X @ X.T - XW @ XW.T, 'fro')**2) + rho * (np.trace(np.ones((d, d)) @ WW) - np.trace(WW))
        obj.append(obj_val)
    
    # Feature selection based on row-wise sum of squared elements
    score = np.sum(W**2, axis=1)
    index = np.argsort(score)[::-1]  # Sort indices in descending order
    
    return W, index, obj
