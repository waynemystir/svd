import numpy as np
from numpy.linalg import norm

from random import normalvariate
from math import sqrt


def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    if type(A) == list: A=np.array(A, dtype=float)
    n, m = A.shape
    x = randomUnitVector(m)
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if np.dot(currentV, lastV) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV

def svd(A, epsilon=1e-10):
    if type(A) == list: A=np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
 
    for i in range(m):
        matrixFor1D = A.copy()
 
        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)
 
        v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
        u_unnormalized = np.dot(A, v)
        sigma = norm(u_unnormalized)  # next singular value
        u = u_unnormalized / sigma
 
        svdSoFar.append((sigma, u, v))
 
    # transform it into matrices of the right shape
    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
 
    return singularValues, us.T, vs
