import numpy as np
import warnings

def GaussElimination(Q,W):
    A = Q.copy()
    B = W.copy()
    if np.shape(A)[0] == np.shape(A)[1]:
        if np.linalg.det(A) != 0:
            size = np.shape(A)[0]
            for j in range(0,size):
                for i in range(j+1,size):
                    if B.all() != 0:
                        B[i] -= ((A[i,j]/A[j,j])*B[j])
                    A[i] -= ((A[i,j]/A[j,j])*A[j])
            S = np.zeros(size,float)
            S[size-1] = B[size-1]/A[size-1,size-1]
            for n in range(0,size-1):
                k = (size-1)-n-1
                p = 0
                for l in range(0,size):
                    p = p + (A[k,l]*S[l])
                S[k] = (B[k] - p)/A[k,k]
            return S
        else:
            warnings.warn("Coefficient matrix is singular. This means there are no solutions or infinite number of solutions.")
    else:
        warnings.warn("Coefficient matrix is not square")

def doolittleLU(A,B):
    if np.shape(A)[0] == np.shape(B)[0]:
        if np.shape(A)[0] == np.shape(A)[1]:
            if np.linalg.det(A) != 0:
                size = np.shape(A)[0]
                U = A.copy()
                L = np.zeros((size,size),float)
                for l in range(0,size):
                    for m in range(l+1,size):
                        L[m,l] = A[m,l]/A[l,l]
                for k in range(0,size):
                    L[k,k] = 1
                for j in range(0,size):
                    for i in range(j+1,size):
                        U[i] = U[i] - L[i,j]*U[j]
                print(np.matmul(L,U))
                Y = GaussElimination(L,B)
                X = GaussElimination(U,Y)
                return X
            else:
                return ["infinite or no solutions"]
        else:
            warnings.warn("Coefficient matrix is not square")
    else:
        warnings.warn("Input coeffecient matrix and constant vector dont have same dimension")

def PivotedGausselimination(A,B):
    if np.shape(A)[0] == np.shape(A)[1]:
        if (np.linalg.det(A)) != 0:
            size = np.shape(A)[0]
            R = np.zeros((size,size),float)
            for i in range(0,size):
                for j in range(0,size):
                    R[i,j] = (A[i,j])/np.amax(np.abs(A[i]))
            for y in range(0,size):
                f = np.amax((R[y:,y]))
                m = np.where(R[:,y]==f)[0][0]
                A[m],A[y] = A[y].copy(),A[m].copy()
                B[m],B[y] = B[y].copy(),B[m].copy()
                R[m],R[y] = R[y].copy(),R[m].copy()
            return GaussElimination(A,B)
        else:
            warnings.warn("Coefficient matrix is singular. This means there are no solutions or infinite number of solutions.")
    else:
        warnings.warn("Coefficient matrix is not square")

def PivoteddoolittleLU(A,B):
    if np.shape(A)[0] == np.shape(A)[1]:
        if (np.linalg.det(A)) != 0:
            size = np.shape(A)[0]
            R = np.zeros((size,size),float)
            for i in range(0,size):
                for j in range(0,size):
                    R[i,j] = (A[i,j])/np.amax(np.abs(A[i]))
            for y in range(0,size):
                f = np.amax((R[y:,y]))
                m = np.where(R[:,y]==f)[0][0]
                A[m],A[y] = A[y].copy(),A[m].copy()
                B[m],B[y] = B[y].copy(),B[m].copy()
                R[m],R[y] = R[y].copy(),R[m].copy()
            return doolittleLU(A,B)
        else:
            warnings.warn("Coefficient matrix is singular. This means there are no solutions or infinite number of solutions.")
    else:
        warnings.warn("Coefficient matrix is not square")
