import numpy as np
import EqSolve as Es
import warnings

def pwrEigenValue(B):
    '''Takes a square matrix of the type numpy.array() and returns its largest Eigenvalue.'''
    if np.shape(B)[0] == np.shape(B)[1]:
        size = np.shape(B)[0]
        w = np.zeros((size),float)
        for u in range(0,size):
            w[u] = 1
        A = B.copy()
        v = w.copy
        zi = np.matmul(B,w)
        modzi = np.linalg.norm(zi)
        vi = zi/modzi
        li = np.matmul(B,vi)
        modli = np.linalg.norm(li)
        vn = li/modli
        sign = 0
        for i in range(150):
            i+=1
            vi = vn
            z = np.matmul(B,vn)
            if(np.sign(np.dot(z,zi))==-1 and sign != 1):
                sign = 1
            modz = np.linalg.norm(z)
            vn = z/modz
            zi = z
        if (np.abs(vi) - np.abs(vn)).all() != 0:
            warnings.warn("Power Method did not converge")
        else:
            if sign == 1:
                return -np.linalg.norm(zi)
            elif sign == 0:
                return np.linalg.norm(zi)
    else:
        warnings.warn("Input matrix is not square")

def pwrEigenVector(B):
    '''Takes a square matrix of the type numpy.array() and returns the Eigen Vector corresponding to the largest Eigen Value.'''
    if np.shape(B)[0] == np.shape(B)[1]:
        size = np.shape(B)[0]
        w = np.zeros((size),float)
        for u in range(0,size):
            w[u] = 1
        A = B.copy()
        v = w.copy
        zi = np.matmul(B,w)
        modzi = np.linalg.norm(zi)
        vi = zi/modzi
        li = np.matmul(B,vi)
        modli = np.linalg.norm(li)
        vn = li/modli
        sign = 0
        for i in range(150):
            i+=1
            vi = vn
            z = np.matmul(B,vn)
            if(np.sign(np.dot(z,zi))==-1 and sign != 1):
                sign = 1
            modz = np.linalg.norm(z)
            vn = z/modz
            zi = z
        if (np.abs(vi) - np.abs(vn)).all() != 0:
            warnings.warn("Power Method did not converge")
        else:
            return(vn/vn[0])
    else:
        warnings.warn("Input matrix is not square")

def invpwrEigenValue(P):
    '''Takes a square matrix of the type numpy.array() and returns its smallest Eigenvalue.'''
    if np.shape(P)[0] == np.shape(P)[1]:
        size = np.shape(P)[0]
        Q = np.zeros((size,1),float)
        for u in range(0,size):
            Q[u,0] = 1
        B = Q.copy()
        A = P.copy()
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
        Yi = Es.PivotedGausselimination(L,B)
        Zi = Es.PivotedGausselimination(U,Yi)
        Vi = Zi/np.linalg.norm(Zi)
        Yn = Es.PivotedGausselimination(L,Vi)
        Zn = Es.PivotedGausselimination(U,Yn)
        Vn = Zn/np.linalg.norm(Zn)
        sign = 0
        for i in range(150):
            Zi = Zn
            Yi = Yn
            Vi = Vn
            Yn = Es.PivotedGausselimination(L,Vi)
            Zn = Es.PivotedGausselimination(U,Yn)
            if(np.sign(np.dot(Zn,Zi))==-1 and sign != 1):
                sign = 1
            Vn = Zn/np.linalg.norm(Zn)
        out = np.linalg.norm(Zn)
        if (np.abs(Vi) - np.abs(Vn)).all() != 0:
            warnings.warn("Inverse Power Method did not converge")
        else:
            if sign == 1:
                return -1/out
            elif sign == 0:
                return 1/out
    else:
        warnings.warn("Input matrix is not square")

def invpwrEigenVector(P):
    '''Takes a square matrix of the type numpy.array() and returns the Eigen Vector corresponding to the smallest Eigen Value.'''
    if np.shape(P)[0] == np.shape(P)[1]:
        size = np.shape(P)[0]
        Q = np.zeros((size,1),float)
        for u in range(0,size):
            Q[u,0] = 1
        B = Q.copy()
        A = P.copy()
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
        Yi = Es.PivotedGausselimination(L,B)
        Zi = Es.PivotedGausselimination(U,Yi)
        Vi = Zi/np.linalg.norm(Zi)
        Yn = Es.PivotedGausselimination(L,Vi)
        Zn = Es.PivotedGausselimination(U,Yn)
        Vn = Zn/np.linalg.norm(Zn)
        sign = 0
        for i in range(150):
            Zi = Zn
            Yi = Yn
            Vi = Vn
            Yn = Es.PivotedGausselimination(L,Vi)
            Zn = Es.PivotedGausselimination(U,Yn)
            if(np.sign(np.dot(Zn,Zi))==-1 and sign != 1):
                sign = 1
            Vn = Zn/np.linalg.norm(Zn)
        out = np.linalg.norm(Zn)
        if (np.abs(Vi) - np.abs(Vn)).all() != 0:
            warnings.warn("Inverse Power Method did not converge")
        else:
            return Vn/Vn[0]
    else:
        warnings.warn("Input matrix is not square")

def EigenVector(B,lamb):
    '''Takes square matrix and the EigenValue in the form (np.array(),EigenValue) and returns the EigenVector.'''
    if np.shape(B)[0] == np.shape(B)[1]:
        size = np.shape(B)[0]
        U = np.zeros((size,size),float)
        for i in range(0,size):
            U[i,i] = lamb
        A = B - U
        H = np.zeros((size-1,size-1),float)
        for i in range(0,size-1):
            for j in range(1,size):
                H[i,j-1] = A[i,j]
        I = np.zeros((size-1),float)
        for k in range(0,size-1):
            I[k] = -A[k,0]
        out = Es.GaussElimination(H,I)
        actout = np.zeros((size),float)
        for i in range(0,size-1):
            actout[i+1] = out[i]
        actout[0] = 1
        return actout
    else:
        warnings.warn("Input matrix is not square")

def pwrEig(B):
    '''Takes a square matrix of the form numpy.array() and returns the largest Eigen Value and the corresponding Eigen Vector in the form (EigenValue,np.array())'''
    if np.shape(B)[0] == np.shape(B)[1]:
        size = np.shape(B)[0]
        w = np.zeros((size),float)
        for u in range(0,size):
            w[u] = 1
        A = B.copy()
        v = w.copy
        zi = np.matmul(B,w)
        modzi = np.linalg.norm(zi)
        vi = zi/modzi
        li = np.matmul(B,vi)
        modli = np.linalg.norm(li)
        vn = li/modli
        sign = 0
        for i in range(150):
            i+=1
            vi = vn
            z = np.matmul(B,vn)
            if(np.sign(np.dot(z,zi))==-1 and sign != 1):
                sign = 1
            modz = np.linalg.norm(z)
            vn = z/modz
            zi = z
        val = 0
        vnnorm = vn/vn[0]
        if (np.abs(vi) - np.abs(vn)).all() != 0:
            warnings.warn("Power Method did not converge")
        else:
            if sign == 1:
                val = -np.linalg.norm(zi)
            elif sign == 0:
                val = np.linalg.norm(zi)
        return val, vnnorm
    else:
        warnings.warn("Input matrix is not square")

def invpwrEig(P):
    '''Takes a square matrix of the form numpy.array() and returns the smallest Eigen Value and the corresponding Eigen Vector in the form (EigenValue,np.array())'''
    if np.shape(P)[0] == np.shape(P)[1]:
        size = np.shape(P)[0]
        Q = np.zeros((size,1),float)
        for u in range(0,size):
            Q[u,0] = 1
        B = Q.copy()
        A = P.copy()
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
        Yi = Es.PivotedGausselimination(L,B)
        Zi = Es.PivotedGausselimination(U,Yi)
        Vi = Zi/np.linalg.norm(Zi)
        Yn = Es.PivotedGausselimination(L,Vi)
        Zn = Es.PivotedGausselimination(U,Yn)
        Vn = Zn/np.linalg.norm(Zn)
        sign = 0
        for i in range(150):
            Zi = Zn
            Yi = Yn
            Vi = Vn
            Yn = Es.PivotedGausselimination(L,Vi)
            Zn = Es.PivotedGausselimination(U,Yn)
            if(np.sign(np.dot(Zn,Zi))==-1 and sign != 1):
                sign = 1
            Vn = Zn/np.linalg.norm(Zn)
        out = np.linalg.norm(Zn)
        val = 0
        if (np.abs(Vi) - np.abs(Vn)).all() != 0:
            warnings.warn("Inverse Power Method did not converge")
        else:
            if sign == 1:
                val = -1/out
            elif sign == 0:
                val = 1/out
        return val,Vn/Vn[0]
    else:
        warnings.warn("Input matrix is not square")

def jacobi(W,tol = 1e-50):
    '''Take a sqaure matrix of the type np.array and returns ALL eigenvalues and eigenvectors in the form array[Eigenvalues],array[Eigenvectors]'''
    A = W.copy()
    if np.shape(A)[0] == np.shape(A)[1]:
        n = np.shape(A)[0]
        P = np.identity(n,float)
        for k in range(2000):
            #Setting the threshold:
            S = 0
            s=0
            for i in range(0,n-1):
                for j in range(i+1,n):
                    S+=np.abs(A[i,j])
            mu = (0.5*S)/(n*(n-1))
            for f in range(0,n-1):
                for g in range(f+1,n):
                    if np.abs(A[f,g]) >= mu:
                        #Calculating phi,t,c,s,tau:
                        diff = A[f,f] - A[g,g]
                        phi = -diff/(2*A[f,g])
                        if phi==0:
                            t = 1
                        else:
                            t = np.sign(phi)/(np.abs(phi) + (phi**2 + 1)**0.5)
                        c = 1/(1 + t**2)**0.5
                        s = t*c
                        tau = s/(1+c)
                        #Performing the rotation:
                        A[f,f] -= t*A[f,g]
                        A[g,g] += t*A[f,g]
                        A[f,g] = A[g,f] = 0
                        for i in range(0,n):
                            temp = P[i,f]
                            P[i,f] = temp - s*(P[i,g] + tau*P[i,f])
                            P[i,g] = P[i,g] + s*(temp - tau*P[i,g])
                            if i!=f and i!=g:
                                temp1 = A[f,i]
                                A[f,i] = A[i,f] = temp1 - s*(A[g,i] + temp1)
                                A[g,i] = A[i,g] = A[g,i] + s*(temp1 - tau*A[g,i])
                            else:
                                continue
            if mu<=tol:
                break;
        #Collecting Eigenvalues:
        Eigenvals = np.zeros(n,float)
        return np.diagonal(A),P
    else:
        warnings.warn("Input matrix is not square")
