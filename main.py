import pprint
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import numpy as np

def dd(X):
    k = np.diag(np.abs(X)) # Find diagonal coefficients
    S = np.sum(np.abs(X), axis=1) - k # Find row sum without diagonal
    if np.all(k > S):
        print ('matrix is diagonally dominant')
        return True
    else:
        print ('NOT diagonally dominant')
        return False

def jacob(matrix, vector, tolerence, max_iter):
    D = np.array([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ])
    L = np.array([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ])
    U = np.array([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ])

    dominant = dd(matrix)
    if(dominant==False):
        return
    else:
        for i in range(0,3):
            for j in range(0,3):
                if(i == j):
                    D[i,j] = matrix[i,j]

        for i in range(0,3):
            for j in range(0,3):
                if(j > i):
                    U[i,j] = matrix[i,j]

        for i in range(0,3):
            for j in range(0,3):
                if(i > j):
                    L[i,j] = matrix[i,j]

        Dnew = np.linalg.inv(D)
        DnewM = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for i in range(0, 3):
            for j in range(0, 3):
                DnewM[i, j] = Dnew[i, j] * (-1)

        LUnew = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for i in range(0, 3):
            for j in range(0, 3):
                LUnew[i, j] = L[i, j] + U[i, j]

        iteration = 1
        x = np.array([0,0,0])
        for count in range(max_iter):
            print(x)
            newX = Dnew.dot(vector - (LUnew.dot(x)))
            if np.allclose(x,newX,rtol=tolerence,atol=0):
                break
            x = newX
            iteration = iteration + 1

        print("Final approximation:")
        print(x)
        print("-------------------------")

def gauss_zaidel(matrix, vector, tolerence, max_iter):
    D = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    L = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    U = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    dominant = dd(matrix)
    if (dominant == False):
        return
    else:
        for i in range(0, 3):
            for j in range(0, 3):
                if (i == j):
                    D[i, j] = matrix[i, j]

        for i in range(0, 3):
            for j in range(0, 3):
                if (j > i):
                    U[i, j] = matrix[i, j]

        for i in range(0, 3):
            for j in range(0, 3):
                if (i > j):
                    L[i, j] = matrix[i, j]

        LDnew = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for i in range(0, 3):
            for j in range(0, 3):
                LDnew[i, j] = L[i, j] + D[i, j]

        LDinv = np.linalg.inv(LDnew)

        iteration = 1
        x = np.array([0, 0, 0])
        for count in range(max_iter):
            print(x)
            newX = LDinv.dot(vector - (U.dot(x)))
            if np.allclose(x, newX, rtol=tolerence, atol=0):
                break
            x = newX
            iteration = iteration + 1

        print("Final approximation:")
        print(x)
        print("-------------------------")


A = np.array([[4,2,0],
              [2,10,4],
              [0,4,5]])

b = np.array([2,6,5])

jacob(A,b,0.001,200)
gauss_zaidel(A, b, 0.001, 200)









