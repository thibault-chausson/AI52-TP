import numpy as np
import matplotlib.pyplot as plt
import pandas as pnd


def solveDG(A, B, Z, alpha, precision, nbmax):
    arret = 0
    i = 0
    res = B - np.dot(A, Z)  # résidu
    while ((np.sqrt(np.dot(res.T, res)) > precision) & (i < nbmax)):
        Grad = np.dot(-A.T, res)  # Gradient
        d = -Grad  # direction
        Z = Z + alpha * d  # nouvelle valeur
        res = B - np.dot(A, Z)
        i = i + 1
    return (Z, i)


def solveDGBB(A, B, Z, alpha, precision, nbmax):  # Méthode de Barzilai-Borwein
    i = 0
    res = B - np.dot(A, Z)  # résidu
    if ((np.sqrt(np.dot(res.T, res)) < precision)):
        return Z
    Grad = np.dot(-A.T, res)  # Gradient
    d = -Grad  # direction

    while ((np.sqrt(np.dot(res.T, res)) > precision) & (i < nbmax)):
        nouvZ = Z + alpha * d  # nouvelle solution
        nouvres = B - np.dot(A, nouvZ)  # nouveau résidu
        nouvGrad = np.dot(-A.T, nouvres)  # nouveau gradient
        Dz = alpha * d
        DGrad = nouvGrad - Grad
        alpha = (np.dot(Dz.T, DGrad)) / np.dot(DGrad.T, DGrad)  # optimization de Alpha
        Z = nouvZ
        res = nouvres  # direction
        Grad = nouvGrad
        d = -Grad
        i = i + 1
    return (Z, i)


def solveDGC(A, B, Z, precision, nbmax):
    i = 0
    r = B - np.dot(A, Z)
    d = r
    res_V = np.dot(r.T, r)
    while ((np.sqrt(res_V) > precision) & (i < nbmax)):
        alpha = (np.dot(r.T, r)) / (np.dot(d.T, np.dot(A, d)))
        Z = Z + alpha * d
        r = B - np.dot(A, Z)
        res_N = np.dot(r.T, r)
        beta = res_N / res_V
        d = r + beta * d
        res_V = res_N
        i = i + 1
    return (Z, i)


n = (int)(input("Donner la valeur de n :"))
A = np.random.uniform(-10, 10, (n, n))
X_Sol = np.random.uniform(-10, 10, (n, 1))
b = np.dot(A, X_Sol)
Init = np.zeros(np.shape(b))
print("\n Voici la solution initiale *************************\n", Init)
print("\n Nous souhaitons résoudre A*X=B")
print("\n Avec A= \n", A)
print("\n et   B= \n", b)
print("\n Voici la solution analytique \n", X_Sol)
Z, k = solveDG(A, b, np.zeros(np.shape(b)), 0.0005, 0.01,
               100000)  # methode de descente du gradient, 0.0005 est le Alpha, 0.01 la précision et 100000 nombre d'étapes maximal
print("\n Voici ce que donne l algorithme \n")
print(Z)
print("Apres", k, " étapes \n")
Z, k = solveDGBB(A, b, np.zeros(np.shape(b)), 3., 0.01,
                 100000)  # methode de Barzilai-Borwein, 3. est le Alpha, 0.01 la précision et 100000 nombre d'étapes maximal
print("\n Voici ce que donne l algorithme avec alpha amélioré \n")
print(Z)
print("Apres", k, " étapes \n")
print(k)
Z, k = solveDGC(np.dot(A.T, A), np.dot(A.T, b), np.zeros(np.shape(b)), 0.01,
                10000)  ##methode de gradient conjugué, A.T*A remplace A pour avoir une matrice symétrique, A.T*b remplace le b, 0.01 la précision et 100000 nombre d'étapes maximal
print("\n Voici ce que donne l algorithme du gradient conjugué  \n")
print(Z)
print("Apres", k, " étapes \n")
print(k)
