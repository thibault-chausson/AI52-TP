# Code pour exécuter la méthode du gradien conjugué

import numpy as np
import math
import matplotlib.pyplot as plt


# Fonction pour déterminer le gradient de la fonction

def gradient(A, B, X):
    return np.dot(A, X) - B


# Initialiser D_0

def calcul_D_0(A, B, X):
    return -gradient(A, B, X)


# Calcul de alpha_K

def calcul_alpha(A, B, X, D):
    return - np.dot(D.T, gradient(A, B, X)) / np.dot(np.dot(D.T, A), D)
    # return np.dot(gradient(A, B, X).T, gradient(A, B, X)) / np.dot(np.dot(D.T, A), D)


# Calcul de beta_K

def calcul_beta(A, B, X, D):
    return np.dot(np.dot(gradient(A, B, X).T, A), D) / np.dot(np.dot(D.T, A), D)


# Calcul de D_K+1

def calcul_D(D, beta, gradient):
    return -gradient + beta * D


# Calcul de X_K+1

def calcul_X(X, alpha, D):
    return X + alpha * D


# Critère d'arrêt

def critere(gradien, nb_iter, taille_espace):
    norme_gradien = math.sqrt(np.dot(gradien.T, gradien))
    return norme_gradien == 0 or nb_iter == taille_espace


# Méthode du gradien conjugué

def methode_gradient(A, B, X_0):
    if A.shape[0] != A.shape[1]:
        print("La matrice A n'est pas carrée")
        return
    else:
        taille_espace = A.shape[0]
        nb_iter = 0

        D_0 = calcul_D_0(A, B, X_0)

        while not critere(gradient(A, B, X_0), nb_iter, taille_espace):
            alpha_K = calcul_alpha(A, B, X_0, D_0)
            X_0 = calcul_X(X_0, alpha_K, D_0)
            beta_K = calcul_beta(A, B, X_0, D_0)
            D_0 = calcul_D(D_0, beta_K, gradient(A, B, X_0))
            nb_iter += 1

        return X_0


# A = np.array([[6, -2], [-2, 2]])
# B = np.array([[0], [-8]])

# A = np.array([[2, 3], [1, 1]])
# B = np.array([[17], [6]])


# A = np.array([[2, -1], [-1, 2]])
# B = np.array([[0], [-3]])
X_0 = np.array([[0], [0]])

print("L'extremum est le point : \n", methode_gradient(A, B, X_0))
