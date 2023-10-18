import numpy as np
import math
import matplotlib.pyplot as plt


def solve(A, B, Z, alpha):
    i = 0
    Y = np.dot(A, Z) - B

    while math.sqrt(np.dot(Y.T, Y)) > 0.0001:
        Z = Z - alpha * np.dot(A, Y)
        Y = np.dot(A, Z) - B
        i = i + 1
    Z = Z - alpha * np.dot(A, Y)
    return (Z, i)


# Gradien conjugué

def solveGC(A, B, Z):
    i = 0
    r = B - np.dot(A, Z)
    H = B - np.dot(A, Z)
    P = r
    res_V = np.dot(r.T, r)
    print("res_V", res_V)
    while (math.sqrt(np.dot(H.T, H)) > 0.001):
        alpha = (res_V) / (np.dot(P.T, np.dot(A, P)))
        Z = Z + alpha * P
        r = r - alpha * np.dot(A, P)
        res_N = np.dot(r.T, r)
        print("Z", Z)
        if (math.sqrt(res_N) < 1e-10):
            break
        beta = res_N / res_V
        P = r + beta * P
        res_V = res_N
        i = i + 1
        H = B - np.dot(A, Z)
    return (Z, i)


A = np.array([[2, 3], [1, 1]])
b = np.array([[17], [6]])

print("res GC", solveGC(A, b, np.array([[0], [0]])))

# A = np.array([[2, 3], [1, 1]])
X_Sol = np.array([[1], [5]])
A_Inv = np.linalg.inv(A)
# b = np.dot(A, X_Sol)
print("B", b)
print("\n Voici la solution recherchée \n")
print(X_Sol)
Z, k = solve(A, b, np.array([[0], [0]]), 0.1)
print("\n Voici ce que donne l algorithme \n")
print(Z)
print("Apres", k, " étapes \n")
print(k)
