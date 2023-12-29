# -*- coding: utf-8 -*-

# Résolution du problème du voyageur de commerce ou TCS à l'aide du recuit simulé
# import de la librairie
import random
import numpy as np

NOMBRE_DE_VILLES = 10
MAX_DISTANCE = 2000
TEMPERATURE = 1000
FACTEUR = 0.99
TOUR = 1000


def generer_ville(nombre_de_ville, max_distance):
    # Données du problème (générées aléatoirement)
    distances = np.zeros((nombre_de_ville, nombre_de_ville))
    for ville in range(nombre_de_ville):
        villes = [i for i in range(nombre_de_ville) if not i == ville]
        for vers_la_ville in villes:
            distances[ville][vers_la_ville] = random.randint(50, max_distance)
            distances[vers_la_ville][ville] = distances[ville][vers_la_ville]
    # print('voici la matrice des distances entres les villes \n', distances)
    return distances


def cal_distance(solution, distances, nombre_de_ville):
    eval_distance = 0
    for i in range(len(solution)):
        origine, destination = solution[i], solution[(i + 1) % nombre_de_ville]
        eval_distance += distances[origine][destination]
    return eval_distance


def voisinage(solution, nombre_de_ville):
    echange = random.sample(range(nombre_de_ville), 2)
    sol_voisine = solution
    (sol_voisine[echange[0]], sol_voisine[echange[1]]) = (sol_voisine[echange[1]], sol_voisine[echange[0]])
    return sol_voisine


def recuit_simule(nombre_de_ville, distances, T, facteur, tour):
    # recuit simulé
    # distances = generer_ville(nombre_de_ville, max_distance)
    solution = random.sample(range(nombre_de_ville), nombre_de_ville)
    cout0 = cal_distance(solution, distances, nombre_de_ville)
    # T_intiale = max_distance / 2
    min_sol = solution
    cout_min_sol = cout0
    for i in range(tour):
        # print('la ', i, 'ème solution = ', solution, ' donne la distance totale= ', cout0, ' la température actuelle =', T)
        T = T * facteur
        for j in range(50):
            nouv_sol = voisinage(solution * 1, nombre_de_ville)
            cout1 = cal_distance(nouv_sol, distances, nombre_de_ville)
            #  print('la ',j,'ème recherche de voisinage de',solution,'donne la solution=' ,nouv_sol,' distance totale= ',cout1)
            if cout1 < cout0:
                cout0 = cout1
                solution = nouv_sol
                if cout1 < cout_min_sol:
                    cout_min_sol = cout1
                    min_sol = solution
            else:
                x = np.random.uniform()
                if x < np.exp((cout0 - cout1) / T):
                    cout0 = cout1
                    solution = nouv_sol

    distance_totale = cal_distance(min_sol, distances, nombre_de_ville)
    # print('voici la solution retenue ', min_sol, ' et son coût ', distance_totale)
    return min_sol, distance_totale


# print(recuit_simule(NOMBRE_DE_VILLES, generer_ville(NOMBRE_DE_VILLES, MAX_DISTANCE), TEMPERATURE, FACTEUR, TOUR))