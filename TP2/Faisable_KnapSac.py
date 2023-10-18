# Résolution du problème du sac à dos ou KP (Knapsack Problem) à l'aide d'algorithme génétique
# import de la librairie

import numpy as np  # utilisation des calculs matriciels
import random as rd  # génération de nombres aléatoires
from random import randint  # génération des nombres aléatoires
import matplotlib.pyplot as plt
import random


def fct_population_initiale(solutions_par_pop, nombre_objets):
    ID_objets = np.arange(0, nombre_objets)  # ID des objets à mettre dans le sac de 1 à 10
    # Créer la population initiale
    pop_size = (solutions_par_pop, ID_objets.shape[0])
    population_initiale = np.random.randint(2, size=pop_size)
    population_initiale = population_initiale.astype(int)

    return ID_objets, population_initiale, pop_size


def correction_initiale(pop, poids, capacite, choix):
    for i in range(len(pop)):
        pop[i] = correction(pop[i], poids, capacite, choix)
    return pop


'''
Pour vérifier si une solution est faisable, on vérifie si la somme des poids des objets sélectionnés est inférieure à la capacité du sac.
Si ce n'est pas viable on corrige la solution
'''


def correction(enfant, poids, capacite, choix):
    if choix == "0":  # on corrige
        P = np.sum(enfant * poids)
        if P < capacite:
            return enfant
        else:
            # Générer une liste d'indices
            indices = list(range(len(enfant)))
            # Mélanger les indices
            random.shuffle(indices)
            compte_capacite = 0
            new_enfant = np.zeros(len(enfant))
            # On parcourt tous les indices
            for i in range(len(indices)):
                # Si la capacité n'est pas dépassée, on ajoute l'objet dans le tableau de 0
                if enfant[indices[i]] == 1 and capacite > compte_capacite + poids[indices[i]]:
                    new_enfant[indices[i]] = 1
                    compte_capacite += poids[indices[i]]
            return new_enfant
    else:  # On corrige pas ca fitness neg
        return enfant


def cal_fitness_0(poids, valeur, population, capacite):
    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * valeur)
        S2 = np.sum(population[i] * poids)

        if S2 <= capacite:
            fitness[i] = S1
        else:
            fitness[i] = 0

    return fitness.astype(int)


def cal_fitness_neg(poids, valeur, population, capacite):
    fitness = np.empty(population.shape[0])

    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * valeur)
        S2 = np.sum(population[i] * poids)

        if S2 <= capacite:
            fitness[i] = S1
        else:
            # Proportionnelle (négative) au dépassement de la capacité
            depassement = S2 / capacite
            fitness[i] = - (depassement * S1)

    return fitness.astype(int)


def cal_fitness(poids, valeur, population, capacite, choix):
    if choix == "0":
        return cal_fitness_0(poids, valeur, population, capacite)
    else:
        return cal_fitness_neg(poids, valeur, population, capacite)


def selection(fitness, nbr_parents, population):
    fitness = list(fitness)
    parents = np.empty((nbr_parents, population.shape[1]))

    for i in range(nbr_parents):
        indice_max_fitness = np.where(fitness == np.max(fitness))
        parents[i, :] = population[indice_max_fitness[0][0], :]
        fitness[indice_max_fitness[0][0]] = -999999

    return parents


def croisement(parents, nbr_enfants, poids, capacite, choix):
    enfants = np.empty((nbr_enfants, parents.shape[1]))
    point_de_croisement = int(parents.shape[1] / 2)  # croisement au milieu
    taux_de_croisement = 0.8
    i = 0

    while (i < nbr_enfants):  # parents.shape[0]
        indice_parent1 = i % parents.shape[0]
        indice_parent2 = (i + 1) % parents.shape[0]
        x = rd.random()
        if x > taux_de_croisement:  # probabilité de parents stériles
            continue
        indice_parent1 = i % parents.shape[0]
        indice_parent2 = (i + 1) % parents.shape[0]
        enfants[i, 0:point_de_croisement] = parents[indice_parent1, 0:point_de_croisement]
        enfants[i, point_de_croisement:] = parents[indice_parent2, point_de_croisement:]
        # On corrige l'enfant
        enfants[i] = correction(enfants[i], poids, capacite, choix)
        i += 1

    return enfants


# La mutation consiste à inverser le bit
def mutation(enfants, poids, capacite, choix):
    mutants = np.empty((enfants.shape))
    taux_mutation = 0.5
    for i in range(mutants.shape[0]):
        random_valeur = rd.random()
        mutants[i, :] = enfants[i, :]
        if random_valeur > taux_mutation:
            continue
        int_random_valeur = randint(0, enfants.shape[1] - 1)  # choisir aléatoirement le bit à inverser
        if mutants[i, int_random_valeur] == 0:
            mutants[i, int_random_valeur] = 1
        else:
            mutants[i, int_random_valeur] = 0
        correction(mutants[i], poids, capacite, choix)
    return mutants


def optimize(poids, valeur, population, pop_size, nbr_generations, capacite, choix):
    sol_opt, historique_fitness = [], []
    nbr_parents = pop_size[0] // 2
    nbr_enfants = pop_size[0] - nbr_parents
    for _ in range(nbr_generations):
        fitness = cal_fitness(poids, valeur, population, capacite, choix)
        historique_fitness.append(fitness)
        parents = selection(fitness, nbr_parents, population)
        enfants = croisement(parents, nbr_enfants, poids, capacite, choix)
        mutants = mutation(enfants, poids, capacite, choix)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print(f'Voici la dernière génération de la population: \n{population}\n')
    fitness_derniere_generation = cal_fitness(poids, valeur, population, capacite, choix)
    print(f'Fitness de la dernière génération: \n{fitness_derniere_generation}\n')
    max_fitness = np.where(fitness_derniere_generation == np.max(fitness_derniere_generation))
    sol_opt.append(population[max_fitness[0][0], :])

    return sol_opt, historique_fitness


def affichage(nbr_generations, capacite_max, poids, valeur, ID_objets, population_initiale, pop_size, choix):
    if choix == "0":  # On corrige
        print("On corrige")
        population_initiale = correction_initiale(population_initiale, poids, capacite_max, choix)

    # affichage des objets: Une instance aléatoire du problème Knapsack
    print('La liste des objet est la suivante :')
    print('ID_objet   Poids   Valeur')
    for i in range(ID_objets.shape[0]):
        print(f'{ID_objets[i]} \t {poids[i]} \t {valeur[i]}')
    print()

    # lancement de l'algorithme génétique
    sol_opt, historique_fitness = optimize(poids, valeur, population_initiale, pop_size, nbr_generations, capacite_max,
                                           choix)

    # affichage du résultat
    print('La solution optimale est:')
    print(sol_opt)
    print('objets n°', [i for i, j in enumerate(sol_opt[0]) if j != 0])

    print(np.asarray(historique_fitness).shape)
    print(f'Avec une valeur de {np.amax(historique_fitness)} € et un poids de {np.sum(sol_opt * poids)} kg')
    print('Les objets qui maximisent la valeur contenue dans le sac sans le dÃ©chirer :')
    objets_selectionnes = ID_objets * sol_opt
    for i in range(objets_selectionnes.shape[1]):
        if ((sol_opt[0][i] == 1)):
            print(f'{objets_selectionnes[0][i]}')

    historique_fitness_moyenne = [np.mean(fitness) for fitness in historique_fitness]
    historique_fitness_max = [np.max(fitness) for fitness in historique_fitness]
    plt.plot(list(range(nbr_generations)), historique_fitness_moyenne, label='Valeurs moyennes')
    plt.plot(list(range(nbr_generations)), historique_fitness_max, label='Valeur maximale')
    plt.legend()
    plt.title('Evolution de la Fitness à travers les générations en Euros')
    plt.xlabel('Générations')
    plt.ylabel('Fitness')
    plt.show()
