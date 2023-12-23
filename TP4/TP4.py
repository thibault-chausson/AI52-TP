#!/usr/bin/env python
# coding: utf-8

# # Algorithme classique du problème du sac à dos

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time as t
import random


# # Les 5 jeux de données

# # Jeu de données 1

# In[2]:


weights_1 = np.array([10, 20, 30, 40, 50])
values_1 = np.array([60, 100, 120, 140, 160])
max_weight_1 = np.sum(weights_1) // 3


# ## Jeu de données 2

# In[3]:


weights_2 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
values_2 = np.array([60, 100, 120, 140, 160, 180, 200, 220, 240, 260])
max_weight_2 = np.sum(weights_2) // 3


# ## Jeu de données 3

# In[4]:


weights_3 = [83, 73, 22, 1, 65, 98, 64, 40, 92, 68, 6, 39, 90, 73, 7, 99, 6, 52, 23, 14]
values_3 = [8, 17, 2, 17, 20, 16, 15, 2, 19, 10, 19, 6, 19, 5, 16, 21, 19, 18, 9, 1]
max_weight_3 = np.sum(weights_3) // 3


# ## Jeu de données 4

# In[5]:


weights_4 = [83, 73, 22, 1, 65, 98, 64, 40, 92, 68, 6, 39, 90, 73, 7, 99, 6, 52, 23, 14, 83, 73, 22, 1, 65, 98, 64, 40,
             92, 68, 6, 39, 90, 73, 7, 99, 6, 52, 23, 14]
values_4 = [8, 17, 2, 17, 20, 16, 15, 2, 19, 10, 19, 6, 19, 5, 16, 21, 19, 18, 9, 1, 8, 17, 2, 17, 20, 16, 15, 2, 19,
            10, 19, 6, 19, 5, 16, 21, 19, 18, 9, 1]
max_weight_4 = np.sum(weights_4) // 3


# ## Jeu de données 5

# In[6]:


weights_5 = [104, 390, 95, 276, 357, 393, 345, 148, 170, 174, 316, 185, 163, 22, 406, 360, 13, 14, 205, 136, 305, 75,
             395, 262, 267, 379, 114, 171, 267, 383, 347, 308, 399, 382, 231, 207, 354, 395, 382, 174]
values_5 = [25, 7, 6, 24, 19, 6, 19, 20, 21, 14, 25, 5, 24, 6, 25, 5, 25, 25, 25, 25, 25, 25, 25, 25, 6, 25, 25, 25, 25,
            25, 25, 25, 25, 25, 6, 25, 25, 25, 25, 25]
max_weight_5 = np.sum(weights_5) // 3


# # Classique

# In[7]:


weights = np.array([10, 20, 30, 40, 50])
values = np.array([60, 100, 120, 140, 160])
max_weight = np.sum(weights) // 3

population_size = 100

population = np.random.randint(2, size=(population_size, weights.shape[0]))
print(population)


# In[8]:


def correction_un(arg_population, arg_max_weight, arg_weights):
    # On corrige un individu
    S2 = np.sum(arg_population * arg_weights)
    if S2 > arg_max_weight:
        # On cherche l'indice du premier 1
        indice = np.where(arg_population == 1)[0][0]
        # On le remplace par un 0
        arg_population[indice] = 0
    return arg_population


def correction(arg_population, arg_max_weight, arg_weights):
    # On corrige la population
    for i in range(arg_population.shape[0]):
        arg_population[i] = correction_un(arg_population[i], arg_max_weight, arg_weights)
    return arg_population


# In[9]:


population = correction(population, max_weight, weights)
print(population)


# In[10]:


def fitness_un(arg_weight, arg_value, arg_population, arg_max_weight):
    # On calcule la fitness d'un individu
    S1 = np.sum(arg_population * arg_value)
    S2 = np.sum(arg_population * arg_weight)
    if S2 > arg_max_weight:
        return 0
    else:
        return S1


def fitness(arg_weight, arg_value, arg_population, arg_max_weight):
    arg_fitness = np.empty(arg_population.shape[0])
    for i in range(arg_population.shape[0]):
        arg_fitness[i] = fitness_un(arg_weight, arg_value, arg_population[i], arg_max_weight)
    return arg_fitness


# In[11]:


fitness(weights, values, population, max_weight)


# In[12]:


def selection_roulette(arg_fitness, arg_population):
    # Selectionner par la roulette les 2 meilleurs individus, ils doivent être différents
    parents = []
    fitness_aux = arg_fitness
    for i in range(2):
        somme = np.sum(fitness_aux)
        fitness_roulette = fitness_aux / somme
        # Tirage aléatoire entre 0 et 1
        tirage = np.random.rand()
        # On cherche l'indice de l'individu qui correspond au tirage et on le stocke dans parents puis on le supprime de la population
        somme = 0
        for j in range(fitness_roulette.shape[0]):
            somme += fitness_roulette[j]
            if somme >= tirage:
                parents.append(arg_population[j])
                arg_population = np.delete(arg_population, j, axis=0)
                arg_fitness = np.delete(arg_fitness, j, axis=0)
                fitness_aux = np.delete(fitness_roulette, j, axis=0)
                break
    return parents


# In[13]:


print(selection_roulette(fitness(weights, values, population, max_weight), population))


# In[14]:


def crossover(parents, arg_max_weight, arg_weights):
    # On choisit un point de croisement aléatoire
    point_croisement = np.random.randint(1, parents[0].shape[0] - 1)
    # On crée les enfants
    enfant1 = np.concatenate((parents[0][:point_croisement], parents[1][point_croisement:]))
    enfant2 = np.concatenate((parents[1][:point_croisement], parents[0][point_croisement:]))
    return correction_un(enfant1, arg_max_weight, arg_weights), correction_un(enfant2, arg_max_weight, arg_weights)


# In[15]:


print(crossover(selection_roulette(fitness(weights, values, population, max_weight), population), max_weight, weights))


# In[16]:


def mutation(enfants, arg_max_weight, arg_weights):
    # On choisit un point de mutation aléatoire
    point_mutation = np.random.randint(0, enfants[0].shape[0])
    # On crée les enfants si le point de mutation est 0 on le change en 1 et inversement
    if enfants[0][point_mutation] == 0:
        enfants[0][point_mutation] = 1
    else:
        enfants[0][point_mutation] = 0
    return correction_un(enfants[0], arg_max_weight, arg_weights), correction_un(enfants[1], arg_max_weight,
                                                                                 arg_weights)


# In[17]:


print(mutation(
    crossover(selection_roulette(fitness(weights, values, population, max_weight), population), max_weight, weights),
    max_weight, weights))


# In[18]:


def evolution(arg_population, arg_fitness, arg_max_weight, arg_weights, arg_values, arg_mutation_rate=0.1):
    # On sélectionne les parents
    parents = selection_roulette(arg_fitness, arg_population)
    # On crée les enfants
    enfants = crossover(parents, arg_max_weight, arg_weights)
    # On mute les enfants avec une probabilité
    if np.random.rand() < arg_mutation_rate:
        enfants = mutation(enfants, arg_max_weight, arg_weights)
    # On ajoute les enfants à la population si ils améliorent le pire individu
    enfants = np.array(enfants)
    fitness_enfants = fitness(arg_weights, arg_values, enfants, arg_max_weight)
    for i in range(2):
        if fitness_enfants[i] > np.min(arg_fitness):
            # On remplace le pire individu par l'enfant
            arg_population[np.argmin(arg_fitness)] = enfants[i]
            arg_fitness[np.argmin(arg_fitness)] = fitness_enfants[i]
    # On calcule la fitness de la nouvelle population
    return arg_population


# In[19]:


print(evolution(population, fitness(weights, values, population, max_weight), max_weight, values, weights))


# In[20]:


def algo_genetique(arg_population, arg_weights, arg_values, arg_max_weight, nb_iterations=1000, seuil_amelioration=0.01,
                   patience=100):
    historique_fitness = []  # Stocker l'historique des fitness maximales
    compteur_sans_amelioration = 0  # Compteur pour les itérations sans amélioration significative

    for _ in range(nb_iterations):
        arg_fitness = fitness(arg_weights, arg_values, arg_population, arg_max_weight)
        fitness_max = np.max(arg_fitness)

        # Mise à jour de l'historique et vérification du critère d'arrêt
        if historique_fitness:
            if fitness_max - historique_fitness[-1] < seuil_amelioration:
                compteur_sans_amelioration += 1
            else:
                compteur_sans_amelioration = 0

        historique_fitness.append(fitness_max)

        if compteur_sans_amelioration >= patience:
            print(f"Arrêt après {_} itérations, car il n'y a pas eu d'amélioration significative.")
            break

        arg_population = evolution(arg_population, arg_fitness, arg_max_weight, arg_weights, arg_values)

    # On récupère le meilleur individu après le dernier cycle
    arg_fitness = fitness(arg_weights, arg_values, arg_population, arg_max_weight)
    return arg_population[np.argmax(arg_fitness)], np.max(arg_fitness), historique_fitness


# In[21]:


algo_genetique(population, weights, values, max_weight)


# # Quantique 

# In[22]:


weights_q = np.array([10, 20])
values_q = np.array([60, 100])
max_weight_q = np.sum(weights_q) // 3


# In[23]:


import numpy as np


def initialiser_q(n, m):
    un_ele = np.full(m, 1 / (2 ** 0.5))
    return np.array([[un_ele, un_ele] for _ in range(n)])


def initialiser_p(n, m, Q):
    fct_r = np.random.rand(n, m)
    return (fct_r >= (Q[:, 0] ** 2)).astype(int)


def evaluer_fitness(P, arg_weights, arg_values, arg_max_weight):
    S1 = np.dot(P, arg_values)
    S2 = np.dot(P, arg_weights)
    return np.where(S2 > arg_max_weight, 0, S1)


def recuperer_meilleur(P, arg_fitness):
    idx_max = np.argmax(arg_fitness)
    return P[idx_max], arg_fitness[idx_max]


def def_gamma(fitness_indi, fitness_max, bit_indi, bit_max):
    conditions = [(bit_indi == 0) & (bit_max == 0),
                  (bit_indi == 0) & (bit_max == 1) & (fitness_indi > fitness_max),
                  (bit_indi == 1) & (bit_max == 0) & (fitness_indi > fitness_max),
                  (bit_indi == 1) & (bit_max == 1) & (fitness_indi > fitness_max)]
    choices = [0, 0.05 * np.pi, 0.025 * np.pi, 0.025 * np.pi]
    return np.select(conditions, choices, default=0.01 * np.pi)


def calcule_alpha_beta(alpha, beta, gamma):
    alpha_prime = alpha * np.cos(gamma) - beta * np.sin(gamma)
    beta_prime = alpha * np.sin(gamma) + beta * np.cos(gamma)
    return alpha_prime, beta_prime


def update(P, Q, B, fct_weights, fct_values, fct_max_weight):
    best_sol, _ = get_best(B)
    fitness = evaluer_fitness(P, fct_weights, fct_values, fct_max_weight)
    for i in range(len(Q)):
        for j in range(len(Q[i][0])):
            alpha, beta = Q[i, 0, j], Q[i, 1, j]
            gamma_fct = def_gamma(fitness[i], get_best(B)[1], P[i, j], best_sol[j])
            alpha_prime, beta_prime = calcule_alpha_beta(alpha, beta, gamma_fct)
            Q[i, 0, j], Q[i, 1, j] = alpha_prime, beta_prime
    return Q


def get_best(B):
    return max(B, key=lambda x: x[1])


def creer_p(Q):
    r = np.random.rand(Q.shape[0], Q.shape[2])
    return (r < (Q[:, 0, :] ** 2)).astype(int)


def algorithme_quantique(n, m, arg_weights, arg_values, arg_max_weight, nb_iterations, seuil_amelioration=0.01,
                         patience=10):
    Q = initialiser_q(n, m)
    B = []
    historique_fitness = []  # Liste pour stocker l'historique des fitness
    compteur_sans_amelioration = 0

    for _ in range(nb_iterations):
        P = creer_p(Q)
        fct_fitness = evaluer_fitness(P, arg_weights, arg_values, arg_max_weight)
        P_best, fct_fitness_best = recuperer_meilleur(P, fct_fitness)

        # Mise à jour de l'historique et vérification du critère d'arrêt
        if historique_fitness:
            if fct_fitness_best - historique_fitness[-1] < seuil_amelioration:
                compteur_sans_amelioration += 1
            else:
                compteur_sans_amelioration = 0
        historique_fitness.append(fct_fitness_best)

        if compteur_sans_amelioration >= patience:
            print(f"Arrêt après {_} itérations, car il n'y a pas eu d'amélioration significative.")
            break

        B.append([P_best, fct_fitness_best])
        Q = update(P, Q, B, arg_weights, arg_values, arg_max_weight)

    return B, historique_fitness


# In[24]:


print(algorithme_quantique(100, len(weights_1), weights_1, values_1, max_weight_1, 1000))


# In[25]:


res_test = algorithme_quantique(100, len(weights_1), weights_1, values_1, max_weight_1, 1000)

print(get_best(res_test[0]))


# # Comparaisons
# 
# Nous allons maintenant comparer les efficacités des codes entre l'algorithme génétique classique et quantique

# In[26]:


taille_population = 50
iteration_max = 1000


# In[27]:


def premiere_occurence_max(arg_histo_fit):
    # Trouver le premier indice où apparaît le maximum
    max_value = max(arg_histo_fit)
    first_max_index = arg_histo_fit.index(max_value)

    return first_max_index


# ## Population 1

# In[28]:


debut_classique_1 = t.time()
population_gene = np.random.randint(2, size=(taille_population, weights_1.shape[0]))
resultat_classique_1 = algo_genetique(population_gene, weights_1, values_1, max_weight_1, 1000)
fin_classique_1 = t.time()

temps_classique_1 = fin_classique_1 - debut_classique_1

historique_fitness_classique_1 = resultat_classique_1[2]
premiere_classique_1 = premiere_occurence_max(historique_fitness_classique_1)


# In[29]:


debut_quantique_1 = t.time()
resultat_quantique_1 = algorithme_quantique(taille_population, len(weights_1), weights_1, values_1, max_weight_1,
                                            iteration_max)
fin_quantique_1 = t.time()

temps_quantique_1 = fin_quantique_1 - debut_quantique_1

historique_fitness_quantique_1 = resultat_quantique_1[1]
premiere_quantique_1 = premiere_occurence_max(historique_fitness_quantique_1)


# ## Population 2

# In[30]:


debut_classique_2 = t.time()
population_gene = np.random.randint(2, size=(taille_population, len(weights_2)))
resultat_classique_2 = algo_genetique(population_gene, weights_2, values_2, max_weight_2, 1000)
fin_classique_2 = t.time()

temps_classique_2 = fin_classique_2 - debut_classique_2

historique_fitness_classique_2 = resultat_classique_2[2]
premiere_classique_2 = premiere_occurence_max(historique_fitness_classique_2)


# In[31]:


debut_quantique_2 = t.time()
resultat_quantique_2 = algorithme_quantique(taille_population, len(weights_2), weights_2, values_2, max_weight_2,
                                            iteration_max)
fin_quantique_2 = t.time()

temps_quantique_2 = fin_quantique_2 - debut_quantique_2

historique_fitness_quantique_2 = resultat_quantique_2[1]
premiere_quantique_2 = premiere_occurence_max(historique_fitness_quantique_2)


# ## Population 3

# In[32]:


debut_classique_3 = t.time()
population_gene = np.random.randint(2, size=(taille_population, len(weights_3)))
resultat_classique_3 = algo_genetique(population_gene, weights_3, values_3, max_weight_3, 1000)
fin_classique_3 = t.time()

temps_classique_3 = fin_classique_3 - debut_classique_3

historique_fitness_classique_3 = resultat_classique_3[2]
premiere_classique_3 = premiere_occurence_max(historique_fitness_classique_3)


# In[33]:


debut_quantique_3 = t.time()
resultat_quantique_3 = algorithme_quantique(taille_population, len(weights_3), weights_3, values_3, max_weight_3,
                                            iteration_max)
fin_quantique_3 = t.time()

temps_quantique_3 = fin_quantique_3 - debut_quantique_3

historique_fitness_quantique_3 = resultat_quantique_3[1]
premiere_quantique_3 = premiere_occurence_max(historique_fitness_quantique_3)


# ## Population 4

# In[34]:


debut_classique_4 = t.time()
population_gene = np.random.randint(2, size=(taille_population, len(weights_4)))
resultat_classique_4 = algo_genetique(population_gene, weights_4, values_4, max_weight_4, 1000)
fin_classique_4 = t.time()

temps_classique_4 = fin_classique_4 - debut_classique_4

historique_fitness_classique_4 = resultat_classique_4[2]
premiere_classique_4 = premiere_occurence_max(historique_fitness_classique_4)


# In[35]:


debut_quantique_4 = t.time()
resultat_quantique_4 = algorithme_quantique(taille_population, len(weights_4), weights_4, values_4, max_weight_4,
                                            iteration_max)
fin_quantique_4 = t.time()

temps_quantique_4 = fin_quantique_4 - debut_quantique_4

historique_fitness_quantique_4 = resultat_quantique_4[1]
premiere_quantique_4 = premiere_occurence_max(historique_fitness_quantique_4)


# ## Population 5

# In[36]:


debut_classique_5 = t.time()
population_gene = np.random.randint(2, size=(taille_population, len(weights_5)))
resultat_classique_5 = algo_genetique(population_gene, weights_5, values_5, max_weight_5, 1000)
fin_classique_5 = t.time()

temps_classique_5 = fin_classique_5 - debut_classique_5

historique_fitness_classique_5 = resultat_classique_5[2]
premiere_classique_5 = premiere_occurence_max(historique_fitness_classique_5)


# In[ ]:


debut_quantique_5 = t.time()
resultat_quantique_5 = algorithme_quantique(taille_population, len(weights_5), weights_5, values_5, max_weight_5,
                                            iteration_max)
fin_quantique_5 = t.time()

temps_quantique_5 = fin_quantique_5 - debut_quantique_5

historique_fitness_quantique_5 = resultat_quantique_5[1]
premiere_quantique_5 = premiere_occurence_max(historique_fitness_quantique_5)


# ## Résultats

# In[ ]:


plt.figure(figsize=(15, 25))

# Sous-graphique 1
plt.subplot(5, 1, 1)
plt.plot(historique_fitness_classique_1, label="Classique 1")
plt.plot(historique_fitness_quantique_1, label="Quantique 1")
# Ajouter les croix
plt.scatter(premiere_classique_1, historique_fitness_classique_1[premiere_classique_1], c='red', marker='x')
plt.scatter(premiere_quantique_1, historique_fitness_quantique_1[premiere_quantique_1], c='blue', marker='x')
plt.text(x=0.6, y=0.1, 
         s=f"Temps Classique: {round(temps_classique_1, 5)}s, Temps Quantique: {round(temps_quantique_1, 5)}s", 
         fontsize=10, transform=plt.gca().transAxes)
plt.title("Population 1")
plt.xlabel("Itérations")
plt.ylabel("Fitness")
plt.legend()
plt.grid()

# Sous-graphique 2
plt.subplot(5, 1, 2)
plt.plot(historique_fitness_classique_2, label="Classique 2")
plt.plot(historique_fitness_quantique_2, label="Quantique 2")
# Ajouter les croix
plt.scatter(premiere_classique_2, historique_fitness_classique_2[premiere_classique_2], c='red', marker='x')
plt.scatter(premiere_quantique_2, historique_fitness_quantique_2[premiere_quantique_2], c='blue', marker='x')
plt.text(x=0.6, y=0.1, 
         s=f"Temps Classique: {round(temps_classique_2, 5)}s, Temps Quantique: {round(temps_quantique_2, 5)}s", 
         fontsize=10, transform=plt.gca().transAxes)
plt.title("Population 2")
plt.xlabel("Itérations")
plt.ylabel("Fitness")
plt.legend()
plt.grid()

# Sous-graphique 3
plt.subplot(5, 1, 3)
plt.plot(historique_fitness_classique_3, label="Classique 3")
plt.plot(historique_fitness_quantique_3, label="Quantique 3")
# Ajouter les croix
plt.scatter(premiere_classique_3, historique_fitness_classique_3[premiere_classique_3], c='red', marker='x')
plt.scatter(premiere_quantique_3, historique_fitness_quantique_3[premiere_quantique_3], c='blue', marker='x')
plt.text(x=0.6, y=0.1, 
         s=f"Temps Classique: {round(temps_classique_3, 5)}s, Temps Quantique: {round(temps_quantique_3, 5)}s", 
         fontsize=10, transform=plt.gca().transAxes)
plt.title("Population 3")
plt.xlabel("Itérations")
plt.ylabel("Fitness")
plt.legend()

# Sous-graphique 4
plt.subplot(5, 1, 4)
plt.plot(historique_fitness_classique_4, label="Classique 4")
plt.plot(historique_fitness_quantique_4, label="Quantique 4")
# Ajouter les croix
plt.scatter(premiere_classique_4, historique_fitness_classique_4[premiere_classique_4], c='red', marker='x')
plt.scatter(premiere_quantique_4, historique_fitness_quantique_4[premiere_quantique_4], c='blue', marker='x')
plt.text(x=0.6, y=0.1, 
         s=f"Temps Classique: {round(temps_classique_4, 5)}s, Temps Quantique: {round(temps_quantique_4, 5)}s", 
         fontsize=10, transform=plt.gca().transAxes)
plt.title("Population 4")
plt.xlabel("Itérations")
plt.ylabel("Fitness")
plt.legend()



plt.show()

