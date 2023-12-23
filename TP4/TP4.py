#!/usr/bin/env python
# coding: utf-8

# # Algorithme classique du problème du sac à dos

# In[1]:


import numpy as np


# # Classique

# In[2]:


weights = np.array([10, 20, 30, 40, 50])
values = np.array([60, 100, 120, 140, 160])
max_weight = np.sum(weights) // 3

population_size = 100

population = np.random.randint(2, size=(population_size, weights.shape[0]))
print(population)


# In[17]:


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


# In[18]:


population = correction(population, max_weight, weights)
print(population)


# In[19]:


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


# In[20]:


fitness(weights, values, population, max_weight)


# In[76]:


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
    


# In[78]:


print(selection_roulette(fitness(weights, values, population, max_weight), population))


# In[82]:


def crossover(parents, arg_max_weight, arg_weights):
    # On choisit un point de croisement aléatoire
    point_croisement = np.random.randint(1, parents[0].shape[0]-1)
    # On crée les enfants
    enfant1 = np.concatenate((parents[0][:point_croisement], parents[1][point_croisement:]))
    enfant2 = np.concatenate((parents[1][:point_croisement], parents[0][point_croisement:]))
    return correction_un(enfant1, arg_max_weight, arg_weights), correction_un(enfant2, arg_max_weight, arg_weights)


# In[28]:


print(crossover(selection_roulette(fitness(weights, values, population, max_weight), population), max_weight, weights))


# In[29]:


def mutation(enfants, arg_max_weight, arg_weights):
    # On choisit un point de mutation aléatoire
    point_mutation = np.random.randint(0, enfants[0].shape[0])
    # On crée les enfants si le point de mutation est 0 on le change en 1 et inversement
    if enfants[0][point_mutation] == 0:
        enfants[0][point_mutation] = 1
    else:
        enfants[0][point_mutation] = 0
    return correction_un(enfants[0], arg_max_weight, arg_weights), correction_un(enfants[1], arg_max_weight, arg_weights)


# In[30]:


print(mutation(crossover(selection_roulette(fitness(weights, values, population, max_weight), population), max_weight, weights), max_weight, weights))


# In[41]:


def evolution(arg_population, arg_fitness, arg_max_weight, arg_weights, arg_mutation_rate=0.1):
    # On sélectionne les parents
    parents = selection_roulette(arg_fitness, arg_population)
    # On crée les enfants
    enfants = crossover(parents, arg_max_weight, arg_weights)
    # On mute les enfants avec une probabilité
    if np.random.rand() < arg_mutation_rate:
        enfants = mutation(enfants, arg_max_weight, arg_weights)
    # On ajoute les enfants à la population si ils améliorent le pire individu
    enfants = np.array(enfants)
    fitness_enfants = fitness(arg_weights, values, enfants, arg_max_weight)
    fitness_population = fitness(arg_weights, values, arg_population, arg_max_weight)
    for i in range(2):
        if fitness_enfants[i] > np.min(fitness_population):
            # On remplace le pire individu par l'enfant
            arg_population[np.argmin(fitness_population)] = enfants[i]
            fitness_population[np.argmin(fitness_population)] = fitness_enfants[i]
    # On calcule la fitness de la nouvelle population
    arg_fitness = fitness_population
    return arg_population


# In[42]:


print(evolution(population, fitness(weights, values, population, max_weight), max_weight, weights))


# In[80]:


def algo_genetique(arg_population, arg_weights, arg_values, arg_max_weight, nb_iterations=1000):
    for _ in range(nb_iterations):
        arg_fitness = fitness(arg_weights, arg_values, arg_population, arg_max_weight)
        arg_population = evolution(arg_population, arg_fitness, arg_max_weight, arg_weights)
    # On récupère le meilleur individu
    arg_fitness = fitness(arg_weights, arg_values, arg_population, arg_max_weight)
    return arg_population[np.argmax(arg_fitness)], np.max(arg_fitness)


# In[83]:


algo_genetique(population, weights, values, max_weight)


# # Les 5 jeux de données

# ## Jeu de données 1

# In[84]:


weights_1 = np.array([10, 20, 30, 40, 50])
values_1 = np.array([60, 100, 120, 140, 160])
max_weight_1 = np.sum(weights) // 3


# ## Jeu de données 2

# In[85]:


weights_2 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
values_2 = np.array([60, 100, 120, 140, 160, 180, 200, 220, 240, 260])
max_weight_2 = np.sum(weights) // 3


# ## Jeu de données 3

# In[86]:


weights_3 = [83, 73, 22, 1, 65, 98, 64, 40, 92, 68, 6, 39, 90, 73, 7, 99, 6, 52, 23, 14]
values_3 = [8, 17, 2, 17, 20, 16, 15, 2, 19, 10, 19, 6, 19, 5, 16, 21, 19, 18, 9, 1]
max_weight_3 = np.sum(weights) // 3


# ## Jeu de données 4

# In[87]:


weights_4 = [83, 73, 22, 1, 65, 98, 64, 40, 92, 68, 6, 39, 90, 73, 7, 99, 6, 52, 23, 14, 83, 73, 22, 1, 65, 98, 64, 40, 92, 68, 6, 39, 90, 73, 7, 99, 6, 52, 23, 14]
values_4 = [8, 17, 2, 17, 20, 16, 15, 2, 19, 10, 19, 6, 19, 5, 16, 21, 19, 18, 9, 1, 8, 17, 2, 17, 20, 16, 15, 2, 19, 10, 19, 6, 19, 5, 16, 21, 19, 18, 9, 1]
max_weight_4 = np.sum(weights) // 3


# ## Jeu de données 5

# In[88]:


weights_5 = [104, 390, 95, 276, 357, 393, 345, 148, 170, 174, 316, 185, 163, 22, 406, 360, 13, 14, 205, 136, 305, 75, 395, 262, 267, 379, 114, 171, 267, 383, 347, 308, 399, 382, 231, 207, 354, 395, 382, 174]
values_5 = [25, 7, 6, 24, 19, 6, 19, 20, 21, 14, 25, 5, 24, 6, 25, 5, 25, 25, 25, 25, 25, 25, 25, 25, 6, 25, 25, 25, 25, 25, 25, 25, 25, 25, 6, 25, 25, 25, 25, 25]
max_weight_5 = np.sum(weights) // 3


# # Quantique 

# In[89]:


weights_q = np.array([10, 20])
values_q = np.array([60, 100])
max_weight_q = np.sum(weights_q) // 3


# In[90]:


def initialiser_q(n, m):
    Q = []
    un_ele = [1/(2**0.5) for _ in range(m)]
    for i in range(n):
        Q.append([un_ele, un_ele])
    return Q


# In[91]:


print(initialiser_q(5, 2))


# In[93]:


def initialiser_p(n, m, Q):
    P = []
    for i in range(n):
        aux = []
        for j in range(m):
            fct_r = np.random.rand()
            if fct_r < Q[i][0][j]**2:
                aux.append(0)
            else:
                aux.append(1)
        P.append(aux)
    return P


# In[94]:


print(initialiser_p(5, 2, initialiser_q(5, 2)))


# In[95]:


def evaluer_fitness(P, arg_weights, arg_values, arg_max_weight):
    tab_fitness = []
    for i in range(len(P)):
        S1 = np.sum(P[i] * arg_values)
        S2 = np.sum(P[i] * arg_weights)
        if S2 > arg_max_weight:
            tab_fitness.append(0)
        else:
            tab_fitness.append(S1)
    return tab_fitness


# In[96]:


print(evaluer_fitness(initialiser_p(5, 2, initialiser_q(5, 2)), weights_q, values_q, max_weight_q))


# In[97]:


def recuperer_meilleur(P, arg_fitness):
    return P[np.argmax(arg_fitness)], np.max(arg_fitness)


# In[98]:


sol = initialiser_p(5, 2, initialiser_q(5, 2))

print(recuperer_meilleur(sol, evaluer_fitness(sol, weights_q, values_q, max_weight_q)))


# In[99]:


# Définition de l'angle gamma suivant la stratégie donnée par Kuk-Hyun Han et Jong-Hwan Kim en 2000

def def_gamma(fitnes_indi, fitness_max, bit_indi, bit_max):
    if bit_indi == 0 and bit_max == 0:
        return 0
    elif bit_indi == 0 and bit_max == 1:
        if fitnes_indi > fitness_max:
            return 0.05 * np.pi
        else:
            return 0
    elif bit_indi == 1 and bit_max == 0:
        if fitnes_indi > fitness_max:
            return 0.025 * np.pi
        else:
            return 0.01 * np.pi
    elif bit_indi == 1 and bit_max == 1:
        if fitnes_indi > fitness_max:
            return 0.025 * np.pi
        else:
            return 0.05 * np.pi


# In[105]:


def get_best(B):
    meilleur_element = max(B, key=lambda x: x[1])
    return meilleur_element


# In[100]:


def calcule_alpha_beta(alpha, beta, gamma):
    alpha_prime = alpha * np.cos(gamma) - beta * np.sin(gamma)
    beta_prime = alpha * np.sin(gamma) + beta * np.cos(gamma)
    return alpha_prime, beta_prime

print(calcule_alpha_beta(0.5, 0.5, np.random.rand() * np.pi))

def update(P, Q, B, fct_weights, fct_values, fct_max_weight):
    for i in range(len(Q)):
        best_sol = get_best(B)[0]
        for j in range(len(Q[i])):
            alpha = Q[i][0][j]
            beta = Q[i][1][j]
            gamma_fct = def_gamma(evaluer_fitness([P[i]], fct_weights, fct_values, fct_max_weight)[0], get_best(B)[1], P[i][j], best_sol[j])
            alpha_prime, beta_prime = calcule_alpha_beta(alpha, beta, gamma_fct)
            Q[i][0][j] = alpha_prime
            Q[i][1][j] = beta_prime
    return Q


# In[101]:


# Créer 𝑃(𝑡) en observant 𝑄(𝑡) et évaluer 𝑃(𝑡) : Sur la base de l'observation de 𝑄(𝑡), les solutions binaires 𝑃(𝑡) sont formées  

def creer_p(Q):
    P = []
    for i in range(len(Q)):
        aux = []
        for j in range(len(Q[i][1])):
            r = np.random.rand()
            if r < Q[i][0][j]**2:
                aux.append(0)
            else:
                aux.append(1)
        P.append(aux)
    return P


# In[102]:


print(creer_p(initialiser_q(5, 2)))


# In[103]:


# Algorithme quantique

def algorithme_quantique(n, m, arg_weights, arg_values, arg_max_weight, nb_iterations):
    Q = initialiser_q(n, m)
    B = []
    for i in range(nb_iterations):
        P = creer_p(Q)
        fct_fitness = evaluer_fitness(P, arg_weights, arg_values, arg_max_weight)
        P_best, fct_fitness_best = recuperer_meilleur(P, fct_fitness)
        B.append([P_best, fct_fitness_best])
        Q = update(P, Q, B, arg_weights, arg_values, arg_max_weight)
    return B


# In[106]:


print(algorithme_quantique(5, 2, weights_q, values_q, max_weight_q, 10))


# In[107]:


print(algorithme_quantique(100, len(weights_1), weights_1, values_1, max_weight_1, 1000))


# In[110]:


print(get_best(algorithme_quantique(100, len(weights_1), weights_1, values_1, max_weight_1, 1000)))


# In[109]:


population_gene = np.random.randint(2, size=(100, weights_1.shape[0]))

print(algo_genetique(population_gene, weights_1, values_1, max_weight_1, 1000))

