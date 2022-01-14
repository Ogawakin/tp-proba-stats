import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy.stats as stats

# Dans ce Tp, on cherche à estimer, via
# intervalle de confiance, l'espérance
# d'un échantillon statistique

def moyenne_empirique(liste):
    """ fonction qui calcule de manière empirique
        la moyenne d'un set de valeur """
    return sum(liste)/len(liste)

def variance_empirique(liste):
    """ fonction qui calcule de manière empirique
        la variance d'un set de valeur """
    variance = 0
    x_bar = moyenne_empirique(liste)

    for i in range(0, len(liste)):
        variance += (liste[i] - x_bar)**2

    return variance/(len(liste) - 1)

# détermination de l'intervalle de confiance à 95% et 99%
# Etant donné que la variance est inconnue,
# on utilisera la deuxième façon de calculer un intervalle
# de confiance présenté sur la fiche du TP

def intervalle_confiance_1(alpha, liste, variance):
    """ calcul d'un intervalle de confiance
        dans le cadre d'une variance connue
        et donc d'une recherche du fractile
        selon la loi N(0, 1)"""

    moyenne = moyenne_empirique(liste)
    ecart_type = m.sqrt(variance)

    # calcul du quartile/fractile pour N(0, 1) et d'ordre 1 - alpha/2
    u = stats.norm.ppf(1 - alpha/2)

    intervalle_gauche = moyenne - u*ecart_type/(m.sqrt(len(liste)))
    intervalle_droit = moyenne + u*ecart_type/(m.sqrt(len(liste)))

    return [round(intervalle_gauche, 3), round(intervalle_droit, 3)]

def intervalle_confiance_2(alpha, liste):
    """ calcul d'un intervalle de confiance
        dans le cadre d'une variance inconnue
        et donc d'une recherche du fractile
        selon la loi de student (St(n-1))"""

    moyenne = moyenne_empirique(liste)
    ecart_type = m.sqrt(variance_empirique(liste))

    # calcul du quartile/fractile pour St((n-1) et d'ordre 1 - alpha/2
    t = stats.t.ppf(1 - alpha/2, df= len(liste) - 1)

    intervalle_gauche = moyenne - t*ecart_type/(m.sqrt(len(liste)))
    intervalle_droit = moyenne + t*ecart_type/(m.sqrt(len(liste)))

    return [round(intervalle_gauche, 3), round(intervalle_droit, 3)]

## Problème 1

poids_kg = [0.499, 0.509, 0.501, 0.494, 0.498, 0.497, 0.504, 0.506, 0.502, 0.496, 0.495, 0.493, 0.507, 0.505, 0.503, 0.491]
poids_g = [85.06, 91.44, 87.93, 89.02, 87.28, 82.34, 86.23, 84.16, 88.56, 90.45, 84.91, 89.90, 85.52, 86.75, 88.54, 87.90]

# moyenne empirique de poids_kg
print("Moyenne empirique de poids_kg = ", moyenne_empirique(poids_kg))

# histogramme des fréquences
plt.hist(poids_kg, color = 'yellow', edgecolor = 'red')
plt.xlabel('valeurs')
plt.ylabel('effectifs')
plt.title('Histogramme des fréquences')
plt.show()

# calcul de différents intervalles de confiances
interval1 = intervalle_confiance_2(0.05, poids_kg)
interval2 = intervalle_confiance_2(0.05, poids_g)
interval3 = intervalle_confiance_2(0.01, poids_kg)
interval4 = intervalle_confiance_2(0.01, poids_g)

print("Pour pots confiture (en kg), intervalle à 95% : ", interval1)
print("Pour pots confiture (en kg), intervalle à 99% : ", interval3)
print("Pour avocats (en g), intervalle à 95% : ", interval2)
print("Pour avocats (en g), intervalle à 99% : ", interval4)

## Problème 2

# selon l'étude, 95/500 sont satisfait par la compagnie
# on peut donc en approximer une moyenne de 95/500
# Donc la fonction moyenne_empirique doit trouver 95/500

population = [1 for k in range(95)] + [0 for k in range(405)]
intervalle = intervalle_confiance_2(0.01, population)
print("\nLa proportion de voyageurs satisfaits se situe entre ", intervalle)

## Problème 3

# Pour cela, on utilise la fonction bernoulli
# du module scipy.stats et plus précisemment
# la fonction bernoulli.rvs qui génèrent un array
# de valeurs de n exdpériences de Bernoulli.
# bernoulli.rvs(p, n)

print("\nTaille de l'échantillon voulue : ", end="")
n = int(input())

# ici, p = 1/2 = 0.5
echantillon = stats.bernoulli.rvs(0.5, size=n)
echantillon = echantillon.tolist()
# variance = p*(1-p) dasn le cadre d'une loi de Bernoulli
variance = 0.5*0.5

print(f"Pour un échantillon de {n} expériences de Bernoulli, ", end="")
print("intervalle de confiance à ", end="")
print("95% pour le paramètre p : ", intervalle_confiance_1(0.05, echantillon, variance))

# plus le nombre d'expérience augmente, plus les bornes se resserent autour
# de la valeur véritable du paramètre

