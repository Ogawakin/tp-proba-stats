import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy.stats as stats

## Problème 1

poids_kg = [0.499, 0.509, 0.501, 0.494, 0.498, 0.497, 0.504, 0.506, 0.502, 0.496, 0.495, 0.493, 0.507, 0.505, 0.503, 0.491]
poids_g = [85.06, 91.44, 87.93, 89.02, 87.28, 82.34, 86.23, 84.16, 88.56, 90.45, 84.91, 89.90, 85.52, 86.75, 88.54, 87.90]


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

    return variance/(len(liste))

plt.hist(poids_kg, color = 'yellow', edgecolor = 'red')
plt.xlabel('valeurs')
plt.ylabel('effectifs')
plt.title('Histogramme des fréquences')
plt.show()

# détermination de l'intervalle de confiance à 95% et 99%
# Etant donné que la variance est inconnue,
# on va procédé à un centrage et réduction de celle-ci,
# (en jouant sur la varaible aléatoire de la loi)
# ce qui donnera sqrt(n)*(moyenne_empirique - moyenne)/ecart_type_empirique

def intervalle_confiance(alpha, liste):
    """ calcul d'un intervalle de confiance de 95%
        ou 99% (aplha = 0.01 ou 0.05) """

    moyenne = moyenne_empirique(liste)
    ecart_type = m.sqrt(variance_empirique(liste))
    t = 0

    # calcul du quartile/fractile pour St((n-1) et d'ordre 1 - alpha/2
    t = stats.t.ppf(1 - alpha/2, df= len(liste) - 1)

    intervalle_gauche = moyenne - t*ecart_type/(m.sqrt(len(liste)))
    intervalle_droit = moyenne + t*ecart_type/(m.sqrt(len(liste)))
    return [round(intervalle_gauche, 3), round(intervalle_droit, 3)]

interval1 = intervalle_confiance(0.05, poids_kg)
interval2 = intervalle_confiance(0.05, poids_g)
interval3 = intervalle_confiance(0.01, poids_kg)
interval4 = intervalle_confiance(0.01, poids_g)

print("Pour pots confiture (en kg), intervalle à 95% : ", interval1)
print("Pour avocats (en g), intervalle à 95% : ", interval2)
print("Pour pots confiture (en kg), intervalle à 99% : ", interval3)
print("Pour avocats (en g), intervalle à 99% : ", interval4)

## Problème 2

# selon l'étude, 95/500 sont satisfait par la compagnie
# on peut donc en approximer une moyenne de 95/500


## Problème 3

# Pour cela, on utilise la fonction bernoulli
# du module scipy.stats et plus précisemment
# la fonction bernoulli.rvs qui génèrent un array
# de valeurs de n exdpériences de Bernoulli.
# bernoulli.rvs(p, n)

print("Taille de l'échantillon voulue : ", end="")
n = int(input())

# ici, p = 1/2 = 0.5
echantillon = stats.bernoulli.rvs(0.5, size=n)
echantillon = echantillon.tolist()

print(f"Pour un échantillon de {n} expériences de Bernoulli ", end="")
print("indépendantes, voici l'intervalle de confiance à ", end="")
print("95% pour le paramètre p : ", intervalle_confiance(0.05, echantillon))

# plus le nombre d'expérience augmente, plus les bornes se resserent autour
# de la valeur véritable du paramètre

