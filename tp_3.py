import numpy as np
import matplotlib.pyplot as plt
import math as m

## problème 1

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

    for i in range(len(liste)-1):
        variance = variance + (liste[i] - x_bar)**2

    return variance

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
    if alpha == 0.05:
        t = 1.746
    elif alpha == 0.01:
        t = 2.583

    intervalle_gauche = moyenne - t*ecart_type/(m.sqrt(len(liste)))
    intervalle_droit = moyenne + t*ecart_type/(m.sqrt(len(liste)))
    return [intervalle_gauche, intervalle_droit]

interval1 = intervalle_confiance(0.05, poids_kg)
interval2 = intervalle_confiance(0.05, poids_g)
interval3 = intervalle_confiance(0.01, poids_kg)
interval4 = intervalle_confiance(0.01, poids_g)
print("intervalle à 95% : ", interval1)
print("intervalle à 95% : ", interval2)
print("intervalle à 98% : ", interval3)
print("intervalle à 98% : ", interval4)

