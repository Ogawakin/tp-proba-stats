# Imports
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Fonctions
def moyenne(x):
    """ Etant donné une liste x, renvoie sa moyenne. """
    return sum(x)/len(x)


def covariance(x, y):
    """ Etant donné deux listes x et y de même taille,
    renvoie leur covariance. """

    assert len(x) == len(y), 'x et y doivent avoir la même taille.'
    n = len(x)

    x_barre = moyenne(x)
    y_barre = moyenne(y)

    somme_cov = 0

    for i in range(n):
        somme_cov += (x[i] - x_barre) * (y[i] - y_barre)

    return somme_cov


def regression_lineaire(x, y):
    """Etant donné deux listes x et y de même taille,
    renvoie leur covariance. """

    assert len(x) == len(y), 'x et y doivent avoir la même taille.'
    n = len(x)

    x_barre = moyenne(x)
    y_barre = moyenne(y)

    beta_1 = covariance(x,y) / covariance(x,x)
    beta_0 = y_barre - beta_1 * x_barre

    return (beta_1, beta_0) # a * beta_1 + beta_0 (ax + b)


# Pourcentage de rendement d'un procédé chimique y_i
# en fonction de la température x_i
x_i = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
y_i = [43, 45, 48, 51, 55, 57, 59, 63, 66, 68]
plt.plot(x_i, y_i)

a, b = regression_lineaire(x_i, y_i)
x = np.linspace(x_i[0], x_i[-1]) # Premier et dernier élément de x_i
plt.plot(x, a * x + b) # tracée en orange

a_2, b_2 = np.polyfit(x_i, y_i, deg=1)
plt.plot(x, a_2 * x + b_2) # tracée en vert, elle superpose EXACTEMENT la droite orange

#plt.show()


# Modèle vectoriel
assert len(x_i) == len(y_i), 'x et y doivent avoir la même taille.'
n = len(x_i)

x_i = np.array(x_i)
y_i = np.array(y_i)

colonne_1 = np.ones((1, n))
A = np.transpose(np.vstack((colonne_1, x_i)))
A_T = np.transpose(A)

formule = A_T.dot(A)
formule = np.linalg.inv(formule)
formule = formule.dot(A_T).dot(y_i) 

print(formule)

b, a = formule
plt.plot(x, a * x + b)

test = np.random.normal(0, 1, 1000)
sm.qqplot(y_i)

plt.show()

