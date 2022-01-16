import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy as sc

## exo 2.1 Loi Binomiale
# def combinaison(k, n):
#     """ calcul de k parmi n"""
#     return m.factorial(n)/(m.factorial(k) * m.factorial(n-k))
#
# def binom(x, n, p):
#     """ calcul pour une valeur
#         x de la valeur de la loi
#         binomiale appliqué à celui-ci """
#     if x > n:
#         return 0
#     else:
#         return combinaison(x, n) * (p**x) * ((1-p)**(n-x))

X = [k for k in range(101)]

# utilisation de scipy.stats.binom.pmf
# qui permet de calculer la fonction de
# densité de probabilité de B(k, n, p) = (n k)p**k(1-p)**n-k
plt.plot(sc.stats.binom.pmf(X, 30, 0.5), label="n = 30, p = 0.5")
plt.plot(sc.stats.binom.pmf(X, 30, 0.7), label="n = 30, p = 0.7")
plt.plot(sc.stats.binom.pmf(X, 50, 0.4), label="n = 50, p = 0.4")
plt.legend()
plt.show()


## exo 2.2 Loi Normale Univariée

# def normal(x, mu, sigma):
#     """ calcul pour une valeur
#         x de la valeur de la loi
#         normale univariée appliqué à celui-ci """
#     return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(((x - mu)/sigma)**2))

X = np.linspace(-15, 15, 2000)

plt.plot(X, sc.stats.norm.pdf(X, 0, 1), label="mu = 0, sigma = 1")
plt.plot(X, sc.stats.norm.pdf(X, 2, 1.5), label="mu = 2, sigma = 1.5")
plt.plot(X, sc.stats.norm.pdf(X, 2, 0.6), label="mu = 2, sigma = 0.6")
plt.legend()
plt.show()


## 2.3.1 Simulation de données à partir d'une loi, cas de a loi normale

def tirage_alea(mu, sigma, n):
    return np.random.normal(mu, sigma, n)


mu = 0
sigma = 1
fig, axs = plt.subplots(5)
fig.suptitle("Various sample\'s histogramm with normal law displayed")
sous_graphique = 0

for n in [100, 1000, 10000, 100000, 1000000]:

    sample = tirage_alea(mu, sigma, n)
    print("moyenne du sample = ", abs(mu - np.mean(sample)))                # 0.0, may vary
    print("écart-type du sample = ", abs(sigma - np.std(sample, ddof=1)))   # 0.1, may vary

    count, bins, ignored = axs[sous_graphique].hist(sample, 30, density=True)
    loi_normale = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))
    axs[sous_graphique].plot(bins, loi_normale,linewidth=2, color='r', label=f"Pour un échantillon de {n}")
    axs[sous_graphique].legend()
    sous_graphique += 1

plt.show()


## 2.4.1 Estimation de la densité :
##  Cas de la loi normale

def moyenne(liste):
    """ Fonction qui calcule de manière empirique
        la moyenne d'un set de valeur """
    return sum(liste)/len(liste)

def ecart_type(liste):
    """ Fonction qui calcule de manière empirique
        la variance d'un set de valeur """
    variance = 0
    x_bar = moyenne(liste)

    for i in range(len(liste)):
        variance = variance + (liste[i] - x_bar)**2

    return m.sqrt(variance/len(liste))

mu = 0
sigma = 1

#création de plusieurs sous-graphiques
figure, axis = plt.subplots(3)
figure.suptitle("Theorical normal law (red) and sampled normal law (blue)")
num_graph = 0       # "numéro" du sous-graphique

for i in [20, 80, 150]:

    sample = tirage_alea(mu, sigma, i)
    moy, sig = moyenne(sample), ecart_type(sample)
    print(f"Moyenne du sample pour n = {i} : ", moy)
    print(f"Ecart-type du sample pour n = {i} : ", sig, "\n")

    X = np.linspace(-5, 5, 100)
    # affichage de la densité estimée
    axis[num_graph].plot(X, 1/(sig * np.sqrt(2 * np.pi)) * np.exp(-(X - moy)**2 / (2 * sig**2)),linewidth=2, label=f"Pour un échantillon de {i}")
    #affichage conjoint de la vraie densité
    axis[num_graph].plot(X, normal(X, mu, sigma),linewidth=2, color='r')
    axis[num_graph].legend()
    #passage au sous-graphique suivant
    num_graph += 1

plt.show()


##  Cas de la loi exponentielle (fonction de densité)

# la fonction de densité de la loi exponentielle : lambda*exp(-x*lambda)

def echantillonage_exp(lambd, n):
    return np.random.exponential(1/lambd, n)

def estimation_lambda(echantillon):
    # Dans le cas d'une loi exponnentielle, la variance est égale à 1/lambda**2
    # donc pour estimer lambda, il faut calculer l'écart-type de l'échantillon

    return 1/ecart_type(echantillon)


print("Valeur de lambda souhaité (diff de 0) = ", end="")
lambd = float(input())

fig, axs = plt.subplots(3)
fig.suptitle(f"Various sample\'s histogramm with theorical exponential law (red) and sampled exponential law (blue) \n lambda = {lambd}")
num_graph = 0

for n in [20, 80, 150]:
    sample = echantillonage_exp(lambd, n)
    estimate_lambda = estimation_lambda(sample)

    count, bins, ignored = axs[num_graph].hist(sample, 30, density=True)

    X = np.linspace(0, 20, 200)
    axs[num_graph].plot(X, estimate_lambda * np.exp(-X * estimate_lambda), linewidth=2, label=f"Pour un échantillon de {n}")
    axs[num_graph].plot(X, lambd * np.exp(-X * lambd),linewidth=2, color='r')
    axs[num_graph].legend()
    num_graph += 1

plt.show()

## Cas de la loi exponentielle (fonction de répartition)

# la fonction de répartition de la loi exponentielle : 1 - exp(-lambda * x)

print("Valeur de lambda souhaité (diff de 0) = ", end="")
lambd = float(input())

fig, axs = plt.subplots(3)
fig.suptitle(f"Sample\'s CDF and Theorical CDF (red) \n lambda = {lambd}")
num_graph = 0

for n in [20, 80, 150]:

    #on crée un échantillon avec la fonction précédente
    sample = echantillonage_exp(lambd, n)
    estimate_lambda = estimation_lambda(sample)

    X = np.linspace(0, 8, 80)
    axs[num_graph].plot(X, 1 - np.exp(-X * estimate_lambda), linewidth=2, label=f"Pour un échantillon de {n}")
    axs[num_graph].plot(X, 1 - np.exp(-X * lambd),linewidth=2, color='r')
    axs[num_graph].legend()
    num_graph += 1

plt.show()







