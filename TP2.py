# à completer
import numpy as np
import math as m
import matplotlib.pyplot as plt
#import scipy as sc

## exo 2.1 Loi Binomiale
def combinaison(k, n):
    return m.factorial(n)/(m.factorial(k) * m.factorial(n-k))

def binom(x, n, p):
    if x > n:
        return 0
    else:
        return combinaison(x, n) * (p**x) * ((1-p)**(n-x))

X = [k for k in range(101)]
Y1 = [binom(k, 30, 0.5) for k in range(101)]
Y2 = [binom(k, 30, 0.7) for k in range(101)]
Y3 = [binom(k, 50, 0.4) for k in range(101)]

plt.plot(X, Y1, label="n = 30, p = 0.5")
plt.plot(X, Y2, label="n = 30, p = 0.7")
plt.plot(X, Y3, label="n = 50, p = 0.4")
plt.legend()
plt.show()


## exo 2.2 Loi Normale Univariée

def normal(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(((x - mu)/sigma)**2))

X = np.linspace(-15, 15, 2000)
Y1 = normal(X, 0, 1)
Y2 = normal(X, 2, 1.5)
Y3 = normal(X, 2, 0.6)

plt.plot(X, Y1, label="mu = 0, sigma = 1")
plt.plot(X, Y2, label="mu = 2, sigma = 1.5")
plt.plot(X, Y3, label="mu = 2, sigma = 0.6")
plt.legend()
plt.show()














