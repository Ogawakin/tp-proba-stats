## exo 1.2 régression linaire et descente de gradient
import numpy as np


def regression_poly(beta_0, esp, x, y, alpha):
    beta_t1 = np.array([0, 0])
    beta_t = beta_0
    assert len(x)==len(y), "les listes ne sont pas de taille égale"
    n = len(x)

    while np.linalg.norm(beta_t-beta_t1) > esp:
        f_beta_0 = 0
        f_beta_1 = 0

        for i in range(n):
            f_beta_0 += ((beta_t[0] + beta_t[1]*x[i]) - y[i])
            f_beta_1 += ((beta_t[0] + beta_t[1]*x[i]) - y[i])*x[i]

        f_beta_0 *= (1/n)
        f_beta_1 *= (1/n)

        beta_aux = beta_t
        beta_t[0] = beta_aux[0] - alpha*f_beta_0
        beta_t[1] = beta_aux[1] - alpha*f_beta_1
        beta_t1 = beta_aux

    return beta_t


x = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
y = [43, 45, 48, 51, 55, 57, 59, 63, 66, 68]
beta = regression_poly(np.array([-1,-1]), 0.001, x, y, 0.1)
print(beta)

# X = np.linspace(x[0], x[len(x)-1])
# Y = np.linspace(y[0], y[len(y)-1])
# #plt.plot(X, Y)
# plt.plot(X, beta[0] + beta[1]*X)
# plt.show()
