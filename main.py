import numpy as np
import matplotlib.pyplot as plt

from MLE import MLE
from MCMC import *

np.random.seed(0)

LAMBDA = 2
SIGMA = 100
DIMENSIONS = 2

data = np.loadtxt("data/Florentine_families.csv", delimiter=",")
n = np.shape(data)[0]  # = 15



def main(Y: np.matrix, n_iter: int = 1000, n_save: int = 100) -> tuple[list[float], np.matrix]:
    """
    Cette fonction prend une matrice Y de données et renvoie une liste de alpha et une matrice Z.
    On initialise Z avec le MLE de Y.
    """
    Z = MLE(Y, DIMENSIONS)
    for i in range(n):
        Z[i] = np.random.multivariate_normal(
            [0 for _ in range(DIMENSIONS)], SIGMA ** (1 / 2) * np.identity(DIMENSIONS)
        )
    alpha = np.random.exponential(2)
    alphas = [alpha]
    Z_liste = [Z]
    for i in range(n_iter):
        alpha, Z = MCMC(alpha, Z, data)
        if i%n_save == 0:
            alphas.append(alpha)
            Z_liste.append(Z)
    return alphas, Z_liste


if __name__ == "__main__":
    alphas, Z_liste = main(data)

    fig, axis = plt.subplots(1, 2)

    axis[0].plot(alphas)
    axis[0].set_title("Evolution de alpha")

    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    for i in range(np.shape(Z_liste)[0]):
        Z = Z_liste[i]
        for j in range(n):
            axis[1].scatter(
                Z[j][0], Z[j][1], c=colors[j], cmap="jet", label=f"Famille {j+1}"
            )
    axis[1].legend()
    axis[1].set_title("Projection des familles sur les 2 axes principaux")
    plt.show()
