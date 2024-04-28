import numpy as np
import matplotlib.pyplot as plt

from MLE import MLE

np.random.seed(0)

data = np.loadtxt("data/Florentine_families.csv", delimiter=",")
n = np.shape(data)[0]  # = 15

DIMENSIONS = 2
SIGMA = 100


def test_Z(alpha: float, Z_tilde: np.matrix, Z: np.matrix, Y: np.matrix) -> float:
    """
    Renvoie la probabilite d'acceptation pour Z
    """
    test = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            eta = alpha - np.linalg.norm(Z[i] - Z[j])
            eta_tilde = alpha - np.linalg.norm(Z_tilde[i] - Z_tilde[j])
            test += (
                Y[i][j] * eta_tilde
                - np.log(1 + np.exp(eta_tilde))
                - Y[i][j] * eta
                - np.log(1 + np.exp(eta))
            )
        test += (np.linalg.norm(Z_tilde[i]) - np.linalg.norm(Z[i])) / (2 * SIGMA)
    return test


def test_alpha(alpha_tilde: float, alpha: float, Z: np.matrix, Y: np.matrix) -> float:
    """
    Renvoie la probabilite d'acceptation pour alpha
    """
    test = 0
    for i in range(len(Z)):
        for j in range(len(Z)):
            if i != j:
                eta = alpha - np.linalg.norm(Z[i] - Z[j])
                eta_tilde = alpha_tilde - np.linalg.norm(Z[i] - Z[j])
                test += (
                    Y[i][j] * eta_tilde
                    - np.log(1 + np.exp(eta_tilde))
                    - Y[i][j] * eta
                    + np.log(1 + np.exp(eta))
                    + 2 * (eta_tilde - eta)
                )
    return test


def MCMC(
    alpha: float, Z: np.matrix, Y: np.matrix, pas: float = 1
) -> tuple[float, np.matrix]:
    """
    Met à jour alpha et Z selon l'algorithme MCMC
    """
    Z_tilde = Z + np.random.normal(0, pas, size=(Z.shape[0], Z.shape[1]))

    test = test_Z(alpha, Z_tilde, Z, Y)
    if np.log(np.random.random()) < test:
        Z = Z_tilde
    alpha_tilde = np.random.exponential(2)

    test = test_alpha(alpha_tilde, alpha, Z, Y)
    if np.log(np.random.random()) < test:
        alpha = alpha_tilde

    return alpha, Z


def main(Y: np.matrix, n_iter: int = 100) -> tuple[list[float], np.matrix]:
    """
    Renvoie les valeurs successives de alpha et Z
    """
    Z = MLE(Y, DIMENSIONS)
    for i in range(n):
        Z[i] = np.random.multivariate_normal([0, 0], 10 * np.identity(2))
    alpha = np.random.exponential(2)
    alphas = [alpha]
    for i in range(n_iter):
        alpha, Z = MCMC(alpha, Z, Y)
        alphas.append(alpha)
    return alphas, Z


if __name__ == "__main__":
    alphas, Z = main(data)

    fig, axis = plt.subplots(1, 2)

    axis[0].plot(alphas)
    axis[0].set_title("Evolution de alpha")

    colors = plt.cm.rainbow(np.linspace(0, 1, 15))
    for i in range(np.shape(Z)[0]):
        axis[1].scatter(
            Z[i, 0], Z[i, 1], c=colors[i], cmap="jet", label=f"Famille {i+1}"
        )
    axis[1].legend()
    axis[1].set_title("Projection des familles sur les 2 axes principaux")
    plt.show()
