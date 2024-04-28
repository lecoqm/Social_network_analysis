import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

from MLE import MLE

np.random.seed(0)

LAMBDA = 2
SIGMA = 100
DIMENSIONS = 2

data = np.loadtxt("data/Florentine_families.csv", delimiter=",")
n = np.shape(data)[0]  # = 15


def pi_1(Z: np.matrix, sigma: float = SIGMA) -> float:
    """
    Renvoie le Log de la loi a priori de Z (a une constante pres)
    """
    return -np.linalg.norm(Z) / (2 * sigma**2)


def pi_2(alpha: float, lambd: float = LAMBDA) -> float:
    """
    Renvoie le Log de la loi a priori de alpha (a une constante pres)
    """
    return -lambd * alpha


def test_Z(
    alpha: float, Z_tilde: np.matrix, Z: np.matrix, Y: np.matrix, sigma: float = SIGMA
) -> float:
    """
    Renvoie la probabilite d'acceptation pour Z
    """
    log_prob = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            eta = alpha - np.linalg.norm(Z[i] - Z[j])
            eta_tilde = alpha - np.linalg.norm(Z_tilde[i] - Z_tilde[j])
            log_prob += (
                Y[i][j] * eta_tilde
                - np.log(1 + np.exp(eta_tilde))
                - Y[i][j] * eta
                - np.log(1 + np.exp(eta))
            )
        log_prob += pi_1(Z_tilde, sigma) - pi_1(Z, sigma)
    return np.exp(log_prob)


def test_alpha(
    alpha_tilde: float, alpha: float, Z: np.matrix, Y: np.matrix, lambd: float = LAMBDA
) -> float:
    """
    Renvoie la probabilite d'acceptation pour alpha
    """
    log_prob = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            eta = alpha - np.linalg.norm(Z[i] - Z[j])
            eta_tilde = alpha_tilde - np.linalg.norm(Z[i] - Z[j])
            log_prob += (
                Y[i][j] * eta_tilde
                - np.log(1 + np.exp(eta_tilde))
                - Y[i][j] * eta
                + np.log(1 + np.exp(eta))
            )
    log_prob += pi_2(alpha_tilde, lambd) - pi_2(alpha, lambd)
    return np.exp(log_prob)


def MCMC(
    alpha: float,
    Z: np.matrix,
    Y: np.matrix,
    pas1: float = 1,
    pas2: float = 0.5,
    lambd: float = LAMBDA,
    sigma: float = SIGMA,
) -> tuple[float, np.matrix]:
    """
    Cette fonction prend un couple (alpha,Z), et renvoie un nouveau couple (alpha,Z).
    Le nouveau Z est tire selon une loi normale centree en Z, de variance pas1**2. Puis accepte, avec pour prior
    la loi normale centree, de variance sigma**2.
    Le nouveau alpha est tire selon une loi normale centree en Z, de variance pas2**2. Puis accepte, avec pour prior
    la loi exponentielle de parametre lambda.
    """
    Z_tilde = Z + np.random.normal(0, pas1, size=Z.shape)
    Z_tilde = Z_tilde - np.mean(Z_tilde, axis=0)
    Omega, _ = orthogonal_procrustes(Z, Z_tilde)

    Z_tilde = np.dot(Z_tilde, Omega)
    alpha_tilde = abs(alpha + np.random.normal(0, pas2))

    prob_Z = test_Z(alpha, Z_tilde, Z, Y, sigma)
    if np.random.random() < prob_Z:
        Z = Z_tilde

    prob_alpha = test_alpha(alpha, alpha, Z, Y, lambd)  # = tout le temps 0 ??
    if np.random.random() < prob_alpha:
        alpha = alpha_tilde
    return alpha, Z


def main(Y: np.matrix, n_iter: int = 1000) -> tuple[list[float], np.matrix]:
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
    for _ in range(n_iter):
        alpha, Z = MCMC(alpha, Z, data)
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

# alpha converge différement de celui du papier mais les familles sont bien séparées.
