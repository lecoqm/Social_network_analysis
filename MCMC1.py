import numpy as np
from scipy.linalg import orthogonal_procrustes
from MLE import MLE

def pi_1(Z, sigma=100):
    '''
    log de la loi à priori des Z (à une constante près)
    '''
    return np.sum(np.linalg.norm(Z)**2 / (2 * sigma**2))

def pi_2(alpha, lamb=2):
    '''
    log de la loi à priori des alpha (à une constante près)
    '''
    return - lamb*alpha

def test_Z(alpha, Z_tilde, Z, Y, sigma=100):
    """
    renvoie le log de probabilité d'acceptation pour Z
    """
    test = 0
    for i in range(len(Z)):
        for j in range(len(Z)):
            if i !=j:
                eta = alpha - np.linalg.norm(Z[i]-Z[j])
                eta_tilde = alpha - np.linalg.norm(Z_tilde[i]-Z_tilde[j])
                test += Y[i][j]*eta_tilde-np.log(1+np.exp(eta_tilde))-Y[i][j]*eta-np.log(1+np.exp(eta))
        test += pi_1(Z_tilde, sigma)-pi_1(Z, sigma)
    return test

def test_alpha(alpha_tilde, alpha, Z, Y, lamb=2):
    """
    renvoie la probabilité d'acceptation pour alpha
    """
    test = 0
    for i in range(len(Z)):
        for j in range(len(Z)):
                if i != j:
                    eta = alpha - np.linalg.norm(Z[i]-Z[j])
                    eta_tilde = alpha_tilde - np.linalg.norm(Z[i]-Z[j])
                    test += Y[i][j]*eta_tilde - np.log(1+np.exp(eta_tilde)) - Y[i][j]*eta + np.log(1+np.exp(eta))
    test *= np.exp(pi_2(alpha_tilde, lamb) - pi_2(alpha, lamb))
    return test

def MCMC(alpha, Z, Y, pas1=1, pas2=0.5, lamb=2, sigma=100):
    """
    Cette fonction prend un couple (alpha,Z), et renvoie un nouveau couple (alpha,Z).
    Le nouveau Z est tiré selon une loi normale centrée en Z, de variance pas1**2. Puis accepté, avec pour prior
    la loi normale centrée, de variance sigma**2.
    Le nouveau alpha est tiré selon une loi normale centrée en Z, de variance pas2**2. Puis accepté, avec pour prior
    la loi exponentielle de paramètre lambda.
    """
    Z_tilde = Z + np.random.normal(0, pas1, size=(Z.shape[0], Z.shape[1]))
    Z_tilde = Z_tilde - np.mean(Z_tilde, axis=0)
    T, d = orthogonal_procrustes(Z, Z_tilde)
    Z_tilde = np.dot(Z_tilde, T)
    test = test_Z(alpha, Z_tilde, Z, Y, sigma)
    if np.random.random() < np.exp(test) :
        Z = Z_tilde
    alpha_tilde = abs(alpha + np.random.normal(0, pas2))
    test = test_alpha(alpha_tilde, alpha, Z, Y, lamb)
    if np.random.random() < test : 
        alpha = alpha_tilde
    return alpha, Z

####################

import matplotlib.pyplot as plt
np.random.seed(12345)

data = np.loadtxt("data/Florentine_families.csv", delimiter=',')

def boucle(Y, iterations):
    Z = MLE(Y,2)
    for i in range(15):
        Z[i] = np.random.multivariate_normal([0,0], 10*np.identity(2))
    alpha = np.random.exponential(2)
    alphas = [alpha]
    for i in range(iterations):
        alpha, Z = MCMC(alpha, Z, Y)
        alphas.append(alpha)
    return alphas, Z

alphas, Z = boucle(data, 5000)
print(alphas)
colors = plt.cm.rainbow(np.linspace(0, 1, 15))
for i in range (np.shape(Z)[0]):
    plt.scatter(Z[i, 0], Z[i, 1], c=colors[i], cmap='jet', label=f'Famille {i+1}')
plt.legend()
plt.show()