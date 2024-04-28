import numpy as np
# import pandas as pd

def test_Z(alpha, Z_tilde, Z, Y):
    test = 0
    for i in range(len(Z)):
        for j in range(len(Z)):
            if i !=j:
                eta = alpha - np.linalg.norm(Z[i]-Z[j])
                eta_tilde = alpha - np.linalg.norm(Z_tilde[i]-Z_tilde[j])
                test += Y[i][j]*eta_tilde-np.log(1+np.exp(eta_tilde))-Y[i][j]*eta-np.log(1+np.exp(eta))
        test += (np.linalg.norm(Z_tilde[i])-np.linalg.norm(Z[i]))/200
    return test

def test_alpha(alpha_tilde, alpha, Z, Y):
    test = 0
    for i in range(len(Z)):
        for j in range(len(Z)):
                if i != j:
                    eta = alpha - np.linalg.norm(Z[i]-Z[j])
                    eta_tilde = alpha_tilde - np.linalg.norm(Z[i]-Z[j])
                    test += Y[i][j]*eta_tilde-np.log(1+np.exp(eta_tilde))-Y[i][j]*eta+np.log(1+np.exp(eta)) + 2*(eta_tilde-eta)
    return test


def MCMC(alpha, Z, Y, pas=1):
    Z_tilde = Z + np.random.normal(0, pas, size=(Z.shape[0], Z.shape[1]))
    test = test_Z(alpha, Z_tilde, Z, Y)
    #print(test)
    if np.log(np.random.random()) < test :
        Z = Z_tilde
    alpha_tilde = np.random.exponential(2)
    test = test_alpha(alpha_tilde, alpha, Z, Y)
    #print(test)
    if np.log(np.random.random()) < test : 
        alpha = alpha_tilde
    return alpha, Z



from MLE import MLE
import matplotlib.pyplot as plt
np.random.seed(12345)

data = np.loadtxt("data/Florentine_families.csv", delimiter=',')
#print(relations)

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

alphas, Z = boucle(data, 100)
plt.plot(alphas)
plt.show()
"""
colors = plt.cm.rainbow(np.linspace(0, 1, 15))
for i in range (np.shape(Z)[0]):
    plt.scatter(Z[i, 0], Z[i, 1], c=colors[i], cmap='jet', label=f'Famille {i+1}')
plt.legend()
plt.show()"""
