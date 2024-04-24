import numpy as np
# import pandas as pd

np.random.seed(12345)

relations = np.loadtxt("data/Florentine_families.csv", delimiter=',')
#print(relations)

def test_Z(alpha, Z_tilde, Z, Y):
    test = 0
    for i in range(len(Z)):
        for j in range(len(Z)):
            if i !=j:
                eta = alpha * (1-np.linalg.norm(Z[i]-Z[j]))
                eta_tilde = alpha * (1-np.linalg.norm(Z_tilde[i]-Z_tilde[j]))
                test += Y[i][j]*eta_tilde-np.log(1+np.exp(eta_tilde))-Y[i][j]*eta-np.log(1+np.exp(eta))
        test += (np.linalg.norm(Z_tilde[i])-np.linalg.norm(Z[i]))/200
    return test

def test_alpha(alpha_tilde, alpha, Z, Y):
    test = 0
    for i in range(len(Z)):
        for j in range(len(Z)):
                if i != j:
                    eta = alpha * (1-np.linalg.norm(Z[i]-Z[j]))
                    eta_tilde = alpha_tilde * (1-np.linalg.norm(Z[i]-Z[j]))
                    test += Y[i][j]*eta_tilde-np.log(1+np.exp(eta_tilde))-Y[i][j]*eta-np.log(1+np.exp(eta)) + 2*(eta_tilde-eta)
    return test


def MCMC(alpha, Z, Y):
    Z_tilde = np.random.multivariate_normal(Z, 5*np.identity(len(Z)))
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

def boucle(Y, taille):
    Z = np.zeros(15)
    for i in range(15):
        Z[i] = np.random.normal(0, 10)
    alpha = np.random.exponential(2)
    for i in range(taille):
        alpha, Z = MCMC(alpha, Z, Y)
    return alpha, Z

for i in range(20):
    alpha, Z = boucle(relations, 100)
    print(alpha)
