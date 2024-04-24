import numpy as np
# import pandas as pd
from math import exp

np.random.seed(12345)

relations = np.loadtxt("data/Florentine_families.csv", delimiter=',')
print(relations)

def test_Z(alpha, Z_tilde, Z, Y):
    test = 1
    for i in range(len(Z)):
        for j in range(len(Z)):
            if i !=j:
                eta = alpha * (1-np.linalg.norm(Z[i]-Z[j]))
                eta_tilde = alpha * (1-np.linalg.norm(Z_tilde[i]-Z_tilde[j]))
                test *= exp(eta_tilde * Y[i][j]) / (1+exp(eta_tilde)) * exp(eta * Y[i][j]) / (1+exp(eta))
                test *= exp((np.linalg.norm(Z)**2 - np.linalg.norm(Z_tilde)**2)/20000)
    return test

def test_alpha(alpha_tilde, alpha, Z, Y):
    test = 1
    for i in range(len(Z)):
        for j in range(len(Z)):
                if i != j:
                    eta = alpha * (1-np.linalg.norm(Z[i]-Z[j]))
                    eta_tilde = alpha_tilde * (1-np.linalg.norm(Z[i]-Z[j]))
                    #print(eta, eta_tilde)
                    #print(alpha)
                    test *= exp(eta_tilde * Y[i][j]) / (1+exp(eta_tilde)) * exp(eta * Y[i][j]) / (1+exp(eta))
                    test *= exp(2*(eta_tilde-eta))
    return test


def MCMC_Florentine(alpha, Z, Y):
    Z_tilde = np.random.normal(Z, 5*np.identity(len(Z)))
    test = test_Z(alpha, Z_tilde, Z, Y)
    print(test)
    if np.random.random() < test :
        Z = Z_tilde
    alpha_tilde = np.random.exponential(2)
    test = test_alpha(alpha_tilde, alpha, Z, Y)
    print(test)
    if np.random.random() < test : 
        alpha = alpha_tilde
    return alpha, Z

def boucle(Y, taille):
    Z = np.random.normal(0, 1000*np.identity(15))
    alpha = np.random.exponential(2)
    for i in range(taille):
        alpha, Z = MCMC_Florentine(alpha, Z, Y)
    return alpha, Z

boucle(relations, 100000)