import numpy as np
from scipy.optimize import minimize

def vraisemblance(D, Y):
    D = np.reshape(D,np.shape(Y))
    vraissemblance = 0
    for i in range (np.shape(Y)[0]):
        for j in range (np.shape(Y)[0]):
            if i != j:
                vraissemblance += Y[i][j] * (1-D[i][j]) - np.log(1+np.exp(1-D[i][j]))
    return - vraissemblance

def contraintes(D):
    n = n = int(np.sqrt(len(D)))
    D = np.reshape(D,(n,n))
    contraintes = []
    for i in range(np.shape(D)[0]):
        for j in range(np.shape(D)[0]):
            for k in range(np.shape(D)[0]):
                print(D)
                contrainte = {'type': 'ineq', 'fun': lambda D, i=i, j=j, k=k: D[i][j]+D[j][k]-D[i][k]}
                contraintes.append(contrainte)
            contrainte = {'type': 'ineq', 'fun': lambda D, i=i, j=j: D[i][j]}
            contraintes.append(contrainte)
    return contraintes

def MLE(Y):
    D = np.zeros(np.shape(Y)[0]**2)
    resultat = minimize(vraisemblance, D, args=(Y,), constraints=contraintes(D), method='SLSQP')
    D = resultat.x.reshape((n, n))
    return D


relations = np.loadtxt("data/Florentine_families.csv", delimiter=',')

D = MLE(relations)
print(D)
