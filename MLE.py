import numpy as np
from sklearn.manifold import MDS
import pandas as pd
import matplotlib.pyplot as plt

def distance_graphe(Y):
    """
    Calcule la distance entre les noeuds comme le nombre minimal
    d'arêtes entre les noeuds
    (renvoie n=nombre de sommets si pas de lien)
    """
    n = np.shape(Y)[0]
    # On détermine notre Y_{1}
    Yr = np.copy(Y)
    # On modifie progressivement notre distance : on a déjà les noeuds à distance de 1 les uns des autres
    D = np.copy(Y)
    D = Y * (Y == 1) + n * (Y == 0)
    for r in range(2, n):
        # On détermine notre Y_{r+1}
        Yr = np.dot(Yr, Y)
        # On modifie notre distance pour les paires de noeuds n'ayant pas de distance plus courte
        D += (r - n) * ((Yr > 0) & (D == n))
    # On met bien des 0 sur la diagonale de D
    np.fill_diagonal(D, 0)
    return D

def MLE(Y, k=2):
    """
    Renvoie une liste de point de dimension k correpondant au MLE de la matrice Y
    """
    D = distance_graphe(Y)
    Z = np.linalg.svd(D)[0][:, :k] # On 
    Z = Z - np.mean(Z, axis=0) # on centre le nuage de points
    return Z

"""
data = np.loadtxt("data/Florentine_families.csv", delimiter=',')

Z = MLE(data)
print(Z)
colors = plt.cm.rainbow(np.linspace(0, 1, 15))
for i in range (np.shape(Z)[0]):
    plt.scatter(Z[i, 0], Z[i, 1], c=colors[i], cmap='jet', label=f'Trajectoire {i+1}')
plt.legend()
plt.show()
"""