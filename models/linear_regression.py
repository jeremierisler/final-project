import numpy as np

def solve_linear_regression(X, y):
    """X et y déjà sous la bonne forme via la fonction flatten"""
    X = np.column_stack((np.ones(X.shape[0]), X)) #on ajoute une colonne de 0 pour le offset

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return beta


def apply_linear_regression(X, beta):
    """prend un jeu d'entrée (deja bonne forme) et les coefficient et calcule la sortie
    """
    X = np.column_stack((np.ones(X.shape[0]), X))

    return X @ beta