"""VAR contrainte par le graphe : une régression Ridge par capteur (cf. docs/var_graphe.tex)."""
import numpy as np
from sklearn.linear_model import Ridge

N_LAGS = 10
CALENDAR = slice(10, 39)  # colonnes jour (5) + heure (24)


def neighborhoods(A, K):
    """Pour chaque capteur, liste des capteurs à au plus K sauts (lui-même compris)."""
    reach = np.linalg.matrix_power(A + np.eye(len(A)), K)
    return [np.flatnonzero(row) for row in reach]


def features(X, sensors):
    """(T, 36, 48) -> (T, p) : 10 débits passés de chaque capteur de `sensors` + calendrier."""
    lags = X[:, sensors, :N_LAGS].reshape(len(X), -1)
    return np.hstack([lags, X[:, 0, CALENDAR]])


def fit(X, Y, A, K=2, alpha=0.3):
    """Entraîne une Ridge par capteur, renvoie les modèles et les voisinages."""
    hoods = neighborhoods(A, K)
    models = [Ridge(alpha).fit(features(X, hood), Y[:, i]) for i, hood in enumerate(hoods)]
    return models, hoods


def predict(X, models, hoods):
    """Prédit tous les capteurs : (T, 36, 48) -> (T, 36)."""
    return np.column_stack([m.predict(features(X, hood)) for m, hood in zip(models, hoods)])


def influence(models, hoods):
    """Matrice (36, 36) : somme des |poids| du capteur j dans la prédiction du capteur i."""
    M = np.zeros((len(hoods), len(hoods)))
    for i, (m, hood) in enumerate(zip(models, hoods)):
        weights = m.coef_[:N_LAGS * len(hood)].reshape(len(hood), N_LAGS)
        M[i, hood] = np.abs(weights).sum(axis=1)
    return M
