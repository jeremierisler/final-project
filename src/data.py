from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

ROOT = Path(__file__).resolve().parent.parent  #on remonte au dossier chemin du dossier projet 3A
DATA_PATH = ROOT / "data" / "traffic_dataset.mat" #chemin et nom du doc avec les datas d'entrées matlab


def _dense(x):
    """Convertit une matrice creuse (on note pas les 0) en tableau classique, 
    car les matrices de datas X sont creuses"""
    return (x.toarray() if issparse(x) else np.asarray(x)).astype(float)
a
def _stack(cells):
    """Cell array MATLAB (1, T) de matrices (36, 48) -> ndarray (T, 36, 48)."""
    """transforme les tableaux bizarre matlab en tableau numpy"""
    return np.stack([_dense(x) for x in cells.squeeze()])


def load_raw(path=DATA_PATH):
    """charge les données matlab du fichier à DATA_PATH, renvoie le dico m qui affilie au nom le tableau de données"""
    m = loadmat(str(path))
    X_tr, X_te = _stack(m["tra_X_tr"]), _stack(m["tra_X_te"])  # (T, 36, 48)
    #on utilise stack car forme bizarre, _dense inclue dans _stack
    Y_tr = _dense(m["tra_Y_tr"]).T                             # (T, 36)
    #T ie transposée, normalement déjà dense mais bon
    Y_te = _dense(m["tra_Y_te"]).T
    #same
    A = _dense(m["tra_adj_mat"])                               # (36, 36)
    assert X_tr.shape[1:] == (36, 48) and Y_tr.shape == X_tr.shape[:2]
    #assert verifie que les données sont cohérentes askip
    return X_tr, Y_tr, X_te, Y_te, A


def flatten(X, Y):
    #transforme les données pour qu'elles soient adaptées à un modèle de scikit-learn
    T, S, F = X.shape
    sensor = np.tile(np.arange(S), T)
    return X.reshape(T * S, F), Y.reshape(T * S), sensor


def train_val_split(X, Y, val_frac=0.2):
   cut = int(len(X) * (1 - val_frac))
    return (X[:cut], Y[:cut]), (X[cut:], Y[cut:])


# def add_neighbors(X, A, n_lags=10):
#     #ajoute la prise en compte des voisins pour chaque capteur, en utilisant la matrice d'adjacence A
#     A_norm = A / np.maximum(A.sum(1, keepdims=True), 1)
#     nb = np.einsum("ij,tjf->tif", A_norm, X[:, :, :n_lags])
#     return np.concatenate([X, nb], axis=2)


def build( path=DATA_PATH, val_frac=0.2):
    X_tr, Y_tr, X_te, Y_te, A = load_raw(path)
    # if variant == "spatial":
    #     X_tr, X_te = add_neighbors(X_tr, A), add_neighbors(X_te, A)
    (Xa, Ya), (Xv, Yv) = train_val_split(X_tr, Y_tr, val_frac) #séparation apprentissage et validation
    Xa, Ya, _ = flatten(Xa, Ya)
    Xv, Yv, _ = flatten(Xv, Yv)
    # if scaled:
    #     sc = StandardScaler().fit(X_tr)
    #     X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    return dict(X_train=Xa, y_train=Ya, X_val=Xv, y_val=Yv)