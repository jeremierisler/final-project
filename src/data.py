from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "traffic_dataset.mat"

def shift_target(X, Y, horizon):
    """horizon=2 : cible fournie (t+2). horizon=1 : volume à t+1 (= Y[i-1]), on perd le 1er instant."""
    if horizon == 2:
        return X, Y
    if horizon == 1:
        return X[1:], Y[:-1]
    raise ValueError("horizon doit valoir 1 ou 2")

def _dense(x):
    """Convertit un objet sparse ou un tableau en ndarray float."""
    return (x.toarray() if issparse(x) else np.asarray(x)).astype(float)


def _stack(cells):
    """Cell array MATLAB (1, T) de matrices (36, 48) -> ndarray (T, 36, 48)."""
    return np.stack([_dense(x) for x in cells.squeeze()])


def load_raw(path=DATA_PATH):
    m = loadmat(str(path))
    X_tr, X_te = _stack(m["tra_X_tr"]), _stack(m["tra_X_te"])  # (T, 36, 48)
    Y_tr = _dense(m["tra_Y_tr"]).T                             # (T, 36)
    Y_te = _dense(m["tra_Y_te"]).T
    A = _dense(m["tra_adj_mat"])                               # (36, 36)
    assert X_tr.shape[1:] == (36, 48) and Y_tr.shape == X_tr.shape[:2]
    return X_tr, Y_tr, X_te, Y_te, A


def flatten(X, Y):
    T, S, F = X.shape
    sensor = np.tile(np.arange(S), T)
    return X.reshape(T * S, F), Y.reshape(T * S), sensor


def train_val_split(X, Y, val_frac=0.2):
    cut = int(len(X) * (1 - val_frac))
    return (X[:cut], Y[:cut]), (X[cut:], Y[cut:])


def add_neighbors(X, A, n_lags=10):
    A_norm = A / np.maximum(A.sum(1, keepdims=True), 1)
    nb = np.einsum("ij,tjf->tif", A_norm, X[:, :, :n_lags])
    return np.concatenate([X, nb], axis=2)


def add_sensor(X, sensor, mode):
    if mode == "none":
        return X
    if mode == "int":                      # pour les arbres, RF et XGBoost
        return np.column_stack([X, sensor])
    if mode == "onehot":                   # pour linéaire, kNN, MLP
        return np.column_stack([X, np.eye(36)[sensor]])
    raise ValueError("sensor_id doit valoir 'none', 'int' ou 'onehot'")


def build(variant="base", scaled=False, val_frac=0.2, horizon=2, sensor_id="none", path=DATA_PATH):
    X_tr, Y_tr, X_te, Y_te, A = load_raw(path)
    X_tr, Y_tr = shift_target(X_tr, Y_tr, horizon)
    X_te, Y_te = shift_target(X_te, Y_te, horizon)
    if variant == "spatial":
        X_tr, X_te = add_neighbors(X_tr, A), add_neighbors(X_te, A)
    (Xa, Ya), (Xv, Yv) = train_val_split(X_tr, Y_tr, val_frac)
    Xa, ya, sa = flatten(Xa, Ya)
    Xv, yv, sv = flatten(Xv, Yv)
    Xt, yt, st = flatten(X_te, Y_te)
    Xa, Xv, Xt = add_sensor(Xa, sa, sensor_id), add_sensor(Xv, sv, sensor_id), add_sensor(Xt, st, sensor_id)
    if scaled:
        sc = StandardScaler().fit(Xa)
        Xa, Xv, Xt = sc.transform(Xa), sc.transform(Xv), sc.transform(Xt)
    return dict(X_train=Xa, y_train=ya, X_val=Xv, y_val=yv, X_test=Xt, y_test=yt,
                X_trainval=np.vstack([Xa, Xv]), y_trainval=np.concatenate([ya, yv]))
