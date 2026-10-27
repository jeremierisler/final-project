from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "traffic_dataset.mat"
T1_PATH = ROOT / "data" / "traffic_dataset_t1.npz"  # label à t+1, cf. make_dataset.py


def _dense(x):
    """Convertit un objet sparse ou un tableau en ndarray float."""
    return (x.toarray() if issparse(x) else np.asarray(x)).astype(float)


def _stack(cells):
    """Cell array MATLAB (1, T) de matrices (36, 48) -> ndarray (T, 36, 48)."""
    return np.stack([_dense(x) for x in cells.squeeze()])


def load_raw(path=DATA_PATH):
    if Path(path).suffix == ".npz":
        d = np.load(path)
        return d["X_tr"], d["Y_tr"], d["X_te"], d["Y_te"], d["A"]
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


def build(variant="base", scaled=False, val_frac=0.2, path=DATA_PATH):
    X_tr, Y_tr, X_te, Y_te, A = load_raw(path)
    if variant == "spatial":
        X_tr, X_te = add_neighbors(X_tr, A), add_neighbors(X_te, A)
    (Xa, Ya), (Xv, Yv) = train_val_split(X_tr, Y_tr, val_frac)
    Xa, ya, _ = flatten(Xa, Ya)
    Xv, yv, _ = flatten(Xv, Yv)
    Xt, yt, _ = flatten(X_te, Y_te)
    if scaled:
        sc = StandardScaler().fit(Xa)
        Xa, Xv, Xt = sc.transform(Xa), sc.transform(Xv), sc.transform(Xt)
    return dict(X_train=Xa, y_train=ya, X_val=Xv, y_val=yv, X_test=Xt, y_test=yt,
                X_trainval=np.vstack([Xa, Xv]), y_trainval=np.concatenate([ya, yv]))