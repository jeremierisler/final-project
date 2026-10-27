"""Construit un dataset avec label à t+1 à partir de traffic_dataset.mat.

X est inchangé (48 colonnes : 0-9 = trafic t-9..t | 10-14 = jour lun-ven | 15-38 = heure
| 39 = nb voies | 40-43 = direction | 44-47 = route). Dans le .mat, Y[t] = trafic à t+2 ;
ici Y[t] = trafic à t+1.
"""
from pathlib import Path
import numpy as np

try:
    from .data import load_raw
except ImportError:
    from data import load_raw

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "traffic_dataset_t1.npz"


def fix_label(X, Y):
    """Y[t] = v(t+2) -> v(t+1) : valeur à t de l'entrée suivante, et Y[T-2] pour la dernière."""
    return np.concatenate([X[1:, :, 9], Y[-2:-1]])


def main(out=OUT_PATH):
    X_tr, Y_tr, X_te, Y_te, A = load_raw()
    assert np.allclose(Y_tr[:-2], X_tr[2:, :, 9]) and np.allclose(Y_te[:-2], X_te[2:, :, 9])
    Ytr, Yte = fix_label(X_tr, Y_tr), fix_label(X_te, Y_te)
    np.savez_compressed(out, X_tr=X_tr, Y_tr=Ytr, X_te=X_te, Y_te=Yte, A=A)
    print(f"{out} : X_tr {X_tr.shape}, Y_tr {Ytr.shape}, X_te {X_te.shape}, Y_te {Yte.shape}")


if __name__ == "__main__":
    main()
