import os, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.data import load_raw, flatten, train_val_split, add_neighbors, build
from src.utils import evaluate, log_result

ok = lambda msg: print("✓", msg)
X_tr, Y_tr, X_te, Y_te, A = load_raw()

# 1. formes et valeurs
assert X_tr.shape == (1261, 36, 48) and X_te.shape == (840, 36, 48)
assert Y_tr.shape == (1261, 36) and Y_te.shape == (840, 36) and A.shape == (36, 36)
for n, a in dict(X_tr=X_tr, X_te=X_te, Y_tr=Y_tr, Y_te=Y_te, A=A).items():
    assert np.isfinite(a).all(), f"NaN/inf dans {n}"
ok("formes et valeurs finies")

# 2. flatten garde X, y et capteur alignés
Xf, yf, s = flatten(X_tr, Y_tr)
for t, i in [(0, 0), (5, 7), (1260, 35)]:
    assert np.allclose(Xf[t*36 + i], X_tr[t, i]) and yf[t*36 + i] == Y_tr[t, i] and s[t*36 + i] == i
ok("flatten aligné")

# 3. split chronologique, sans mélange ni perte
(Xa, Ya), (Xv, Yv) = train_val_split(X_tr, Y_tr)
assert len(Xa) + len(Xv) == 1261 and np.array_equal(np.vstack([Ya, Yv]), Y_tr)
ok("split train/val")

# 4. voisins : recalcul à la main pour un capteur et un instant
Xn = add_neighbors(X_tr, A)
assert Xn.shape == (1261, 36, 58) and np.allclose(Xn[:, :, :48], X_tr)
i, t = 3, 10
assert np.allclose(Xn[t, i, 48:], A[i] @ X_tr[t, :, :10] / max(A[i].sum(), 1))
ok("add_neighbors")

# 5. build : tailles et test intact
d = build("base")
assert d["X_train"].shape == (36288, 48) and d["X_val"].shape == (9108, 48)
assert d["X_test"].shape == (30240, 48) and d["X_trainval"].shape == (45396, 48)
assert build("spatial")["X_train"].shape == (36288, 58)
assert np.array_equal(d["y_test"], Y_te.reshape(-1))
ok("build")

# 6. scaler : ajusté sur le train uniquement
ds = build("base", scaled=True)
sd = ds["X_train"].std(0)
assert np.allclose(ds["X_train"].mean(0), 0, atol=1e-6) and np.allclose(sd[sd > 0], 1)
assert np.abs(ds["X_test"].mean(0)).max() > 1e-6   # le test n'a pas servi au fit
ok("scaler")

# 7. métriques sur des cas calculés à la main
y = np.array([1., 2., 3.])
assert evaluate(y, y) == {"MAE": 0.0, "RMSE": 0.0}
m = evaluate(np.array([0., 0.]), np.array([1., 3.]))
assert np.isclose(m["MAE"], 2) and np.isclose(m["RMSE"], np.sqrt(5))
ok("evaluate")

# 8. log_result : virgule dans params, en-tête écrit une seule fois
log_result("_test", "base", "n=200,depth=10", "val", {"MAE": 1.0, "RMSE": 2.0})
log_result("_test", "spatial", "n=300", "val", {"MAE": 1.5, "RMSE": 2.5})
df = pd.read_csv("results/_test.csv")
assert len(df) == 2 and df.loc[0, "params"] == "n=200,depth=10"
os.remove("results/_test.csv")
ok("log_result")
print("TOUS LES TESTS AUTOMATIQUES PASSENT")