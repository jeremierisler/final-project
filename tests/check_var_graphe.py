import sys
sys.path.insert(0, ".")
import numpy as np
from src.data import load_raw
from src.var_graphe import neighborhoods, features, fit, predict, influence

ok = lambda msg: print("✓", msg)
X, Y, _, _, A = load_raw()
X, Y = X[:300], Y[:300]

# 1. voisinages : petit graphe en chaîne 0 - 1 - 2 - 3
C = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], float)
assert [list(h) for h in neighborhoods(C, 0)] == [[0], [1], [2], [3]]
assert [list(h) for h in neighborhoods(C, 1)] == [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3]]
assert list(neighborhoods(C, 2)[0]) == [0, 1, 2]
hoods = neighborhoods(A, 1)
assert list(hoods[4]) == [4, 5, 21, 25, 26, 31] and sum(map(len, hoods)) == 122
ok("neighborhoods")

# 2. features : débits des bons capteurs, puis calendrier identique partout
assert np.array_equal(X[:, 0, 10:39], X[:, 35, 10:39])
Z = features(X, hoods[4])
assert Z.shape == (300, 89)
assert np.array_equal(Z[7, 10:20], X[7, 5, :10]) and np.array_equal(Z[7, -29:], X[7, 0, 10:39])
ok("features")

# 3. fit retrouve un vrai modèle sans bruit : y_i = 2 v_i(t) - v_voisin(t) + 0.5
Y_fake = 2 * X[:, :, 9] + 0.5
Y_fake[:, 4] -= X[:, 5, 9]
models, hoods = fit(X, Y_fake, A, K=1, alpha=1e-8)
assert np.allclose(predict(X, models, hoods), Y_fake, atol=1e-5)
ok("fit / predict")

# 4. la contrainte : un capteur hors du voisinage a un poids nul
M = influence(*fit(X, Y, A, K=1, alpha=0.3))
assert M.shape == (36, 36) and np.array_equal(M > 0, (A + np.eye(36)) > 0)
ok("influence et contrainte du graphe")

# 5. K=1 sur la solution exacte de docs/var_graphe.tex
Zc = Z - Z.mean(0)
yc = Y[:, 4] - Y[:, 4].mean()
theta = np.linalg.solve(Zc.T @ Zc + 0.3 * np.eye(89), Zc.T @ yc)
models, _ = fit(X, Y, A, K=1, alpha=0.3)
assert np.allclose(models[4].coef_, theta, atol=1e-6)
ok("solution exacte de la Ridge")
print("TOUS LES TESTS AUTOMATIQUES PASSENT")
