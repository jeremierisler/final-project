# Prévision du trafic routier à court terme

Mini-projet *Data Analytics & Machine Learning*, CentraleSupélec, dominante ROC, 2026-2027. **Groupe 19** : Madeleine Lemaistre, Jérémie Risler, Thibault Millet, Enzo Thessieu, Arthur de Kermel.

Nous prédisons le débit de 36 capteurs autoroutiers **15 et 30 minutes à l'avance**, à partir de la base UCI *Traffic Flow Forecasting*, et comparons six familles de modèles sur un protocole commun. Ce README est le mode d'emploi du code. L'analyse complète est dans le rapport.

## Sommaire

1. [Démarrage rapide](#1-démarrage-rapide)
2. [Organisation du dépôt](#2-organisation-du-dépôt)
3. [Les notebooks](#3-les-notebooks)
4. [Tester d'autres réglages](#4-tester-dautres-réglages)
5. [Données et protocole](#5-données-et-protocole)
6. [Résultats](#6-résultats)
7. [Remarques](#7-remarques)

## 1. Démarrage rapide

Le code a été testé avec **Python 3.11.5** sous Windows. Toutes les commandes se lancent depuis la racine du dépôt.

```bash
# 1. Installer les dépendances
pip install numpy scipy scikit-learn pandas matplotlib plotly nbformat xgboost jupyter

# 2. Créer la cible à 15 minutes (data/traffic_dataset_t1.npz)
python src/make_dataset.py

# 3. Vérifier que tout fonctionne
python tests/check_setup.py        # pipeline de données
python tests/check_var_graphe.py   # modèle VAR

# 4. Ouvrir les notebooks
jupyter notebook notebooks/
```

Chaque test affiche `TOUS LES TESTS AUTOMATIQUES PASSENT` en cas de succès. L'étape 2 est facultative : les notebooks créent ce fichier eux-mêmes s'il manque.

Exécutez ensuite un notebook du début à la fin (*Run All*). Ils sont indépendants les uns des autres et peuvent être lancés dans n'importe quel ordre. Ouvrez-les depuis le dossier `notebooks/` : ils retrouvent le code de `src/` par un chemin relatif.

<details>
<summary>Versions testées</summary>

| Paquet | Version |
|---|---|
| numpy | 1.26.1 |
| scipy | 1.13.1 |
| scikit-learn | 1.5.2 |
| pandas | 2.1.3 |
| matplotlib | 3.8.1 |
| plotly | 5.18.0 |
| nbformat | 5.11.1 |
| xgboost | 3.2.0 |

`nbformat` est nécessaire pour afficher les graphiques Plotly dans un notebook.
</details>

## 2. Organisation du dépôt

```text
final-project/
├── data/
│   └── traffic_dataset.mat              données d'origine (UCI)
├── notebooks/                           un notebook par modèle (section 3)
├── src/
│   ├── data.py                          chargement, découpage chronologique, normalisation, moyenne des voisins
│   ├── make_dataset.py                  construction de la cible à t+1 (15 min)
│   ├── utils.py                         métriques (RMSE, MAE) et enregistrement des scores
│   ├── var_graphe.py                    modèle VAR contraint par le graphe
│   ├── data_jerem.py                    variante de data.py utilisée par les notebooks 00 et 07
│   └── utils_jerem.py                   variante de utils.py utilisée par les notebooks 00 et 07
├── tests/
│   ├── check_setup.py                   tests automatiques du pipeline de données (data_jerem.py)
│   ├── check_var_graphe.py              tests automatiques du modèle VAR
│   ├── affichage.py, test_y.py          scripts d'exploration du fichier .mat
├── results/                             scores enregistrés par les notebooks, un CSV par modèle
└── docs/
    ├── traffic_dataset_t1_explorateur.html   toutes les variables, capteur par capteur (s'ouvre dans un navigateur)
    └── section_mlp.tex, figures/        section MLP du rapport et sa figure
```

## 3. Les notebooks

Les durées ont été mesurées sur un ordinateur portable.

| Notebook | Contenu | Auteur | Durée |
|---|---|---|---|
| `00_eda_preprocessing` | exploration des données, références (moyenne, persistance), premiers modèles | Jérémie | < 1 min |
| `01_notebook_model` | première forêt aléatoire, cible à 30 min ; remplacé par `07_arbres` | Jérémie | < 1 min |
| `02_mlp` | réseau de neurones à une couche cachée : variantes, nombre de neurones, activation, ensemble de 5 graines | Thibault | **long** : plusieurs dizaines de minutes |
| `03_regression_lineaire` | moindres carrés : modèle commun, modèle par capteur, ajout des voisins | Madeleine | ≈ 2 min |
| `04_var_graphe` | VAR contraint par le graphe : réglage de $K$ et $\alpha$, matrice d'influence, test aux deux horizons | Arthur | < 1 min |
| `07_arbres` | forêt aléatoire et XGBoost : recherche de réglages, validation glissante, importance des variables, test | Jérémie | ≈ 16 min |
| `09_SADL_optimisation` | SADL-I, méthode de l'article d'origine, optimisée par ADMM | Enzo | ≈ 7 min |

Chaque notebook suit le protocole de la section 5 : réglage sur la validation, puis une seule évaluation sur le test, à 15 et 30 minutes.

## 4. Tester d'autres réglages

Modifiez la variable indiquée, puis relancez le notebook en entier.

| Notebook | Où modifier | Exemple |
|---|---|---|
| `02_mlp` | cellule *Préparation* : `SEEDS` ; arguments de `run(...)` : `variant` (`base` / `spatial`), `cible` (`debit` / `variation`), `h` (neurones), `act` (`tanh` / `relu`) | `run("spatial", "variation", h=50)` |
| `03_regression_lineaire` | sections *Modèle local* et *Prise en compte des voisins* : nombre de sauts dans le graphe | voisinage à 3 sauts |
| `04_var_graphe` | cellule des fonctions utilitaires : `voisinages` et `alphas` ; section *Lecture du modèle* : `capteur` | `"3 sauts": (A, 3)` dans `voisinages` |
| `07_arbres` | section *Recherche de réglages* : grille de la forêt (`n`, profondeur, feuille minimale, fraction de variables) et tirages d'XGBoost | ajouter 500 arbres à la grille |
| `09_SADL_optimisation` | cellule d'optimisation : `sigma` et `lamda` ; balayages de $\sigma$ et $\lambda$ dans les cellules commentées en fin de notebook | `sigma = 5.0` |

Dans `04_var_graphe`, la meilleure configuration est sélectionnée automatiquement sur la validation : il suffit d'ajouter des valeurs.

Pour un usage hors notebook, le modèle VAR s'emploie en quatre lignes :

```python
from src.data import load_raw, T1_PATH
from src import var_graphe

X_tr, Y_tr, X_te, Y_te, A = load_raw(T1_PATH)                     # tableaux (instants, capteurs, variables)
models, hoods = var_graphe.fit(X_tr, Y_tr, A, K=2, alpha=0.3)     # une régression Ridge par capteur
Y_pred = var_graphe.predict(X_te, models, hoods)                  # (instants, 36)
```

## 5. Données et protocole

**Données.** Le fichier `data/traffic_dataset.mat` contient 1 261 instants d'entraînement et 840 instants de test, qui les prolongent directement, pour 36 capteurs. Un instant correspond à 15 minutes. Pour chaque instant et chaque capteur, il y a 48 variables :

| Colonnes | Variable |
|---|---|
| 0–9 | 10 derniers débits, de $t-9$ à $t$ (colonne 9 = débit actuel) |
| 10–14 | jour de la semaine, lundi à vendredi (one-hot) |
| 15–38 | heure de la journée (one-hot) |
| 39 | nombre de voies |
| 40–47 | direction et route (one-hot) |

Le fichier contient aussi la matrice d'adjacence `A` (36 × 36) entre capteurs.

**Écart avec la documentation UCI.** La cible fournie est le débit à $t+2$, soit **30 minutes**, et non 15 minutes comme annoncé. `src/make_dataset.py` reconstruit une cible à $t+1$ (15 minutes) sans perdre d'instant :

| Horizon | Fichier | Constante dans `src/data.py` |
|---|---|---|
| 15 min | `data/traffic_dataset_t1.npz` (généré) | `T1_PATH` |
| 30 min | `data/traffic_dataset.mat` (d'origine) | `DATA_PATH` |

**Protocole commun.**
- **Découpage chronologique :** les 80 % premiers instants d'entraînement servent à l'apprentissage, les 20 % suivants à la validation (choix des réglages), et le jeu de test n'est utilisé qu'une fois, après réentraînement sur entraînement + validation.
- **Pas de mélange :** deux instants consécutifs partagent 9 débits sur 10, donc mélanger les lignes fausserait l'évaluation.
- **Métriques :** RMSE et MAE, en débit normalisé (entre 0 et 1).
- **Référence :** la *persistance*, qui prédit le dernier débit connu.