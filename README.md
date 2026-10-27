# Prévision du trafic routier — mini-projet DAML

> **Document en cours de rédaction.** Les sections marquées *À compléter* seront remplies au fur et à mesure des expériences.

Mini-projet du cours *Data Analytics & Machine Learning* (CentraleSupélec, dominante ROC, 2026-2027).

## 1. Le projet

**Objectif** : prédire le débit de véhicules mesuré par 36 capteurs routiers **15 minutes à l'avance**, à partir de l'historique récent du capteur, du moment (jour, heure) et des caractéristiques de la route.

**Formulation** : régression supervisée. Pour chaque instant t et chaque capteur i, on apprend une fonction f telle que

  ŷᵢ(t+1) = f(xᵢ(t)),  avec xᵢ(t) ∈ ℝ⁴⁸ (variables décrites en section 2)

en minimisant l'erreur quadratique. Chaque couple (instant, capteur) est un exemple : environ 45 000 exemples d'entraînement et 30 000 de test.

**Référence à battre** : la *persistance*, ŷ(t+1) = y(t). À 15 minutes, le trafic varie peu et cette référence est déjà forte. Un modèle n'a d'intérêt que s'il fait mieux.

**Métriques** : RMSE et MAE, dans l'unité du débit normalisé (entre 0 et 1).

## 2. Le dataset

**Source** : *Traffic Flow Forecasting*, UCI Machine Learning Repository. Débits mesurés toutes les 15 minutes par 36 capteurs sur deux autoroutes de la région de Washington D.C. / Virginie du Nord.

> L. Zhao, O. Gkountouna, D. Pfoser (2019). *Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints.* ACM Trans. Spatial Algorithms Syst. 5(3), 19. https://doi.org/10.1145/3339823

**Contenu de `data/traffic_dataset.mat`**

| Variable | Forme | Contenu |
|---|---|---|
| `tra_X_tr` | 1 × 1261 cellules, chacune 36 × 48 (creuse) | variables d'entrée, entraînement |
| `tra_Y_tr` | 36 × 1261 | cible, entraînement |
| `tra_X_te` | 1 × 840 cellules, chacune 36 × 48 | variables d'entrée, test |
| `tra_Y_te` | 36 × 840 | cible, test |
| `tra_adj_mat` | 36 × 36 | matrice d'adjacence entre capteurs |

Chaque cellule correspond à **un instant** : une ligne par capteur, une colonne par variable.

**Les 48 variables (indices Python)**

| Colonnes | Variable | Varie selon |
|---|---|---|
| 0–9 | débits passés, de t−9 (col. 0) à t (col. 9) | instant et capteur |
| 10–14 | jour de la semaine, one-hot (lundi à vendredi) | instant |
| 15–38 | heure de la journée, one-hot (0 à 23 h) | instant |
| 39 | nombre de voies (1 à 5) | capteur |
| 40–43 | direction, one-hot | capteur |
| 44–47 | route, one-hot | capteur |

Les débits sont **déjà normalisés** entre 0 et 1 dans le fichier fourni.

## 3. Écarts avec la documentation et retraitement

La documentation du dataset est courte et ne correspond pas exactement au fichier. Avant de modéliser, on a donc vérifié la structure réelle des données, ligne par ligne :

1. **48 variables, et non 47.** La documentation annonce 7 colonnes pour le jour et 1 pour le nom de la route. Le fichier contient **5 colonnes de jour** et **4 colonnes one-hot de route**, d'où 10 + 5 + 24 + 1 + 4 + 4 = 48.

2. **Aucun week-end.** Seuls les jours ouvrés sont présents : 96 instants par jour (4 × 24), du lundi au vendredi. Le jeu d'entraînement couvre environ 13 jours ouvrés, le jeu de test environ 8,75.

3. **La cible fournie est à t+2 (30 min), et non à t+1 (15 min).** La documentation indique *« predict the traffic volume 15 minutes into the future »*. Pourtant, d'une ligne à la suivante, les 10 débits passés se décalent exactement d'un cran (une ligne = 15 min), et `tra_Y(:, k)` est égal au débit « t » de la ligne **k+2** sur 100 % des lignes, en entraînement comme en test. Ce n'est le cas pour la ligne k+1 que dans moins de 1 % des cas (des coïncidences).

4. **Le test suit directement l'entraînement.** La première ligne de test prolonge la dernière ligne d'entraînement, et la relation ci-dessus reste vraie à la jonction.

5. **L'horodatage est imparfait, mais pas les débits.** La série des débits est continue (aucune rupture dans le décalage des retards). En revanche, les variables jour et heure sont parfois irrégulières : certaines heures ne comptent que 3 instants, certains jours 94 au lieu de 96, et le test commence au milieu d'une journée.

**Retraitement** : `src/make_dataset.py` génère `data/traffic_dataset_t1.npz`, où **X est inchangé** et la cible est recalculée à **t+1** :

  Y[k] = X[k+1, :, 9]  (débit « t » de l'instant suivant)

Pour la dernière ligne, on prend `Y[T-2]` du `.mat` d'origine, qui contient exactement ce débit. Aucune ligne n'est donc perdue. Ce fichier n'est pas versionné (`.gitignore`) : il est régénéré automatiquement par les notebooks s'il est absent.

**Visualisation** : `docs/traffic_dataset_t1_explorateur.html` affiche toutes les données de ce fichier, capteur par capteur (48 variables et la cible t+1, entraînement et test). Il s'ouvre dans n'importe quel navigateur, sans installation.

## 4. Organisation du dépôt

```
final-project/
├── data/
│   ├── traffic_dataset.mat          # données d'origine
│   └── traffic_dataset_t1.npz       # généré : cible à t+1 (non versionné)
├── docs/
│   └── traffic_dataset_t1_explorateur.html   # visualisation des variables
├── notebooks/
│   ├── 00_eda_preprocessing.ipynb   # exploration des données (à compléter)
│   ├── 01_notebook_model.ipynb      # forêt aléatoire
│   └── 02_mlp.ipynb                 # réseau de neurones (MLP)
├── src/
│   ├── data.py                      # chargement, découpage, normalisation, variante spatiale
│   ├── make_dataset.py              # génération du dataset t+1
│   └── utils.py                     # métriques (MAE, RMSE), journal des résultats
└── tests/
    └── check_setup.py               # vérifications automatiques du pipeline de données
```

**Protocole commun à tous les modèles** (`src/data.build`) :
- découpage **chronologique** : train = 80 % premiers instants du jeu d'entraînement, validation = 20 % suivants, test = jeu de test d'origine, utilisé une seule fois à la fin ;
- normalisation (`StandardScaler`) apprise sur le train uniquement ;
- variante `spatial` : ajoute, pour chaque capteur, la moyenne des 10 débits passés de ses voisins (matrice d'adjacence), soit 58 variables.

## 5. Mode d'emploi

*À compléter (versions exactes, commandes testées).*

```bash
pip install numpy scipy scikit-learn pandas matplotlib jupyter
cd final-project
python src/make_dataset.py        # crée data/traffic_dataset_t1.npz
python tests/check_setup.py       # vérifie le pipeline de données
jupyter notebook notebooks/       # puis exécuter les notebooks dans l'ordre
```

Réglages modifiables :
- `02_mlp.ipynb` : nombre de neurones `h`, activation `act`, variante (`base` / `spatial`), cible (`debit` / `variation`), graines `SEEDS`.

## 6. Modèles et résultats

*À compléter.*

| Modèle | Variante | RMSE val | RMSE test | MAE test |
|---|---|---|---|---|
| Persistance | — | | | |
| Régression linéaire | | | | |
| Forêt aléatoire | | | | |
| MLP | | | | |

## 7. Limites et perspectives

*À compléter.* Pistes déjà identifiées :
- chaque exemple (instant, capteur) est traité indépendamment : l'historique n'est vu qu'à travers les 10 débits passés fournis ;
- la topologie du réseau routier n'est exploitée que par une moyenne des voisins, alors que l'article d'origine en fait le cœur de la méthode ;
- modèles séquentiels (LSTM) ou sur graphe, et prédiction à plusieurs pas (t+1, t+2…) pour mesurer la dégradation avec l'horizon.
