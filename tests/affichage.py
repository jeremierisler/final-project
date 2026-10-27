from scipy.io import loadmat
import pandas as pd

data = loadmat("data/traffic_dataset.mat")

# Afficher les noms des variables
print(data.keys())
df = pd.DataFrame(data["tra_X_tr"])
print(df.head(10))