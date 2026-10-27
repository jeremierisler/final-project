from scipy.io import loadmat
m = loadmat("data/traffic_dataset.mat")
for k, v in m.items():
    if not k.startswith("__"):
        first = v.ravel()[0] if hasattr(v, "ravel") and v.dtype == object else None
        print(k, type(v).__name__, getattr(v, "shape", None), getattr(v, "dtype", None),
              "| 1er élément :", type(first).__name__, getattr(first, "shape", None))