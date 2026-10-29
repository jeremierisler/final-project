import csv
import os
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
def evaluate(y_true, y_pred):
    e = y_true - y_pred
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))}


def log_result(model, variant, params, split, metrics):
    os.makedirs(ROOT / "results", exist_ok=True)
    path = ROOT / "results" / f"{model}.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "model", "variant", "params", "split", "MAE", "RMSE"])
        w.writerow([time.strftime("%d/%m %H:%M"), model, variant, params, split,
                    metrics["MAE"], metrics["RMSE"]])