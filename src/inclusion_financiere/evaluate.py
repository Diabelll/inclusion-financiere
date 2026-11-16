"""Métriques et résumés d'évaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import COST_FN, COST_FP

NEG_METRICS = {"brier", "logloss"}  # renvoyées en négatif par scikit-learn


def summarize(res: dict, name: str, metrics: list[str]) -> pd.Series:
    """Transforme le dictionnaire de cross_validate en une ligne de tableau.

    Moyenne et écart-type (ddof=1) de chaque métrique, en train et en validation,
    scores « neg_ » remis au signe positif, et temps d'entraînement.
    """
    row = {"model": name}
    for m in metrics:
        sign = -1 if m in NEG_METRICS else 1
        te = sign * np.asarray(res[f"test_{m}"])
        row[f"{m}_val_mean"] = te.mean()
        row[f"{m}_val_std"] = te.std(ddof=1)
        if f"train_{m}" in res:
            tr = sign * np.asarray(res[f"train_{m}"])
            row[f"{m}_train_mean"] = tr.mean()
            row[f"{m}_train_std"] = tr.std(ddof=1)
    row["fit_time_mean"] = np.mean(res["fit_time"])
    row["fit_time_std"] = np.std(res["fit_time"], ddof=1)
    return pd.Series(row)


def expected_calibration_error(y_true, p, n_bins=10):
    y_true, p = np.asarray(y_true), np.asarray(p)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return ece


def cout_total(y_true, y_pred, c_fp=COST_FP, c_fn=COST_FN):
    _tn, fp, fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp * c_fp + fn * c_fn


def seuil_theorique(c_fp=COST_FP, c_fn=COST_FN):
    """t* = C_FP / (C_FP + C_FN), valable si les probabilités sont calibrées."""
    return c_fp / (c_fp + c_fn)


def bootstrap_ci(y_true, p, metric, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, p = np.asarray(y_true), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        vals.append(metric(y_true[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])
