"""
seleccion.py
------------
Selección/diagnóstico de variables. Todo se calcula SOLO con train (salvo la importancia por
permutación, que se mide en validación):

    ranking_mi              información mutua X vs Y        (relevancia, no lineal)
    anova_f                 estadístico F X vs Y            (relevancia lineal entre clases)
    pares_correlacionados   |r| entre X1 y X2               (redundancia)
    vif / vif_iterativo     factor de inflación de la varianza (colinealidad)
    importancia_permutacion caída de F1 macro al barajar una columna (sobre validación)
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance


def imputar_con_train(X_train: pd.DataFrame) -> pd.DataFrame:
    """Imputa con la mediana de train (para diagnóstico). Columnas totalmente vacías -> 0."""
    return X_train.fillna(X_train.median(numeric_only=True)).fillna(0.0)


def ranking_mi(X: pd.DataFrame, y, semilla: int = 0) -> pd.Series:
    mi = mutual_info_classif(imputar_con_train(X), y, random_state=semilla)
    return pd.Series(mi, index=X.columns, name="info_mutua").sort_values(ascending=False)


def anova_f(X: pd.DataFrame, y) -> pd.Series:
    Xi = imputar_con_train(X)
    cte = Xi.std() == 0
    f = pd.Series(np.nan, index=X.columns, name="F")
    f[~cte] = f_classif(Xi.loc[:, ~cte], y)[0]
    return f.sort_values(ascending=False)


def pares_correlacionados(X: pd.DataFrame, umbral: float = 0.95) -> pd.DataFrame:
    r = imputar_con_train(X).corr().abs()
    filas = [(a, b, r.loc[a, b]) for i, a in enumerate(r.columns) for b in r.columns[i + 1:]
             if r.loc[a, b] > umbral]
    return pd.DataFrame(filas, columns=["var_1", "var_2", "|r|"]).sort_values("|r|", ascending=False)


def vif(X: pd.DataFrame) -> pd.Series:
    A = imputar_con_train(X).to_numpy(float)
    sd = A.std(axis=0)
    ok = sd > 1e-12
    Z = (A[:, ok] - A[:, ok].mean(axis=0)) / sd[ok]
    n, p = Z.shape
    out = np.full(p, np.nan)
    for j in range(p):
        y = Z[:, j]
        M = np.c_[np.ones(n), np.delete(Z, j, axis=1)]
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        res = y - M @ beta
        r2 = 1 - (res @ res) / max(y @ y, 1e-12)
        out[j] = min(1.0 / max(1 - r2, 1e-12), 1e6)
    s = pd.Series(np.nan, index=X.columns, name="VIF")
    s[X.columns[ok]] = out
    return s.sort_values(ascending=False)


def vif_iterativo(X: pd.DataFrame, umbral: float = 10.0):
    """Elimina, una a una, la variable con mayor VIF hasta que todas queden por debajo del umbral."""
    cols, eliminadas = list(X.columns), []
    while len(cols) > 2:
        v = vif(X[cols]).dropna()
        if v.empty or v.iloc[0] <= umbral:
            break
        eliminadas.append((v.index[0], float(v.iloc[0])))
        cols.remove(v.index[0])
    return cols, eliminadas


def importancia_permutacion(pipe, X_val: pd.DataFrame, y_val, n_repeats: int = 10, semilla: int = 0):
    """`pipe` ya ajustado con train; mide en validación cuánto cae el F1 macro al barajar cada columna."""
    r = permutation_importance(pipe, X_val, y_val, scoring="f1_macro",
                               n_repeats=n_repeats, random_state=semilla)
    return (pd.DataFrame({"caida_f1": r.importances_mean, "std": r.importances_std}, index=X_val.columns)
            .sort_values("caida_f1", ascending=False))
