"""
transformadores.py
------------------
Transformadores con ESTADO compatibles con sklearn. Todos aprenden sus parámetros en `fit`
(percentiles, asimetrías, límites IQR, correlaciones) y los aplican en `transform`.
Dentro de un sklearn.Pipeline, `fit` solo ve la data de entrenamiento (o el pliegue de train
en la validación cruzada), que es lo que evita la fuga de información.
"""
import numpy as np
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


def _nombres(input_features, n):
    return (np.asarray(input_features, dtype=object) if input_features is not None
            else np.array([f"x{i}" for i in range(n)], dtype=object))


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta cada columna a los percentiles [p_bajo, p_alto] APRENDIDOS EN TRAIN."""

    def __init__(self, p_bajo=5, p_alto=95):
        self.p_bajo, self.p_alto = p_bajo, p_alto

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.n_features_in_ = X.shape[1]
        self.lo_ = np.nanpercentile(X, self.p_bajo, axis=0)
        self.hi_ = np.nanpercentile(X, self.p_alto, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return _nombres(input_features, self.n_features_in_)


class Log1pAsimetricas(BaseEstimator, TransformerMixin):
    """Aplica log1p SOLO a las columnas no negativas con asimetría (skew) > umbral en train."""

    def __init__(self, umbral_asimetria=1.0):
        self.umbral_asimetria = umbral_asimetria

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.n_features_in_ = X.shape[1]
        cols = []
        for j in range(X.shape[1]):
            x = X[~np.isnan(X[:, j]), j]
            if len(x) > 3 and x.min() >= 0 and np.ptp(x) > 0 and skew(x) > self.umbral_asimetria:
                cols.append(j)
        self.cols_ = np.array(cols, dtype=int)
        return self

    def transform(self, X):
        X = np.array(X, float, copy=True)
        if len(self.cols_):
            X[:, self.cols_] = np.log1p(np.clip(X[:, self.cols_], 0, None))
        return X

    def get_feature_names_out(self, input_features=None):
        return _nombres(input_features, self.n_features_in_)


class MarcasOutlier(BaseEstimator, TransformerMixin):
    """Bandera 0/1 de atípico (regla IQR con límites aprendidos en train). Solo crea bandera para
    columnas con al menos `tasa_min` de atípicos en train. Devuelve ÚNICAMENTE las banderas."""

    def __init__(self, k=1.5, tasa_min=0.01):
        self.k, self.tasa_min = k, tasa_min

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.n_features_in_ = X.shape[1]
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lo_, self.hi_ = q1 - self.k * iqr, q3 + self.k * iqr
        tasa = np.nanmean((X < self.lo_) | (X > self.hi_), axis=0)
        self.cols_ = np.where((tasa >= self.tasa_min) & (iqr > 0))[0]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        c = self.cols_
        return ((X[:, c] < self.lo_[c]) | (X[:, c] > self.hi_[c])).astype(float)

    def get_feature_names_out(self, input_features=None):
        nom = _nombres(input_features, self.n_features_in_)
        return np.array([f"{nom[j]}_atipico" for j in self.cols_], dtype=object)


class CiclicoGrados(BaseEstimator, TransformerMixin):
    """Codifica ángulos en grados como (sen, cos): 359 y 1 grados quedan cerca (como diciembre-enero)."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        r = np.radians(np.asarray(X, float))
        return np.hstack([np.sin(r), np.cos(r)])

    def get_feature_names_out(self, input_features=None):
        nom = _nombres(input_features, self.n_features_in_)
        return np.array([f"{c}_sin" for c in nom] + [f"{c}_cos" for c in nom], dtype=object)


class FiltroColinealidad(BaseEstimator, TransformerMixin):
    """Entre cada par de columnas con |r| > umbral elimina la que menos información mutua tiene con
    el objetivo. Correlaciones e información mutua se calculan SOLO con train."""

    def __init__(self, umbral=0.95, semilla=0):
        self.umbral, self.semilla = umbral, semilla

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.n_features_in_ = X.shape[1]
        sd = X.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
        mi = (mutual_info_classif(X, y, random_state=self.semilla) if y is not None else sd)
        eliminar = set(np.where(sd == 0)[0])                     # constantes
        pares = [(abs(r[i, j]), i, j) for i in range(X.shape[1]) for j in range(i + 1, X.shape[1])
                 if abs(r[i, j]) > self.umbral]
        for _, i, j in sorted(pares, reverse=True):
            if i in eliminar or j in eliminar:
                continue
            eliminar.add(i if mi[i] < mi[j] else j)
        self.keep_ = np.array([j for j in range(X.shape[1]) if j not in eliminar], dtype=int)
        self.eliminadas_ = sorted(eliminar)
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return _nombres(input_features, self.n_features_in_)[self.keep_]
