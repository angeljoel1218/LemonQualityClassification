"""
remuestreo.py
-------------
Variante de DATOS: balanceo de clases por remuestreo, aplicado SOLO al conjunto de entrenamiento.

`ConRemuestreo` envuelve a un clasificador y remuestrea dentro de `fit`. Como en validación cruzada
`fit` solo recibe el pliegue de entrenamiento, el remuestreo nunca contamina validación ni test
(el error clásico es balancear TODO el dataset antes de partir: las copias de un mismo fruto
quedarían a ambos lados y las métricas saldrían infladas).

    sobre : duplica al azar frutos de las clases minoritarias hasta igualar a la mayoritaria
    sub   : descarta al azar frutos de las clases mayoritarias hasta igualar a la minoritaria
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.metaestimators import available_if


def indices_remuestreo(y, modo: str, semilla: int = 0) -> np.ndarray:
    y = np.asarray(y)
    clases, cuentas = np.unique(y, return_counts=True)
    objetivo = cuentas.max() if modo == "sobre" else cuentas.min()
    rng = np.random.default_rng(semilla)
    idx = []
    for c, n in zip(clases, cuentas):
        ic = np.where(y == c)[0]
        if modo == "sobre":
            extra = rng.choice(ic, size=objetivo - n, replace=True) if objetivo > n else np.array([], int)
            idx.append(np.concatenate([ic, extra]))
        elif modo == "sub":
            idx.append(rng.choice(ic, size=objetivo, replace=False))
        else:
            raise ValueError("modo debe ser 'sobre' o 'sub'")
    return np.concatenate(idx)


class ConRemuestreo(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, modo="sobre", semilla=0):
        self.estimator, self.modo, self.semilla = estimator, modo, semilla

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        idx = indices_remuestreo(y, self.modo, self.semilla)
        self.estimator_ = clone(self.estimator).fit(X[idx], y[idx])
        self.classes_ = self.estimator_.classes_
        self.n_entrenamiento_ = len(idx)
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X))

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))
