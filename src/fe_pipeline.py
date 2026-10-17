"""
fe_pipeline.py
--------------
Construye los sklearn.Pipeline de cada experimento. TODA transformación con estado vive dentro
del Pipeline, así que en validación cruzada se ajusta solo con el pliegue de entrenamiento
("divide primero, transforma después").

Base común a todos los experimentos (no se ablaciona): imputación por mediana + escalado estándar,
porque kNN y regresión logística no son comparables sin escalar (ver 'A_sin_escala' para medir su efecto).

    A        columnas originales
    +B       winsorización + log1p (asimétricas) + banderas de atípicos y de faltantes + ángulos cíclicos
    +C       features de dominio (se crean sin estado antes de partir)
    +S       filtro de colinealidad
"""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.fe_columnas import verificar_sin_fuga
from src.transformadores import (CiclicoGrados, FiltroColinealidad, Log1pAsimetricas,
                                 MarcasOutlier, Winsorizador)


def _imp():
    return SimpleImputer(strategy="median", keep_empty_features=True)


def crear_modelo(nombre: str, semilla: int, mp: dict):
    if nombre == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if nombre == "knn":
        return KNeighborsClassifier(n_neighbors=mp.get("knn_k", 5))
    if nombre == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=semilla)
    if nombre == "rf":
        return RandomForestClassifier(n_estimators=mp.get("rf_arboles", 300), class_weight="balanced",
                                      random_state=semilla, n_jobs=1)
    raise ValueError(f"Modelo desconocido: {nombre}")


def construir_pipeline(cols, conjuntos: str, modelo: str, semilla: int, cfg: dict,
                       escalar: bool = True, control_fuga: str | None = None) -> Pipeline:
    """
    cols       columnas numéricas de entrada (ya verificadas contra fuga)
    conjuntos  p. ej. 'A', 'A+B', 'A+C', 'A+B+C', 'A+B+C+S'
    control_fuga  None | 'onehot_lote' | 'te_lote'  -> SOLO para el experimento de control de fuga
    """
    letras = set(conjuntos.split("+"))
    tr = cfg["transformaciones"]
    verificar_sin_fuga(cols)

    usa_b = "B" in letras
    angulares = [c for c in cols if c.endswith("_deg")] if usa_b else []
    lineales = [c for c in cols if c not in angulares]

    pasos = [("imp", _imp())]
    if usa_b:
        pasos += [("winsor", Winsorizador(*tr["winsor"])),
                  ("log1p", Log1pAsimetricas(tr["umbral_asimetria"]))]
    if escalar:
        pasos.append(("esc", StandardScaler()))
    ramas = [("num", Pipeline(pasos), lineales)]
    if usa_b:
        ramas += [
            ("atip", Pipeline([("imp", _imp()), ("flags", MarcasOutlier(tr["outlier_k"], tr["outlier_tasa_min"]))]),
             lineales),
            ("falt", MissingIndicator(features="missing-only", error_on_new=False), lineales),
        ]
        if angulares:
            ramas.append(("ciclo", Pipeline([("imp", _imp()), ("cic", CiclicoGrados())]), angulares))
    if control_fuga == "onehot_lote":
        ramas.append(("lote", OneHotEncoder(handle_unknown="ignore"), ["lote"]))
    elif control_fuga == "te_lote":
        ramas.append(("lote", TargetEncoder(target_type="multiclass", random_state=semilla), ["lote"]))

    ct = ColumnTransformer(ramas, remainder="drop", verbose_feature_names_out=False)
    etapas = [("fe", ct)]
    if "S" in letras:
        etapas.append(("sel", FiltroColinealidad(tr["colinealidad_umbral"], semilla)))
    etapas.append(("clf", crear_modelo(modelo, semilla, cfg.get("modelo_params", {}))))
    return Pipeline(etapas)
