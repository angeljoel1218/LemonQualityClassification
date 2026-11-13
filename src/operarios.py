"""
operarios.py
------------
Comparación con operarios en ejercicio (Plan de tesis v4, secciones 4.6.5 y 4.7.4; hipótesis H4).

Entrada: `data/raw/operarios_clasificaciones.csv` (una fila por fruto x operario x ronda):
    id_fruto, operario, ronda, categoria, tiempo_s
      ronda 1 = clasificación de los 400 frutos de prueba (a ciegas respecto del estándar y del sistema, cronometrada)
      ronda 2 = repetición de una submuestra de 50 frutos tras un intervalo (repetibilidad intra-operario)
Los operarios son un grupo DISTINTO del panel (sin participación en la rúbrica, el panel ni la anotación); al menos tres.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src import estadistica as est
from src.eda import normalizar_categorias
from src.ingesta import leer_csv

RUTA_OPERARIOS = C.ROOT / "data" / "raw" / "operarios_clasificaciones.csv"


def cargar_operarios(ruta: Path = RUTA_OPERARIOS) -> pd.DataFrame:
    df = leer_csv(ruta, numericas=["ronda", "tiempo_s"])
    falt = [c for c in ("id_fruto", "operario", "categoria") if c not in df.columns]
    if falt:
        raise SystemExit(f"operarios_clasificaciones.csv: faltan columnas {falt}")
    if "ronda" not in df:
        df["ronda"] = 1
    df["ronda"] = df["ronda"].fillna(1).astype(int)
    df["categoria"] = normalizar_categorias(df["categoria"])
    return df


def predicciones_por_operario(df: pd.DataFrame, ids, ronda: int = 1) -> dict:
    """{operario: array de categorías alineado con `ids`}; solo operarios que clasificaron TODOS los frutos de `ids`."""
    q = df[df["ronda"] == ronda]
    out = {}
    for o, g in q.groupby("operario"):
        s = g.drop_duplicates("id_fruto").set_index("id_fruto")["categoria"].reindex(list(ids))
        if s.notna().all():
            out[o] = s.to_numpy(object)
    return out


def resumen(df: pd.DataFrame, ids, y_ref, minimo_operarios: int = 3) -> dict:
    """Por operario: κ y F1 macro frente al estándar, exactitud, tiempo por fruto y repetibilidad; y el promedio de κ (κ_op)."""
    ids = list(ids)
    preds = predicciones_por_operario(df, ids)
    filas = {}
    for o, p in preds.items():
        k = est.kappa_cohen(y_ref, p)
        g = df[(df["operario"] == o) & (df["ronda"] == 1)]
        r1 = df[(df["operario"] == o) & (df["ronda"] == 1)].set_index("id_fruto")["categoria"]
        r2 = df[(df["operario"] == o) & (df["ronda"] == 2)].set_index("id_fruto")["categoria"]
        comun = r1.index.intersection(r2.index)
        filas[o] = dict(kappa=float(k), f1_macro=float(est.ic_estratificado(y_ref, p, "f1_macro", B=200)["estimado"]),
                        exactitud=float(np.mean(np.asarray(y_ref, object) == np.asarray(p, object))),
                        tiempo_s_mediana=float(g["tiempo_s"].median()) if "tiempo_s" in g and g["tiempo_s"].notna().any() else np.nan,
                        n_repetidos=int(len(comun)),
                        repetibilidad_kappa=float(est.kappa_cohen(r1[comun], r2[comun])) if len(comun) >= 2 else np.nan)
    ks = [v["kappa"] for v in filas.values() if np.isfinite(v["kappa"])]
    return dict(operarios=filas, n_operarios=len(filas), kappa_media=float(np.mean(ks)) if ks else np.nan,
                kappa_rango=float(np.ptp(ks)) if len(ks) > 1 else np.nan, suficientes=len(filas) >= minimo_operarios, preds=preds)
