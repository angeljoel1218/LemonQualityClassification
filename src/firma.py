"""
firma.py
--------
"Firma de datos": estadísticos por variable y por subconjunto (media, p10/p50/p90, varianza,
% de faltantes) para dejar constancia de CON QUÉ datos se corrió cada experimento y para detectar
deriva (cambio de distribución) entre train/val/test y entre lotes/fechas.

    firma(df, cols)                       tabla de estadísticos
    firma_por_subconjunto(df, cols)       dict {subset: tabla} + hash corto
    deriva_entre_subconjuntos(df, cols)   KS train vs val / test (p < 0.01 = deriva)
    deriva_por_grupo(df, cols, grupo)     diferencia estandarizada (SMD) de cada lote/fecha vs. el resto
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def firma(df: pd.DataFrame, cols) -> pd.DataFrame:
    x = df[list(cols)]
    return pd.DataFrame({
        "n": x.count(), "faltantes_%": (100 * x.isna().mean()).round(2),
        "media": x.mean(), "std": x.std(), "varianza": x.var(),
        "p10": x.quantile(.10), "p50": x.quantile(.50), "p90": x.quantile(.90),
    })


def firma_por_subconjunto(df: pd.DataFrame, cols, col_subset: str = "subset"):
    firmas = {s: firma(g, cols) for s, g in df.groupby(col_subset)}
    plano = {s: f.round(6).to_dict() for s, f in firmas.items()}
    h = hashlib.sha256(json.dumps(plano, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return firmas, h


def deriva_entre_subconjuntos(df: pd.DataFrame, cols, col_subset: str = "subset") -> pd.DataFrame:
    tr = df[df[col_subset] == "train"]
    filas = []
    for c in cols:
        for s in ("val", "test"):
            o = df[df[col_subset] == s]
            a, b = tr[c].dropna(), o[c].dropna()
            if len(a) > 2 and len(b) > 2:
                ks = ks_2samp(a, b)
                filas.append((c, s, ks.statistic, ks.pvalue))
    t = pd.DataFrame(filas, columns=["variable", "vs_train", "KS", "p_valor"])
    t["deriva"] = t["p_valor"] < 0.01
    return t.sort_values("p_valor")


def deriva_por_grupo(df: pd.DataFrame, cols, grupo: str) -> pd.DataFrame:
    """SMD = (media del grupo - media del resto) / desviación global. |SMD| > 0.5: el grupo es distinguible
    por esa variable (posible atajo de contexto)."""
    filas = {}
    for g, sub in df.groupby(grupo):
        rest = df[df[grupo] != g]
        sd = df[list(cols)].std().replace(0, np.nan)
        filas[g] = (sub[list(cols)].mean() - rest[list(cols)].mean()) / sd
    return pd.DataFrame(filas).T
