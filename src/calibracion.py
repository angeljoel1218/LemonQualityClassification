"""
calibracion.py
--------------
Evaluacion del calibre ecuatorial (hipotesis H2 / objetivo 2 de la tesis).

Meta de H2: MAE <= 1,0 mm y RMSE <= 1,5 mm frente al calibrador, con el error reportado en los calibres criticos
34,5; 35; 36; 37; 38; 39 y 42 mm (secciones 3.1.3 y 4.6.4) y el grafico de Bland y Altman (1986).

El calibre en mm sale de la mascara y de la escala de la sesion (ecuacion 2: d_mm = s * d_px, con
s = d_ref_mm / d_ref_px, objeto de referencia en el plano del fruto). Esa escala vive en
`data/raw/calibracion_sesiones.csv`. SI ESE ARCHIVO NO EXISTE, `diam_ecu_mm` queda vacio y H2 no se puede medir.

Mientras no exista, `escala_provisional` ofrece un SUSTITUTO de diagnostico: estima mm/px de cada sesion
(lote, fecha) como la mediana de calibre_mm_ref / diam_ecu_px usando SOLO los frutos de train. NO reemplaza a la
calibracion del protocolo (seccion 4.5.3): usa la etiqueta como patron y no detecta paralaje ni inclinacion.
Sirve para saber si el diametro en pixeles contiene informacion de calibre y para dejar el codigo de la
evaluacion listo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CALIBRES_CRITICOS = [34.5, 35.0, 36.0, 37.0, 38.0, 39.0, 42.0]      # seccion 3.1.3 y matriz de consistencia (H2)
TOLERANCIA_CRITICA_MM = 0.5                                           # ventana +- alrededor de cada calibre critico


def escala_provisional(train: pd.DataFrame, col_px: str = "diam_ecu_px", col_mm: str = "calibre_mm_ref") -> dict:
    """mm/px por sesion (lote, fecha), estimado con la MEDIANA de mm/px de los frutos de train de esa sesion."""
    d = train.dropna(subset=[col_px, col_mm]).copy()
    d["mm_px"] = d[col_mm] / d[col_px]
    return {k: float(v) for k, v in d.groupby(["lote", "fecha"])["mm_px"].median().items()}


def aplicar_escala(df: pd.DataFrame, escala: dict, col_px: str = "diam_ecu_px") -> pd.Series:
    """diam_mm = s(sesion) * diam_px. Sesiones sin escala quedan en NaN."""
    s = pd.Series([escala.get((l, f), np.nan) for l, f in zip(df["lote"], df["fecha"])], index=df.index)
    return s * df[col_px]


def metricas_calibre(estimado, referencia) -> dict:
    """MAE, RMSE, sesgo y limites de acuerdo de Bland-Altman (sesgo +- 1,96 DE de las diferencias)."""
    e = np.asarray(estimado, float) - np.asarray(referencia, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, sesgo=np.nan, loa_inf=np.nan, loa_sup=np.nan)
    sd = float(e.std(ddof=1)) if len(e) > 1 else 0.0
    return dict(n=int(len(e)), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                sesgo=float(e.mean()), loa_inf=float(e.mean() - 1.96 * sd), loa_sup=float(e.mean() + 1.96 * sd))


def error_por_calibre_critico(estimado, referencia, criticos=CALIBRES_CRITICOS,
                              tol: float = TOLERANCIA_CRITICA_MM) -> pd.DataFrame:
    """MAE y RMSE de los frutos cuyo calibre de referencia esta a <= `tol` mm de cada calibre critico."""
    est, ref = np.asarray(estimado, float), np.asarray(referencia, float)
    filas = []
    for c in criticos:
        m = np.isfinite(est) & np.isfinite(ref) & (np.abs(ref - c) <= tol)
        mt = metricas_calibre(est[m], ref[m])
        filas.append(dict(calibre_critico_mm=c, n=mt["n"], mae=mt["mae"], rmse=mt["rmse"], sesgo=mt["sesgo"]))
    return pd.DataFrame(filas)


def categoria_por_calibre(mm) -> np.ndarray:
    """Categoria por calibre con los cortes continuos 35 / 37 / 39 (sin Fabrica, que depende de defecto y color)."""
    mm = np.asarray(mm, float)
    out = np.where(mm >= 39.0, "Extra", np.where(mm >= 37.0, "Primera", np.where(mm >= 35.0, "Segunda", "Tercera"))).astype(object)
    out[~np.isfinite(mm)] = None
    return out


def cambios_de_categoria(estimado, referencia) -> dict:
    """Cuantos frutos cambian de categoria por calibre al usar el calibre estimado en vez del medido (no-Fabrica)."""
    ce, cr = categoria_por_calibre(estimado), categoria_por_calibre(referencia)
    ok = np.array([a is not None and b is not None for a, b in zip(ce, cr)])
    cambia = (ce != cr) & ok
    return dict(n=int(ok.sum()), cambian=int(cambia.sum()), pct=float(100 * cambia.sum() / max(ok.sum(), 1)))
