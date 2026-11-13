"""
calibracion.py
--------------
Evaluacion del calibre ecuatorial (hipotesis H1 / objetivo OE1 del Plan de tesis v4).

Meta de H1: MAE <= 0,50 mm (el LIMITE SUPERIOR del IC95 % no excede 0,50), RMSE <= 0,70 mm (limite superior del IC tambien
<= 0,70) y exactitud dentro de una categoria adyacente >= 0,98 (k = 1 Tercera .. 4 Extra; |k_est - k_ref| <= 1). La
concordancia exacta de categoria por calibre se reporta con su IC como dato DESCRIPTIVO, sin umbral. El error se reporta
ademas en los calibres criticos 34,5; 35; 36; 37; 38; 39 y 42 mm y con el grafico de Bland y Altman (1986).

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

from src import config as _C  # noqa: E402

CALIBRES_CRITICOS = list(_C.CFG["calibres_criticos_mm"])               # 34,5; 35; 36; 37; 38; 39 y 42 mm
TOLERANCIA_CRITICA_MM = float(_C.CFG["tolerancia_critica_mm"])        # ventana +- alrededor de cada calibre critico


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
    """Categoria por calibre con los intervalos semiabiertos del plan v4 (35 / 37 / 39; > 42 = Fuera_de_alcance)."""
    from src.regla_comercial import categoria_por_calibre_vec
    return categoria_por_calibre_vec(mm)


def cambios_de_categoria(estimado, referencia) -> dict:
    """Cuantos frutos cambian de categoria por calibre al usar el calibre estimado en vez del medido."""
    ce, cr = categoria_por_calibre(estimado), categoria_por_calibre(referencia)
    ok = np.array([a is not None and b is not None for a, b in zip(ce, cr)], dtype=bool)
    cambia = np.array([(a is not None and b is not None and a != b) for a, b in zip(ce, cr)], dtype=bool)
    return dict(n=int(ok.sum()), cambian=int(cambia.sum()), pct=float(100 * cambia.sum() / max(ok.sum(), 1)))


def evaluar_h1(estimado, referencia, cat_ref=None, B: int | None = None) -> dict:
    """H1 completa: MAE/RMSE/adyacente con IC por bootstrap (estratificado por categoria si se pasa `cat_ref`), error en calibres
    criticos y cumplimiento de cada criterio. Cada criterio se cumple si el limite (superior para MAE/RMSE, estimado para la
    exactitud adyacente) satisface el umbral del plan; el p-valor de cada uno alimenta la familia de Holm (contrastes.py)."""
    from src import estadistica as est
    h1 = _C.EST["h1"]
    r = est.ic_calibre(estimado, referencia, cat_ref, B)
    if r.get("n", 0) == 0:
        return dict(n=0, cumple=False)
    r["criterios"] = dict(
        mae=dict(valor=r["mae"], limite=r["mae_ic"][1], umbral=h1["mae_max"], cumple=bool(r["mae_ic"][1] <= h1["mae_max"]), p=r["p_mae"]),
        rmse=dict(valor=r["rmse"], limite=r["rmse_ic"][1], umbral=h1["rmse_max"], cumple=bool(r["rmse_ic"][1] <= h1["rmse_max"]), p=r["p_rmse"]),
        adyacente=dict(valor=r["adyacente"], limite=r["adyacente_ic"][0], umbral=h1["adyacente_min"],
                       cumple=bool(r["adyacente"] >= h1["adyacente_min"]), p=r["p_adyacente"]))
    r["cumple"] = all(c["cumple"] for c in r["criterios"].values())
    r["p_h1"] = max(c["p"] for c in r["criterios"].values())          # "su valor p es el mayor de los de sus criterios"
    r["criticos"] = error_por_calibre_critico(estimado, referencia).to_dict("records")
    return r


def figura_bland_altman(estimado, referencia, ruta=None, titulo="Bland-Altman del calibre (imagen - vernier)"):
    """Grafico de Bland y Altman (1986): diferencia vs media, con sesgo y limites de acuerdo (+-1,96 DE)."""
    import matplotlib.pyplot as plt
    e, r = np.asarray(estimado, float), np.asarray(referencia, float)
    ok = np.isfinite(e) & np.isfinite(r)
    m, d = (e[ok] + r[ok]) / 2, e[ok] - r[ok]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.scatter(m, d, s=10, alpha=.6, color="#7a1f2b")
    bias, sd = d.mean(), d.std(ddof=1) if len(d) > 1 else 0.0
    for y, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
        ax.axhline(y, color="k", ls=ls, lw=1)
    ax.set_xlabel("media de ambos metodos (mm)"); ax.set_ylabel("imagen - vernier (mm)"); ax.set_title(titulo, fontsize=10); ax.grid(alpha=.25)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=130)
        plt.close(fig)
    return fig
