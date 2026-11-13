"""
atributos.py
-------------
Agrega a NIVEL DE FRUTO los atributos medidos en cada vista (Plan de tesis v4, Tabla 3.3 y Tabla 4.2):

    Calibre  d_est = max( media(ancho 0°, ancho 180°) , media(ancho 90°, ancho 270°) )      [mm]
             (las vistas opuestas proyectan casi la misma anchura -> redundancia; se toma el máximo porque el calibrador
             mide el DIÁMETRO MÁXIMO). Se reporta también la mediana de las cuatro anchuras (estimador alternativo).
    Defecto  A_def de cada vista = área de la unión de máscaras / área visible; severidad S por vista (área + tipo);
             el fruto toma el MÁXIMO entre las 5 vistas ("basta una vista", incluido el pedúnculo).
    Color    P_am = proporción de píxeles sanos con matiz h*_ab <= h_am, agrupando los píxeles de TODAS las vistas.
             L*, a*, b* medios solo como descriptivos.

Este módulo trabaja sobre la tabla por vista (una fila por imagen) que producen `extraer_features.py` (segmentación clásica)
o `inferir_hibrido.py` (segmentación de YOLOv8-seg / Mask R-CNN). Con el mismo código se evalúan E1 (1 vs 5 vistas) y el resto.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as C
from src import regla_comercial as rc
from src.features_imagen import p_amarillo_desde_hist

PARES_OPUESTOS = (("0", "180"), ("90", "270"))


def calibre_estimado(anchos: dict) -> tuple:
    """(d_est, d_mediana4) a partir de {vista: ancho} de las vistas ecuatoriales (en mm o px). NaN si no hay datos.
    Con un par incompleto se usa la vista disponible del par."""
    def media_par(par):
        v = [anchos.get(x) for x in par if x in anchos and np.isfinite(anchos[x])]
        return float(np.mean(v)) if v else np.nan
    pares = [media_par(p) for p in PARES_OPUESTOS]
    d_est = float(np.nanmax(pares)) if np.isfinite(pares).any() else np.nan
    todos = [anchos[v] for v in C.VISTAS_EQ if v in anchos and np.isfinite(anchos[v])]
    return d_est, (float(np.median(todos)) if todos else np.nan)


def severidad_de_vista(fila: pd.Series, u: rc.UmbralesRegla) -> float:
    """S de una vista a partir de su A_def (`defecto_ratio`) y de los tipos presentes (`tipos_defecto`)."""
    area = fila.get("defecto_ratio", np.nan)
    if not np.isfinite(area):
        return np.nan
    tipos = [t for t in str(fila.get("tipos_defecto", "") or "").split(";") if t and t != "nan"]
    return float(rc.severidad_vista(area, tipos, u))


def p_amarillo_fruto(g: pd.DataFrame, h_am: float) -> float:
    """P_am del fruto: píxeles amarillos / píxeles sanos sumados sobre las vistas usadas (histogramas de matiz por vista)."""
    if "hue_hist" not in g or g["hue_hist"].isna().all():
        return float(g["p_amarillo"].mean()) if "p_amarillo" in g else np.nan
    tot = np.zeros(1)
    for h in g["hue_hist"].dropna():
        v = np.asarray(json.loads(h) if isinstance(h, str) else h, float)
        tot = v if tot.shape != v.shape else tot + v
    return p_amarillo_desde_hist(tot, h_am)


def agregar_atributos_fruto(g: pd.DataFrame, u: rc.UmbralesRegla | None = None, vistas=None) -> pd.Series:
    """Atributos del fruto a partir de las filas por vista de UN fruto (`g`). `vistas` restringe las vistas usadas (E1)."""
    u = u or rc.UmbralesRegla()
    g = g[g["seg_ok"].fillna(False).astype(bool)] if "seg_ok" in g else g
    if vistas is not None:
        g = g[g["vista"].astype(str).isin([str(v) for v in vistas])]
    eq = g[g["vista"].astype(str).isin(C.VISTAS_EQ)]
    r = {}
    for suf, col in (("mm", "ancho_mm"), ("px", "ancho_px")):
        if col in eq and eq[col].notna().any():
            r[f"d_est_{suf}"], r[f"d_med4_{suf}"] = calibre_estimado(dict(zip(eq["vista"].astype(str), eq[col])))
        else:
            r[f"d_est_{suf}"] = r[f"d_med4_{suf}"] = np.nan
    sev = g.apply(lambda f: severidad_de_vista(f, u), axis=1) if len(g) else pd.Series(dtype=float)
    r["severidad_max"] = float(sev.max()) if len(sev) else np.nan
    r["A_def_max"] = float(g["defecto_ratio"].max()) if len(g) and "defecto_ratio" in g else np.nan
    r["p_amarillo"] = p_amarillo_fruto(g, u.h_am) if len(g) else np.nan
    for c in ("L_media", "a_media", "b_media"):
        r[c] = float(eq[c].mean()) if len(eq) and c in eq else np.nan
    r["n_vistas_validas"] = int(len(g))
    return pd.Series(r)


def agregar_tabla(vistas: pd.DataFrame, u: rc.UmbralesRegla | None = None, vistas_usadas=None) -> pd.DataFrame:
    """Una fila por fruto con los atributos agregados (usa solo `vistas_usadas` si se indica)."""
    u = u or rc.UmbralesRegla()
    out = (vistas.groupby("id_fruto").apply(lambda g: agregar_atributos_fruto(g, u, vistas_usadas), include_groups=False)
           .reset_index())
    meta = vistas.groupby("id_fruto")[[c for c in ("lote", "fecha") if c in vistas]].first().reset_index()
    return meta.merge(out, on="id_fruto")


def clasificar_tabla(vistas: pd.DataFrame, u: rc.UmbralesRegla | None = None, vistas_usadas=None) -> pd.DataFrame:
    """Atributos + categoría de la regla jerárquica para cada fruto (columna `pred`)."""
    u = u or rc.UmbralesRegla()
    t = agregar_tabla(vistas, u, vistas_usadas)
    t["pred"] = rc.clasificar_df(t, u).to_numpy()
    t["motivo_fabrica"] = [rc.motivo_fabrica(s, p, u) for s, p in zip(t["severidad_max"], t["p_amarillo"])]
    return t
