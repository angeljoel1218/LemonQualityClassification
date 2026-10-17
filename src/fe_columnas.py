"""
fe_columnas.py
--------------
Definición ÚNICA de los conjuntos de features, de las columnas prohibidas (fuga) y de las
features de dominio. Todo el resto del código importa desde aquí.

Conjuntos (para las ablaciones "cambia una cosa por vez"):
    A  Variables originales: medidas directas extraídas de las imágenes (geometría, color,
       defecto, textura), ya agregadas por fruto.
    B  Transformaciones/codificación (se ajustan SOLO con train): winsorización, log1p,
       banderas de atípicos y de faltantes, codificación cíclica de ángulos.
    C  Dominio: features con sentido agronómico/comercial (croma, matiz, índice de amarillez,
       distancia a los umbrales de calibre, clase de color ordinal, etc.).
    S  Selección: filtro de colinealidad (se ajusta SOLO con train).
"""
import numpy as np
import pandas as pd

from src.regla_comercial import CALIBRE_EXTRA, CALIBRE_PRIMERA, CALIBRE_SEGUNDA

# ----------------------------------------------------------------------------- Conjunto A
COLS_GEOMETRIA = [
    "diam_ecu_px", "diam_ecu_mm",          # diámetro ecuatorial (= calibre), eje menor de la vista ecuatorial
    "long_polar_px", "long_polar_mm",      # longitud polar, eje mayor de la vista ecuatorial
    "aspecto",                             # long_polar / diam_ecu
    "area_px", "circularidad", "solidez", "excentricidad",
    "asimetria_cv_diam",                   # variación del diámetro entre las 4 vistas ecuatoriales
    "ped_diam_px", "ped_circularidad", "ped_area_ratio",   # vista del pedúnculo
]
COLS_COLOR = [
    "L_media", "a_media", "b_media", "L_std", "a_std", "b_std",
    "L_p10", "L_p90", "entropia_L", "disp_b_vistas",
]
COLS_DEFECTO = ["defecto_max", "defecto_media", "n_vistas_defecto", "mancha_max", "n_manchas_total"]
COLS_TEXTURA = ["lap_var", "grad_media"]
COLS_A = COLS_GEOMETRIA + COLS_COLOR + COLS_DEFECTO + COLS_TEXTURA

# ----------------------------------------------------------------------------- Conjunto C
COLS_C = [
    "croma", "hue_deg", "indice_amarillez", "razon_a_b", "color_clase_ord",
    "cat_calibre_ord", "dist_umbral_calibre_mm", "es_fronterizo",
    "defecto_alto_flag", "frac_vistas_defecto",
]

# ------------------------------------------------- Metadatos de calidad (NO son features por defecto)
COLS_META = ["n_vistas_validas", "frac_seg_ok", "nitidez", "brillo_fondo", "t_extraccion_ms"]

# ------------------------------------------------- Columnas que JAMÁS pueden ser entrada del modelo
#   * categoria_ref, calibre_mm_ref, color_panel, severidad_defecto_panel, tipo_defecto:
#     la categoría se DERIVA de ellas -> usarlas como entrada es fuga directa del objetivo.
#   * lote, fecha, id_fruto, subset: permiten atajos (el modelo aprende el contexto, no el fruto).
PROHIBIDAS = {
    "id_fruto", "lote", "fecha", "subset", "y", "cat",
    "categoria_ref", "calibre_mm_ref", "color_panel", "severidad_defecto_panel", "tipo_defecto",
}


def verificar_sin_fuga(columnas, permitir_fuga: bool = False):
    """Lanza un error si alguna columna prohibida está entre las entradas del modelo."""
    malas = sorted(set(columnas) & PROHIBIDAS)
    if malas and not permitir_fuga:
        raise ValueError(
            f"FUGA DE INFORMACION: {malas} no pueden ser entrada del modelo "
            "(derivan del objetivo o permiten atajos de contexto).")
    return malas


def columnas_para(conjuntos, usar_calidad: bool = False):
    """Columnas de entrada para una combinación de conjuntos, p. ej. 'A+C'."""
    cols = list(COLS_A)
    if "C" in conjuntos:
        cols += COLS_C
    if usar_calidad:
        cols += COLS_META
    verificar_sin_fuga(cols)
    return cols


# ----------------------------------------------------------------------------- Creación (dominio)
def crear_features_dominio(df: pd.DataFrame, dom: dict) -> pd.DataFrame:
    """Crea el conjunto C a partir del A. Es SIN ESTADO (no aprende nada de los datos), por lo
    que puede aplicarse antes de partir sin riesgo de fuga. Los umbrales vienen de configs/fe.yaml
    y deben fijarse en el piloto con el panel.
    """
    d = df.copy()
    umbrales = np.array(dom.get("umbrales_calibre_mm",
                                [CALIBRE_SEGUNDA, CALIBRE_PRIMERA, CALIBRE_EXTRA]), dtype=float)
    eps = 1e-6
    # --- color: croma, matiz (ángulo), amarillez
    d["croma"] = np.sqrt(d["a_media"] ** 2 + d["b_media"] ** 2)
    d["hue_deg"] = np.degrees(np.arctan2(d["b_media"], d["a_media"])) % 360.0
    d["indice_amarillez"] = d["b_media"] / (d["L_media"].abs() + eps)
    d["razon_a_b"] = d["a_media"] / (d["b_media"].abs() + eps)
    # clase de color ordinal (orden real: 0 verde < 1 verde-amarillo < 2 amarillo)
    h = d["hue_deg"]
    clase = np.where(h >= dom.get("hue_verde_min", 123.0), 0,
                     np.where(h >= dom.get("hue_amarillo_max", 103.0), 1, 2)).astype(float)
    d["color_clase_ord"] = np.where(h.isna(), np.nan, clase)
    # --- calibre: categoría comercial por calibre estimado y distancia a los umbrales
    mm = d["diam_ecu_mm"]
    d["cat_calibre_ord"] = np.where(mm.isna(), np.nan,
                                    (mm >= umbrales[0]).astype(float) + (mm >= umbrales[1]) + (mm >= umbrales[2]))
    dist = np.abs(mm.to_numpy(dtype=float)[:, None] - umbrales[None, :]).min(axis=1)
    d["dist_umbral_calibre_mm"] = dist
    d["es_fronterizo"] = np.where(mm.isna(), np.nan, (dist <= dom.get("fronterizo_mm", 0.5)).astype(float))
    # --- defecto
    d["defecto_alto_flag"] = (d["defecto_max"] >= dom.get("theta_def", 0.10)).astype(float)
    d["frac_vistas_defecto"] = d["n_vistas_defecto"] / 5.0
    return d
