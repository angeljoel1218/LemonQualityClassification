"""
regla_comercial.py
-------------------
Regla jerarquica (determinista, no aprendida) que asigna la categoria comercial final a partir de los
atributos agregados del fruto (atributos.py). Es la Figura 4.2 del plan de tesis.

Orden de evaluacion (de mayor a menor prioridad):
    1. ¿Algun defecto con severidad >= theta_def en cualquier vista?      -> Fabrica
    2. ¿Color/madurez "amarillo" (matiz CIELAB por debajo de theta_col)?  -> Fabrica
    3. ¿Calibre >= 39.0 mm?                                              -> Extra
    4. ¿37.0 <= calibre < 39.0 mm?                                       -> Primera
    5. ¿35.0 <= calibre < 37.0 mm?                                       -> Segunda
    6. calibre < 35.0 mm                                                 -> Tercera

Sobre los intervalos (Tabla 1.1 de la tesis): tal como estan escritos (Extra 39-42, Primera 37-38, Segunda 35-36,
Tercera < 35) dejan valores sin categoria (p. ej. 36,5 mm) y no fijan limite superior para frutos > 42 mm. Esta
regla usa intervalos CONTINUOS con cortes en 35, 37 y 39 mm y sin limite superior; es la propuesta de la seccion
4.5.2 que el panel y el centro de acopio deben validar antes del etiquetado definitivo.

Umbrales:
    theta_def  fraccion de superficie con defecto (o severidad) desde la cual el fruto es Fabrica.
    theta_col  matiz CIELAB (grados) POR DEBAJO del cual el fruto se considera amarillo/sobremaduro.
               En CIELAB el verde esta cerca de 120-135 grados y el amarillo cerca de 90 grados.
Se fijan con el panel en el piloto y se ajustan sobre VALIDACION (seccion 4.6.1, paso A.4); no cambian despues
de iniciada la recoleccion definitiva. Ver `calibrar_umbrales` y `configs/fe.yaml` (bloque `dominio`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

CALIBRE_EXTRA = 39.0
CALIBRE_PRIMERA = 37.0
CALIBRE_SEGUNDA = 35.0
CATEGORIAS = ["Extra", "Primera", "Segunda", "Tercera", "Fabrica"]


@dataclass
class UmbralesRegla:
    theta_def: float = 0.10
    theta_col: float = 103.0


def hue_cielab(color_lab: tuple) -> float:
    """Matiz (grados, 0-360) a partir de (L*, a*, b*)."""
    _, a, b = color_lab
    return math.degrees(math.atan2(b, a)) % 360.0


def _es_color_amarillo(color_lab: tuple, theta_col: float) -> bool:
    """Amarillo/sobremaduro si el matiz CIELAB queda por debajo de theta_col (grados)."""
    return hue_cielab(color_lab) < theta_col


def clasificar(calibre_mm: float, color_lab: tuple, severidad_defecto_max: float,
               umbrales: UmbralesRegla) -> str:
    """Aplica el arbol de decision jerarquico a UN fruto y devuelve la categoria final."""
    if severidad_defecto_max >= umbrales.theta_def:
        return "Fabrica"
    if _es_color_amarillo(color_lab, umbrales.theta_col):
        return "Fabrica"
    if calibre_mm >= CALIBRE_EXTRA:
        return "Extra"
    if calibre_mm >= CALIBRE_PRIMERA:
        return "Primera"
    if calibre_mm >= CALIBRE_SEGUNDA:
        return "Segunda"
    return "Tercera"


def es_fabrica_df(df: pd.DataFrame, theta_def: float, theta_col: float,
                  col_defecto: str = "defecto_max", col_a: str = "a_media", col_b: str = "b_media") -> np.ndarray:
    """Version vectorizada de los pasos 1 y 2 (Fabrica por defecto o por color). No necesita el calibre."""
    hue = np.degrees(np.arctan2(df[col_b].to_numpy(float), df[col_a].to_numpy(float))) % 360.0
    d = df[col_defecto].to_numpy(float)
    return (d >= theta_def) | (hue < theta_col)


def clasificar_df(df: pd.DataFrame, umbrales: UmbralesRegla, col_calibre: str = "diam_ecu_mm",
                  col_defecto: str = "defecto_max") -> pd.Series:
    """Regla completa sobre una tabla de frutos. Si falta el calibre (NaN) el fruto no-Fabrica queda sin categoria."""
    fab = es_fabrica_df(df, umbrales.theta_def, umbrales.theta_col, col_defecto)
    mm = df[col_calibre].to_numpy(float)
    cat = np.where(mm >= CALIBRE_EXTRA, "Extra",
          np.where(mm >= CALIBRE_PRIMERA, "Primera",
          np.where(mm >= CALIBRE_SEGUNDA, "Segunda", "Tercera"))).astype(object)
    cat[np.isnan(mm)] = None
    cat[fab] = "Fabrica"
    return pd.Series(cat, index=df.index, name="categoria_regla")


def calibrar_umbrales(df: pd.DataFrame, y_fabrica, rejilla_def=None, rejilla_col=None,
                      col_defecto: str = "defecto_max") -> pd.DataFrame:
    """Rejilla de (theta_def, theta_col) y desempeno de la regla para detectar Fabrica (seccion 4.6.1, paso A.4).

    Devuelve una fila por par con recall, precision y F1 de Fabrica. La eleccion (p. ej. maximizar F1 con
    recall >= 0,90) se hace sobre VALIDACION; el TEST no interviene.
    """
    y = np.asarray(y_fabrica, bool)
    rejilla_def = np.arange(0.00, 0.31, 0.01) if rejilla_def is None else rejilla_def
    rejilla_col = np.arange(80.0, 131.0, 2.5) if rejilla_col is None else rejilla_col
    filas = []
    for td in rejilla_def:
        for tc in rejilla_col:
            p = es_fabrica_df(df, td, tc, col_defecto)
            tp = int((p & y).sum())
            rec = tp / max(int(y.sum()), 1)
            pre = tp / max(int(p.sum()), 1)
            f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
            filas.append(dict(theta_def=float(td), theta_col=float(tc), recall=rec, precision=pre, f1=f1,
                              n_pred=int(p.sum())))
    return pd.DataFrame(filas)
