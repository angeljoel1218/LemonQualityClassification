"""Pruebas de importancia de variables.  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import importancia as im  # noqa: E402
from src.extraer_features import cargar_cfg  # noqa: E402
from src.fe_columnas import COLS_A, COLS_C, DIMENSION_DE, dimension_de  # noqa: E402


def test_todas_las_variables_tienen_dimension():
    assert [c for c in COLS_A + COLS_C if c not in DIMENSION_DE] == []
    assert dimension_de("ruido_control") == "Control (ruido)"


def test_fuente_recupera_la_variable_original():
    assert im.fuente("hue_deg_sin") == "hue_deg" and im.fuente("defecto_max_atipico") == "defecto_max"
    assert im.fuente("missingindicator_diam_ecu_mm") == "diam_ecu_mm" and im.fuente("b_media") == "b_media"


def _largo(medias, sd=0.01, n=15, seed=0):
    rng = np.random.default_rng(seed)
    filas = []
    for v, m in medias.items():
        for i in range(n):
            filas.append((i // 5, i % 5, v, m + rng.normal(0, sd)))
    return pd.DataFrame(filas, columns=["semilla", "pliegue", "variable", "caida_f1"])


def test_resumen_marca_solo_lo_que_supera_al_ruido():
    r = im.resumen_permutacion(_largo({"defecto_max": 0.20, "lap_var": 0.0, im.RUIDO: 0.0}))
    assert bool(r.loc["defecto_max", "supera_ruido"]) and not bool(r.loc["lap_var", "supera_ruido"])
    assert r.index[0] == "defecto_max" and not bool(r.loc[im.RUIDO, "supera_ruido"])


def test_estabilidad_topk_no_cuenta_variables_sin_aporte():
    largo = _largo({"a": 0.3, "b": 0.0, "c": 0.0, "d": 0.0}, sd=0.0)
    t = im.estabilidad_topk(largo, k=3)
    assert t["a"] == 100.0 and t["b"] == 0.0           # los empates en cero no cuentan como "top"


def _datos_sinteticos(n=150, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: rng.normal(size=n) for c in COLS_A + COLS_C})
    base = rng.normal(size=n) * 5 + 40
    d["diam_ecu_px"] = base * 5 + rng.normal(0, .2, n)       # señal real de calibre...
    d["diam_ecu_mm"] = d["diam_ecu_px"] * 0.2                # ...REPETIDA en otra columna (redundancia)
    d["y"] = pd.cut(base, [-np.inf, 37, 40, 43, np.inf], labels=["Tercera", "Segunda", "Primera", "Extra"]).astype(str)
    return d


def test_permutacion_agrupada_detecta_la_redundancia_que_la_individual_subestima():
    cfg = cargar_cfg(); cfg["modelo_params"] = {"rf_arboles": 40}
    d = _datos_sinteticos(); d[im.RUIDO] = np.random.default_rng(1).normal(size=len(d))
    cols = ["diam_ecu_px", "diam_ecu_mm", "L_media", "a_media", "defecto_max", im.RUIDO]
    ind = im.resumen_permutacion(im.permutacion_oof(d, cols, "A", "rf", cfg, [0], 3, 3))
    dim = im.permutacion_dimensiones_oof(d, cols, "A", "rf", cfg, [0], 3, 3)
    g = dim[dim.clase == "macro"].groupby("dimension")["caida_f1"].mean()
    assert g["Calibre"] > 0.2                                        # el grupo completo sí sostiene la predicción
    assert g["Calibre"] > 1.2 * ind.loc[["diam_ecu_px", "diam_ecu_mm"], "media"].max()   # y supera a cualquiera suelta
    assert abs(g["Control (ruido)"]) < 0.1 and g["Color"] < g["Calibre"]
