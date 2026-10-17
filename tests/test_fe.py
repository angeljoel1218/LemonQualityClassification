"""Pruebas de las garantías anti-fuga y de la extracción. Ejecutar:  python -m pytest -q"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import fe_experimentos as fx  # noqa: E402
from src.extraer_features import cargar_cfg  # noqa: E402
from src.fe_columnas import COLS_A, COLS_C, crear_features_dominio, verificar_sin_fuga  # noqa: E402
from src.fe_pipeline import construir_pipeline  # noqa: E402
from src.features_imagen import features_vista  # noqa: E402
from src.firma import firma_por_subconjunto  # noqa: E402
from src.particion import asignar_subconjuntos, verificar_particion  # noqa: E402
from src.transformadores import FiltroColinealidad, MarcasOutlier, Winsorizador  # noqa: E402

CATS = ["Extra", "Primera", "Segunda", "Tercera", "Fabrica"]


def df_sintetico(n=200, semilla=0):
    rng = np.random.default_rng(semilla)
    d = pd.DataFrame({c: rng.normal(size=n) for c in COLS_A})
    d["y"] = rng.choice(CATS, n)
    d["categoria_ref"] = d["y"]
    d["id_fruto"] = [f"F{i:04d}" for i in range(n)]
    d["lote"] = rng.choice(["L1", "L2", "L3"], n)
    d["fecha"] = rng.choice(["20260901", "20260902", "20260903", "20260904"], n)
    d["t_extraccion_ms"], d["frac_seg_ok"] = 100.0, 1.0
    d["diam_ecu_mm"] = rng.uniform(30, 45, n)
    d["defecto_max"] = rng.uniform(0, 0.3, n)
    d["n_vistas_defecto"] = rng.integers(0, 5, n)
    d["L_media"] = rng.uniform(50, 70, n)
    return d


def test_guardia_bloquea_columnas_del_objetivo():
    for mala in ("categoria_ref", "calibre_mm_ref", "color_panel", "severidad_defecto_panel", "lote", "fecha"):
        with pytest.raises(ValueError):
            verificar_sin_fuga(COLS_A + [mala])
    verificar_sin_fuga(COLS_A + COLS_C)          # los conjuntos A y C son legítimos


def test_ningun_conjunto_contiene_columnas_prohibidas():
    from src.fe_columnas import PROHIBIDAS
    assert not (set(COLS_A) | set(COLS_C)) & PROHIBIDAS


def test_winsorizador_usa_solo_percentiles_de_train():
    tr = np.arange(100, dtype=float).reshape(-1, 1)
    w = Winsorizador(5, 95).fit(tr)
    fuera = w.transform(np.array([[1000.0], [-1000.0]]))
    assert fuera[0, 0] == pytest.approx(np.percentile(tr, 95))
    assert fuera[1, 0] == pytest.approx(np.percentile(tr, 5))


def test_banderas_outlier_aprenden_limites_en_train():
    tr = np.r_[np.random.default_rng(0).normal(size=200), [50.0, -50.0]].reshape(-1, 1)
    m = MarcasOutlier(1.5, 0.005).fit(tr)
    assert m.transform(np.array([[0.0], [60.0]])).ravel().tolist() == [0.0, 1.0]


def test_escalador_del_pipeline_no_ve_validacion():
    cfg = cargar_cfg()
    d = df_sintetico()
    tr, va = d.iloc[:150], d.iloc[150:].copy()
    va["L_media"] += 1000                         # una validación extrema no debe mover el escalado
    pipe = construir_pipeline(COLS_A, "A", "logreg", 0, cfg)
    pipe.fit(tr[COLS_A], tr["y"])
    esc = pipe.named_steps["fe"].named_transformers_["num"].named_steps["esc"]
    j = COLS_A.index("L_media")
    assert esc.mean_[j] == pytest.approx(tr["L_media"].mean())
    pipe.predict(va[COLS_A])                      # y se puede aplicar sin reajustar
    assert esc.mean_[j] == pytest.approx(tr["L_media"].mean())


def test_filtro_colinealidad_elimina_duplicadas():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 1))
    X = np.hstack([a, a * 2 + 1e-6 * rng.normal(size=(200, 1)), rng.normal(size=(200, 1))])
    f = FiltroColinealidad(0.95).fit(X, rng.integers(0, 2, 200))
    assert len(f.keep_) == 2


def test_particion_por_fruto_sin_repetidos_y_estratificada():
    d = df_sintetico(300)
    part = asignar_subconjuntos(d)
    assert part["id_fruto"].is_unique
    assert verificar_particion(d, part) == []
    assert asignar_subconjuntos(d).equals(part)       # determinista


def test_features_vista_mide_el_diametro_ecuatorial():
    img = np.full((480, 640, 3), 215, np.uint8)
    cv2.ellipse(img, (320, 240), (150, 100), 20, 0, 360, (60, 170, 90), -1)   # eje menor = 200 px
    r = features_vista(img, mm_por_px=0.2)
    assert r["seg_ok"] and abs(r["eje_menor_px"] - 200) < 4
    assert abs(r["eje_menor_mm"] - 40) < 1.0


def test_features_vista_detecta_defecto():
    img = np.full((480, 640, 3), 215, np.uint8)
    cv2.ellipse(img, (320, 240), (150, 110), 0, 0, 360, (60, 200, 170), -1)
    sin = features_vista(img.copy())["defecto_ratio"]
    cv2.circle(img, (300, 230), 35, (30, 50, 90), -1)
    con = features_vista(img)["defecto_ratio"]
    assert con > sin + 0.02


def test_dominio_es_sin_estado_y_coherente():
    cfg = cargar_cfg()
    d = crear_features_dominio(df_sintetico(), cfg["dominio"])
    assert set(COLS_C) <= set(d.columns)
    assert d["cat_calibre_ord"].isin([0, 1, 2, 3]).all()
    assert crear_features_dominio(df_sintetico(), cfg["dominio"]).equals(d)


def test_firma_es_estable():
    d = df_sintetico().assign(subset=lambda x: np.where(np.arange(len(x)) < 140, "train", "val"))
    assert firma_por_subconjunto(d, COLS_A)[1] == firma_por_subconjunto(d, COLS_A)[1]


def test_test_set_solo_se_evalua_una_vez(tmp_path, monkeypatch):
    cfg = cargar_cfg()
    d = df_sintetico(240)
    d["subset"] = asignar_subconjuntos(d).set_index("id_fruto").loc[d["id_fruto"], "subset"].to_numpy()
    d = crear_features_dominio(d, cfg["dominio"])
    monkeypatch.setattr(fx, "cargar_datos", lambda *a, **k: d)
    monkeypatch.setattr(fx, "LOGS", tmp_path)
    fx.evaluar_test(tmp_path, "A|logreg", cfg)
    with pytest.raises(SystemExit):
        fx.evaluar_test(tmp_path, "A|logreg", cfg)
