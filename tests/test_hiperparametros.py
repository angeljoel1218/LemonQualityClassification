"""Pruebas de la busqueda de hiperparametros (sin fuga y con presupuesto fijo).  python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import busqueda_hiperparametros as bh  # noqa: E402
from src.extraer_features import cargar_cfg  # noqa: E402
from src.fe_columnas import COLS_A  # noqa: E402
from src.remuestreo import ConRemuestreo  # noqa: E402

CFG = bh.cargar_cfg_hp()


def test_mismo_presupuesto_para_todos_los_modelos_y_sin_repetidos():
    for m in CFG["modelos"]:
        c = bh.candidatos(m, CFG, 12, 0)
        claves = {json.dumps(x, sort_keys=True) for x in c}
        assert len(c) == len(claves) == 12, m


def test_candidatos_deterministas():
    assert bh.candidatos("rf", CFG, 8, 0) == bh.candidatos("rf", CFG, 8, 0)
    assert bh.candidatos("rf", CFG, 8, 0) != bh.candidatos("rf", CFG, 8, 1)


def test_los_pesos_de_clase_solo_se_ofrecen_a_modelos_que_los_soportan():
    assert "pesos" not in {c["balanceo"] for c in bh.candidatos("knn", CFG, 40, 0)}
    assert "pesos" in {c["balanceo"] for c in bh.candidatos("rf", CFG, 40, 0)}


def test_el_espacio_cubre_los_niveles_de_la_tabla_3_4_y_el_modelo_profundo_esta_definido():
    assert CFG["balanceo"] == ["ninguno", "pesos", "sobre", "sub"]
    ep = CFG["espacio_profundo"]
    assert ep["optimizador"] == "adamw" and ep["scheduler"] == "cosine" and ep["early_stopping"]["monitor"] == "val_loss"
    assert {"gamma", "alpha"} <= set(ep["focal_loss"]) and {"lambda_seg", "lambda_col", "lambda_reg"} <= set(ep["lambdas_multitarea"])


def test_construir_aplica_hiperparametros_y_balanceo_dentro_de_fit():
    cfg = {**cargar_cfg(), "modelo_params": {}}
    cols = COLS_A[:6]
    p = bh.construir("knn", dict(n_neighbors=9, weights="distance", p=1, balanceo="sobre"), cols, "A", 0, cfg)
    clf = p.named_steps["clf"]
    assert isinstance(clf, ConRemuestreo) and clf.estimator.n_neighbors == 9 and clf.estimator.p == 1
    p = bh.construir("rf", dict(n_estimators=50, max_depth=3, min_samples_leaf=2, max_features="sqrt", balanceo="pesos"), cols, "A", 0, cfg)
    assert p.named_steps["clf"].class_weight == "balanced" and p.named_steps["clf"].max_depth == 3
    p = bh.construir("rf", dict(n_estimators=50, max_depth=None, min_samples_leaf=1, max_features="sqrt", balanceo="ninguno"), cols, "A", 0, cfg)
    assert p.named_steps["clf"].class_weight is None


def test_la_busqueda_no_toca_test(tmp_path):
    """Con datos sinteticos: la busqueda corre, usa solo train/val y registra la huella de la particion."""
    rng = np.random.default_rng(0)
    n, cats = 150, ["Extra", "Primera", "Segunda", "Tercera", "Fabrica"]
    d = pd.DataFrame({c: rng.normal(size=n) for c in COLS_A})
    y = np.array(cats * (n // 5))
    d["diam_ecu_mm"] = 34 + 2 * np.searchsorted(cats, y) / 2 + rng.normal(0, .3, n)
    d["a_media"], d["b_media"] = -20 + rng.normal(0, 1, n), 30 + rng.normal(0, 1, n)
    d["id_fruto"] = [f"F{i:03d}" for i in range(n)]
    d["lote"], d["fecha"] = rng.choice(["L1", "L2", "L3"], n), rng.choice(["a", "b", "c", "d"], n)
    d["categoria_ref"], d["calibre_mm_ref"] = y, d["diam_ecu_mm"]
    d["color_panel"], d["severidad_defecto_panel"] = "verde", 0.0
    d["frac_seg_ok"], d["t_extraccion_ms"], d["n_vistas_validas"] = 1.0, 100.0, 5
    d["completo"], d["n_vistas"], d["origen"] = True, 5, "ok"
    d["subset"] = np.where(np.arange(n) % 10 < 7, "train", np.where(np.arange(n) % 10 < 9, "val", "test"))
    proc = tmp_path / "proc"; proc.mkdir()
    d[["id_fruto", "lote", "fecha"] + COLS_A + ["frac_seg_ok", "t_extraccion_ms", "n_vistas_validas"]].to_csv(proc / "features_frutos.csv", index=False)
    d[["id_fruto", "lote", "fecha", "calibre_mm_ref", "categoria_ref", "color_panel", "severidad_defecto_panel", "completo", "n_vistas", "origen"]].to_csv(proc / "indice_frutos.csv", index=False)
    d[["id_fruto", "subset"]].to_csv(proc / "particion.csv", index=False)
    cfg = dict(CFG, semillas=[0], cv_folds=3, presupuesto_por_modelo=2, modelos=["knn", "gnb"])
    import src.busqueda_hiperparametros as mod
    logs0, mod.LOGS = mod.LOGS, tmp_path / "logs"
    try:
        r = bh.ejecutar(proc, cfg, verbose=False)
    finally:
        mod.LOGS = logs0
    assert len(r["trials"]) == 4 and set(r["mejores"].modelo) == {"knn", "gnb"}
    assert r["trials"].columns.str.startswith("val_").any() and (r["dir"] / "trials.csv").exists()
    assert r["trials"].loc[:, "cv_f1_macro_media"].between(0, 1).all()
