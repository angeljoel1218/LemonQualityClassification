"""Pruebas del baseline y de las variantes de datos.  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import modelo_baseline as mb  # noqa: E402
from src.extraer_features import agregar_por_fruto  # noqa: E402
from src.remuestreo import ConRemuestreo, indices_remuestreo  # noqa: E402


def test_sobremuestreo_iguala_clases_y_conserva_originales():
    y = np.array(["a"] * 30 + ["b"] * 10 + ["c"] * 5)
    idx = indices_remuestreo(y, "sobre", 0)
    assert pd.Series(y[idx]).value_counts().nunique() == 1          # todas con 30
    assert set(range(len(y))) <= set(idx.tolist())                  # no se pierde ningún original


def test_submuestreo_iguala_a_la_minoritaria_sin_repetir():
    y = np.array(["a"] * 30 + ["b"] * 10 + ["c"] * 5)
    idx = indices_remuestreo(y, "sub", 0)
    assert pd.Series(y[idx]).value_counts().eq(5).all()
    assert len(set(idx.tolist())) == len(idx)


def test_remuestreo_es_determinista():
    y = np.array(["a"] * 20 + ["b"] * 4)
    assert (indices_remuestreo(y, "sobre", 3) == indices_remuestreo(y, "sobre", 3)).all()


def test_con_remuestreo_solo_ve_lo_que_recibe_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3)); y = rng.permutation(np.array(["a"] * 50 + ["b"] * 10))
    m = ConRemuestreo(LogisticRegression(), "sobre", 0).fit(X[:40], y[:40])   # "pliegue de entrenamiento"
    assert m.n_entrenamiento_ == 2 * max((y[:40] == "a").sum(), (y[:40] == "b").sum())
    assert m.predict_proba(X[40:]).shape == (20, 2)                           # y se aplica sin reajustar


def test_t_corregida_valores_conocidos():
    d = np.array([0.1, 0.12, 0.08, 0.11, 0.09, 0.1, 0.13, 0.07, 0.1, 0.1])
    J, m, v = len(d), d.mean(), d.var(ddof=1)
    t_esp = m / np.sqrt((1 / J + 0.25 / 0.75) * v)
    t, p = mb.t_corregida(d, n_train=75, n_test=25)
    assert t == pytest.approx(t_esp) and 0 < p < 1
    assert mb.t_corregida(np.zeros(10), 75, 25) == (0.0, 1.0)               # sin diferencia -> p = 1


def test_holm():
    assert mb.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert mb.holm([0.5]) == pytest.approx([0.5])


def _vistas_sinteticas(n_frutos=6):
    filas = []
    for i in range(n_frutos):
        for v in ["0", "90", "180", "270", "pedunculo"]:
            filas.append(dict(id_fruto=f"F{i}", lote="L1", fecha="20260901", vista=v, seg_ok=True,
                              eje_menor_px=100.0 + (50 if v == "0" else 0), eje_mayor_px=120.0, eje_menor_mm=20.0,
                              eje_mayor_mm=24.0, aspecto=1.2, area_px=9000.0, circularidad=.9, solidez=.98,
                              excentricidad=.5, diam_equiv_px=105.0, L_media=60.0, a_media=-20.0, b_media=50.0,
                              L_std=1., a_std=1., b_std=1., L_p10=58., L_p90=62., entropia_L=1.,
                              defecto_ratio=0.0 if v != "270" else 0.2, mancha_max_ratio=0.0, n_manchas=0,
                              lap_var=10., grad_media=5., nitidez=100., brillo_fondo=90., t_ms=100.0))
    return pd.DataFrame(filas)


def test_variante_de_vistas_usa_solo_las_vistas_indicadas():
    v = _vistas_sinteticas()
    una = agregar_por_fruto(v[v.vista.isin(["0"])])
    cinco = agregar_por_fruto(v)
    assert una["diam_ecu_px"].iloc[0] == 150.0 and cinco["diam_ecu_px"].iloc[0] == 100.0   # mediana de 4 vs 1 vista
    assert una["defecto_max"].iloc[0] == 0.0 and cinco["defecto_max"].iloc[0] == 0.2        # el defecto de 270 solo se ve con 5
    assert una["t_extraccion_ms"].iloc[0] == 100.0 and cinco["t_extraccion_ms"].iloc[0] == 500.0


def test_datos_con_vistas_conserva_frutos_y_orden():
    v = _vistas_sinteticas()
    base = pd.DataFrame({"id_fruto": [f"F{i}" for i in reversed(range(6))], "lote": "L1", "fecha": "x",
                         "y": "a", "subset": "train"})
    out = mb.datos_con_vistas(base, v, ["0", "180"], 0.02)
    assert out["id_fruto"].tolist() == base["id_fruto"].tolist()


def _cv(config, media, ruido, n=15, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"config": config, "semilla": np.repeat([0, 1, 2], 5), "pliegue": np.tile(range(5), 3),
                         "n_train_fold": 60, "n_test_fold": 15,
                         "f1_macro": media + rng.normal(0, ruido, n), "f1_pos": media + rng.normal(0, ruido, n)})


def test_comparar_detecta_mejora_clara_y_no_inventa_diferencias():
    cfg = {"alfa": 0.05}
    res = pd.DataFrame([dict(config="BASE|knn", variante="BASE", familia="BASE", modelo="knn"),
                        dict(config="V1_x|knn", variante="V1_x", familia="V1", modelo="knn"),
                        dict(config="V2_x|knn", variante="V2_x", familia="V2", modelo="knn")])
    base = _cv("BASE|knn", 0.60, 0.0)
    base["f1_macro"] = np.tile([0.58, 0.60, 0.62], 5)[:15]            # variación real entre pliegues
    mejor = base.assign(config="V1_x|knn", f1_macro=base["f1_macro"] + 0.20 + np.random.default_rng(1).normal(0, .005, 15))
    igual = base.assign(config="V2_x|knn")
    c = mb.comparar(pd.concat([base, mejor, igual]), res, cfg)
    f = c[c.metrica == "f1_macro"].set_index("variante")
    assert f.loc["V1_x", "veredicto"] == "mejora"
    assert f.loc["V2_x", "veredicto"] == "sin evidencia" and f.loc["V2_x", "dif_media"] == 0


def test_dummy_solo_aparece_como_referencia():
    cfg = mb.cargar_cfg_baseline()
    exp = mb.lista_experimentos(cfg)
    assert [e["config"] for e in exp if e["modelo"] == "dummy"] == ["BASE|dummy"]
    assert {"V1_1vista", "V2_sobremuestreo", "V2_submuestreo"} <= {e["variante"] for e in exp}


def test_informe_renderiza_markdown_y_word(tmp_path):
    from src import informe
    B = [("banner", "demo"), ("h1", "Uno"), ("p", "texto"), ("bullets", ["a", "b"]),
         ("table", pd.DataFrame({"A": ["1", "2"], "B": ["x", "y"]}), "tabla de prueba"), ("nota", "nota")]
    md = informe.a_markdown(B, "Titulo", "figuras")
    assert "## Uno" in md and "| A | B |" in md
    informe.a_docx(B, "Titulo", tmp_path / "x.docx", "autores")
    assert (tmp_path / "x.docx").stat().st_size > 5000
