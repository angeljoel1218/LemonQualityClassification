"""Pruebas de la estadistica de la tesis, la regla comercial y la calibracion del calibre.  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import calibracion as cb  # noqa: E402
from src import estadistica as est  # noqa: E402
from src import evidencia_hipotesis as eh  # noqa: E402
from src import regla_comercial as rc  # noqa: E402


def test_kappa_perfecto_y_azar():
    y = np.array(list("aabbab" * 10))
    assert est.kappa_cohen(y, y) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert abs(est.kappa_cohen(y, rng.permutation(y))) < 0.35
    assert np.isnan(est.kappa_cohen(["a"] * 5, ["a"] * 5))


def test_bootstrap_es_determinista_y_cubre_el_estimado():
    rng = np.random.default_rng(1)
    y = rng.choice(["a", "b", "c"], 120)
    p = np.where(rng.random(120) < 0.8, y, rng.choice(["a", "b", "c"], 120))
    a = est.bootstrap_pareado(y, p, B=200, semilla=3)
    b = est.bootstrap_pareado(y, p, B=200, semilla=3)
    assert a == b and a["ic_inf"] <= a["estimado"] <= a["ic_sup"] and a["B"] == 200


def test_bootstrap_de_diferencia_detecta_mejora_clara():
    rng = np.random.default_rng(2)
    y = rng.choice(["a", "b"], 200)
    bueno = np.where(rng.random(200) < 0.95, y, rng.choice(["a", "b"], 200))
    malo = rng.choice(["a", "b"], 200)
    d = est.bootstrap_pareado(y, bueno, malo, B=300)
    assert d["excluye_cero"] and d["estimado"] > 0.3 and est.no_inferioridad(d, 0.05)


def test_verifica_umbral_usa_el_limite_inferior():
    assert est.verifica_umbral_ic(dict(estimado=.92, ic_inf=.88, ic_sup=.96), .90).startswith("estimado >= umbral")
    assert est.verifica_umbral_ic(dict(estimado=.95, ic_inf=.91, ic_sup=.98), .90).startswith("cumple")
    assert est.verifica_umbral_ic(dict(estimado=.80, ic_inf=.7, ic_sup=.9), .90) == "no cumple"


def test_mcnemar_exacto_y_simetria():
    a = np.array([1] * 12 + [0] * 3 + [1] * 5, bool)      # A acierta 12 que B falla
    b = np.array([0] * 12 + [1] * 3 + [1] * 5, bool)
    r = est.mcnemar(a, b)
    assert (r["b"], r["c"]) == (12, 3) and r["prueba"] == "exacta" and r["p"] == pytest.approx(0.0352, abs=1e-3)
    assert est.mcnemar(b, a)["p"] == pytest.approx(r["p"])
    assert est.mcnemar(a, a)["p"] == 1.0


def test_wilson_y_holm():
    lo, hi = est.wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=2e-3) and hi == pytest.approx(0.596, abs=2e-3)
    assert est.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_f1_par_y_por_clase():
    y = np.array(["Primera"] * 5 + ["Segunda"] * 5 + ["Extra"] * 5)
    p = np.array(["Primera"] * 5 + ["Primera"] * 5 + ["Extra"] * 5)
    assert est.f1_clase(y, p, "Extra") == 1.0
    assert est.f1_par(y, p) < 0.8 and est.recall_clase(y, p, "Segunda") == 0.0


# ------------------------------------------------------------------ regla comercial
def test_regla_comercial_jerarquica():
    u = rc.UmbralesRegla(0.10, 103.0)
    verde = (60, -25, 30)
    assert rc.clasificar(40, verde, 0.0, u) == "Extra"
    assert rc.clasificar(37.5, verde, 0.0, u) == "Primera" and rc.clasificar(36.5, verde, 0.0, u) == "Segunda"   # sin huecos
    assert rc.clasificar(30, verde, 0.0, u) == "Tercera" and rc.clasificar(45, verde, 0.0, u) == "Extra"       # sin techo
    assert rc.clasificar(40, verde, 0.2, u) == "Fabrica"                       # defecto manda sobre el calibre
    assert rc.clasificar(40, (60, -2, 30), 0.0, u) == "Fabrica"                # amarillo manda sobre el calibre


def test_regla_vectorizada_coincide_con_la_escalar():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(dict(diam_ecu_mm=rng.uniform(30, 44, 60), defecto_max=rng.uniform(0, .3, 60),
                           a_media=rng.uniform(-30, 5, 60), b_media=rng.uniform(5, 45, 60), L_media=60.0))
    u = rc.UmbralesRegla(0.12, 100.0)
    v = rc.clasificar_df(df, u)
    e = [rc.clasificar(r.diam_ecu_mm, (r.L_media, r.a_media, r.b_media), r.defecto_max, u) for r in df.itertuples()]
    assert v.tolist() == e


def test_calibrar_umbrales_encuentra_la_regla_perfecta():
    rng = np.random.default_rng(1)
    n = 200
    fab = rng.random(n) < .3
    df = pd.DataFrame(dict(defecto_max=np.where(fab, rng.uniform(.2, .4, n), rng.uniform(0, .05, n)),
                           a_media=-25.0, b_media=30.0))
    g = rc.calibrar_umbrales(df, fab, np.arange(0, .31, .05), np.array([90.0, 100.0]))
    mejor = g.sort_values("f1", ascending=False).iloc[0]
    assert mejor.f1 == pytest.approx(1.0) and 0.05 <= mejor.theta_def <= 0.2


# ------------------------------------------------------------------ calibracion del calibre
def test_escala_provisional_usa_solo_train_y_la_aplica_por_sesion():
    tr = pd.DataFrame(dict(lote=["L1"] * 3 + ["L2"] * 3, fecha="x", diam_ecu_px=[400, 410, 390, 600, 610, 590],
                           calibre_mm_ref=[40, 41, 39, 36, 36.6, 35.4]))
    esc = cb.escala_provisional(tr)
    assert esc[("L1", "x")] == pytest.approx(0.1) and esc[("L2", "x")] == pytest.approx(0.06)
    nuevo = pd.DataFrame(dict(lote=["L2", "L9"], fecha="x", diam_ecu_px=[500, 500]))
    mm = cb.aplicar_escala(nuevo, esc)
    assert mm.iloc[0] == pytest.approx(30.0) and np.isnan(mm.iloc[1])


def test_metricas_y_error_por_calibre_critico():
    ref = np.array([35.0, 35.2, 37.0, 39.0, 42.0])
    estim = ref + np.array([1.0, -1.0, 0.5, 0.0, np.nan])
    m = cb.metricas_calibre(estim, ref)
    assert m["n"] == 4 and m["mae"] == pytest.approx(0.625) and m["sesgo"] == pytest.approx(0.125)
    t = cb.error_por_calibre_critico(estim, ref).set_index("calibre_critico_mm")
    assert t.loc[35.0, "n"] == 2 and np.isnan(t.loc[42.0, "mae"]) and list(t.index) == cb.CALIBRES_CRITICOS
    assert cb.cambios_de_categoria([37.4], [36.9])["cambian"] == 1


def test_e4_cuenta_defectos_omitidos_con_una_y_cinco_vistas():
    filas = []
    for i in range(10):                                   # defecto visible solo en la vista 90 para los 10 frutos
        for v in ["0", "90", "180", "270", "pedunculo"]:
            filas.append(dict(id_fruto=f"F{i}", vista=v, seg_ok=True, defecto_ratio=0.2 if v == "90" else 0.0))
    vistas = pd.DataFrame(filas)
    frutos = pd.DataFrame(dict(id_fruto=[f"F{i}" for i in range(10)], y="Fabrica", severidad_defecto_panel=1.0, subset="train"))
    r = eh.e4_defectos_omitidos(vistas, frutos)
    assert r["omitidos_1vista"] == 10 and r["omitidos_5vistas"] == 0 and r["reduccion_pp"] == pytest.approx(100.0)
    assert r["mcnemar"]["p"] < 0.01 and r["cumple"]
