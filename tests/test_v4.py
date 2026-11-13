"""Pruebas de los modulos del Plan de tesis v4 (sin torch ni GPU).  python -m pytest -q"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anotacion as an  # noqa: E402
from src import atributos as at  # noqa: E402
from src import config as C  # noqa: E402
from src import contrastes as ct  # noqa: E402
from src import entrenar_directo as ed  # noqa: E402
from src import estadistica as est  # noqa: E402
from src import operarios as op  # noqa: E402
from src import panel as pn  # noqa: E402
from src import particion as P  # noqa: E402
from src import potencia as pw  # noqa: E402
from src import preprocesamiento as pp  # noqa: E402
from src import protocolo  # noqa: E402
from src import regla_comercial as rc  # noqa: E402
from src import segmentacion as sg  # noqa: E402
from src.features_imagen import ancho_perpendicular, features_vista  # noqa: E402

CATS = C.CATEGORIAS


def test_configuracion_del_plan():
    assert (C.TOTAL_VALIDOS, C.TOTAL_TRAIN, C.TOTAL_VAL, C.TOTAL_TEST) == (1500, 880, 220, 400)
    assert C.EST["bootstrap_B"] == 5000 and C.EST["delta_kappa"] == 0.05 and C.CALIBRE_MAXIMO == 42.0
    for c, q in C.CUOTAS.items():
        assert q["train"] + q["val"] + q["test"] == q["validos"] and q["test"] == 80


# ------------------------------------------------------------------ atributos
def test_calibre_estimado_es_el_maximo_de_los_promedios_de_vistas_opuestas():
    d, med = at.calibre_estimado({"0": 38.0, "180": 38.4, "90": 37.0, "270": 37.2, "pedunculo": 50})
    assert d == pytest.approx(38.2) and med == pytest.approx(37.6)
    assert at.calibre_estimado({"0": 36.0, "90": 37.0})[0] == 37.0          # pares incompletos: se usa la vista disponible
    assert np.isnan(at.calibre_estimado({})[0])


def _vistas_fruto(fid, anchos, area=0.0, tipos="", hue_hist=None):
    hh = json.dumps(hue_hist or ([0] * 25 + [100] + [0] * 46))              # todo verde (bin de 125-130 grados)
    return pd.DataFrame([dict(id_fruto=fid, lote="L1", fecha="20260901", vista=v, seg_ok=True, ancho_mm=a, ancho_px=a * 8, defecto_ratio=area if v == "90" else 0.0,
                              tipos_defecto=tipos if v == "90" else "", hue_hist=hh, L_media=60, a_media=-20, b_media=30)
                         for v, a in anchos.items()])


def test_agregacion_por_fruto_basta_una_vista_y_p_am_agrupado():
    v = _vistas_fruto("F1", {"0": 38, "90": 37, "180": 38, "270": 37, "pedunculo": 36}, area=0.07)
    r = at.agregar_atributos_fruto(v)
    assert r["d_est_mm"] == pytest.approx(38.0) and r["severidad_max"] == 2 and r["p_amarillo"] == pytest.approx(0.0)
    t = at.clasificar_tabla(v, rc.UmbralesRegla(theta_def=2))
    assert t["pred"].iloc[0] == "Fabrica" and "defecto" in t["motivo_fabrica"].iloc[0]
    uno = at.agregar_atributos_fruto(v, vistas=["0"])
    assert uno["severidad_max"] == 0                                         # el defecto estaba en la vista 90: con 1 vista no se ve
    amarillo = _vistas_fruto("F2", {"0": 40, "90": 40, "180": 40, "270": 40, "pedunculo": 40}, hue_hist=[100] + [0] * 71)
    assert at.agregar_atributos_fruto(amarillo, rc.UmbralesRegla(h_am=103.0))["p_amarillo"] > 0.9


def test_ancho_perpendicular_mide_el_eje_menor_aunque_gire():
    m = np.zeros((480, 640), np.uint8)
    cv2.ellipse(m, (320, 240), (150, 100), 35, 0, 360, 255, -1)              # eje menor 200 px, rotado 35 grados
    ancho, largo = ancho_perpendicular(m)
    assert ancho == pytest.approx(200, abs=4) and largo == pytest.approx(300, abs=4)


def test_features_vista_entrega_p_amarillo_e_histograma():
    img = np.full((480, 640, 3), 215, np.uint8)
    cv2.ellipse(img, (320, 240), (150, 120), 0, 0, 360, (40, 220, 235), -1)    # amarillo (BGR)
    f = features_vista(img, mm_por_px=0.2)
    assert f["seg_ok"] and f["p_amarillo"] > 0.8 and len(json.loads(f["hue_hist"])) == 72
    verde = np.full((480, 640, 3), 215, np.uint8)
    cv2.ellipse(verde, (320, 240), (150, 120), 0, 0, 360, (60, 170, 90), -1)
    assert features_vista(verde, mm_por_px=0.2)["p_amarillo"] < 0.05


# ------------------------------------------------------------------ partición
def _frutos(cuotas, n_lotes=4, n_fechas=4, semilla=0):
    rng = np.random.default_rng(semilla)
    filas, k = [], 0
    for c, n in cuotas.items():
        for _ in range(n):
            k += 1
            filas.append(dict(id_fruto=f"F{k:05d}", lote=f"L{rng.integers(1, n_lotes + 1)}", fecha=f"2026090{rng.integers(1, n_fechas + 1)}", categoria_ref=c,
                              calibre_mm_ref=rng.uniform(30, 41)))
    return pd.DataFrame(filas)


def test_particion_con_cuotas_completas_880_220_400_y_prueba_balanceada():
    d = _frutos(C.CUOTAS_VALIDAS)
    part, meta = P.particionar(d, reservado=None, detalle=True)
    assert not meta["provisional"] and part["id_fruto"].is_unique
    t = P.tabla_asignacion(d, part)
    assert t[["train", "val", "test"]].sum().tolist() == [880, 220, 400]
    assert (t["test"] == 80).all() and t.loc["Extra", "train"] == 192 and t.loc["Tercera", "train"] == 152 and t.loc["Fabrica", "val"] == 38
    assert P.verificar_particion(d, part) == []
    assert P.particionar(d, reservado=None).equals(part)                    # determinista


def test_particion_proporcional_provisional_mantiene_prueba_balanceada():
    d = _frutos({c: 100 for c in CATS})
    part, meta = P.particionar(d, reservado=None, detalle=True)
    t = P.tabla_asignacion(d, part)
    assert meta["provisional"] and (t["test"] == t["test"].iloc[0]).all() and t["test"].iloc[0] >= 20
    assert any("PROVISIONAL" in a for a in meta["avisos"])


def test_particion_excluye_piloto_y_frutos_mayores_de_42mm():
    d = _frutos({c: 60 for c in CATS})
    d["piloto"] = 0
    d.loc[:9, "piloto"] = 1
    d.loc[10:14, "calibre_mm_ref"] = 43.0
    part = P.particionar(d, reservado=None)
    assert not set(d.loc[:14, "id_fruto"]) & set(part["id_fruto"])


def test_lote_reservado_solo_va_a_prueba_y_respeta_las_cuotas():
    d = _frutos(C.CUOTAS_VALIDAS, n_lotes=8, n_fechas=4)
    part, meta = P.particionar(d, reservado="auto", detalle=True)
    if meta["reservado"] is None:
        pytest.skip("sin lote elegible con esta semilla")
    m = d.merge(part, on="id_fruto")
    res = m[m["reservado"] == 1]
    assert (res["subset"] == "test").all() and (res.groupby("categoria_ref").size() <= 80).all()
    col, val = meta["reservado"]["tipo"], meta["reservado"]["valor"]
    assert not ((m[col] == val) & (m["subset"] != "test")).any()                 # ningun fruto del lote reservado en train/val
    assert P.verificar_particion(d, part) == []


def test_lote_que_vaciaria_una_categoria_no_es_elegible():
    d = _frutos({c: 100 for c in CATS})
    d.loc[d["categoria_ref"] == "Tercera", "lote"] = "L9"                      # toda una categoria en un lote
    ok, why = P.reservado_elegible(d.assign(cat=d["categoria_ref"]), "lote", "L9", P.objetivos_por_categoria(d["categoria_ref"].value_counts().to_dict())[0])
    assert not ok


# ------------------------------------------------------------------ estadística v4
def test_f1_macro_por_matriz_coincide_con_sklearn():
    from sklearn.metrics import cohen_kappa_score, f1_score
    rng = np.random.default_rng(0)
    y = rng.choice(CATS, 300)
    p = np.where(rng.random(300) < .8, y, rng.choice(CATS, 300))
    cm = est.matriz_confusion_idx(est._a_indices(y, CATS), est._a_indices(p, CATS), 5)
    assert est.f1_macro_cm(cm, 5) == pytest.approx(f1_score(y, p, average="macro", labels=CATS))
    assert float(est.kappa_cm(cm)) == pytest.approx(cohen_kappa_score(y, p))


def test_bootstrap_estratificado_conserva_n_por_categoria_y_es_determinista():
    y = np.repeat(CATS, [10, 20, 30, 40, 50])
    idx = est._indices_estratificados(est._a_indices(y, CATS), 50, 1)
    yb = est._a_indices(y, CATS)[idx]
    assert all((yb == k).sum(1).tolist() == [n] * 50 for k, n in enumerate([10, 20, 30, 40, 50]))
    p = np.where(np.random.default_rng(0).random(150) < .9, y, "Extra")
    assert est.ic_estratificado(y, p, B=300)["ic_inf"] == est.ic_estratificado(y, p, B=300)["ic_inf"]


def test_fleiss_ac1_y_holm_escalonado():
    cal = np.array([["a", "a", "a"]] * 10 + [["a", "a", "b"]] * 5, dtype=object)
    assert est.kappa_fleiss(np.array([["a", "b", "a"]] * 20, dtype=object), ["a", "b"]) < 0.9
    perfecto = np.array([["a", "a", "a"]] * 5 + [["b", "b", "b"]] * 5, dtype=object)
    assert est.kappa_fleiss(perfecto, ["a", "b"]) == pytest.approx(1.0) and est.ac1_gwet(perfecto, ["a", "b"]) == pytest.approx(1.0)
    # AC1 es mas alto que kappa cuando la prevalencia es muy desigual
    desbal = np.array([["a", "a", "a"]] * 90 + [["a", "a", "b"]] * 5 + [["b", "a", "b"]] * 5, dtype=object)
    assert est.ac1_gwet(desbal, ["a", "b"]) > est.kappa_fleiss(desbal, ["a", "b"])
    h = est.holm_escalonado({"H1": .001, "H2": .2, "H3": .01, "H4": .03}, 0.05)
    assert h["H1"]["rechaza"] and h["H3"]["rechaza"] and not h["H4"]["rechaza"] and not h["H2"]["rechaza"]
    assert h["H1"]["umbral"] == pytest.approx(0.05 / 4) and h["H3"]["umbral"] == pytest.approx(0.05 / 3)
    # tras el primer no rechazo se detiene: H4 (p=.03 > .025) frena aunque H2 > umbral final
    assert not est.holm_escalonado({"a": .001, "b": .04, "c": .04, "d": .04}, 0.05)["c"]["rechaza"]


def test_exactitud_adyacente_y_kappa_ponderado():
    ref = ["Tercera", "Segunda", "Primera", "Extra", "Fabrica"]
    pred = ["Segunda", "Segunda", "Extra", "Tercera", "Fabrica"]          # Primera->Extra adyacente; Extra->Tercera salta 3; Fabrica no cuenta
    assert est.exactitud_adyacente(ref, pred) == pytest.approx(3 / 4)
    assert est.kappa_ponderado_calibre(ref, ref) == pytest.approx(1.0)


def test_reponderar_por_prevalencia_natural():
    y = np.repeat(CATS, 80)
    p = np.where((y == "Fabrica") & (np.arange(400) % 4 == 0), "Extra", y)           # Fabrica con recall 0,75, el resto perfecto
    uniforme = est.metricas_ponderadas(y, p, {c: 0.2 for c in CATS})
    raro = est.metricas_ponderadas(y, p, {"Extra": .4, "Primera": .3, "Segunda": .2, "Tercera": .05, "Fabrica": .05})
    assert raro["exactitud"] > uniforme["exactitud"]


def test_ic_calibre_cumple_o_no_segun_el_error():
    rng = np.random.default_rng(1)
    ref = rng.uniform(33, 41, 300)
    bueno = est.ic_calibre(ref + rng.normal(0, .2, 300), ref, B=500)
    malo = est.ic_calibre(ref + rng.normal(0, 1.2, 300), ref, B=500)
    assert bueno["p_mae"] < 0.05 < malo["p_mae"] and bueno["mae_ic"][1] < 0.5


# ------------------------------------------------------------------ contrastes
def test_contrastes_aceptan_un_sistema_claramente_bueno_y_marcan_sin_evidencia():
    rng = np.random.default_rng(3)
    y = np.repeat(CATS, 80)
    mk = lambda a: np.where(rng.random(400) < a, y, rng.choice(CATS, 400))
    h5 = mk(.95)
    h1 = np.where((y == "Fabrica") & (rng.random(400) < .4), "Extra", h5)
    ref = rng.uniform(33, 41, 400)
    r = ct.evaluar(y, h5, h1, mk(.8), ref + rng.normal(0, .25, 400), ref, {f"o{i}": mk(.85) for i in range(3)}, B=800)
    assert r["HG"]["acepta"] and all(r["decision"][k]["estado"] == "se acepta" for k in ("H1", "H2", "H3", "H4"))
    r2 = ct.evaluar(y, h5, B=300)
    assert all(r2["decision"][k]["estado"] == "sin evidencia" for k in ("H1", "H2", "H3", "H4"))


def test_h3_no_se_acepta_si_el_directo_es_igual():
    rng = np.random.default_rng(4)
    y = np.repeat(CATS, 80)
    p = np.where(rng.random(400) < .9, y, rng.choice(CATS, 400))
    r = ct.evaluar(y, p, p, p, B=500)
    assert r["H3"]["diferencia"] == 0 and r["decision"]["H3"]["estado"] == "no se acepta"


# ------------------------------------------------------------------ potencia
def test_formulas_de_tamano_de_muestra_del_plan():
    assert pw.n_h1(0.40, 0.30) == 56 and pw.n_h1(0.45, 0.30) == 223 and pw.n_h1(0.45, 0.34) == 286
    assert pw.n_h4(0.70) == 245 and pw.n_h4(0.50) == 409
    h2 = pw.potencia_h2(80, 0.25, 0.20)
    assert h2["EE"] == pytest.approx(0.051, abs=1e-3) and h2["potencia"] == pytest.approx(0.83, abs=0.01)


def test_simulacion_de_potencia_reproduce_el_anexo_2_rapido():
    r90 = pw.potencia_hg(0.90, MC=150, B=300)
    r88 = pw.potencia_hg(0.88, MC=150, B=300)
    assert 0.8 <= r90["potencia"] <= 0.98 and 0.3 <= r88["potencia"] <= 0.7 and r90["error_estandar"] == pytest.approx(0.0147, abs=0.004)
    h3 = pw.potencia_h3(0.05, MC=120, B=300)
    assert 0.65 <= h3["potencia"] <= 0.95


# ------------------------------------------------------------------ panel y operarios
def _panel(sev, cal, ruido=0.0, semilla=0):
    rng = np.random.default_rng(semilla)
    ids = [f"F{i:03d}" for i in range(len(cal))]
    etq = pd.DataFrame(dict(id_fruto=ids, calibre_mm_ref=cal))
    filas = []
    for e in "ABC":
        for ronda, n in ((1, len(ids)), (2, 50)):
            for i in range(n):
                s = sev[i] if rng.random() > ruido else rng.integers(0, 4)
                filas.append(dict(id_fruto=ids[i], evaluador=e, ronda=ronda, color="verde", severidad_nivel=s))
    return pd.DataFrame(filas), etq


def test_panel_kappa_repetibilidad_y_adjudicacion():
    rng = np.random.default_rng(0)
    cal = rng.uniform(30, 41, 100)
    sev = np.where(rng.random(100) < .2, 3, 0)
    pan, etq = _panel(sev, cal)
    p = pn.categoria_por_evaluador(pan, etq)
    r = pn.criterio_piloto(p)
    assert r["cumple"] and r["acuerdo"]["kappa_fleiss"] == pytest.approx(1.0)
    ruidoso, etq = _panel(sev, cal, ruido=0.45, semilla=2)
    r2 = pn.criterio_piloto(pn.categoria_por_evaluador(ruidoso, etq))
    assert not r2["cumple"] and "rúbrica" in r2["accion"]
    adj = pn.adjudicar(pn.categoria_por_evaluador(pan, etq))
    assert (adj["via"] == "unanime").all() and adj["categoria_panel"].notna().all()


def test_adjudicacion_sin_mayoria_usa_el_cuarto_evaluador():
    filas = [dict(id_fruto="F1", evaluador=e, ronda=1, rol="panel", cat_eval=c) for e, c in zip("ABC", ["Extra", "Primera", "Segunda"])]
    filas.append(dict(id_fruto="F1", evaluador="D", ronda=1, rol="adjudicador", cat_eval="Primera"))
    r = pn.adjudicar(pd.DataFrame(filas))
    assert r["categoria_panel"].iloc[0] == "Primera" and r["via"].iloc[0] == "adjudicador"


def test_resumen_de_operarios():
    rng = np.random.default_rng(1)
    ids = [f"F{i:03d}" for i in range(80)]
    y = rng.choice(CATS, 80)
    filas = []
    for o in ("o1", "o2", "o3"):
        for i, f in enumerate(ids):
            filas.append(dict(id_fruto=f, operario=o, ronda=1, categoria=y[i] if rng.random() < .9 else "Extra", tiempo_s=30.0))
        for i, f in enumerate(ids[:50]):
            filas.append(dict(id_fruto=f, operario=o, ronda=2, categoria=y[i], tiempo_s=30.0))
    r = op.resumen(pd.DataFrame(filas), ids, y)
    assert r["n_operarios"] == 3 and r["suficientes"] and 0.7 < r["kappa_media"] < 1 and all(v["n_repetidos"] == 50 for v in r["operarios"].values())


# ------------------------------------------------------------------ protocolo, preprocesamiento, segmentación, anotación
def test_la_prueba_se_registra_una_vez_y_se_detecta_su_modificacion(tmp_path, monkeypatch):
    monkeypatch.setattr(protocolo, "RUTA_REGISTRO", tmp_path / "reg.json")
    f = tmp_path / "pred.csv"
    f.write_text("a,b\n1,2\n")
    protocolo.registrar_artefacto("t", f)
    assert protocolo.verificar_artefacto("t", f)["ok"]
    with pytest.raises(SystemExit):
        protocolo.registrar_artefacto("t", f)
    f.write_text("a,b\n1,3\n")
    assert not protocolo.verificar_artefacto("t", f)["ok"]
    protocolo.registrar_artefacto("t", f, forzar=True)
    assert json.loads((tmp_path / "reg.json").read_text())["t"]["reemplazos"]


def test_la_prueba_exige_umbrales_y_modelos_congelados(tmp_path, monkeypatch):
    monkeypatch.setattr(protocolo, "RUTA_MODELOS", tmp_path / "m.json")
    monkeypatch.setattr(rc, "RUTA_UMBRALES", tmp_path / "u.json")
    with pytest.raises(SystemExit):
        protocolo.exigir_listo_para_prueba()


def test_escala_desde_damero_y_correccion_de_brillo():
    sq = 40
    img = np.full((7 * sq + 80, 10 * sq + 80, 3), 255, np.uint8)
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                img[40 + r * sq:40 + (r + 1) * sq, 40 + c * sq:40 + (c + 1) * sq] = 0
    e = pp.escala_desde_damero(img, (9, 6), 25.0)
    assert e["ok"] and e["mm_por_px"] == pytest.approx(25 / 40, rel=0.01) and e["anisotropia"] == pytest.approx(1.0, abs=0.02)
    base = np.full((300, 400, 3), 200, np.uint8)
    cv2.ellipse(base, (200, 150), (80, 60), 0, 0, 360, (60, 170, 90), -1)
    oscura = (base * 0.6).astype(np.uint8)
    assert pp.nivel_fondo(pp.aplicar(oscura, "brillo")) == pytest.approx(pp.L_FONDO_REF, abs=2)
    assert pp.perturbar_luminancia(base, 1.1).mean() > base.mean() > pp.perturbar_luminancia(base, 0.9).mean()
    assert np.array_equal(pp.aplicar(base, "ninguno"), base)


def test_poligonos_ida_y_vuelta_y_seleccion_de_modelo():
    m = np.zeros((480, 640), np.uint8)
    cv2.ellipse(m, (320, 240), (150, 100), 0, 0, 360, 255, -1)
    pols = sg.poligonos_desde_txt.__wrapped__ if hasattr(sg.poligonos_desde_txt, "__wrapped__") else None
    lineas = sg.mascara_a_poligono_txt(m, 0)
    assert len(lineas) == 1 and lineas[0].startswith("0 ")
    pts = np.array(lineas[0].split()[1:], float).reshape(-1, 2) * [640, 480]
    m2 = sg.mascara_desde_poligonos([(0, pts)], 480, 640)
    assert sg.iou_mascaras(m, m2) > 0.98 and sg.dice_mascaras(m, m2) > 0.99
    cand = pd.DataFrame(dict(nombre=["yolo", "mask"], iou_medio=[0.950, 0.955], mae_calibre_mm=[0.30, 0.32], latencia_ms_mediana=[20.0, 90.0], tiempo_entrenamiento_s=[100, 400]))
    sel = sg.seleccionar_modelo(cand)
    assert sel["elegido"]["nombre"] == "yolo"                                  # igualdad practica (dMAE 0,02 < 0,05): menor costo
    cand.loc[1, "mae_calibre_mm"] = 0.45
    assert sg.seleccionar_modelo(cand)["elegido"]["nombre"] == "mask"          # el MAE ya no es equivalente: gana el mayor IoU


def test_instancia_fruto_descarta_la_carta():
    fruto, carta = np.zeros((100, 100), np.uint8), np.zeros((100, 100), np.uint8)
    fruto[20:80, 20:80] = 255
    carta[0:20, 0:20] = 255
    assert sg.instancia_fruto([carta, fruto], [0.99, 0.9]).sum() == fruto.sum()


def test_submuestra_de_defectos_400_desarrollo_y_100_prueba():
    d = _frutos({c: 200 for c in CATS})
    part = P.particionar(d, reservado=None)
    s = an.submuestra_defectos(d, part)
    g = s.groupby(["grupo", "categoria"]).size()
    assert (g.xs("desarrollo") == 80).all() and (g.xs("prueba") == 20).all() and len(s) == 500
    assert set(s[s["grupo"] == "prueba"]["subset"]) == {"test"} and set(s[s["grupo"] == "desarrollo"]["subset"]) <= {"train", "val"}
    assert len(an.muestra_control_calidad(s["id_fruto"])) == 50
    assert an.estimar_horas()["horas_total"] == pytest.approx(104.17, abs=0.1)


# ------------------------------------------------------------------ directo (lógica pura)
def test_busqueda_del_directo_tiene_el_mismo_presupuesto_para_todas_las_variantes():
    for b in ed.BALANCEOS:
        hs = ed.espacio_directo(8, b)
        assert len(hs) == 8 and all(h["balanceo"] == b for h in hs)
    assert all(h["pesos"] for h in ed.espacio_directo(8, "pesos")) and all(h["gamma"] > 0 for h in ed.espacio_directo(8, "focal"))
    assert all(h["aumento"] == "dirigido" for h in ed.espacio_directo(8, "aumento")) and not any(h["pesos"] or h["gamma"] for h in ed.espacio_directo(8, "ninguno"))
    # los hiperparametros de base son los MISMOS en todas las variantes (comparacion justa)
    base = [(h["lr"], h["weight_decay"], h["batch_size"]) for h in ed.espacio_directo(8, "ninguno")]
    assert base == [(h["lr"], h["weight_decay"], h["batch_size"]) for h in ed.espacio_directo(8, "focal")]


def test_seleccion_de_semilla_mediana_y_mejor_configuracion():
    assert ed.elegir_semilla_mediana(pd.DataFrame(dict(semilla=[0, 1, 2], val_f1_macro=[.80, .90, .85]))) == 2
    assert ed.elegir_mejor_config(pd.DataFrame(dict(val_f1_macro=[.8, .9, .9], val_loss=[.5, .4, .3]))) == 2
    w = ed.pesos_de_clase(["Extra"] * 50 + ["Fabrica"] * 10)
    assert w[CATS.index("Fabrica")] > w[CATS.index("Extra")]


def test_a_cuadrado_no_deforma():
    img = np.full((480, 640, 3), 100, np.uint8)
    assert ed.a_cuadrado(img, 224).shape == (224, 224, 3)
