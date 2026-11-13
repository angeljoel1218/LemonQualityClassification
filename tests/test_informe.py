"""Flujo de analisis -> informe con artefactos SINTETICOS (sin imagenes ni torch).  python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src import experimentos as ex  # noqa: E402
from src import informe_parcial as ip  # noqa: E402
from src import particion as P  # noqa: E402
from src import protocolo  # noqa: E402
from src import regla_comercial as rc  # noqa: E402

BASE = ROOT / "informe" / "plantilla" / "Informe_Parcial_PI2_Plan_de_Tesis_Limon_UNI.docx"
CATS = C.CATEGORIAS
RANGOS = {"Extra": (39.2, 41.8), "Primera": (37.2, 38.8), "Segunda": (35.2, 36.8), "Tercera": (30, 34.8), "Fabrica": (33, 41)}


def _proyecto(tmp_path, n_por_cat=40, con_resultados=True):
    rng = np.random.default_rng(0)
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir(); proc.mkdir()
    filas, vistas = [], []
    k = 0
    for c in CATS:
        for _ in range(n_por_cat):
            k += 1
            fid = f"F{k:04d}"
            cal = rng.uniform(*RANGOS[c])
            lote, fecha = f"L{rng.integers(1, 4)}", f"2026090{rng.integers(1, 5)}"
            filas.append(dict(id_fruto=fid, lote=lote, fecha=fecha, calibre_mm_ref=round(cal, 1), categoria_ref=c, color_panel="verde", severidad_defecto_panel=0.75 if c == "Fabrica" else 0.0))
            for v in C.VISTAS:
                vistas.append(dict(id_fruto=fid, lote=lote, fecha=fecha, vista=v, seg_ok=True, legible=True, ancho_mm=cal + rng.normal(0, .15),
                                   defecto_ratio=(0.2 if (c == "Fabrica" and v in ("90", "pedunculo")) else 0.0), tipos_defecto="", L_media=60.0, a_media=-20.0, b_media=30.0,
                                   hue_hist=json.dumps([0] * 25 + [100] + [0] * 46), t_total_ms=30.0))
    et = pd.DataFrame(filas)
    et.to_csv(raw / "etiquetas.csv", index=False)
    idx = et.assign(completo=True, origen="ok", n_vistas=5)
    idx.to_csv(proc / "indice_frutos.csv", index=False)
    pd.DataFrame([dict(id_fruto=f, lote=l, fecha=d, vista=v, ruta=f"x/{l}_{d}_{f}_{v}.jpg") for f, l, d in zip(et.id_fruto, et.lote, et.fecha) for v in C.VISTAS]).to_csv(proc / "indice_imagenes.csv", index=False)
    part, meta = P.particionar(et, reservado=None, detalle=True)
    part.to_csv(proc / "particion.csv", index=False)
    (proc / "particion_meta.json").write_text(json.dumps(meta, default=str), encoding="utf-8")
    return raw, proc, pd.DataFrame(vistas), et, part


def _parchear(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "DIR_INF", tmp_path / "inf"); monkeypatch.setattr(ex, "DIR_DIR", tmp_path / "dir"); monkeypatch.setattr(ex, "DIR_RES", tmp_path / "res")
    monkeypatch.setattr(protocolo, "RUTA_REGISTRO", tmp_path / "reg.json")
    monkeypatch.setattr(rc, "RUTA_UMBRALES", tmp_path / "u.json")
    monkeypatch.setattr(ip, "LOGS", tmp_path / "logs")


def _artefactos(tmp_path, vistas, et, part):
    rng = np.random.default_rng(1)
    m = vistas.merge(part, on="id_fruto")
    for sub, nombres in (("val", ["base", "e3_brillo", "e5_luz_mas"]), ("test", ["final"])):
        for n in nombres:
            d = ex.DIR_INF / n
            d.mkdir(parents=True, exist_ok=True)
            m[m["subset"] == sub].drop(columns=["subset", "reservado"]).to_csv(d / f"vistas_{sub}.csv", index=False)
    protocolo.registrar_artefacto("hibrido_final_test", ex.DIR_INF / "final" / "vistas_test.csv")
    y = et.merge(part, on="id_fruto")
    for b in ("ninguno", "focal"):
        d = ex.DIR_DIR / f"directo_{b}"
        d.mkdir(parents=True, exist_ok=True)
        for sub in ("val", "test"):
            c = y[y["subset"] == sub]
            p = np.where(rng.random(len(c)) < .8, c["categoria_ref"], rng.choice(CATS, len(c)))
            pd.DataFrame(dict(id_fruto=c["id_fruto"], y=c["categoria_ref"], pred=p, t_ms=10.0)).to_csv(d / f"pred_{sub}.csv", index=False)
        json.dump(dict(nombre=f"directo_{b}", balanceo=b, semillas=[dict(semilla=s, val_f1_macro=.8 + .01 * s) for s in range(3)], val_f1_macro_media=.81, val_f1_macro_std=.01,
                       tiempo_total_s=600.0, curvas_semilla_mediana=[dict(epoca=i + 1, loss_train=1 / (i + 1), loss_val=1 / (i + 1) + .1, f1_val=.5 + .05 * i) for i in range(6)]),
                  open(d / "resumen.json", "w"))
    # operarios (3) y panel no se necesitan para el flujo minimo


def test_analizar_sin_artefactos_no_inventa_resultados(tmp_path, monkeypatch):
    raw, proc, vistas, et, part = _proyecto(tmp_path)
    _parchear(monkeypatch, tmp_path)
    r = ex.analizar(proc, raw, B=200, verbose=False)
    assert r["prueba"]["disponible"] is False and r["desarrollo"]["E3"] == [] and r["desarrollo"]["E4"] == []


def test_flujo_completo_sintetico_hasta_las_conclusiones(tmp_path, monkeypatch):
    raw, proc, vistas, et, part = _proyecto(tmp_path)
    _parchear(monkeypatch, tmp_path)
    _artefactos(tmp_path, vistas, et, part)
    r = ex.analizar(proc, raw, "final", "directo_ninguno", B=300, verbose=False)
    p = r["prueba"]
    assert p["disponible"] and p["huella"]["ok"] and set(p["contrastes"]["decision"]) == {"H1", "H2", "H3", "H4"}
    assert p["contrastes"]["decision"]["H4"]["estado"] == "sin evidencia"                 # no hay operarios: no se inventa
    assert p["contrastes"]["H2"]["recall_5vistas"] > p["contrastes"]["H2"]["recall_1vista"]  # el defecto solo esta en las vistas 90 y pedunculo
    assert {x["balanceo"] for x in r["desarrollo"]["E4"]} == {"ninguno", "focal"} and r["desarrollo"]["E3"] and [x["perturbacion"] for x in r["desarrollo"]["E5"]] == ["luz_mas"]
    assert r["desarrollo"]["sobreajuste"]["disponible"]
    # la prueba modificada despues de registrarla se detecta
    f = ex.DIR_INF / "final" / "vistas_test.csv"
    f.write_text(f.read_text() + "\n")
    assert not protocolo.verificar_artefacto("hibrido_final_test", f)["ok"]


@pytest.mark.skipif(not BASE.exists(), reason="falta el informe base en informe/plantilla")
def test_el_informe_se_completa_sin_inventar(tmp_path, monkeypatch):
    import docx
    raw, proc, vistas, et, part = _proyecto(tmp_path)
    _parchear(monkeypatch, tmp_path)
    r0 = ip.completar(BASE, tmp_path / "sin_resultados.docx", proc, raw, mes="noviembre", docente="Dr. Prueba")
    d0 = docx.Document(str(tmp_path / "sin_resultados.docx"))
    t0 = "\n".join(p.text for p in d0.paragraphs)
    assert "Lima, noviembre de" in t0 and "Docente del curso: Dr. Prueba" in t0
    assert "[Pendiente: conclusiones sobre HG y H1–H4" in t0                             # sin prueba: las conclusiones quedan pendientes
    assert {"6.2", "8.1", "8.2", "8.3", "9.1"} <= set(r0["completadas"]) and "6.1" in r0["completadas"]
    _artefactos(tmp_path, vistas, et, part)
    ex.analizar(proc, raw, "final", "directo_ninguno", B=300, verbose=False)
    r1 = ip.completar(BASE, tmp_path / "con_resultados.docx", proc, raw)
    d1 = docx.Document(str(tmp_path / "con_resultados.docx"))
    t1 = "\n".join(p.text for p in d1.paragraphs) + "\n".join(c.text for t in d1.tables for row in t.rows for c in row.cells)
    assert "Hipótesis general:" in t1 and "PROVISIONAL" in t1                             # partición provisional declarada
    assert "H4 (no inferioridad frente a operarios): sin evidencia" in t1
    assert "[Pendiente: conclusiones sobre HG y H1–H4" not in t1
