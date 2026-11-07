"""La matriz de consistencia debe ser internamente consistente.  python -m pytest -q"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import matriz_consistencia as mc  # noqa: E402

D = yaml.safe_load((ROOT / "docs" / "matriz_consistencia.yaml").read_text(encoding="utf-8"))
E = D["especificos"]


def _ids(campo, patron):
    return {m.group(1) for f in E for m in [re.match(patron, f[campo])] if m}


def test_estan_todos_los_problemas_hipotesis_y_objetivos():
    assert _ids("problema", r"P(\d)\.") == {str(i) for i in range(1, 9)}
    assert _ids("hipotesis", r"H(\d)\.") == {str(i) for i in range(1, 9)}
    assert {m.group(1) for f in E for m in [re.match(r"OE(\d)\.", f["objetivo"])] if m} == {str(i) for i in range(1, 9)}


def test_ninguna_hipotesis_ni_problema_se_repite():
    h = [m.group(1) for f in E for m in [re.match(r"H(\d)\.", f["hipotesis"])] if m]
    p = [m.group(1) for f in E for m in [re.match(r"P(\d)\.", f["problema"])] if m]
    assert len(h) == len(set(h)) and len(p) == len(set(p))


def test_cada_fila_tiene_variables_e_indicadores():
    for f in E + [D["general"]]:
        assert f["independientes"] and f["dependientes"] and f["indicadores"], f.get("id", "general")


def test_los_objetivos_se_corresponden_uno_a_uno_con_problemas_e_hipotesis():
    """Anexo del Plan de tesis corregido: fila i = P_i, OE_i, H_i."""
    for f in E:
        n = f["id"]
        assert f["problema"].startswith(f"P{n}.") and f["objetivo"].startswith(f"OE{n}.") and f["hipotesis"].startswith(f"H{n}."), n


def test_la_resolucion_de_entrada_ya_no_es_variable():
    """Tabla 3.7 corregida: la resolución de entrada fue retirada de las variables."""
    assert "Resolución de entrada" not in str(D)


def test_la_hipotesis_con_umbral_numerico_tiene_indicador_numerico():
    for f in E:
        if re.match(r"H\d\.", f["hipotesis"]):
            assert any(re.search(r"\d", i) for i in f["indicadores"]), f["hipotesis"][:30]


def test_la_trazabilidad_cubre_todas_las_hipotesis_con_estado_valido():
    claves = {r[0] for r in D["trazabilidad"]}
    assert {f"H{i}" for i in range(1, 9)} <= {c.split(" ")[0] for c in claves} and "Hipótesis general" in claves
    assert {r[3] for r in D["trazabilidad"]} <= set(mc.COLOR_ESTADO)


def test_genera_word_y_markdown(tmp_path):
    r = mc.generar(ROOT / "docs" / "matriz_consistencia.yaml", tmp_path, tmp_path / "m.md")
    assert r["docx"].stat().st_size > 8000 and "| Problema | Objetivos |" in r["md"].read_text(encoding="utf-8")


def test_el_informe_incluye_la_matriz_como_anexo_horizontal(tmp_path):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from src import informe
    informe.a_docx([("h1", "Uno"), ("p", "texto")], "Titulo", tmp_path / "i.docx", "autores", matriz=D)
    doc = Document(str(tmp_path / "i.docx"))
    assert len(doc.sections) == 2 and doc.sections[1].orientation == WD_ORIENT.LANDSCAPE
    assert doc.sections[0].orientation != WD_ORIENT.LANDSCAPE
    assert any("Anexo A. Matriz de consistencia" in p.text for p in doc.paragraphs)
    assert len(doc.tables) >= 2                                   # la matriz y la trazabilidad
    assert mc.a_markdown(D, anexo=True).startswith("## Anexo A")


def test_el_readme_y_el_repositorio_contienen_la_matriz():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/Matriz_de_Consistencia.docx" in readme and "docs/matriz_consistencia.md" in readme
    assert "notebooks/Matriz_Consistencia.ipynb" in readme
    assert (ROOT / "docs" / "Matriz_de_Consistencia.docx").exists()      # el Word debe estar DENTRO del repo


# ------------------------------------------------------------------ notebook de la matriz
def _md_cells(path):
    import nbformat
    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)
    return [c.source for c in nb.cells if c.cell_type == "markdown"]


def test_el_notebook_contiene_la_matriz_completa():
    nbk = ROOT / "notebooks" / "Matriz_Consistencia.ipynb"
    assert nbk.exists()
    texto = "\n".join(_md_cells(nbk))
    assert "Problema general" in texto and "Objetivo general" in texto and "Hipótesis general" in texto
    for i in range(1, 9):
        assert f"H{i}." in texto and f"P{i}." in texto and f"OE{i}" in texto, i
    assert "Trazabilidad" in texto and "Observaciones de consistencia" in texto
    assert "Var. independiente(s):" in texto and "Var. dependiente(s):" in texto


def test_el_notebook_esta_sincronizado_con_el_yaml(tmp_path):
    """Si se edita el YAML y no se reconstruye el notebook, esta prueba falla."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import construir_notebook_matriz as cn
    nuevo = cn.construir(ROOT / "docs" / "matriz_consistencia.yaml", tmp_path / "m.ipynb")
    assert _md_cells(nuevo) == _md_cells(ROOT / "notebooks" / "Matriz_Consistencia.ipynb")


def _datos_minimos(tmp_path, n=40):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    proc = tmp_path / "proc"; proc.mkdir()
    cat = rng.choice(["Extra", "Primera", "Segunda", "Tercera", "Fabrica"], n)
    ref = rng.uniform(33, 44, n)
    ids = [f"F{i:03d}" for i in range(n)]
    pd.DataFrame(dict(id_fruto=ids, lote=rng.choice(["L1", "L2", "L3"], n), fecha=rng.choice(["20260901", "20260902", "20260903", "20260904"], n),
                      calibre_mm_ref=ref, categoria_ref=cat, completo=True)).to_csv(proc / "indice_frutos.csv", index=False)
    pd.DataFrame(dict(id_fruto=ids, diam_ecu_mm=ref + rng.normal(0, .3, n), t_extraccion_ms=rng.uniform(100, 150, n))
                 ).to_csv(proc / "features_frutos.csv", index=False)
    pd.DataFrame(dict(id_fruto=ids, subset=np.where(np.arange(n) < 28, "train", np.where(np.arange(n) < 34, "val", "test")))
                 ).to_csv(proc / "particion.csv", index=False)
    return proc


def test_evidencia_actual_mide_lo_que_se_puede_y_marca_pendiente_lo_demas(tmp_path):
    proc = _datos_minimos(tmp_path)
    ev = mc.evidencia_actual(proc, logs=tmp_path / "logs")
    assert set(ev["Estado"]) <= set(mc.COLOR_EVIDENCIA) and list(ev.columns)[:2] == ["Hipótesis/objetivo", "Indicador"]
    est = lambda h, ind: ev[(ev["Hipótesis/objetivo"] == h) & ev["Indicador"].str.contains(ind)]["Estado"].iloc[0]
    assert est("Datos", "Frutos con las 5 vistas") == "No cumple"            # 40 < 1 500
    assert est("Datos", "Lotes distintos") == "Cumple" and est("Datos", "Fechas distintas") == "Cumple"
    assert est("H2", "MAE") == "Cumple"                                    # ruido de 0,3 mm -> MAE < 1 mm
    assert est("H8", "Latencia") == "Parcial"
    assert est("H3", "defectos") == "Pendiente" and est("H6", "F1 macro") == "Pendiente"   # sin corridas


def test_evidencia_ignora_corridas_de_otra_particion(tmp_path):
    import json
    import pandas as pd
    proc = _datos_minimos(tmp_path)
    d = tmp_path / "logs" / "baseline_runs" / "r1"; d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"meta": {"huella_particion": "otra"}, "cfg": {}}), encoding="utf-8")
    pd.DataFrame([dict(modelo="knn", variante="V1_1vista", metrica="f1_macro", dif_media=.1, veredicto="mejora")]).to_csv(d / "comparaciones.csv", index=False)
    ev = mc.evidencia_actual(proc, logs=tmp_path / "logs")
    assert ev[ev["Hipótesis/objetivo"] == "H3"]["Estado"].iloc[0] == "Pendiente"
