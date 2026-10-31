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


def test_la_hipotesis_con_umbral_numerico_tiene_indicador_numerico():
    for f in E:
        if re.match(r"H\d\.", f["hipotesis"]):
            assert any(re.search(r"\d", i) for i in f["indicadores"]), f["hipotesis"][:30]


def test_la_trazabilidad_cubre_todas_las_hipotesis_con_estado_valido():
    claves = {r[0] for r in D["trazabilidad"]}
    assert {f"H{i}" for i in range(1, 9)} <= claves and "Hipótesis general" in claves
    assert {r[3] for r in D["trazabilidad"]} <= set(mc.COLOR_ESTADO)


def test_genera_word_y_markdown(tmp_path):
    r = mc.generar(ROOT / "docs" / "matriz_consistencia.yaml", tmp_path, tmp_path / "m.md")
    assert r["docx"].stat().st_size > 8000 and "| Problema | Objetivos |" in r["md"].read_text(encoding="utf-8")
