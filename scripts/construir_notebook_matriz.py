"""
construir_notebook_matriz.py
----------------------------
Construye notebooks/Matriz_Consistencia.ipynb a partir de docs/matriz_consistencia.yaml.
La matriz queda escrita en celdas de texto (visible y editable sin ejecutar nada, tambien en GitHub) y el notebook
anade una seccion que calcula con tus datos cuanto de cada indicador ya se cumple.

Uso (tras editar el YAML):   python scripts/construir_notebook_matriz.py
"""
import sys
from pathlib import Path

import nbformat as nbf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import matriz_consistencia as mc  # noqa: E402


def construir(yaml_path: Path = ROOT / "docs" / "matriz_consistencia.yaml", salida: Path = ROOT / "notebooks" / "Matriz_Consistencia.ipynb"):
    d = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    P = mc.partes_markdown(d)
    M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.cells = [
        M(f"""# {d['titulo']}

**Proyecto:** {d['proyecto']}

**Autores:** {d['autores']} · **Asesora:** {d['asesora']}

*{d['leyenda']}*

Este notebook contiene (1) la **matriz de consistencia** completa en el formato *Problema · Objetivos · Hipótesis · Variables (independientes y dependientes) · Indicadores*,
(2) la **trazabilidad** de cada hipótesis hacia el repositorio, (3) las **observaciones de consistencia** para revisar con la asesora y
(4) la **evidencia actual**: cuánto de cada indicador se puede medir hoy con los datos y corridas del repositorio (incluye alertas de datos: concentración por sesión, calibre en mm, campos faltantes).

> El texto de las secciones 1 a 3 se puede editar aquí mismo. Para mantener sincronizados el Word y el Markdown, la fuente única es `docs/matriz_consistencia.yaml`;
> tras editarla: `python src/matriz_consistencia.py` (Word y Markdown) y `python scripts/construir_notebook_matriz.py` (este notebook)."""),
        M("## 1. Matriz de consistencia\n\n### 1.1 Fila general\n\n" + P["general"]),
        M("### 1.2 Filas específicas\n\nCada fila corresponde a un objetivo específico y su problema e hipótesis (P1–OE1–H1 … P8–OE8–H8, como en el Anexo del Plan de tesis); las ablaciones E1 a E6 se indican en cada variable independiente.\n\n" + P["especificos"]),
        M("## 2. Trazabilidad con el repositorio\n\nQué contrasta cada hipótesis y en qué estado está su evidencia (**Implementado**, **Parcial** o **Pendiente**).\n\n" + P["trazabilidad"]),
        M("## 3. Observaciones de consistencia (para revisar con la asesora)\n\n" + P["observaciones"]),
        M("""## 4. Evidencia actual frente a los indicadores

Compara los indicadores de la matriz con lo que **hoy** se puede medir. Estados: **Cumple / No cumple** (medido), **Parcial** (hay una medida parcial o un sustituto, p. ej. modelos clásicos en lugar del modelo profundo), **Pendiente** (falta implementar o recolectar), **Informativo**.
Las corridas guardadas con otra partición se ignoran. Con `MODO_DEMO = True` se usan imágenes sintéticas: sirve para ver el funcionamiento, **no** para concluir."""),
        C("""import sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src import matriz_consistencia as mc, ingesta

MODO_DEMO = False                     # <-- True solo con imágenes sintéticas
RAW  = ROOT / "data" / ("demo/raw" if MODO_DEMO else "raw")
PROC = ROOT / "data" / ("demo/processed" if MODO_DEMO else "processed")
if not (PROC / "indice_frutos.csv").exists():
    ingesta.main(RAW, PROC)           # genera los índices si faltan
if MODO_DEMO:
    print("[AVISO] MODO_DEMO: datos sintéticos; las cifras NO son resultados de la investigación.")"""),
        C("""ev = mc.evidencia_actual(PROC, raw=RAW)
pd.set_option("display.max_colwidth", None)
mc.estilo_evidencia(ev)"""),
        C("""resumen = ev.groupby("Estado").size().reindex(["Cumple", "No cumple", "Parcial", "Pendiente", "Informativo"]).fillna(0).astype(int)
print("Indicadores por estado:", resumen.to_dict())
print(f"\\nMedidos hoy (cumple + no cumple): {int(resumen['Cumple'] + resumen['No cumple'])} de {len(ev)} | "
      f"pendientes o parciales: {int(resumen['Pendiente'] + resumen['Parcial'])}")"""),
        M("""## 5. Cómo se mantiene al día
1. **Avanza la investigación:** actualiza el estado de cada hipótesis en `docs/matriz_consistencia.yaml` (sección `trazabilidad`) y sus umbrales cuando se ratifiquen tras el piloto.
2. **Regenera** el Word y el Markdown (`python src/matriz_consistencia.py`) y este notebook (`python scripts/construir_notebook_matriz.py`).
3. La sección 4 se actualiza sola al ejecutar el notebook: lee los datos y las corridas más recientes del repositorio (EDA, Feature Engineering, importancia de variables y baseline).
4. El informe (`python src/informe.py`) incluye la matriz como **Anexo A**."""),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    salida.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(salida))
    return salida


if __name__ == "__main__":
    print("Notebook generado:", construir())
