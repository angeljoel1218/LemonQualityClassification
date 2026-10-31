"""
matriz_consistencia.py
----------------------
Genera la MATRIZ DE CONSISTENCIA (Word horizontal + Markdown) a partir de docs/matriz_consistencia.yaml.
El contenido se edita en el YAML (problemas, objetivos, hipotesis, variables, indicadores, observaciones y
trazabilidad) y se regenera; asi el Word nunca queda desactualizado respecto al texto fuente.

Formato:  Problema | Objetivos | Hipotesis | Variables (independientes y dependientes) | Indicadores
          fila GENERAL + una fila por objetivo/ablacion especifica; luego trazabilidad con el repositorio y
          observaciones de consistencia.

Uso:
    python src/matriz_consistencia.py
    python src/matriz_consistencia.py --yaml docs/matriz_consistencia.yaml --salida informe
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENCABEZADO = ["Problema", "Objetivos", "Hipótesis", "Variables (independientes y dependientes)", "Indicadores"]
ANCHOS = [4.7, 5.8, 4.6, 5.6, 5.6]                      # cm (suma 26.3; util en A4 horizontal con 1.4 cm de margen = 26.9)
COLOR_ESTADO = {"Implementado": "DDEEDD", "Parcial": "FFF1CC", "Pendiente": "F6D5D5"}


# ------------------------------------------------------------------------------------ contenido por celda
def _vars(f: dict):
    out = [("Var. independiente(s):", True)] + [("• " + x, False) for x in f.get("independientes", [])]
    out += [("Var. dependiente(s):", True)] + [("• " + x, False) for x in f.get("dependientes", [])]
    if f.get("intervinientes"):
        out += [("Var. interviniente(s):", True)] + [("• " + x, False) for x in f["intervinientes"]]
    return out


def _texto(t: str, prefijo: str = ""):
    partes = [(x, False) for x in t.split("\n")]
    if prefijo:
        partes.insert(0, (prefijo, True))
    return partes


def filas(d: dict) -> list:
    """Lista de (tipo, [celda1..celda5]); cada celda es una lista de (texto, negrita)."""
    g = d["general"]
    out = [("banda", "GENERAL"),
           ("general", [_texto(g["problema"], "Problema general"), _texto(g["objetivo"], "Objetivo general"),
                     _texto(g["hipotesis"], "Hipótesis general"), _vars(g),
                     [("• " + x, False) for x in g["indicadores"]]]),
           ("banda", "ESPECÍFICOS")]
    for f in d["especificos"]:
        out.append(("fila", [_texto(f["problema"]), _texto(f["objetivo"]), _texto(f["hipotesis"]), _vars(f),
                             [("• " + x, False) for x in f["indicadores"]]]))
    return out


# ------------------------------------------------------------------------------------ Markdown
def a_markdown(d: dict) -> str:
    br = lambda partes: "<br>".join(f"**{t}**" if n else t for t, n in partes).replace("|", "\\|")
    L = [f"# {d['titulo']}\n", f"**Proyecto:** {d['proyecto']}  ", f"**Autores:** {d['autores']}  ",
         f"**Asesora:** {d['asesora']}\n", f"*{d['leyenda']}*\n",
         "| " + " | ".join(ENCABEZADO) + " |", "|" + "---|" * 5]
    for tipo, c in filas(d):
        if tipo == "banda":
            L.append(f"| **{c}** | | | | |")
        else:
            L.append("| " + " | ".join(br(x) for x in c) + " |")
    L += ["", "## Trazabilidad con el repositorio\n", "| Hipótesis / objetivo | Qué lo contrasta | Evidencia en el repositorio | Estado |", "|---|---|---|---|"]
    L += ["| " + " | ".join(r) + " |" for r in d["trazabilidad"]]
    L += ["", "## Observaciones de consistencia (para revisar con la asesora)\n"]
    L += [f"{i}. {o}" for i, o in enumerate(d["observaciones"], 1)]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------ Word
def a_docx(d: dict, ruta: Path):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(1.4); sec.top_margin = sec.bottom_margin = Cm(1.4)
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    for n, tam in (("Heading 1", 14), ("Heading 2", 12)):
        h = doc.styles[n]; h.font.name = "Calibri"; h.font.size = Pt(tam); h.font.bold = True
        h.font.color.rgb = RGBColor(0x7A, 0x1F, 0x2B); h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

    def fill(cell, color):
        tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color); tcPr.append(shd)

    def fila_props(row, header=False):
        trPr = row._tr.get_or_add_trPr()
        if header:
            trPr.append(OxmlElement("w:tblHeader"))
        trPr.append(OxmlElement("w:cantSplit"))

    def llenar(cell, partes, fs=8, color=None, centro=False):
        cell.text = ""
        for i, (texto, neg) in enumerate(partes):
            p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
            p.paragraph_format.space_after = Pt(1.5)
            if centro:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(texto); r.bold = neg; r.font.size = Pt(fs)
            if color:
                r.font.color.rgb = color

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(d["titulo"].upper()); r.bold = True; r.font.size = Pt(18); r.font.color.rgb = RGBColor(0x7A, 0x1F, 0x2B)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(d["proyecto"]).italic = True
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"{d['autores']}  ·  Asesora: {d['asesora']}").font.size = Pt(9)
    p = doc.add_paragraph(); r = p.add_run(d["leyenda"]); r.italic = True; r.font.size = Pt(8)

    tb = doc.add_table(rows=1, cols=5); tb.style = "Table Grid"; tb.autofit = False
    for j, h in enumerate(ENCABEZADO):
        c = tb.rows[0].cells[j]; llenar(c, [(h, True)], 9, RGBColor(255, 255, 255), centro=True); fill(c, "7A1F2B"); c.width = Cm(ANCHOS[j])
    fila_props(tb.rows[0], header=True)
    for tipo, cont in filas(d):
        row = tb.add_row()
        if tipo != "general":                 # la fila general es muy alta: se deja fluir entre paginas
            fila_props(row)
        if tipo == "banda":
            m = row.cells[0].merge(row.cells[4]); llenar(m, [(cont, True)], 9, centro=True); fill(m, "F5E6E8")
        else:
            for j, partes in enumerate(cont):
                llenar(row.cells[j], partes); row.cells[j].width = Cm(ANCHOS[j])

    doc.add_page_break()
    doc.add_heading("Trazabilidad con el repositorio", 1)
    t2 = doc.add_table(rows=1, cols=4); t2.style = "Table Grid"; t2.autofit = False
    an = [4.0, 7.0, 12.2, 3.1]
    for j, h in enumerate(["Hipótesis / objetivo", "Qué lo contrasta", "Evidencia en el repositorio", "Estado"]):
        c = t2.rows[0].cells[j]; llenar(c, [(h, True)], 9, RGBColor(255, 255, 255)); fill(c, "7A1F2B"); c.width = Cm(an[j])
    fila_props(t2.rows[0], header=True)
    for r_ in d["trazabilidad"]:
        row = t2.add_row(); fila_props(row)
        for j, v in enumerate(r_):
            llenar(row.cells[j], [(v, j == 0 or j == 3)], 8.5); row.cells[j].width = Cm(an[j])
        fill(row.cells[3], COLOR_ESTADO.get(r_[3], "FFFFFF"))
    doc.add_heading("Observaciones de consistencia (para revisar con la asesora)", 1)
    for o in d["observaciones"]:
        doc.add_paragraph(o, style="List Number")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(ruta))


def generar(yaml_path: Path, salida: Path, md_path: Path | None = None) -> dict:
    d = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    dx = salida / "Matriz_de_Consistencia.docx"
    a_docx(d, dx)
    md = md_path or (yaml_path.parent / "matriz_consistencia.md")
    md.write_text(a_markdown(d), encoding="utf-8")
    return dict(docx=dx, md=md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=ROOT / "docs" / "matriz_consistencia.yaml")
    ap.add_argument("--salida", type=Path, default=ROOT / "informe")
    a = ap.parse_args()
    r = generar(a.yaml, a.salida)
    print(f"Matriz generada:\n  {r['docx']}\n  {r['md']}")
