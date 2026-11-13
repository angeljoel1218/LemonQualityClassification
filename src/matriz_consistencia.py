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
    python src/matriz_consistencia.py --salida docs        (valor por defecto: docs/Matriz_de_Consistencia.docx)
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
def _celda_md(partes) -> str:
    return "<br>".join(f"**{t}**" if n else t for t, n in partes).replace("|", "\\|")


def _celda_html(partes) -> str:
    import html
    return "<br>".join(f"<b>{html.escape(t)}</b>" if n else html.escape(t) for t, n in partes)


def tabla_html(d: dict, tipo: str) -> str:
    """Tabla HTML (alineacion superior, sin colores que dependan del visor) para 'general' o 'especificos'.
    Se usa HTML en el notebook porque las celdas son muy altas y el markdown las centra verticalmente."""
    anchos = ["16%", "20%", "16%", "24%", "24%"]
    cab = "".join(f'<th width="{a}" style="text-align:left;vertical-align:top;background:#F5E6E8">{h}</th>'
                  for h, a in zip(ENCABEZADO, anchos))
    cuerpo = []
    for t, c in filas(d):
        if t == "banda" or (t == "general") != (tipo == "general"):
            continue
        tds = "".join(f'<td valign="top" style="text-align:left;vertical-align:top;font-size:0.9em">{_celda_html(x)}</td>' for x in c)
        cuerpo.append(f"<tr>{tds}</tr>")
    return f'<table style="width:100%;table-layout:fixed"><thead><tr>{cab}</tr></thead><tbody>{"".join(cuerpo)}</tbody></table>'


def partes_markdown(d: dict) -> dict:
    """Fragmentos para el notebook: la matriz (general y especificos) como tablas HTML con alineacion superior;
    trazabilidad y observaciones como markdown."""
    traz = ["| Hipótesis / objetivo | Qué lo contrasta | Evidencia en el repositorio | Estado |", "|---|---|---|---|"]
    traz += ["| " + " | ".join(r) + " |" for r in d["trazabilidad"]]
    return dict(general=tabla_html(d, "general"), especificos=tabla_html(d, "especificos"), trazabilidad="\n".join(traz),
                observaciones="\n".join(f"{i}. {o}" for i, o in enumerate(d["observaciones"], 1)))


def a_markdown(d: dict, anexo: bool = False) -> str:
    """anexo=True: sin titulo propio y con encabezados un nivel mas abajo (para incluirla dentro del informe)."""
    br = lambda partes: "<br>".join(f"**{t}**" if n else t for t, n in partes).replace("|", "\\|")
    h1, h2 = ("## Anexo A. Matriz de consistencia\n", "###") if anexo else (f"# {d['titulo']}\n", "##")
    L = [h1, f"**Proyecto:** {d['proyecto']}  ", f"**Autores:** {d['autores']}  ",
         f"**Asesora:** {d['asesora']}\n", f"*{d['leyenda']}*\n",
         "| " + " | ".join(ENCABEZADO) + " |", "|" + "---|" * 5]
    for tipo, c in filas(d):
        if tipo == "banda":
            L.append(f"| **{c}** | | | | |")
        else:
            L.append("| " + " | ".join(br(x) for x in c) + " |")
    L += ["", f"{h2} Trazabilidad con el repositorio\n", "| Hipótesis / objetivo | Qué lo contrasta | Evidencia en el repositorio | Estado |", "|---|---|---|---|"]
    L += ["| " + " | ".join(r) + " |" for r in d["trazabilidad"]]
    L += ["", f"{h2} Observaciones de consistencia (para revisar con la asesora)\n"]
    L += [f"{i}. {o}" for i, o in enumerate(d["observaciones"], 1)]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------ Word
def construir_en(doc, d: dict, con_titulo: bool = True):
    """Agrega la matriz (tabla + trazabilidad + observaciones) a un Document que YA esta en hoja horizontal."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

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

    if con_titulo:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(d["titulo"].upper()); r.bold = True; r.font.size = Pt(18); r.font.color.rgb = RGBColor(0x7A, 0x1F, 0x2B)
    else:
        doc.add_heading("Anexo A. Matriz de consistencia", 1)
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
    doc.add_heading("Trazabilidad con el repositorio", 2 if not con_titulo else 1)
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
    doc.add_heading("Observaciones de consistencia (para revisar con la asesora)", 2 if not con_titulo else 1)
    for o in d["observaciones"]:
        doc.add_paragraph(o, style="List Number")


def a_docx(d: dict, ruta: Path):
    from docx import Document
    from docx.enum.section import WD_ORIENT
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
    construir_en(doc, d, con_titulo=True)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(ruta))


# ------------------------------------------------------------------------------------ evidencia actual
COLOR_EVIDENCIA = {"Cumple": "#DDEEDD", "No cumple": "#F6D5D5", "Parcial": "#FFF1CC", "Pendiente": "#EEEEEE",
                   "Informativo": "#DCE9F7"}


def evidencia_actual(procesados: Path, logs: Path | None = None, raw: Path | None = None):
    """Compara los indicadores de la matriz v4 con lo que HOY se puede medir con los datos y corridas del repositorio.
    Estados: Cumple / No cumple (medido) · Parcial · Pendiente · Informativo. Nunca usa la PRUEBA salvo que ya esté registrada en
    logs/resultados (generada una sola vez por src/experimentos.py)."""
    import numpy as np
    import pandas as pd

    from src import config as C, eda, experimentos as ex
    from src.particion import cargar_meta

    kw = {"dtype": {"id_fruto": str, "lote": str, "fecha": str}}
    fr = pd.read_csv(procesados / "indice_frutos.csv", **kw)
    filas = []

    def add(h, ind, meta, val, est_, nota=""):
        filas.append((h, ind, meta, val, est_, nota))

    completos = int(fr["completo"].sum()) if "completo" in fr else len(fr)
    add("Datos", "Frutos con las 5 vistas", f"≥ {C.TOTAL_VALIDOS}", str(completos), "Cumple" if completos >= C.TOTAL_VALIDOS else "No cumple")
    t = eda.tabla_cuotas(fr)
    falta = ", ".join(f"{k} {int(v)}" for k, v in t["faltan"].items() if v > 0)
    add("Datos", "Cuotas por categoría (320/320/320/270/270)", "5 de 5 categorías", f"{int(t['cumple'].sum())} de 5" + (f"; faltan: {falta}" if falta else ""),
        "Cumple" if t["cumple"].all() else "No cumple")
    cob = eda.cobertura_lote_fecha(fr)
    add("Datos", "Lotes distintos", f"≥ {eda.MIN_LOTES}", str(cob["n_lotes"]), "Cumple" if cob["cumple_lotes"] else "No cumple")
    add("Datos", "Fechas distintas", f"≥ {eda.MIN_FECHAS}", str(cob["n_fechas"]), "Cumple" if cob["cumple_fechas"] else "No cumple")
    conc = eda.concentracion_por_sesion(fr)
    add("Datos", f"Ningún lote ni fecha aporta > {eda.MAX_PCT_SESION:.0f} % de una categoría", "todas las categorías",
        f"{int(conc['cumple_4_3'].sum())} de {len(conc)} cumplen (máx. lote {conc['pct_lote'].max():.0f} %, fecha {conc['pct_fecha'].max():.0f} %)",
        "Cumple" if conc["cumple_4_3"].all() else "No cumple", "Si no se cumple, la sesión puede predecir la categoría")
    meta = cargar_meta(procesados)
    if meta:
        add("Datos", "Partición por cuotas (880/220/400) con prueba balanceada", "no provisional",
            "PROVISIONAL" if meta.get("provisional") else "completa", "No cumple" if meta.get("provisional") else "Cumple",
            "; ".join(meta.get("avisos", [])[:1]))
    else:
        add("Datos", "Partición congelada", "particion.csv", "sin partición", "Pendiente", "Ejecute src/particion.py")

    r = ex.cargar_ultimo()
    p = (r or {}).get("prueba", {})
    ctr = p.get("contrastes", {}) if p.get("disponible") else {}
    dec = ctr.get("decision", {})
    est_de = lambda k: {"se acepta": "Cumple", "no se acepta": "No cumple"}.get(dec.get(k, {}).get("estado"), "Pendiente")
    f4 = lambda x: "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
    if ctr:
        h1 = ctr["H1"]
        add("H1", "MAE / RMSE / exactitud adyacente del calibre", "≤ 0,50 mm / ≤ 0,70 mm / ≥ 0,98",
            f"MAE {f4(h1.get('mae'))} mm, RMSE {f4(h1.get('rmse'))} mm, adyacente {f4(h1.get('adyacente'))}" if h1.get("n") else "sin calibre", est_de("H1"), "Prueba")
        h2 = ctr["H2"]
        add("H2", "Incremento del recall de Fábrica (5 vistas − 1 vista)", "≥ 15 pp, límite inferior IC > 0",
            f"{f4(h2.get('diferencia_pp'))} pp (IC inf {f4(h2.get('ic_inf'))})" if "diferencia" in h2 else "sin evidencia", est_de("H2"), "Prueba")
        h3 = ctr["H3"]
        add("H3", "Diferencia de F1 macro híbrido − directo", "límite inferior IC > 0 (≥ 0,05 relevancia)",
            f"{f4(h3.get('diferencia'))} (IC inf {f4(h3.get('ic_inf'))})" if "diferencia" in h3 else "sin evidencia", est_de("H3"), "Prueba")
        h4 = ctr["H4"]
        add("H4", "κ_sis − κ_op", "límite inferior IC ≥ −0,05", f"{f4(h4.get('estimado'))} (IC inf {f4(h4.get('ic_inf'))})" if "estimado" in h4 else "sin operarios",
            est_de("H4"), "Prueba")
        hg = ctr["HG"]
        add("General", "F1 macro (límite inferior del IC95 %)", "≥ 0,85", f"{f4(hg['f1_macro'])} (IC inf {f4(hg['ic_inf'])})", "Cumple" if hg["acepta"] else "No cumple", "Prueba")
    else:
        for h, ind, meta_ in (("H1", "MAE / RMSE / exactitud adyacente del calibre", "≤ 0,50 / ≤ 0,70 mm / ≥ 0,98"),
                              ("H2", "Incremento del recall de Fábrica", "≥ 15 pp"), ("H3", "F1 macro híbrido − directo", "IC inf > 0"),
                              ("H4", "κ_sis − κ_op", "≥ −0,05"), ("General", "F1 macro (IC inf)", "≥ 0,85")):
            add(h, ind, meta_, "la prueba aún no se evalúa", "Pendiente", "Se evalúa UNA sola vez con modelos y umbrales congelados")
    des = (r or {}).get("desarrollo", {})
    hb = des.get("hibrido_base_val")
    if hb:
        add("Desarrollo", "Híbrido en validación (informativo)", "—", f"F1 macro {f4(hb['f1_macro'])}; MAE {f4(hb.get('mae'))} mm", "Informativo", "No es resultado confirmatorio")
    return pd.DataFrame(filas, columns=["Hipótesis/objetivo", "Indicador", "Meta", "Valor actual", "Estado", "Nota"])


def estilo_evidencia(df):
    """Styler de pandas con el estado coloreado (para el notebook)."""
    return (df.style.map(lambda v: f"background-color: {COLOR_EVIDENCIA.get(v, 'white')}; font-weight: bold", subset=["Estado"])
              .set_properties(**{"text-align": "left", "white-space": "normal"}).hide(axis="index"))


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
    ap.add_argument("--salida", type=Path, default=ROOT / "docs")
    a = ap.parse_args()
    r = generar(a.yaml, a.salida)
    print(f"Matriz generada:\n  {r['docx']}\n  {r['md']}")
