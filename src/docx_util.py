"""
docx_util.py
------------
Utilidades para EDITAR el informe Word de la plantilla (python-docx + XML): párrafos con minimarcado (**negrita**, *cursiva*, _{subíndice}),
tablas con el estilo de la plantilla, figuras y localización de secciones por su título. Las usa `informe_parcial.py`.
"""
from __future__ import annotations

import copy
import re

import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.table import Table
from docx.text.paragraph import Paragraph

XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def _rpr(bold=False, italic=False, size=None, color=None, sub=False):
    r = OxmlElement("w:rPr")
    if bold:
        r.append(OxmlElement("w:b"))
    if italic:
        r.append(OxmlElement("w:i"))
    if color:
        c = OxmlElement("w:color"); c.set(qn("w:val"), color); r.append(c)
    if size:
        s = OxmlElement("w:sz"); s.set(qn("w:val"), str(int(size * 2))); r.append(s)
        s2 = OxmlElement("w:szCs"); s2.set(qn("w:val"), str(int(size * 2))); r.append(s2)
    if sub:
        v = OxmlElement("w:vertAlign"); v.set(qn("w:val"), "subscript"); r.append(v)
    l = OxmlElement("w:lang"); l.set(qn("w:val"), "es-PE"); r.append(l)
    return r


def add_runs(p, text, bold=False, italic=False, size=None, color=None):
    text = re.sub(r"\b([Lab])\*(?=[\s,.)]|$)", lambda m: m.group(1) + "\x00", text)
    for part in re.split(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\})", text):
        if not part:
            continue
        b, i, sb = bold, italic, False
        if part.startswith("_{") and part.endswith("}"):
            part, sb = part[2:-1], True
        elif part.startswith("**") and part.endswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            part, i = part[1:-1], True
        r = OxmlElement("w:r"); r.append(_rpr(b, i, size, color, sb))
        t = OxmlElement("w:t"); t.text = part.replace("\x00", "*"); t.set(XML_SPACE, "preserve"); r.append(t)
        p.append(r)


def mk_p(text, style=None, bold=False, italic=False, size=None, color=None, keep_next=False, after=None, align=None):
    p = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    if style:
        ps = OxmlElement("w:pStyle"); ps.set(qn("w:val"), style); ppr.append(ps)
    if keep_next:
        ppr.append(OxmlElement("w:keepNext"))
    if after is not None:
        sp = OxmlElement("w:spacing"); sp.set(qn("w:after"), str(after)); ppr.append(sp)
    if align:
        jc = OxmlElement("w:jc"); jc.set(qn("w:val"), align); ppr.append(jc)
    p.append(ppr)
    add_runs(p, text, bold, italic, size, color)
    return p


def set_cell(cell, text, bold=False, size=9):
    tc = cell._tc
    ps = tc.findall(qn("w:p"))
    for extra in ps[1:]:
        tc.remove(extra)
    p0 = ps[0]
    for ch in list(p0):
        if ch.tag != qn("w:pPr"):
            p0.remove(ch)
    lines = str(text).split("\n")
    add_runs(p0, lines[0], bold=bold, size=size)
    for ln in lines[1:]:
        np_ = OxmlElement("w:p")
        add_runs(np_, ln, bold=bold, size=size)
        tc.append(np_)


def fmt_table(doc, tbl_el, widths=None):
    t = Table(tbl_el, doc)
    tblPr = tbl_el.find(qn("w:tblPr"))
    if widths:
        tw = tblPr.find(qn("w:tblW"))
        tw.set(qn("w:w"), str(sum(widths))); tw.set(qn("w:type"), "dxa")
        old = tblPr.find(qn("w:tblLayout"))
        if old is not None:
            tblPr.remove(old)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
        look = tblPr.find(qn("w:tblLook"))
        (look.addprevious(lay) if look is not None else tblPr.append(lay))
        grid = tbl_el.find(qn("w:tblGrid"))
        for g in list(grid):
            grid.remove(g)
        for w in widths:
            g = OxmlElement("w:gridCol"); g.set(qn("w:w"), str(w)); grid.append(g)
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:cantSplit")) is None:
            trPr.insert(0, OxmlElement("w:cantSplit"))
        if ri == 0 and trPr.find(qn("w:tblHeader")) is None:
            trPr.append(OxmlElement("w:tblHeader"))
        for ci, cell in enumerate(row.cells):
            tcPr = cell._tc.get_or_add_tcPr()
            if widths:
                tcw = tcPr.find(qn("w:tcW"))
                if tcw is None:
                    tcw = OxmlElement("w:tcW"); tcPr.insert(0, tcw)
                tcw.set(qn("w:w"), str(widths[ci])); tcw.set(qn("w:type"), "dxa")
            if ri == 0:
                for o in tcPr.findall(qn("w:shd")):
                    tcPr.remove(o)
                shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E7EBEE")
                tcPr.append(shd)
            for p in cell._tc.findall(qn("w:p")):
                ppr = p.find(qn("w:pPr"))
                if ppr is None:
                    ppr = OxmlElement("w:pPr"); p.insert(0, ppr)
                for o in ppr.findall(qn("w:spacing")):
                    ppr.remove(o)
                sp = OxmlElement("w:spacing"); sp.set(qn("w:before"), "40"); sp.set(qn("w:after"), "40"); sp.set(qn("w:line"), "240"); sp.set(qn("w:lineRule"), "auto")
                idx = 0
                for k, ch in enumerate(list(ppr)):
                    if ch.tag in (qn("w:pStyle"), qn("w:keepNext"), qn("w:keepLines")):
                        idx = k + 1
                ppr.insert(idx, sp)


def fill_table(doc, tbl_el, header, rows, widths=None):
    """Reescribe una tabla existente con `header` + `rows` (añade o quita filas según haga falta)."""
    t = Table(tbl_el, doc)
    need = len(rows) + 1
    while len(t.rows) < need:
        tbl_el.append(copy.deepcopy(t.rows[-1]._tr)); t = Table(tbl_el, doc)
    while len(t.rows) > need:
        tbl_el.remove(t.rows[-1]._tr); t = Table(tbl_el, doc)
    if header:
        for c, h in zip(t.rows[0].cells, header):
            set_cell(c, h, bold=True)
    for r, vals in zip(t.rows[1:], rows):
        for c, v in zip(r.cells, vals):
            set_cell(c, v)
    fmt_table(doc, tbl_el, widths)


def new_table(doc, header, rows, widths):
    t = doc.add_table(rows=len(rows) + 1, cols=len(header))
    t.style = doc.styles["Table Grid"]
    el = t._tbl
    el.getparent().remove(el)
    tblPr = el.find(qn("w:tblPr"))
    jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); tblPr.find(qn("w:tblW")).addnext(jc)
    for c, h in zip(t.rows[0].cells, header):
        set_cell(c, h, bold=True)
    for r, vals in zip(t.rows[1:], rows):
        for c, v in zip(r.cells, vals):
            set_cell(c, v)
    fmt_table(doc, el, widths)
    return el


def mk_fig(doc, ruta, ancho_in=5.2):
    p = doc.add_paragraph()
    p.alignment = 1
    p.add_run().add_picture(str(ruta), width=Inches(ancho_in))
    el = p._p
    el.getparent().remove(el)
    ppr = el.find(qn("w:pPr"))
    if ppr is not None:
        ps = ppr.find(qn("w:pStyle"))
        kn = OxmlElement("w:keepNext")
        (ps.addnext(kn) if ps is not None else ppr.insert(0, kn))
    return el


# ------------------------------------------------------------------------------------ localizar secciones
def _nivel(p):
    m = re.match(r"Heading (\d)", p.style.name) if p.style is not None else None
    return int(m.group(1)) if m else None


def cuerpo(doc):
    return list(doc.element.body)


def buscar_heading(doc, prefijo: str):
    """Elemento <w:p> del encabezado cuyo texto empieza por `prefijo` (None si no existe)."""
    for el in cuerpo(doc):
        if el.tag == qn("w:p"):
            p = Paragraph(el, doc)
            if _nivel(p) and p.text.strip().startswith(prefijo):
                return el
    return None


def elementos_seccion(doc, heading_el):
    """Elementos entre el encabezado y el siguiente encabezado (cualquier nivel)."""
    els, activo = [], False
    for el in cuerpo(doc):
        if el is heading_el:
            activo = True
            continue
        if activo:
            if el.tag == qn("w:p") and _nivel(Paragraph(el, doc)):
                break
            els.append(el)
    return els


def es_marcador(doc, el) -> bool:
    """Párrafos que son INDICACIÓN de la plantilla o marcadores de contenido pendiente."""
    if el.tag != qn("w:p"):
        return False
    p = Paragraph(el, doc)
    t = p.text.strip()
    return (p.style is not None and p.style.name == "Indicacion") or t.startswith("[Redactar") or t.startswith("[Pendiente")


def reemplazar_marcadores(doc, heading_el, bloques: list, quitar_indicacion=True, solo_pendientes_de=None):
    """Inserta `bloques` (elementos XML) donde estaba el primer marcador de la sección y elimina los marcadores
    (INDICACIÓN, [Redactar...], [Pendiente...]). Si no hay marcadores, inserta al final de la sección."""
    els = elementos_seccion(doc, heading_el)
    marc = [e for e in els if es_marcador(doc, e) and (quitar_indicacion or Paragraph(e, doc).style.name != "Indicacion")]
    if solo_pendientes_de is not None:
        marc = [e for e in marc if solo_pendientes_de in Paragraph(e, doc).text or Paragraph(e, doc).style.name == "Indicacion"]
    if marc:
        ref = marc[0]
        ancla = ref.getprevious()
        for e in marc:
            e.getparent().remove(e)
        pos = ancla if ancla is not None else heading_el
    else:
        pos = els[-1] if els else heading_el
    for b in bloques:
        pos.addnext(b)
        pos = b
    return pos


def siguiente_tabla(doc, heading_el, k: int = 0):
    ts = [e for e in elementos_seccion(doc, heading_el) if e.tag == qn("w:tbl")]
    return ts[k] if len(ts) > k else None


def reemplazar_texto_parrafo(doc, empieza_por: str, nuevo: str):
    for el in cuerpo(doc):
        if el.tag == qn("w:p"):
            p = Paragraph(el, doc)
            if p.text.strip().startswith(empieza_por):
                runs = p.runs
                runs[0].text = nuevo
                for r in runs[1:]:
                    r._r.getparent().remove(r._r)
                return True
    return False


def siguiente_numero_tabla(doc) -> int:
    n = 0
    for p in doc.paragraphs:
        m = re.match(r"Tabla (\d+)\.", p.text.strip())
        if m:
            n = max(n, int(m.group(1)))
    return n + 1
