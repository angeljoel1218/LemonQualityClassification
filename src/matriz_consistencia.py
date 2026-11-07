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
    """Compara los indicadores de la matriz con lo que HOY se puede medir con los datos y corridas del repositorio.
    Estados: Cumple / No cumple (medido) · Parcial (hay una medida parcial o un sustituto) · Pendiente (falta
    implementar o recolectar) · Informativo. Las corridas de otra particion se ignoran. Nunca usa TEST."""
    import json

    import numpy as np
    import pandas as pd

    from src import calibracion, eda, estadistica as est, evidencia_hipotesis as eh, fe_experimentos as fx, informe
    from src.extraer_features import cargar_cfg
    from src.fe_columnas import columnas_para
    from src.particion import huella_particion

    logs = Path(logs) if logs else ROOT / "logs"
    kw = {"dtype": {"id_fruto": str, "lote": str, "fecha": str}}
    fr = pd.read_csv(procesados / "indice_frutos.csv", **kw)
    fr["cat"] = eda.normalizar_categorias(fr["categoria_ref"])
    ff = pd.read_csv(procesados / "features_frutos.csv", **kw) if (procesados / "features_frutos.csv").exists() else None
    part = pd.read_csv(procesados / "particion.csv", dtype={"id_fruto": str}) if (procesados / "particion.csv").exists() else None
    h_actual = huella_particion(part) if part is not None else None
    filas = []

    def add(h, ind, meta, val, est_, nota=""):
        filas.append((h, ind, meta, val, est_, nota))

    def corrida(nombre, marca="config.json"):
        d = informe._ultimo(logs / nombre, marca)
        if d is None:
            return None
        meta = json.loads((d / "config.json").read_text(encoding="utf-8"))["meta"]
        return d if (h_actual and meta.get("huella_particion") == h_actual) else None

    # ---- Datos (Tabla 4.1, 4.3 y 4.4)
    completos = int(fr["completo"].sum()) if "completo" in fr else len(fr)
    add("Datos", "Frutos con las 5 vistas", "≥ 1 500", str(completos), "Cumple" if completos >= 1500 else "No cumple")
    t = eda.tabla_cuotas(fr)
    falta = ", ".join(f"{k} {int(v)}" for k, v in t["faltan"].items() if v > 0)
    add("Datos", "Cuotas mínimas por categoría (250/250/250/200/200)", "5 de 5 categorías",
        f"{int(t['cumple'].sum())} de 5" + (f"; faltan: {falta}" if falta else ""), "Cumple" if t["cumple"].all() else "No cumple")
    cob = eda.cobertura_lote_fecha(fr)
    add("Datos", "Lotes distintos", "≥ 3", str(cob["n_lotes"]), "Cumple" if cob["cumple_lotes"] else "No cumple")
    add("Datos", "Fechas distintas", "≥ 4", str(cob["n_fechas"]), "Cumple" if cob["cumple_fechas"] else "No cumple")
    conc = eda.concentracion_por_sesion(fr)
    add("Datos", "Ninguna categoría concentrada en un solo lote ni en una sola fecha (4.3)", "≤ 50 % por lote y por fecha",
        f"{int(conc['cumple_4_3'].sum())} de {len(conc)} categorías cumplen (máx. lote {conc['pct_lote'].max():.0f} %, fecha {conc['pct_fecha'].max():.0f} %)",
        "Cumple" if conc["cumple_4_3"].all() else "No cumple", "Si no se cumple, la sesión puede predecir la categoría")
    if part is not None:
        add("Datos", "Un mismo fruto no aparece en dos subconjuntos", "0 repetidos", str(int(part["id_fruto"].duplicated().sum())),
            "Cumple" if not part["id_fruto"].duplicated().any() else "No cumple")
    else:
        add("Datos", "Un mismo fruto no aparece en dos subconjuntos", "0 repetidos", "sin partición", "Pendiente", "Ejecute src/particion.py")

    df = None
    if ff is not None and part is not None:
        try:
            df = fx.cargar_datos(procesados, cargar_cfg())
        except Exception:
            df = None
    if df is not None:
        a = eh.atajo_sesion(df, columnas_para({"A"}))
        add("Datos", "Piso 'solo conoce la sesión': F1 macro en validación", "≪ F1 de los modelos",
            f"{a['f1_macro_por_sesion']:.2f} (dummy ≈ 0,07); las features identifican la sesión con exactitud {a['exactitud_sesion_desde_features']:.2f}",
            "Informativo", "Un modelo solo vale si supera este piso con claridad")

    # ---- H1: panel
    panel = (raw / "panel_evaluaciones.csv") if raw else None
    add("H1", "κ de Fleiss / repetibilidad intra-evaluador del panel", "≥ 0,75 / ≥ 0,80",
        "hay datos del panel" if panel is not None and panel.exists() else "sin panel_evaluaciones.csv", "Pendiente",
        "Falta el script que calcula κ de Fleiss, Cohen y AC1 sobre el panel")

    # ---- H2: calibre
    if df is not None:
        tr = df[df.subset == "train"]
        if df["diam_ecu_mm"].notna().any():
            e_, r_, nota, estado_ok = df["diam_ecu_mm"], df["calibre_mm_ref"], "Con features clásicas; no es el modelo profundo", True
        else:
            e_, r_ = calibracion.aplicar_escala(df, calibracion.escala_provisional(tr)), df["calibre_mm_ref"]
            nota, estado_ok = "SUSTITUTO: sin calibracion_sesiones.csv; escala por sesión estimada en train (no es la calibración del protocolo)", False
        m = calibracion.metricas_calibre(e_[df.subset != "train"], r_[df.subset != "train"])
    elif ff is not None and "diam_ecu_mm" in ff and ff["diam_ecu_mm"].notna().any():
        b_ = ff.merge(fr[["id_fruto", "calibre_mm_ref"]], on="id_fruto")
        m, nota, estado_ok = calibracion.metricas_calibre(b_["diam_ecu_mm"], b_["calibre_mm_ref"]), "Con features clásicas; no es el modelo profundo", True
    else:
        m, nota, estado_ok = None, "", False
    if m is not None and np.isfinite(m["mae"]):
        add("H2", "MAE del calibre (imagen vs calibrador)", "≤ 1,0 mm", f"{m['mae']:.2f} mm (n = {m['n']})",
            ("Cumple" if m["mae"] <= 1.0 else "No cumple") if estado_ok else "Parcial", nota)
        add("H2", "RMSE del calibre", "≤ 1,5 mm", f"{m['rmse']:.2f} mm", ("Cumple" if m["rmse"] <= 1.5 else "No cumple") if estado_ok else "Parcial",
            "Bland–Altman y error por calibre crítico en Feature_Engineering §2")
    else:
        add("H2", "MAE / RMSE del calibre", "≤ 1,0 / ≤ 1,5 mm", "sin calibre en mm", "Pendiente",
            "Falta calibracion_sesiones.csv o calibre_mm_ref; ejecute extraer_features.py")
    crit = eda.frutos_cerca_de_criticos(fr)
    sin = crit.loc[crit.frutos == 0, "calibre_critico_mm"].tolist()
    add("H2", "Frutos disponibles en los calibres críticos (34,5; 35; 36; 37; 38; 39; 42 mm)", "error reportable en los 7",
        f"sin frutos en: {sin}" if sin else "los 7 tienen frutos", "No cumple" if sin else "Cumple")

    # ---- H3 (E4)
    vistas_p = procesados / "features_vistas.csv"
    if vistas_p.exists() and part is not None and "severidad_defecto_panel" in fr:
        v = pd.read_csv(vistas_p, dtype={"id_fruto": str, "lote": str, "fecha": str, "vista": str})
        e4 = eh.e4_defectos_omitidos(v, fr.merge(part, on="id_fruto"), subconjuntos=["train", "val"])
        mc = e4["mcnemar"]
        add("H3", "Reducción de Fábrica-por-defecto omitida, 1 vista (0°) vs 5 vistas", "≥ 15 pp y McNemar significativo",
            f"{e4['reduccion_pp']:.1f} pp ({e4['pct_omitidos_1vista']:.0f} % → {e4['pct_omitidos_5vistas']:.0f} %), p = {mc['p']:.2f}",
            "Cumple" if e4["cumple"] else "No cumple", "Detector clásico de defectos; severidad panel ≥ 0,5 provisional")
    else:
        add("H3", "Reducción de defectos omitidos, 1 vs 5 vistas", "≥ 15 pp", "sin features por vista", "Pendiente")

    # ---- H4 (E1)
    if df is not None:
        h4 = eh.h4_linea_base(df[df.subset == "train"], columnas_para({"A"}))
        add("H4", "Línea base sin preprocesamiento: F1 color/madurez y F1 Primera–Segunda", "+5 / +3 pp sobre esta base",
            f"{h4['f1_color']:.2f} / {h4['f1_primera_segunda']:.2f}", "Parcial", "Falta src/preprocesamiento.py (carta de color) para medir la ganancia")
    else:
        add("H4", "F1 de color/madurez y del par Primera–Segunda", "+5 / +3 pp", "sin features", "Pendiente")

    # ---- H5 (E3)
    bl = corrida("baseline_runs", "resultados.csv")
    if bl is not None:
        res = pd.read_csv(bl / "resultados.csv")
        h5 = eh.h5_por_clase(res)
        if len(h5) and "Δ F1 Tercera" in h5:
            h5 = h5[h5["variante"] != "BASE"]
            txt = "; ".join(f"{x.modelo}/{x.variante.replace('V2_', '')}: ΔF1 Tercera {x['Δ F1 Tercera']:+.2f}, ΔF1 Fábrica {x['Δ F1 Fabrica']:+.2f}" for _, x in h5.iterrows())
            add("H5", "ΔF1 de Tercera y Fábrica con balanceo (sobre/submuestreo)", "+8 pp", txt or "sin variantes", "Parcial",
                "Faltan focal loss y aumento dirigido (modelo profundo); pesos de clase en Hiperparametros.ipynb")
    else:
        add("H5", "ΔF1 de Tercera y Fábrica con balanceo", "+8 pp", "sin corrida de baseline de esta partición", "Pendiente")

    # ---- H6
    add("H6", "F1 macro de atributos+regla frente a clasificador directo", "+3 pp; IC95 % excluye 0", "modelos profundos sin entrenar",
        "Pendiente", "Regla jerárquica y bootstrap pareado listos; faltan MVCNN y multitarea")
    imp = corrida("importancia_runs")
    if imp is not None:
        from src import importancia
        o = importancia.cargar_run(imp)
        m0 = list(o["perm"])[0]
        dt = o["dim_tabla"][m0]
        add("H6 (apoyo)", "Dimensión que más sostiene el F1 macro", "coherente con la regla comercial (calibre y defecto/color)",
            f"{dt['macro'].idxmax()} (caída {dt['macro'].max():.2f}, {m0})", "Informativo", "Importancia con modelos clásicos")
    hp = corrida("hiperparametros_runs")
    if hp is not None:
        mj = pd.read_csv(hp / "mejores.csv")
        top = mj.sort_values("cv_f1_macro_media", ascending=False).iloc[0]
        add("Hiperparámetros", "Mejor candidato por CV (presupuesto fijo por modelo)", "—",
            f"{top.modelo} ({top.balanceo}): F1 macro CV {top.cv_f1_macro_media:.3f}, VAL {top.val_f1_macro:.3f}", "Informativo",
            "Ver comparación pareada contra el baseline en Hiperparametros.ipynb §8")

    # ---- H7
    add("H7", "κ del sistema − κ del operario", "≥ −0,05 (límite inferior del IC95 %)", "sin evaluaciones de operarios", "Pendiente",
        "src/estadistica.py ya trae no_inferioridad, McNemar y Holm")

    # ---- H8
    if ff is not None and "t_extraccion_ms" in ff:
        med, p95 = float(ff["t_extraccion_ms"].median()), float(ff["t_extraccion_ms"].quantile(.95))
        add("H8", "Latencia mediana por fruto", "≤ 200 ms (todas las vistas, GPU)", f"extracción de features: mediana {med:.0f} ms, p95 {p95:.0f} ms",
            "Parcial", "Solo la extracción clásica en CPU; falta el modelo profundo en GPU y la prueba de estabilidad (E6)")
    else:
        add("H8", "Latencia mediana por fruto", "≤ 200 ms", "sin features extraídas", "Pendiente", "Ejecute extraer_features.py")

    # ---- General: estimaciones informativas en VAL con el mejor baseline (nunca TEST)
    gen = "requiere el modelo profundo y las 5 categorías"
    estado = "Pendiente"
    if bl is not None and (bl / "predicciones_val.csv").exists():
        res = pd.read_csv(bl / "resultados.csv")
        b = res[(res.variante == "BASE") & (res.modelo != "dummy")]
        mejor = b.loc[b.cv_f1_macro_media.idxmax(), "modelo"]
        pv = pd.read_csv(bl / "predicciones_val.csv", dtype=str)
        y, p_ = pv["y"].to_numpy(), pv[f"BASE|{mejor}"].to_numpy()
        f1 = est.bootstrap_pareado(y, p_, B=1000)
        rec = est.bootstrap_pareado(y, p_, metrica=lambda a_, b_: est.recall_clase(a_, b_, "Fabrica"), B=1000)
        kap = est.bootstrap_pareado(y, p_, metrica=est.kappa_cohen, B=1000)
        gen = (f"VAL con {mejor} (informativo): F1 macro {f1['estimado']:.2f} [IC inf {f1['ic_inf']:.2f}], recall Fábrica {rec['estimado']:.2f} "
               f"[{rec['ic_inf']:.2f}], κ {kap['estimado']:.2f} [{kap['ic_inf']:.2f}]")
        estado = "Informativo"
    add("General", "F1 macro / recall de Fábrica / κ (límite inferior del IC95 %, B = 1000)", "≥ 0,90 / ≥ 0,90 / ≥ 0,80", gen, estado,
        "Se evalúa una sola vez en TEST con el sistema final; no usar estas cifras como resultado")
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
