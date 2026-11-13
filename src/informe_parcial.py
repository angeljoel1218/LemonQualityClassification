"""
informe_parcial.py
------------------
COMPLETA las secciones pendientes del informe Word (Informe_Parcial_PI2_Plan_de_Tesis_Limon_UNI.docx) con los datos y resultados REALES
del repositorio. Regla de oro: solo escribe lo que se puede obtener de los archivos y las corridas; lo que falta queda marcado como
[Pendiente: ...] y NO se rellena con valores supuestos. Se puede ejecutar varias veces (siempre parte del informe base):
cada vez que haya resultados nuevos se vuelve a generar.

    python src/informe_parcial.py --base informe/plantilla/Informe_Parcial_PI2_Plan_de_Tesis_Limon_UNI.docx \
        --salida informe/Informe_Parcial_completado.docx --mes noviembre --docente "Nombre del docente"

Secciones que completa (si hay evidencia): portada (mes, docente), 5.1 (producto de preprocesamiento), 6.1 (versión/hash, fechas),
6.2 (diccionario de datos y EDA), 6.3 (si existe docs/etica.md), 6.4, 7.2 (tamaños reales), 7.6, 7.7, 8 (intro), 8.1-8.5, 9 (intro),
9.1, 9.2 (columna de evidencia), 9.3, Anexo D (registro de corridas). Opcionales: docs/autorizaciones.md (licencia y autorizaciones),
docs/etica.md (sección 6.3), data/raw/panel_evaluaciones.csv, operarios_clasificaciones.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import docx  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

from src import config as C  # noqa: E402
from src import docx_util as du  # noqa: E402
from src import eda  # noqa: E402
from src import experimentos as ex  # noqa: E402
from src import regla_comercial as rc  # noqa: E402
from src.ingesta import leer_csv  # noqa: E402
from src.particion import cargar_meta, cargar_particion, huella_particion, tabla_asignacion  # noqa: E402

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "setiembre", "octubre", "noviembre", "diciembre"]
LOGS = ROOT / "logs"


def n2(x, d=2):
    """Número con coma decimal ('-' si falta)."""
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "-"
    return f"{x:.{d}f}".replace(".", ",")


def ms(m, s, d=3):
    return f"{n2(m, d)} ± {n2(s, d)}"


def pend(txt):
    return ("pend", txt)


def P(t):
    return ("p", t)


def B(t):
    return ("b", t)


# =====================================================================================================================
# recolección de hechos
# =====================================================================================================================
def _ultimo(dir_runs: Path, marca="resultados.csv"):
    c = sorted(dir_runs.glob(f"*/{marca}")) if dir_runs.exists() else []
    return c[-1].parent if c else None


def _meta(d: Path) -> dict:
    return json.loads((d / "config.json").read_text(encoding="utf-8")).get("meta", {}) if d and (d / "config.json").exists() else {}


def recolectar(procesados: Path, raw: Path) -> dict:
    h = dict(avisos=[])
    kw = dict(dtype={"id_fruto": str, "lote": str, "fecha": str})
    h["frutos"] = pd.read_csv(procesados / "indice_frutos.csv", **kw) if (procesados / "indice_frutos.csv").exists() else None
    h["imgs"] = pd.read_csv(procesados / "indice_imagenes.csv", dtype=str) if (procesados / "indice_imagenes.csv").exists() else None
    h["part"] = cargar_particion(procesados) if (procesados / "particion.csv").exists() else None
    h["part_meta"] = cargar_meta(procesados)
    h["huella_part"] = huella_particion(h["part"]) if h["part"] is not None else None
    h["resultados"] = ex.cargar_ultimo()
    h["piloto"] = json.loads((LOGS / "piloto.json").read_text(encoding="utf-8")) if (LOGS / "piloto.json").exists() else None
    for k, d in (("baseline", "baseline_runs"), ("hp", "hiperparametros_runs"), ("fe", "fe_runs")):
        run = _ultimo(LOGS / d, "resultados.csv" if k != "hp" else "mejores.csv")
        if run is not None and h["huella_part"] and _meta(run).get("huella_particion") != h["huella_part"]:
            h["avisos"].append(f"Se ignoró la última corrida de {d} ({run.name}): se hizo con otra partición. Vuelva a ejecutarla.")
            run = None
        h[k] = run
    h["bl"] = pd.read_csv(h["baseline"] / "resultados.csv") if h["baseline"] else None
    h["bl_cmp"] = pd.read_csv(h["baseline"] / "comparaciones.csv") if h["baseline"] and (h["baseline"] / "comparaciones.csv").exists() else None
    h["bl_cm"] = json.loads((h["baseline"] / "matrices_confusion_val.json").read_text(encoding="utf-8")) if h["baseline"] else None
    h["bl_meta"] = _meta(h["baseline"])
    h["hp_mejores"] = pd.read_csv(h["hp"] / "mejores.csv") if h["hp"] else None
    h["panel"] = None
    try:
        from src import panel as pn
        if (raw / "panel_evaluaciones.csv").exists():
            h["panel"] = pn.cargar_panel(raw / "panel_evaluaciones.csv")
    except SystemExit:
        pass
    h["raw"], h["procesados"] = raw, procesados
    return h


def huella_dataset(raw: Path) -> str | None:
    """SHA-256 de etiquetas.csv + nombres y tamaños de las imágenes (identifica la versión exacta del conjunto de datos)."""
    et = raw / "etiquetas.csv"
    if not et.exists():
        return None
    hs = hashlib.sha256(et.read_bytes())
    for p in sorted((raw / "imagenes").glob("*.*")) if (raw / "imagenes").exists() else []:
        hs.update(f"{p.name}:{p.stat().st_size}".encode())
    return hs.hexdigest()


# =====================================================================================================================
# secciones: cada función devuelve una lista de bloques ('p'|'b'|'pend'|'t'|'fig')
# =====================================================================================================================
def s61(h):
    fr, imgs = h["frutos"], h["imgs"]
    if fr is None or imgs is None:
        return [pend("versión del conjunto de datos y fechas de adquisición; ejecute src/ingesta.py")]
    hd = huella_dataset(h["raw"])
    fechas = sorted(set(imgs["fecha"].dropna().astype(str)))
    b = [P(f"**Versión del conjunto de datos.** Huella SHA-256 {hd[:16]}… del archivo etiquetas.csv más los nombres y tamaños de {len(imgs)} imágenes "
           f"({len(fr)} frutos; {int(fr['completo'].sum())} con las cinco vistas)." if hd else "**Versión del conjunto de datos.** No se pudo calcular la huella (falta etiquetas.csv)."),
         P(f"**Fecha de adquisición.** Capturas registradas entre {fechas[0]} y {fechas[-1]} (AAAAMMDD en el nombre de archivo): {len(fechas)} fechas y "
           f"{imgs['lote'].nunique()} lotes." if fechas else "**Fecha de adquisición.** No hay imágenes indexadas.")]
    aut = ROOT / "docs" / "autorizaciones.md"
    if aut.exists() and aut.read_text(encoding="utf-8").strip():
        b.append(P("**Licencia y autorizaciones.** " + aut.read_text(encoding="utf-8").strip().replace("\n", " ")))
    else:
        b.append(pend("licencia y autorizaciones de uso de los datos y de acceso a los centros de acopio; escríbalas en docs/autorizaciones.md y vuelva a generar el informe"))
    return b


DESC = {"id_fruto": ("texto", "Código único del fruto; vincula sus 5 imágenes"), "lote": ("texto", "Lote de cosecha (solo agrupa; no entra al modelo)"),
        "fecha": ("texto AAAAMMDD", "Fecha de captura (solo agrupa; no entra al modelo)"), "calibre_mm_ref": ("mm", "Diámetro ecuatorial medido con calibrador (vernier)"),
        "categoria_ref": ("categórica", "Categoría de referencia (variable objetivo): Extra, Primera, Segunda, Tercera, Fabrica"),
        "color_panel": ("categórica", "Color/madurez según el panel: verde, verde_amarillo, amarillo"),
        "severidad_defecto_panel": ("0-1 (escala histórica)", "Severidad del peor defecto según el panel"),
        "severidad_nivel_panel": ("ordinal 0-3", "Severidad según la rúbrica de la Tabla 4.2"), "tipo_defecto": ("categórica", "Tipo de defecto"),
        "piloto": ("0/1", "1 = fruto de la prueba piloto (fuera de la muestra definitiva)")}


def s62(h):
    fr = h["frutos"]
    if fr is None:
        return [pend("diccionario de datos y análisis exploratorio; ejecute src/ingesta.py")]
    rows = []
    for c in fr.columns:
        if c in ("vistas_presentes", "n_vistas", "vistas_faltantes", "completo", "origen"):
            continue
        s = fr[c]
        tipo, desc = DESC.get(c, ("", "(sin descripción en el diccionario)"))
        if pd.api.types.is_numeric_dtype(s):
            rango = f"{n2(s.min())} a {n2(s.max())}" if s.notna().any() else "-"
        else:
            u = s.dropna().unique()
            rango = ", ".join(map(str, sorted(u)[:6])) + (" …" if len(u) > 6 else "")
        rows.append([c, tipo or str(s.dtype), desc, rango, f"{100 * s.isna().mean():.1f}".replace(".", ",") + " %"])
    t = eda.tabla_cuotas(fr)
    cob = eda.cobertura_lote_fecha(fr)
    cal = eda.resumen_calidad_etiquetas(fr)["conteo"]
    inco, front = eda.inconsistencias_calibre_categoria(fr), eda.fronterizos(fr, 0.5)
    conc = eda.concentracion_por_sesion(fr)
    b = [P("**Diccionario de datos (etiquetas.csv).**"),
         ("t", "Diccionario de datos de etiquetas.csv", ["Variable", "Tipo", "Descripción", "Valores o rango", "Faltantes"], rows, [1500, 1300, 3200, 2000, 1000]),
         P(f"**Tamaño y balance.** {len(fr)} frutos indexados, {int(fr['completo'].sum())} con las cinco vistas. Frente a las cuotas válidas de la Tabla 4.1: " +
           "; ".join(f"{k} {int(v['frutos'])} de {int(v['cuota_min'])}" for k, v in t.iterrows()) + f". Cumplen la cuota {int(t['cumple'].sum())} de 5 categorías."),
         P(f"**Cobertura.** {cob['n_lotes']} lotes y {cob['n_fechas']} fechas de captura (el plan pide al menos {eda.MIN_LOTES} y {eda.MIN_FECHAS}). "
           + ("Ninguna categoría supera el " + f"{eda.MAX_PCT_SESION:.0f} % en un solo lote o fecha." if conc["cumple_4_3"].all() else
              f"{int((~conc['cumple_4_3']).sum())} de {len(conc)} categorías superan el {eda.MAX_PCT_SESION:.0f} % en un solo lote o fecha (máximo por lote "
              f"{conc['pct_lote'].max():.0f} %, por fecha {conc['pct_fecha'].max():.0f} %; {int((conc['pct_sesion'] >= 99).sum())} categorías provienen de una sola sesión): "
              "la sesión podría predecir la categoría.")),
         P(f"**Anomalías.** {int(cal['capturas_incompletas'])} capturas incompletas; {len(inco)} etiquetas no-Fábrica que contradicen su calibre; {len(front)} frutos "
           f"fronterizos (a ±0,5 mm de un límite de 35, 37 o 39 mm; son los más sensibles al error de medición); {int(cal['calibre_fuera_de_rango(25-60mm)'])} calibres fuera de 25–60 mm; "
           f"{int((fr['calibre_mm_ref'] > C.CALIBRE_MAXIMO).sum()) if 'calibre_mm_ref' in fr else 0} frutos con calibre mayor de 42,0 mm (excluidos de la muestra definitiva).")]
    if len(set(fr["categoria_ref"].dropna())) > 1:
        cr = {c: eda.asociacion_categoria_metadato(fr, c) for c in ("lote", "fecha")}
        b.append(P(f"**Riesgo de sesgo de contexto.** V de Cramér entre categoría y lote = {n2(cr['lote'])}; entre categoría y fecha = {n2(cr['fecha'])} "
                   "(0 = independiente; valores altos indican que el contexto de captura predice la categoría)."))
    faltan = [c for c in ("fecha_cosecha", "horas_cosecha_captura", "productor", "tipo_defecto") if c not in fr.columns]
    if faltan:
        b.append(P("**Campos que el plan pide y no existen todavía:** " + ", ".join(faltan) + "."))
    return b


def s63(h):
    et = ROOT / "docs" / "etica.md"
    if et.exists() and et.read_text(encoding="utf-8").strip():
        return [P(l.strip()) for l in et.read_text(encoding="utf-8").split("\n\n") if l.strip()]
    return None          # se conserva la INDICACIÓN y el marcador de la plantilla


def s64(h):
    cfg = {}
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "configs" / "fe.yaml").read_text(encoding="utf-8"))
    except Exception:
        pass
    tr = cfg.get("transformaciones", {})
    u = rc.cargar_umbrales()
    cong = rc.RUTA_UMBRALES.exists()
    b = [B("**Orden.** La partición por fruto se genera y se congela antes de cualquier cálculo con estado (huella " + (h["huella_part"] or "sin partición") + ")."),
         B(f"**Ajustado exclusivamente con entrenamiento (clasificadores clásicos).** Imputación por mediana, escalado, winsorización (percentiles {tr.get('winsor', '-')}), "
           f"log1p de variables asimétricas (umbral {tr.get('umbral_asimetria', '-')}), banderas de atípicos (IQR × {tr.get('outlier_k', '-')}) y filtro de colinealidad "
           f"(|r| > {tr.get('colinealidad_umbral', '-')}) viven dentro de un Pipeline: en la validación cruzada se ajustan solo con el pliegue de entrenamiento."),
         B("**Balanceo.** Los pesos de clase, el sobremuestreo y el submuestreo se calculan y aplican solo con el entrenamiento (dentro de `fit`); el aumento dirigido se aplica solo a Tercera y Fábrica del entrenamiento."),
         B("**Clasificador directo.** La normalización de entrada usa las constantes de ImageNet del backbone preentrenado (media y desviación fijas): no se estiman con estos datos."),
         B("**Preprocesamiento por imagen o por sesión.** La corrección de brillo (fondo del light box) y la de color (carta) se estiman con la propia imagen o sesión, nunca con etiquetas ni con otros frutos; la escala mm/px sale del patrón de cada sesión."),
         B(f"**Umbrales de la regla.** θ_{{def}} = {u.theta_def}, p_{{am}} = {n2(u.p_am)}, h_{{am}} = {n2(u.h_am, 1)}° "
           + (f"(CONGELADOS; huella {json.loads(rc.RUTA_UMBRALES.read_text(encoding='utf-8')).get('huella')})." if cong else "(valores de trabajo PROVISIONALES: se fijan con el piloto y se congelan con `python src/piloto.py congelar`).") +
           " Se calibran con piloto/validación y nunca con la prueba."),
         B("**Prueba.** No interviene en ninguna selección: se evalúa una sola vez y su archivo de predicciones se registra con su huella SHA-256 (logs/prueba/registro_prueba.json) antes de integrar panel y operarios.")]
    if not (h["raw"] / "calibracion_sesiones.csv").exists():
        b.append(B("**Aviso.** Falta data/raw/calibracion_sesiones.csv: el calibre en mm solo puede estimarse con una escala PROVISIONAL derivada de las etiquetas de entrenamiento (diagnóstico, no es la calibración del protocolo)."))
    return b


def _n_img(df_part, h):
    imgs = h["imgs"]
    if imgs is None or h["part"] is None:
        return {}
    m = imgs.merge(h["part"], on="id_fruto")
    return m.groupby("subset").size().to_dict()


def s72(doc, h, heading):
    """Añade los tamaños reales a la tabla de particiones y un párrafo con huella y estado."""
    tb = du.siguiente_tabla(doc, heading, 0)
    if tb is None or h["part"] is None or h["frutos"] is None:
        return
    ta = tabla_asignacion(h["frutos"], h["part"])
    n_img = _n_img(None, h)
    tot = ta[["train", "val", "test"]].sum()
    t = docx.table.Table(tb, doc)
    mapa = {"Entrenamiento": "train", "Validación": "val", "Prueba": "test"}
    for r in t.rows[1:]:
        nombre = r.cells[0].text.strip()
        for k, sub in mapa.items():
            if nombre.startswith(k):
                old = r.cells[2].text.strip().split("\n")[0]
                du.set_cell(r.cells[2], old + f"\nReal: {int(tot[sub])} frutos ({n_img.get(sub, 0)} imágenes)")
    meta = h["part_meta"]
    txt = f"**Partición real.** Huella {h['huella_part']}; semilla {meta.get('semilla', C.CFG['semilla_particion'])}. "
    if meta.get("provisional"):
        txt += "**PROVISIONAL:** no se alcanzaron las cuotas de la Tabla 4.1 (320/320/320/270/270); se aplicaron sus proporciones a los frutos disponibles y la prueba se mantiene balanceada (" + \
               ", ".join(f"{c} {int(ta.loc[c, 'test'])}" for c in ta.index if c in ta.index) + " frutos). No sirve para la evaluación confirmatoria. "
    res = meta.get("reservado")
    txt += (f"Reservado para la prueba: {res['tipo']} {res['valor']}." if res else "No hay un lote ni una fecha elegibles para reservar: el análisis por lote se omite y se declara como limitación.")
    p = du.mk_p(txt)
    tb.addnext(p)


def s76(doc, h, heading):
    tb = du.siguiente_tabla(doc, heading, 0)
    if tb is None:
        return
    t = docx.table.Table(tb, doc)
    n = C.CFG["busqueda"]["configs_clasificador_directo"]
    rel = {"Estrategia de balanceo": (3, f"{n} configuraciones por variante de balanceo (1 semilla c/u) y 3 semillas para la elegida; mismo presupuesto en todas"),
           "Optimizador y entrenamiento": (3, f"Hasta 50 épocas por corrida; parada temprana con paciencia 5 u 8; AdamW con decaimiento coseno"),
           "Umbrales": (1, "θ_{def} ∈ {1, 2, 3}; p_{am} ∈ [0,05; 0,80] (paso 0,05); h_{am} ∈ [80°, 130°] (la rejilla de h_{am} usa las proporciones por matiz guardadas)"),
           "Otros hiperparámetros": (1, "Clasificador directo: tasa de aprendizaje ∈ [1e-5, 1e-3] y weight decay ∈ [1e-5, 1e-1] (escala log), batch ∈ {8, 16}, paciencia ∈ {5, 8}. YOLOv8-seg: lr0 ∈ [1e-4, 1e-2], weight decay ∈ [1e-5, 1e-2], batch ∈ {8, 16}, hsv_v ∈ {0,2; 0,4}, flipud ∈ {0; 0,5}")}
    for r in t.rows[1:]:
        nombre = r.cells[0].text.strip()
        for k, (col, txt) in rel.items():
            if nombre.startswith(k) and "[Completar" in r.cells[col].text:
                du.set_cell(r.cells[col], txt)


def _git():
    try:
        if (ROOT / ".git").exists():
            c = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
            d = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
            return c + (" (con cambios sin confirmar)" if d else "")
    except Exception:
        pass
    return None


def s77(h):
    def v(m):
        try:
            return __import__(m).__version__
        except Exception:
            return None
    libs = ", ".join(f"{n} {x}" for n, x in (("numpy", v("numpy")), ("pandas", v("pandas")), ("scikit-learn", v("sklearn")), ("scipy", v("scipy")),
                                              ("OpenCV", v("cv2")), ("PyTorch", v("torch")), ("torchvision", v("torchvision")), ("ultralytics", v("ultralytics"))) if x)
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        ram = None
    git = _git()
    b = [B(f"**Código y datos.** " + (f"Commit {git}; " if git else "El proyecto no es un repositorio git (use `git init` para registrar el commit); ") +
           f"huella de la configuración {C.hash_config()}; huella de la partición {h['huella_part'] or '-'}" + (f"; huella del conjunto de datos {huella_dataset(h['raw'])[:16]}…" if huella_dataset(h['raw']) else "") + "."),
         B(f"**Entorno.** Python {platform.python_version()} en {platform.system()} {platform.release()}; {libs}."),
         B(f"**Hardware.** CPU: {platform.processor() or platform.machine()} ({os.cpu_count()} núcleos)" + (f"; RAM {ram:.0f} GB" if ram else "") + (f"; GPU: {gpu}" if gpu else "; GPU no disponible en esta ejecución") + "."),
         B(f"**Semillas.** Partición {C.CFG['semilla_particion']}; entrenamiento {C.CFG['busqueda']['semillas_entrenamiento']}; bootstrap {C.EST['semilla']} (B = {C.EST['bootstrap_B']}); simulación de potencia 20261112 y 20261113."),
         B("**Comando de reproducción.** `python src/ingesta.py` → `python src/particion.py` → `python src/extraer_features.py` → `python src/modelo_baseline.py` → "
           "`python src/segmentacion.py entrenar|elegir` → `python src/piloto.py congelar` → `python src/inferir_hibrido.py --subset val` → `python src/entrenar_directo.py buscar` → "
           "`python src/experimentos.py congelar` → `python src/inferir_hibrido.py --subset test` → `python src/entrenar_directo.py predecir --subset test` → "
           "`python src/experimentos.py analizar` → `python src/informe_parcial.py`."),
         B("**Registro de corridas.** Anexo D (construido de los logs: config.json de cada corrida con huella de partición y de configuración)."),
         B("**Pruebas automáticas.** `python -m pytest -q` (anti-fuga, estadística, partición, regla, protocolo, informe).")]
    return b


# ------------------------------------------------------------------------------------ resultados
def _rutas(p: Path):
    return str(p.relative_to(ROOT)) if ROOT in p.resolve().parents else str(p)


def etapas(h):
    pr, raw = h["procesados"], h["raw"]
    an = lambda t: len(list((ROOT / "data" / "anotaciones" / t).glob("*.txt"))) if (ROOT / "data" / "anotaciones" / t).exists() else 0
    seg = LOGS / "segmentacion"
    prov = h["part_meta"].get("provisional")
    R = h["resultados"] or {}
    E = lambda c, ok, par="Pendiente": "Completado" if c else par
    dirs = lambda d: sorted((LOGS / d).glob("*/")) if (LOGS / d).exists() else []
    return [
        ("Ingesta e índices", "src/ingesta.py", E((pr / "indice_frutos.csv").exists(), 1), f"{len(h['frutos']) if h['frutos'] is not None else 0} frutos, {len(h['imgs']) if h['imgs'] is not None else 0} imágenes"),
        ("Partición congelada por cuotas", "src/particion.py", ("Parcial" if prov else "Completado") if h["part"] is not None else "Pendiente",
         ("PROVISIONAL; huella " + h["huella_part"]) if h["part"] is not None and prov else (h["huella_part"] or "sin partición")),
        ("Atributos por vista (segmentación clásica de respaldo)", "src/extraer_features.py", E((pr / "features_frutos.csv").exists(), 1), "features_frutos.csv"),
        ("Escala mm/px por sesión", "src/preprocesamiento.py", E((raw / "calibracion_sesiones.csv").exists(), 1), "calibracion_sesiones.csv" if (raw / "calibracion_sesiones.csv").exists() else "falta calibracion_sesiones.csv"),
        ("Corrección de brillo y color (E3)", "src/preprocesamiento.py", "Parcial" if not (ROOT / "configs" / "sesiones.yaml").exists() else "Completado",
         "implementado; configs/sesiones.yaml " + ("existe" if (ROOT / "configs" / "sesiones.yaml").exists() else "no existe (sin carta de color)")),
        ("Baseline clásico y variantes (V1, V2)", "src/modelo_baseline.py", E(h["baseline"] is not None, 1), _rutas(h["baseline"]) if h["baseline"] else "sin corrida de esta partición"),
        ("Búsqueda de hiperparámetros (clásicos)", "src/busqueda_hiperparametros.py", E(h["hp"] is not None, 1), _rutas(h["hp"]) if h["hp"] else "sin corrida de esta partición"),
        ("Anotación de contornos del fruto", "src/anotacion.py", E(an("fruto") >= 1, 1) if an("fruto") < len(h["imgs"] if h["imgs"] is not None else []) else "Completado", f"{an('fruto')} etiquetas (meta: una por imagen)"),
        ("Máscaras de defecto (submuestra de 500 frutos)", "src/anotacion.py", E(an("defectos") >= 1, 1), f"{an('defectos')} etiquetas (meta: 2 500 imágenes)"),
        ("Segmentación entrenada y elegida", "src/segmentacion.py", E((seg / "fruto" / "modelo_elegido.json").exists(), 1), "modelo_elegido.json" if (seg / "fruto" / "modelo_elegido.json").exists() else "sin entrenar"),
        ("Umbrales de la rúbrica congelados", "src/regla_comercial.py", E(rc.RUTA_UMBRALES.exists(), 1), "configs/umbrales_congelados.json" if rc.RUTA_UMBRALES.exists() else "valores de trabajo provisionales"),
        ("Fiabilidad del panel (κ de Fleiss)", "src/panel.py", E(h["panel"] is not None, 1), "panel_evaluaciones.csv" if h["panel"] is not None else "falta panel_evaluaciones.csv"),
        ("Prueba piloto", "src/piloto.py", E(h["piloto"] is not None, 1), "logs/piloto.json" if h["piloto"] else "sin ejecutar"),
        ("Inferencia híbrida en validación", "src/inferir_hibrido.py", E((LOGS / "inferencia" / "base" / "vistas_val.csv").exists(), 1), "logs/inferencia/base" if (LOGS / "inferencia" / "base").exists() else "sin ejecutar"),
        ("Clasificador directo (ResNet-50, 5 vistas)", "src/entrenar_directo.py", E(any((d / "resumen.json").exists() for d in dirs("directo")), 1), f"{sum((d / 'resumen.json').exists() for d in dirs('directo'))} variantes entrenadas"),
        ("Comparación con operarios", "src/operarios.py", E((raw / "operarios_clasificaciones.csv").exists(), 1), "operarios_clasificaciones.csv" if (raw / "operarios_clasificaciones.csv").exists() else "falta el archivo"),
        ("Evaluación de la prueba (una sola vez)", "src/experimentos.py", E(bool(R.get("prueba", {}).get("disponible")), 1), "registrada con huella" if R.get("prueba", {}).get("disponible") else "no evaluada"),
        ("API de inferencia (FastAPI)", "src/api_inferencia.py", "Parcial", "implementada; requiere los modelos congelados y fastapi para ejecutarse"),
    ]


def s81(h):
    et = etapas(h)
    comp, par = sum(e[2] == "Completado" for e in et), sum(e[2] == "Parcial" for e in et)
    return [P(f"Corte: {date.today().strftime('%d/%m/%Y')}. De {len(et)} etapas del pipeline, {comp} están completadas, {par} parciales y {len(et) - comp - par} pendientes. "
              "El estado se detecta de los archivos y corridas del repositorio (no se declara una etapa completada si falta su salida verificable)."),
            ("t", "Estado de implementación del pipeline", ["Etapa", "Módulo", "Estado", "Evidencia verificable"], [list(e) for e in et], [2700, 1900, 1300, 3100])]


def s82(h):
    """Filas de la tabla de comparación cuantitativa."""
    rows = []
    bl = h["bl"]
    if bl is not None:
        base = bl[bl["variante"] == "BASE"]
        d = base[base["modelo"] == "dummy"]
        for r in base.itertuples():
            dec = "Referencia (clase mayoritaria)" if r.modelo == "dummy" else ("Δ vs dummy = " + n2(r.cv_f1_macro_media - float(d["cv_f1_macro_media"].iloc[0]), 3) if len(d) else "-")
            rows.append([f"Baseline {r.modelo} (variables A)", "F1 macro en validación cruzada (5 pliegues × 3 semillas, solo train)", ms(r.cv_f1_macro_media, r.cv_f1_macro_std),
                         f"ajuste {n2(r.cv_fit_s_media, 2)} s; {n2(r.latencia_total_ms_fruto, 0)} ms/fruto", dec])
        cmp_ = h["bl_cmp"]
        if cmp_ is not None:
            m = cmp_[cmp_["metrica"] == "f1_macro"]
            for r in m.itertuples():
                rr = bl[(bl["modelo"] == r.modelo) & (bl["variante"] == r.variante)]
                if len(rr):
                    rr = rr.iloc[0]
                    rows.append([f"{r.variante} ({r.modelo})", "F1 macro CV; Δ pareado vs BASE (prueba t corregida + Holm)", ms(rr.cv_f1_macro_media, rr.cv_f1_macro_std),
                                 f"{n2(rr.latencia_total_ms_fruto, 0)} ms/fruto", f"Δ {n2(r.dif_media, 3)}; p(Holm) {n2(r.p_holm, 3)}: {r.veredicto}"])
    if h["hp_mejores"] is not None:
        for r in h["hp_mejores"].itertuples():
            rows.append([f"Mejor candidato {r.modelo} (balanceo {r.balanceo})", "F1 macro CV con presupuesto fijo por modelo", f"{n2(r.cv_f1_macro_media, 3)} ± {n2(r.cv_f1_macro_std, 3)}", "-",
                         f"VAL F1 macro {n2(r.val_f1_macro, 3)}"])
    R = (h["resultados"] or {}).get("desarrollo", {})
    if R.get("hibrido_base_val"):
        x = R["hibrido_base_val"]
        rows.append(["Híbrido (validación)", f"F1 macro, bootstrap estratificado B = {h['resultados']['B']}", f"{n2(x['f1_macro'], 3)} [{n2(x['ic'][0], 3)}; {n2(x['ic'][1], 3)}]", "-", "Informativo (no es la prueba)"])
    if R.get("directo"):
        d = R["directo"]
        rows.append(["Directo ResNet-50 (validación)", "F1 macro de validación, 3 semillas", ms(d["val_f1_macro_media"], d["val_f1_macro_std"]), f"{n2(d.get('tiempo_total_s', 0) / 60, 1)} min de entrenamiento",
                     "Informativo (no es la prueba)"])
    if not rows:
        rows.append(["[Pendiente]", "Sin corridas de esta partición", "-", "-", "Ejecute src/modelo_baseline.py"])
    return rows


def s83(h):
    """Filas de la tabla de ablaciones (Componente cambiado | Control | Resultado | Interpretación)."""
    rows, R = [], h["resultados"] or {}
    P_ = R.get("prueba", {})
    D = R.get("desarrollo", {})
    c = P_.get("contrastes", {}) if P_.get("disponible") else {}
    h2 = c.get("H2", {})
    if "diferencia" in h2:
        rows.append(["E1 · 5 vistas (0°, 90°, 180°, 270°, pedúnculo)", "1 vista ecuatorial (0°); mismo modelo", f"recall de Fábrica {n2(h2['recall_1vista'])} → {n2(h2['recall_5vistas'])} ({n2(h2['diferencia_pp'], 1)} pp; IC95 % [{n2(h2['ic_inf'])}; {n2(h2['ic_sup'])}]; McNemar p = {n2(h2['p'], 4)}; n = {h2['n_fabrica']})",
                     "Confirmatorio (H2): " + R["prueba"]["contrastes"]["decision"]["H2"]["estado"] + f" (umbral {C.EST['h2']['incremento_pp_min']:.0f} pp)"])
    else:
        rows.append(["E1 · 5 vistas", "1 vista (0°)", "[Pendiente: la prueba aún no se evalúa]", "Confirmatorio (H2)"])
    h3 = c.get("H3", {})
    if "diferencia" in h3:
        rows.append(["E2 · Arquitectura híbrida", "Clasificador directo ResNet-50 (5 vistas)", f"F1 macro {n2(h3['f1_directo'], 3)} → {n2(h3['f1_hibrido'], 3)} (Δ {n2(h3['diferencia'], 3)}; IC95 % [{n2(h3['ic_inf'], 3)}; {n2(h3['ic_sup'], 3)}])",
                     "Confirmatorio (H3): " + R["prueba"]["contrastes"]["decision"]["H3"]["estado"] + ("; relevancia práctica (≥ 0,05)" if h3["relevancia_practica"] else "; sin relevancia práctica (< 0,05)") +
                     ". El híbrido usa supervisión adicional (contornos y máscaras) que el directo no usa."])
    else:
        rows.append(["E2 · Arquitectura híbrida", "Clasificador directo", "[Pendiente: la prueba aún no se evalúa]", "Confirmatorio (H3)"])
    e3 = D.get("E3") or []
    if e3:
        txt = "; ".join(f"{x['variante']}: F1 macro {n2(x['f1_macro'], 3)}, F1 sobremadurez {n2(x['f1_madurez'], 3)}, F1 Primera–Segunda {n2(x['f1_primera_segunda'], 3)}" for x in e3)
        rows.append(["E3 · Corrección de brillo / brillo y color", "Sin corrección", txt + " (validación)", "Exploratorio: se reportan diferencias con IC; sin contraste de hipótesis"])
    else:
        rows.append(["E3 · Preprocesamiento", "Sin corrección", "[Pendiente: sin corridas e3_* en validación]", "Exploratorio"])
    e4 = D.get("E4") or []
    if e4:
        txt = "; ".join(f"{x['balanceo']}: F1 macro {n2(x['f1_macro'], 3)}, F1 Tercera {n2(x['f1_tercera'], 3)}, F1 Fábrica {n2(x['f1_fabrica'], 3)}" for x in e4)
        rows.append(["E4 · Balanceo (pesos / focal / aumento / combinación)", "Sin balanceo", txt + " (validación)", "Exploratorio"])
    else:
        rows.append(["E4 · Balanceo", "Sin balanceo", "[Pendiente: sin variantes del clasificador directo]", "Exploratorio"])
    e5 = D.get("E5") or []
    if e5:
        rows.append(["E5 · Perturbaciones leves de luminancia", "Condición controlada", "; ".join(f"{x['perturbacion']}: ΔF1 {n2(x['delta_f1'], 3)}, conserva categoría {n2(100 * x['conserva_categoria'], 1)} %" for x in e5) + " (validación)", "Exploratorio"])
    else:
        rows.append(["E5 · Perturbaciones de luminancia", "Condición controlada", "[Pendiente: sin corridas e5_*]", "Exploratorio"])
    cmp_ = h["bl_cmp"]
    if cmp_ is not None:
        m = cmp_[(cmp_["metrica"] == "f1_macro") & cmp_["variante"].str.startswith("V1_")]
        if len(m):
            rows.append(["V1 · Número de vistas (baseline clásico)", "5 vistas", "; ".join(f"{r.variante.replace('V1_', '')} ({r.modelo}): Δ {n2(r.dif_media, 3)}, {r.veredicto}" for r in m.itertuples()) + " (CV)",
                         "Referencia previa con descriptores clásicos"])
    return rows


def _fig_confusion(h, ruta: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = h["resultados"] or {}
    cm = lab = tit = None
    if R.get("prueba", {}).get("disponible"):
        mc = R["prueba"]["matriz_confusion"]
        cm, lab, tit = np.array(mc["matriz"]), mc["labels"], "Prueba — híbrido (5 vistas)"
    elif h["bl_cm"]:
        bl = h["bl"][(h["bl"]["variante"] == "BASE") & (h["bl"]["modelo"] != "dummy")].sort_values("cv_f1_macro_media", ascending=False)
        if len(bl):
            x = h["bl_cm"][bl.iloc[0]["config"]]
            cm, lab, tit = np.array(x["matriz"]), x["labels"], f"Validación — baseline {bl.iloc[0]['modelo']}"
    if cm is None:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(cm, cmap="Reds")
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicho"); ax.set_ylabel("referencia"); ax.set_title(tit, fontsize=9)
    fig.tight_layout(); fig.savefig(ruta, dpi=140); plt.close(fig)
    return tit


def s84(h, figdir: Path):
    R = h["resultados"] or {}
    D, Pr = R.get("desarrollo", {}), R.get("prueba", {})
    b = []
    so = D.get("sobreajuste")
    if so and so.get("disponible"):
        b.append(B(f"**Brecha entrenamiento–validación (clasificador directo, semilla mediana).** En la mejor época ({so['mejor_epoca']} de {so['epocas']}): pérdida de entrenamiento {n2(so['loss_train'], 3)}, "
                   f"de validación {n2(so['loss_val'], 3)} (brecha {n2(so['brecha_loss'], 3)}); F1 macro de validación {n2(so['f1_val_mejor_epoca'], 3)}. "
                   f"Variabilidad entre semillas (F1 macro de validación): {', '.join(n2(x, 3) for x in so['f1_val_semillas'])} (DE {n2(so['f1_val_std'], 3)})."))
    else:
        b.append(pend("brecha entrenamiento–validación del clasificador directo (sin curvas de entrenamiento)"))
    bl = h["bl"]
    if bl is not None:
        x = bl[(bl["variante"] == "BASE") & (bl["modelo"] != "dummy")]
        b.append(B("**Brecha CV–validación (baseline clásico).** " + "; ".join(f"{r.modelo}: CV {n2(r.cv_f1_macro_media, 3)} frente a validación {n2(r.val_f1_macro, 3)}" for r in x.itertuples()) + "."))
    if Pr.get("disponible"):
        pc = pd.DataFrame(Pr["por_categoria"])
        b.append(B("**Desempeño por categoría en la prueba (slicing).** " + "; ".join(f"{r.categoria}: F1 {n2(r.f1)} (precisión {n2(r.precision)}, recall {n2(r.recall)}, n = {r.n})" for r in pc.itertuples()) + "."))
        lr = Pr.get("lote_reservado", {})
        if lr.get("disponible"):
            b.append(B("**Lote o fecha reservado (secundario).** " + "; ".join(f"{k}: recall {n2(v['recall'])} (Wilson [{n2(v['wilson'][0])}; {n2(v['wilson'][1])}], n = {v['n']})" for k, v in lr["categorias"].items())
                       + (f"; categorías omitidas por tener menos de {C.EST['umbral_categorias_lote_reservado']} frutos: {', '.join(lr['categorias_omitidas'])}" if lr["categorias_omitidas"] else "") + "."))
        else:
            b.append(B("**Lote o fecha reservado.** No hubo lote ni fecha elegibles: el análisis por lote se omite y se declara como limitación."))
        if Pr.get("errores"):
            e = pd.DataFrame(Pr["errores"]).head(6)
            b.append(B("**Casos de fallo (primeros de la lista).** " + "; ".join(f"{r.id_fruto}: referencia {r.y}, predicho {r.pred} (d_est {n2(r.d_est_mm, 1)} mm, calibre medido {n2(r.calibre_mm_ref, 1)} mm, S {n2(r.severidad_max, 0)}, P_am {n2(r.p_amarillo)})" for r in e.itertuples()) + ". Lista completa en resultados.json."))
    else:
        b.append(pend("desempeño por categoría, lote reservado y casos de fallo sobre la PRUEBA (aún no se evalúa)"))
    e5 = D.get("E5") or []
    if e5:
        b.append(B("**Robustez a la luminancia (validación).** " + "; ".join(f"{x['perturbacion']}: ΔF1 {n2(x['delta_f1'], 3)}, {n2(100 * x['conserva_categoria'], 1)} % conserva la categoría" for x in e5) + "."))
    lat = Pr.get("latencia_hibrido") if Pr.get("disponible") else None
    if lat and lat.get("mediana_ms") is not None:
        b.append(B(f"**Latencia por fruto (híbrido, cinco vistas).** Mediana {n2(lat['mediana_ms'], 0)} ms, percentil 95 {n2(lat['p95_ms'], 0)} ms (requisito de ingeniería: ≤ {C.EST['latencia_ms_max']:.0f} ms)."))
    tit = _fig_confusion(h, figdir / "fig_confusion_informe.png")
    if tit:
        b.append(("fig", figdir / "fig_confusion_informe.png", f"Matriz de confusión: {tit}."))
    return b


def s85(h):
    fr, R = h["frutos"], h["resultados"] or {}
    n = int(fr["completo"].sum()) if fr is not None else 0
    b = [B(f"**Tamaño muestral.** {n} frutos con cinco vistas frente a los {C.TOTAL_VALIDOS} del plan; " + (
        "la partición es PROVISIONAL y la prueba tiene menos frutos por categoría que los 80 de los cálculos de potencia: los IC serán más anchos y las conclusiones no son confirmatorias." if h["part_meta"].get("provisional", True)
        else "la prueba tiene 80 frutos por categoría, como en los cálculos de potencia (que usan el α nominal sin corregir por Holm).") + ")")]
    if fr is not None:
        conc = eda.concentracion_por_sesion(fr)
        if not conc["cumple_4_3"].all():
            b.append(B(f"**Contaminación por sesión.** {int((~conc['cumple_4_3']).sum())} categorías concentran más del {eda.MAX_PCT_SESION:.0f} % en un lote o una fecha; "
                       "un modelo puede aprender la sesión (iluminación, escala) en lugar del fruto y sobreestimar su desempeño. Debe corregirse con la recolección dirigida y diversificada."))
    b += [B("**Generalización.** Las conclusiones se limitan a una viabilidad técnica preliminar en condiciones controladas (light box, faja estática, varias vistas); no se validó en faja en movimiento, "
            "con oclusión, polvo, cambios de luz ni hardware de borde. El análisis por lote reservado es secundario."),
          B("**Estándar de referencia.** La categoría depende de intervalos de calibre propuestos que el panel y el centro de acopio deben validar y de una rúbrica provisional (a1–a3, θ_{def}, h_{am}, p_{am}); "
            + ("los umbrales ya están congelados." if rc.RUTA_UMBRALES.exists() else "los umbrales aún NO están congelados.") + " La fiabilidad del panel " + ("se midió." if h["panel"] is not None else "aún no se midió (no hay panel_evaluaciones.csv).")),
          B("**Comparación con el clasificador directo.** El híbrido emplea supervisión adicional (contornos y máscaras de defecto) que el directo no usa; esa diferencia forma parte de la arquitectura comparada."),
          B("**Reproducibilidad.** Semillas, huellas de partición y de configuración, y registro SHA-256 de la prueba (sección 7.7 y Anexo D)."),
          B("**Interpretabilidad.** La arquitectura híbrida expone calibre, severidad, P_{am} y el motivo de cada decisión de Fábrica; el clasificador directo no (se puede añadir Grad-CAM)."),
          B("**Múltiples contrastes.** La familia {H1–H4} usa Holm; los tamaños de muestra se calcularon sin ajustar, por lo que la potencia efectiva de los contrastes más exigentes es algo menor.")]
    for a in h["avisos"]:
        b.append(B("**Aviso de consistencia.** " + a))
    return b


def s9intro(h):
    R = h["resultados"] or {}
    if R.get("prueba", {}).get("disponible"):
        return [P("Con la evaluación única de la prueba se presentan los contrastes confirmatorios (sección 9.1). Quedan pendientes las etapas indicadas en la sección 8.1.")]
    return [P("A la fecha de corte no se ha evaluado la prueba: no hay evidencia confirmatoria sobre la hipótesis general ni sobre H1–H4. Lo disponible (datos, baseline clásico, "
              "protocolo e infraestructura) permite continuar con las etapas pendientes de la sección 8.1.")]


def s91(h):
    R = h["resultados"] or {}
    Pr = R.get("prueba", {})
    if not Pr.get("disponible"):
        return [B("**Hipótesis general (F1 macro ≥ 0,85):** indeterminada; la prueba aún no se evalúa."),
                B("**H1–H4:** indeterminadas; requieren la prueba y, para H4, las clasificaciones de al menos tres operarios."),
                B("**Lo que sí se puede afirmar hoy:** el protocolo, la regla jerárquica, la estadística confirmatoria y el flujo de cómputo están implementados y probados con datos sintéticos; el baseline clásico "
                  "con descriptores de imagen" + (" supera a la referencia de clase mayoritaria (ver 8.2)" if h["bl"] is not None else " aún no se ejecutó con esta partición") + ", sin que eso constituya evidencia de la hipótesis."),
                pend("conclusiones sobre HG y H1–H4 (se completarán al evaluar la prueba)")]
    c = Pr["contrastes"]
    prov = "; **resultado no confirmatorio: la partición es PROVISIONAL**" if R.get("particion_provisional") else ""
    hg = c["HG"]
    b = [B(f"**Hipótesis general:** F1 macro {n2(hg['f1_macro'], 3)} (IC95 % [{n2(hg['ic_inf'], 3)}; {n2(hg['ic_sup'], 3)}], B = {R['B']}); umbral ≥ {C.EST['hg_f1_macro_min']:.2f} sobre el límite inferior → "
            f"{'se acepta' if hg['acepta'] else 'no se acepta'}{prov}.")]
    for k, nom in (("H1", "calibre"), ("H2", "multi-vista"), ("H3", "híbrido vs directo"), ("H4", "no inferioridad frente a operarios")):
        d = c["decision"].get(k, {})
        est_ = d.get("estado", "sin evidencia")
        if est_ == "sin evidencia":
            b.append(B(f"**{k} ({nom}):** sin evidencia (faltan datos para contrastarla)."))
            continue
        hh = d["holm"]
        b.append(B(f"**{k} ({nom}):** {est_} (p = {n2(hh['p'], 4)} frente al umbral de Holm {n2(hh['umbral'], 4)}; criterios puntuales {'cumplidos' if d['criterios_puntuales'] else 'no cumplidos'}){prov}."))
    b.append(B("Las diferencias se interpretan solo dentro de las condiciones controladas del estudio; la no inferioridad no implica superioridad."))
    return b


def s92_evidencia(doc, h, heading):
    tb = du.siguiente_tabla(doc, heading, 0)
    if tb is None:
        return
    fr, R = h["frutos"], h["resultados"] or {}
    t = docx.table.Table(tb, doc)
    cuota = ""
    if fr is not None:
        ta = eda.tabla_cuotas(fr)
        cuota = "Cuotas actuales: " + ", ".join(f"{k} {int(v['frutos'])}/{int(v['cuota_min'])}" for k, v in ta.iterrows())
    dd = (R.get("desarrollo") or {}).get("directo") or {}
    mins = f"{dd['tiempo_total_s'] / 60:.0f} min de entrenamiento en total" if dd.get("tiempo_total_s") else "sin corridas de entrenamiento"
    ev = {"Variabilidad": cuota or "sin datos", "Tiempo de iteración": mins,
          "Disponibilidad del panel": (f"{h['panel']['evaluador'].nunique()} evaluadores en panel_evaluaciones.csv" if h["panel"] is not None else "sin panel_evaluaciones.csv"),
          "Deterioro": "no se registra el tiempo cosecha–captura" if (fr is not None and "horas_cosecha_captura" not in fr.columns) else "registrado"}
    for r in t.rows[1:]:
        for k, v in ev.items():
            if r.cells[0].text.strip().startswith(k) and "[Completar" in r.cells[3].text:
                du.set_cell(r.cells[3], f"{v}. Responsable: [Pendiente]")


def s93(h):
    et = etapas(h)
    pend_ = [e for e in et if e[2] != "Completado"]
    crit = {"Ingesta": "—", "Partición": "1 500 frutos válidos (320/320/320/270/270); prueba de 80 por categoría; ningún lote o fecha > 35 % de una categoría",
            "Escala": "Escala mm/px por sesión con el patrón; error < 0,2 mm al medir un objeto de dimensión conocida",
            "Corrección": "Carta de color y patrón en cada encuadre; configs/sesiones.yaml completo", "Anotación de contornos": "IoU medio entre anotadores ≥ 0,70 en el 10 % de control",
            "Máscaras": "500 frutos (400 de desarrollo + 100 de prueba); IoU entre anotadores ≥ 0,70", "Segmentación": "Selección con validación: mayor IoU medio del fruto; a igualdad (ΔMAE < 0,05 mm), menor costo",
            "Umbrales": "κ de Fleiss ≥ 0,75 y repetibilidad ≥ 0,80 en el piloto; umbrales congelados antes de la recolección definitiva",
            "Fiabilidad": "κ de Fleiss ≥ 0,75; repetibilidad ≥ 0,80", "Prueba piloto": "Criterios (a)–(c) de la sección 4.6.7", "Inferencia": "Latencia mediana ≤ 200 ms por fruto con cinco vistas",
            "Clasificador": "3 semillas; semilla de F1 macro mediano; mismo presupuesto que el híbrido", "Comparación": "Al menos 3 operarios; 400 frutos de prueba; repetibilidad con 50 frutos",
            "Evaluación": "Una sola vez, con modelos y umbrales congelados; huella SHA-256 registrada", "API": "Devuelve categoría, valores y máscaras para las 5 vistas"}
    rows = []
    for e in pend_[:8]:
        k = next((v for kk, v in crit.items() if e[0].startswith(kk)), "Salida verificable en logs/")
        rows.append([e[0], e[3], k, "[Pendiente: no definida en el plan]"])
    return rows or [["Redacción y sustentación", "Informe final", "Todas las etapas completadas", "[Pendiente: no definida en el plan]"]]


def registro_corridas(h, maximo=30) -> list:
    filas = []
    for d, etapa, col in (("baseline_runs", "baseline", "cv_f1_macro_media"), ("fe_runs", "FE", "cv_f1_macro_media"), ("hiperparametros_runs", "hiperparámetros", "cv_f1_macro_media")):
        for run in sorted((LOGS / d).glob("*/")) if (LOGS / d).exists() else []:
            m = _meta(run)
            csv = run / ("mejores.csv" if d == "hiperparametros_runs" else "resultados.csv")
            met = "-"
            if csv.exists():
                r = pd.read_csv(csv)
                if col in r:
                    met = f"F1 macro CV máx. {n2(r[col].max(), 3)}"
            filas.append([run.name, f"{etapa}; partición {m.get('huella_particion', '-')}; config {m.get('config_hash', '-')}", f"semillas 0,1,2; CV 5 pliegues", met, _rutas(run)])
    for run in sorted((LOGS / "segmentacion").glob("*/trials_*.csv")) if (LOGS / "segmentacion").exists() else []:
        r = pd.read_csv(run)
        filas.append([run.stem, "segmentación; " + run.parent.name, f"{len(r)} configuraciones", f"IoU medio máx. {n2(r['iou_medio'].max(), 3)}" if "iou_medio" in r else "-", _rutas(run)])
    for run in sorted((LOGS / "directo").glob("*/resumen.json")) if (LOGS / "directo").exists() else []:
        r = json.loads(run.read_text(encoding="utf-8"))
        filas.append([run.parent.name, f"directo; balanceo {r.get('balanceo')}; config {r.get('config_elegida')}", f"semilla mediana {r.get('semilla_mediana')}", f"F1 macro val {n2(r.get('val_f1_macro_media'), 3)}", _rutas(run)])
    for run in sorted((LOGS / "resultados").glob("*/resultados.json")) if (LOGS / "resultados").exists() else []:
        r = json.loads(run.read_text(encoding="utf-8"))
        filas.append([r["run_id"], f"análisis; partición {r.get('huella_particion')}; config {r.get('config_hash')}", f"B = {r.get('B')}", "contrastes" if r.get("prueba", {}).get("disponible") else "solo desarrollo", _rutas(run)])
    pw = LOGS / "potencia" / "anexo2.json"
    if pw.exists():
        filas.append(["potencia", "Anexo 2", "semillas 20261112 / 20261113", "simulación Monte Carlo", _rutas(pw)])
    return filas[-maximo:]


# =====================================================================================================================
# aplicación sobre el documento
# =====================================================================================================================
def _bloques_a_xml(doc, bloques, tabla0: list):
    out = []
    for b in bloques:
        k = b[0]
        if k == "p":
            out.append(du.mk_p(b[1]))
        elif k == "b":
            out.append(du.mk_p(b[1], style="Listaconvietas"))
        elif k == "pend":
            out.append(du.mk_p("[Pendiente: " + b[1] + "]", italic=True, color="9C2A00"))
        elif k == "t":
            out.append(du.mk_p(f"Tabla {tabla0[0]}. {b[1]}", bold=True, size=9.5, keep_next=True, after=60))
            tabla0[0] += 1
            out.append(du.new_table(doc, b[2], b[3], b[4]))
            out.append(du.mk_p("", after=60))
        elif k == "fig":
            out.append(du.mk_fig(doc, b[1]))
            out.append(du.mk_p(b[2], italic=True, size=9, align="center", after=120))
    return out


def completar(base: Path, salida: Path, procesados: Path = ROOT / "data" / "processed", raw: Path = ROOT / "data" / "raw",
              mes: str | None = None, docente: str | None = None) -> dict:
    doc = docx.Document(str(base))
    h = recolectar(procesados, raw)
    figdir = ROOT / "informe" / "figuras"
    figdir.mkdir(parents=True, exist_ok=True)
    num = [du.siguiente_numero_tabla(doc)]
    hecho, omitido = [], []

    def H(prefijo):
        el = du.buscar_heading(doc, prefijo)
        if el is None:
            omitido.append(f"no se encontró el encabezado '{prefijo}'")
        return el

    def rellenar(prefijo, bloques, **kw):
        el = H(prefijo)
        if el is None or bloques is None:
            if bloques is None and el is not None:
                omitido.append(f"{prefijo}: sin datos; se conserva el marcador de la plantilla")
            return
        du.reemplazar_marcadores(doc, el, _bloques_a_xml(doc, bloques, num), **kw)
        hecho.append(prefijo)

    # --- portada
    mes = mes or MESES[date.today().month - 1]
    du.reemplazar_texto_parrafo(doc, "Lugar y fecha:", f"Lugar y fecha: Lima, {mes} de {date.today().year}")
    if docente:
        du.reemplazar_texto_parrafo(doc, "Docente del curso:", f"Docente del curso: {docente}")
    # --- cronograma: producto de preprocesamiento
    for t in doc.tables:
        for r in t.rows:
            if r.cells[0].text.strip().startswith("Preprocesamiento, aumento y balanceo") and "[Completar" in r.cells[-1].text:
                du.set_cell(r.cells[-1], "Variantes de preprocesamiento (ninguno, brillo, brillo y color) y de balanceo (E3, E4) aplicables con los módulos del repositorio")
    # --- Parte II
    rellenar("6.1", s61(h), solo_pendientes_de="licencia")
    rellenar("6.2", s62(h), solo_pendientes_de="diccionario")
    s = s63(h)
    if s:
        rellenar("6.3", s)
    rellenar("6.4", s64(h), solo_pendientes_de="detalle de qué estadísticos")
    e = H("7.2")
    if e is not None:
        s72(doc, h, e); hecho.append("7.2")
    e = H("7.6")
    if e is not None:
        s76(doc, h, e); hecho.append("7.6")
    rellenar("7.7", s77(h), solo_pendientes_de="repositorio")
    # --- resultados
    R = h["resultados"] or {}
    corte = R.get("fecha", date.today().isoformat())
    rellenar("8. Resultados", [P(f"Período de corte: {corte[:10]}. Todas las cifras provienen de los archivos y corridas del repositorio (logs/, data/processed/); lo que no se ha obtenido figura como "
                                 "[Pendiente] y no se presenta como resultado. " + ("Los experimentos confirmatorios (E1, E2) se evaluaron una sola vez sobre la prueba congelada; E3–E5 son exploratorios y usan solo validación." if R.get("prueba", {}).get("disponible")
                                                                                       else "La prueba aún no se evalúa: las cifras de desarrollo son informativas y no confirmatorias.") + (" " + " ".join(h["avisos"]) if h["avisos"] else ""))])
    rellenar("8.1", s81(h))
    el = H("8.2")
    if el is not None:
        tb = du.siguiente_tabla(doc, el, 0)
        du.fill_table(doc, tb, ["Experimento", "Métrica por fold / corrida", "Media ± DE", "Tiempo / coste", "Decisión"], s82(h), [1900, 2300, 1500, 1500, 1800])
        du.reemplazar_marcadores(doc, el, [du.mk_p("Resumen media ± desviación estándar de cada experimento; los resultados por pliegue y semilla están en `cv_pliegues.csv` de cada corrida (logs/baseline_runs). "
                                                   "Los valores de validación (VAL) son informativos; ninguna cifra de la prueba se usa para elegir configuraciones.")])
        hecho.append("8.2")
    el = H("8.3")
    if el is not None:
        tb = du.siguiente_tabla(doc, el, 0)
        du.fill_table(doc, tb, ["Componente cambiado", "Control", "Resultado", "Interpretación"], s83(h), [2000, 1700, 3300, 2000])
        du.reemplazar_marcadores(doc, el, [du.mk_p("Cada ablación cambia un solo componente y deja el resto constante. E1 y E2 son confirmatorias; E3, E4 y E5 son exploratorias (solo validación, con intervalos de confianza y sin contraste de hipótesis).")])
        hecho.append("8.3")
    rellenar("8.4", s84(h, figdir))
    rellenar("8.5", s85(h))
    rellenar("9. Conclusiones", s9intro(h))
    rellenar("9.1", s91(h))
    el = H("9.2")
    if el is not None:
        s92_evidencia(doc, h, el); hecho.append("9.2")
    el = H("9.3")
    if el is not None:
        tb = du.siguiente_tabla(doc, el, 0)
        du.fill_table(doc, tb, ["Actividad", "Entregable", "Criterio de aceptación", "Fecha"], s93(h), [2400, 2200, 3000, 1400])
        du.reemplazar_marcadores(doc, el, [du.mk_p("Las actividades son las etapas del pipeline aún no completadas (sección 8.1); los criterios de aceptación son los del plan de tesis. Fechas y responsables no están definidos en el plan.")])
        hecho.append("9.3")
    el = H("Anexo D")
    if el is not None:
        tb = du.siguiente_tabla(doc, el, 0)
        filas = registro_corridas(h)
        if filas:
            du.fill_table(doc, tb, ["run_id", "commit / datos", "seed / split", "métrica", "artefacto"], filas, [1500, 2600, 1700, 1500, 1700]); hecho.append("Anexo D")
        else:
            omitido.append("Anexo D: no hay corridas registradas")
    # --- corrección tipográfica del encabezado de la matriz de confusión del Anexo F
    for t in doc.tables:
        c0 = t.rows[0].cells[0]
        if c0.text.strip().replace("\\\\", "\\").startswith("Verdadera \\ Predicha") and "\\\\" in c0.text:
            du.set_cell(c0, "Verdadera \\ Predicha", bold=True)
    salida.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(salida))
    return dict(salida=str(salida), completadas=hecho, omitidas=omitido, avisos=h["avisos"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=ROOT / "informe" / "plantilla" / "Informe_Parcial_PI2_Plan_de_Tesis_Limon_UNI.docx")
    ap.add_argument("--salida", type=Path, default=ROOT / "informe" / "Informe_Parcial_completado.docx")
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--mes", default=None)
    ap.add_argument("--docente", default=None)
    a = ap.parse_args()
    r = completar(a.base, a.salida, a.procesados, a.raw, a.mes, a.docente)
    print("Informe generado:", r["salida"])
    print("Secciones completadas:", ", ".join(r["completadas"]))
    for o in r["omitidas"]:
        print("[omitido]", o)
