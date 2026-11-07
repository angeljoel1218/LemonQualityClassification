"""
informe.py
----------
Genera el INFORME (Word + Markdown) con: datos, EDA, Feature Engineering, baseline y variantes de datos.
Lee los resultados reales de data/processed y logs/, asi que se regenera con tus datos sin copiar numeros.

    informe/Informe_EDA_FE_Baseline.docx
    informe/Informe_EDA_FE_Baseline.md
    informe/figuras/*.png

Uso:
    python src/informe.py
    python src/informe.py --procesados data/demo/processed --salida informe_demo

El informe contiene la lectura AUTOMATICA de los numeros (hechos). La interpretacion y las decisiones
las completa el equipo en el recuadro "Interpretacion del equipo".
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import eda, importancia  # noqa: E402
from src.fe_columnas import COLS_A, COLS_C  # noqa: E402
from src.particion import huella_particion  # noqa: E402

PAL = {"Extra": "#2e7d32", "Primera": "#7cb342", "Segunda": "#fbc02d", "Tercera": "#ef6c00", "Fabrica": "#7a1f2b"}
VINO = "#7a1f2b"
N_VISTAS = {"BASE": 5, "V1_4ecuatoriales": 4, "V1_2vistas": 2, "V1_1vista": 1}


# =================================================================================== carga
def _ultimo(dir_runs: Path, marca: str = "resultados.csv"):
    c = sorted(dir_runs.glob(f"*/{marca}")) if dir_runs.exists() else []
    return c[-1].parent if c else None


def cargar(procesados: Path) -> dict:
    kw = dict(dtype={"id_fruto": str, "lote": str, "fecha": str})
    c = {"procesados": procesados}
    c["frutos"] = pd.read_csv(procesados / "indice_frutos.csv", **kw)
    c["imgs"] = pd.read_csv(procesados / "indice_imagenes.csv", **kw)
    c["frutos"]["cat"] = eda.normalizar_categorias(c["frutos"]["categoria_ref"])
    for k, f in (("ff", "features_frutos.csv"), ("part", "particion.csv")):
        c[k] = pd.read_csv(procesados / f, **kw) if (procesados / f).exists() else None
    c["fe_dir"], c["bl_dir"] = _ultimo(ROOT / "logs" / "fe_runs"), _ultimo(ROOT / "logs" / "baseline_runs")
    c["imp_dir"] = _ultimo(ROOT / "logs" / "importancia_runs", "config.json")
    c["avisos"] = []
    h_actual = huella_particion(c["part"]) if c["part"] is not None else None

    def _meta(d):
        return json.loads((d / "config.json").read_text(encoding="utf-8"))["meta"]

    # una corrida solo es valida para estos datos si usa la MISMA particion congelada
    for clave, nombre in (("fe_dir", "Feature Engineering"), ("bl_dir", "baseline"),
                           ("imp_dir", "importancia de variables")):
        if c[clave] is not None and h_actual and _meta(c[clave]).get("huella_particion") != h_actual:
            c["avisos"].append(f"Se ignoró la última corrida de {nombre} ({c[clave].name}) porque se hizo con otra partición/otros datos. "
                               "Vuelva a ejecutarla con los datos actuales.")
            c[clave] = None
    c["imp"] = importancia.cargar_run(c["imp_dir"]) if c["imp_dir"] else None
    c["fe"] = pd.read_csv(c["fe_dir"] / "resultados.csv") if c["fe_dir"] else None
    c["fe_meta"] = _meta(c["fe_dir"]) if c["fe_dir"] else {}
    if c["bl_dir"]:
        c["bl"] = pd.read_csv(c["bl_dir"] / "resultados.csv")
        c["cmp"] = pd.read_csv(c["bl_dir"] / "comparaciones.csv")
        c["bl_cm"] = json.loads((c["bl_dir"] / "matrices_confusion_val.json").read_text(encoding="utf-8"))
        c["bl_meta"] = _meta(c["bl_dir"])
    else:
        c["bl"] = c["cmp"] = c["bl_cm"] = None
        c["bl_meta"] = {}
    return c


# =================================================================================== calculos EDA / FE
def calcular_eda(c: dict) -> dict:
    fr, imgs = c["frutos"], c["imgs"]
    e = {"calidad": eda.resumen_calidad_etiquetas(fr), "cuotas": eda.tabla_cuotas(fr),
         "cob": eda.cobertura_lote_fecha(fr), "inco": eda.inconsistencias_calibre_categoria(fr),
         "front": eda.fronterizos(fr, 0.5)}
    e["n_clases"] = fr["cat"].nunique()
    e["cramer"] = {col: (eda.asociacion_categoria_metadato(fr, col) if e["n_clases"] > 1 else np.nan)
                   for col in ("lote", "fecha")}
    met = eda.calcular_metricas_imagenes(imgs)
    met["borrosa"] = eda.marcar_atipicas(met, "nitidez") & (met["nitidez"] < met["nitidez"].median())
    e["met"] = met
    return e


def validar_extraccion(c: dict):
    if c["ff"] is None:
        return None
    ex = [x for x in ("calibre_mm_ref", "severidad_defecto_panel", "color_panel") if x in c["frutos"]]
    m = c["ff"].merge(c["frutos"][["id_fruto"] + ex], on="id_fruto")
    out = {"m": m}
    v = m.dropna(subset=["diam_ecu_mm", "calibre_mm_ref"]) if "diam_ecu_mm" in m and "calibre_mm_ref" in m else m.iloc[0:0]
    if len(v) > 2:
        err = v["diam_ecu_mm"] - v["calibre_mm_ref"]
        out.update(mae=float(err.abs().mean()), rmse=float(np.sqrt((err ** 2).mean())), sesgo=float(err.mean()), n=len(v))
    if "severidad_defecto_panel" in m and m["severidad_defecto_panel"].notna().sum() > 3:
        out["rho"] = float(m["defecto_max"].corr(m["severidad_defecto_panel"], method="spearman"))
    return out


# =================================================================================== figuras
def _guardar(fig, ruta):
    fig.tight_layout(); fig.savefig(ruta, dpi=130); plt.close(fig)
    return ruta


def hacer_figuras(c: dict, e: dict, val, figdir: Path) -> dict:
    figdir.mkdir(parents=True, exist_ok=True)
    F = {}
    t = e["cuotas"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(t.index, t["frutos"], color=[PAL[x] for x in t.index], label="frutos actuales")
    ax.plot(t.index, t["cuota_min"], "k_", ms=26, mew=2, label="cuota mínima")
    ax.set_ylabel("frutos"); ax.legend(); ax.grid(alpha=.25)
    F["cuotas"] = _guardar(fig, figdir / "fig_cuotas.png")

    fr = c["frutos"].dropna(subset=["calibre_mm_ref"])
    if len(fr):
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        cats = [x for x in eda.CATEGORIAS if (fr["cat"] == x).any()]
        bp = ax.boxplot([fr.loc[fr.cat == x, "calibre_mm_ref"] for x in cats], patch_artist=True)
        ax.set_xticklabels(cats)
        for p, x in zip(bp["boxes"], cats):
            p.set_facecolor(PAL[x]); p.set_alpha(.7)
        for u in (35, 37, 39):
            ax.axhline(u, color="k", ls="--", lw=1)
        ax.set_ylabel("calibre de referencia (mm)"); ax.grid(alpha=.25)
        F["calibre"] = _guardar(fig, figdir / "fig_calibre.png")

    if val and "m" in val:
        m = val["m"]
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
        v = m.dropna(subset=["diam_ecu_mm", "calibre_mm_ref"]) if "calibre_mm_ref" in m else m.iloc[0:0]
        if len(v):
            axs[0].scatter(v["calibre_mm_ref"], v["diam_ecu_mm"], s=10, alpha=.7, color=VINO)
            lim = [v["calibre_mm_ref"].min() - 1, v["calibre_mm_ref"].max() + 1]
            axs[0].plot(lim, lim, "k--", lw=1)
            axs[0].set_xlabel("calibre medido (mm)"); axs[0].set_ylabel("calibre de la imagen (mm)")
        if "severidad_defecto_panel" in m:
            axs[1].scatter(m["severidad_defecto_panel"], m["defecto_max"], s=10, alpha=.7, color="#555")
            axs[1].set_xlabel("severidad del panel"); axs[1].set_ylabel("defecto de la imagen")
        for a in axs: a.grid(alpha=.25)
        F["validacion"] = _guardar(fig, figdir / "fig_validacion.png")

    fe = c["fe"]
    if fe is not None:
        r = fe[~fe.config.str.startswith(("REF", "CONTROL", "A_sin"))].copy()
        r["conj"], r["modelo"] = r.config.str.split("|").str[0], r.config.str.split("|").str[1]
        mod, conj = list(r.modelo.unique()), list(dict.fromkeys(r.conj))
        w = .8 / len(mod)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for i, m_ in enumerate(mod):
            s = r[r.modelo == m_].set_index("conj").reindex(conj)
            ax.bar(np.arange(len(conj)) + i * w, s.cv_f1_macro_media, w, yerr=s.cv_f1_macro_std, capsize=2, label=m_)
        ref = fe.loc[fe.config.str.startswith("REF"), "cv_f1_macro_media"]
        if len(ref): ax.axhline(ref.iloc[0], color="k", ls=":", label="referencia (dummy)")
        ax.set_xticks(np.arange(len(conj)) + w * (len(mod) - 1) / 2); ax.set_xticklabels(conj)
        ax.set_ylabel("F1 macro (CV en train)"); ax.grid(alpha=.25); ax.legend(ncol=4, fontsize=8, loc="lower center")
        F["fe"] = _guardar(fig, figdir / "fig_fe_ablacion.png")

    if c["imp"] is not None:
        m0 = list(c["imp"]["perm"])[0]
        F["imp_perm"] = _guardar(importancia.fig_permutacion(c["imp"]["perm"][m0], f"Permutación fuera de muestra ({m0})"),
                                 figdir / "fig_importancia_permutacion.png")
        F["imp_dim"] = _guardar(importancia.fig_dimensiones(c["imp"]["dim_tabla"][m0], f"Caída de F1 al permutar cada dimensión ({m0})"),
                                figdir / "fig_importancia_dimensiones.png")

    bl = c["bl"]
    if bl is not None:
        orden = ["BASE", "V1_1vista", "V1_2vistas", "V1_4ecuatoriales", "V2_sobremuestreo", "V2_submuestreo"]
        col = {"BASE": "#888", "V1": "#1f77b4", "V2": "#ff7f0e"}
        mods = [m_ for m_ in bl.modelo.unique() if m_ != "dummy"]
        fig, axs = plt.subplots(1, len(mods), figsize=(3.6 * len(mods) + .6, 3.6), sharey=True, squeeze=False)
        dummy = bl[bl.modelo == "dummy"]["cv_f1_macro_media"]
        for ax, m_ in zip(axs[0], mods):
            s = bl[bl.modelo == m_].set_index("variante").reindex([o for o in orden if o in set(bl.variante)])
            ax.bar(range(len(s)), s.cv_f1_macro_media, yerr=s.cv_f1_macro_std, capsize=2,
                   color=[col[v[:2]] if v != "BASE" else col["BASE"] for v in s.index])
            if len(dummy): ax.axhline(dummy.iloc[0], color="k", ls=":", lw=1)
            ax.set_xticks(range(len(s))); ax.set_xticklabels([v.replace("V1_", "").replace("V2_", "") for v in s.index],
                                                             rotation=45, ha="right", fontsize=8)
            ax.set_title(m_); ax.grid(alpha=.25)
        axs[0][0].set_ylabel("F1 macro (CV en train)")
        F["variantes"] = _guardar(fig, figdir / "fig_variantes.png")

        fig, ax = plt.subplots(figsize=(6.2, 3.4)); ax2 = ax.twinx()
        vs = bl[bl.variante.isin(N_VISTAS)].copy(); vs["nv"] = vs.variante.map(N_VISTAS)
        for m_ in mods:
            s = vs[vs.modelo == m_].sort_values("nv")
            ax.errorbar(s.nv, s.cv_f1_macro_media, yerr=s.cv_f1_macro_std, marker="o", capsize=2, label=m_)
        lat = vs[vs.modelo == mods[0]].sort_values("nv")
        ax2.bar(lat.nv, lat.latencia_total_ms_fruto, width=.5, alpha=.15, color="k")
        ax.set_xlabel("imágenes usadas por fruto"); ax.set_ylabel("F1 macro (CV en train)")
        ax2.set_ylabel("latencia por fruto (ms, barras)"); ax.set_xticks(sorted(vs.nv.unique())); ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        F["vistas"] = _guardar(fig, figdir / "fig_vistas_latencia.png")

        b = bl[(bl.variante == "BASE") & (bl.modelo != "dummy")].sort_values("cv_f1_macro_media", ascending=False)
        if len(b):
            mejor = b.iloc[0]; cm = c["bl_cm"][mejor.config]
            M, lab = np.array(cm["matriz"]), cm["labels"]
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            ax.imshow(M, cmap="Reds")
            ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8)
            for i in range(len(lab)):
                for j in range(len(lab)):
                    ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="white" if M[i, j] > M.max() / 2 else "black")
            ax.set_xlabel("predicho"); ax.set_ylabel("real"); ax.set_title(f"Validación - {mejor.modelo}", fontsize=10)
            F["confusion"] = _guardar(fig, figdir / "fig_confusion.png")
    return F


# =================================================================================== bloques del informe
f3 = lambda x: f"{x:.3f}"
pm = lambda r, m: f"{r[f'cv_{m}_media']:.3f} ± {r[f'cv_{m}_std']:.3f}"


def construir(c: dict, e: dict, val, F: dict, demo: bool) -> list:
    B = []
    fr, cuotas, cal = c["frutos"], e["cuotas"], e["calidad"]["conteo"]
    n_f, n_i = len(fr), len(c["imgs"])
    if demo:
        B.append(("banner", "ATENCIÓN: este informe se generó con DATOS SINTÉTICOS de demostración (elipses que imitan limones). "
                            "Sirve para ver el formato y validar el flujo; las cifras NO son resultados de la investigación."))
    for aviso in c["avisos"]:
        B.append(("banner", aviso))
    B.append(("h1", "1. Resumen"))
    items = [f"Datos: {n_f} frutos y {n_i} imágenes; {e['n_clases']} de 5 categorías presentes; {e['cob']['n_lotes']} lotes y {e['cob']['n_fechas']} fechas.",
             f"EDA: {int(cal['capturas_incompletas'])} capturas incompletas, {int(e['met']['duplicada'].sum())} imágenes duplicadas, "
             f"{int(e['met']['borrosa'].sum())} muy borrosas y {len(e['inco'])} etiquetas que contradicen el calibre."]
    if int((~cuotas['cumple']).sum()):
        items.append("Cuotas mínimas: faltan frutos en " + ", ".join(f"{k} ({int(v)})" for k, v in cuotas[~cuotas['cumple']]['faltan'].items()) + ".")
    if val and "mae" in val:
        items.append(f"Feature Engineering: {len(COLS_A)} variables originales y {len(COLS_C)} de dominio; calibre extraído con MAE {val['mae']:.2f} mm "
                     f"(meta ≤ 1,0 mm) y sesgo {val['sesgo']:+.2f} mm.")
    if c["fe"] is not None:
        r = c["fe"][~c["fe"].config.str.startswith(("REF", "CONTROL", "A_sin"))].sort_values("cv_f1_macro_media", ascending=False).iloc[0]
        items.append(f"Mejor configuración de FE por validación cruzada: {r.config} (F1 macro {pm(r, 'f1_macro')}).")
    if c["imp"] is not None:
        _m0 = list(c["imp"]["perm"])[0]; _dt = c["imp"]["dim_tabla"][_m0]
        items.append(f"Importancia de variables: la dimensión que más sostiene el F1 macro es {_dt['macro'].idxmax()} (caída de {_dt['macro'].max():.2f} al permutarla).")
    if c["bl"] is not None:
        b = c["bl"][(c["bl"].variante == "BASE") & (c["bl"].modelo != "dummy")].sort_values("cv_f1_macro_media", ascending=False).iloc[0]
        d = c["bl"][c["bl"].modelo == "dummy"].iloc[0]
        items.append(f"Baseline: {b.modelo} con F1 macro {pm(b, 'f1_macro')} frente a {f3(d.cv_f1_macro_media)} de la referencia (clase mayoritaria).")
        k = c["cmp"][c["cmp"].metrica == "f1_macro"]
        mej = k[k.veredicto == "mejora"]
        items.append("Variantes de datos: " + (
            "mejoran con evidencia estadística: " + ", ".join(f"{r.variante} ({r.modelo}, Δ {r.dif_media:+.3f})" for r in mej.itertuples()) + "."
            if len(mej) else "ninguna mostró una mejora estadísticamente demostrada sobre el baseline (Holm, α = 0,05); "
                             "las diferencias observadas están dentro de la variabilidad entre pliegues."))
    items.append("Anexo A: matriz de consistencia (problema, objetivos, hipótesis, variables e indicadores) con el estado de la evidencia de cada hipótesis en este repositorio.")
    B.append(("bullets", items))

    # ------------------------------------------------------------------ datos
    B.append(("h1", "2. Datos"))
    t = cuotas.rename_axis("Categoría").reset_index().rename(columns={"frutos": "Frutos", "cuota_min": "Cuota mínima", "faltan": "Faltan", "cumple": "Cumple"})
    t["Cumple"] = t["Cumple"].map({True: "Sí", False: "No"})
    B.append(("table", t[["Categoría", "Frutos", "Cuota mínima", "Faltan", "Cumple", "%"]], "Frutos por categoría frente a la cuota mínima."))
    B.append(("fig", F["cuotas"], "Figura 1. Frutos por categoría y cuota mínima."))
    B.append(("p", "Unidad de análisis: el fruto (5 imágenes: 4 ecuatoriales y 1 de pedúnculo). Variable objetivo: categoría de referencia construida con "
                   "calibre medido, evaluación ciega del panel y adjudicación. Lote y fecha solo agrupan; no son entradas del modelo."))

    # ------------------------------------------------------------------ EDA
    B.append(("h1", "3. Análisis exploratorio de datos (EDA)"))
    B.append(("h2", "3.1 Calidad de los datos"))
    q = e["calidad"].rename_axis("Verificación").reset_index().rename(columns={"conteo": "Casos"})
    q["Verificación"] = q["Verificación"].str.replace("_", " ")
    B.append(("table", q, "Verificaciones de calidad de etiquetas e imágenes."))
    met = e["met"]
    B.append(("p", f"Imágenes: {int((~met['legible']).sum())} ilegibles, {int(met['duplicada'].sum())} duplicadas exactas y {int(met['borrosa'].sum())} muy borrosas "
                   f"(nitidez atípicamente baja). Resoluciones: " + ", ".join(f"{a}x{b}" for (a, b) in met.groupby(['ancho', 'alto']).size().index.tolist()) + "."))
    B.append(("h2", "3.2 Calibre y consistencia con la categoría"))
    if "calibre" in F:
        B.append(("fig", F["calibre"], "Figura 2. Calibre de referencia por categoría; líneas = umbrales de 35, 37 y 39 mm."))
    B.append(("p", f"Etiquetas no-Fábrica que contradicen su calibre: {len(e['inco'])}. Frutos fronterizos (a ±0,5 mm de un umbral): {len(e['front'])} "
                   f"({100 * len(e['front']) / max(n_f, 1):.1f} %); son los más sensibles al error de medición y exigen la mayor precisión del sistema."))
    B.append(("h2", "3.3 Cobertura y riesgo de atajos"))
    cr = e["cramer"]
    txt = (f"V de Cramér entre categoría y lote = {cr['lote']:.2f}; entre categoría y fecha = {cr['fecha']:.2f} (0 = independiente, 1 = totalmente asociado). "
           "Valores altos indicarían que el contexto de captura predice la categoría (atajo)." if e["n_clases"] > 1 else
           "Con una sola categoría no se puede evaluar la asociación entre categoría y lote/fecha.")
    B.append(("p", txt))
    g = met.merge(fr[["id_fruto", "lote"]], on="id_fruto", how="left" if "lote" not in met else "left", suffixes=("", "_f")) if "lote" not in met else met
    tl = g.groupby("lote")[["brillo", "nitidez", "saturacion"]].mean().round(1).reset_index().rename(
        columns={"lote": "Lote", "brillo": "Brillo", "nitidez": "Nitidez", "saturacion": "Saturación"})
    B.append(("table", tl, "Brillo, nitidez y saturación medios por lote (diferencias grandes = iluminación distinta entre sesiones)."))
    B.append(("h2", "3.4 Hallazgos y acciones"))
    acc = []
    if int(cal["capturas_incompletas"]): acc.append(("Capturas incompletas", int(cal["capturas_incompletas"]), "Aplicar el protocolo: 1 vista ecuatorial faltante = captura parcial; falta el pedúnculo o más de una ecuatorial = excluir."))
    if int(met["duplicada"].sum()): acc.append(("Imágenes duplicadas", int(met["duplicada"].sum()), "Eliminar duplicados y verificar el proceso de captura."))
    if int(met["borrosa"].sum()): acc.append(("Imágenes muy borrosas", int(met["borrosa"].sum()), "Recapturar los frutos afectados."))
    if len(e["inco"]): acc.append(("Etiqueta contradice calibre", len(e["inco"]), "Revisar con el panel y remedir el calibre."))
    if int((~cuotas["cumple"]).sum()): acc.append(("Cuotas sin cumplir", int(cuotas["faltan"].sum()), "Recolección dirigida de las categorías deficitarias."))
    if e["n_clases"] < 5: acc.append(("Categorías ausentes", 5 - e["n_clases"], "Sin ellas no se puede entrenar ni evaluar el clasificador."))
    B.append(("table", pd.DataFrame(acc, columns=["Hallazgo", "Casos", "Acción propuesta"]) if acc else
              pd.DataFrame([("Sin hallazgos críticos", 0, "Continuar.")], columns=["Hallazgo", "Casos", "Acción propuesta"]), "Hallazgos del EDA y acciones."))

    # ------------------------------------------------------------------ FE
    B.append(("h1", "4. Feature Engineering"))
    B.append(("p", f"Cada fruto se resume en {len(COLS_A)} variables originales (geometría, color CIELAB sobre superficie sana, defectos, textura; agregadas por "
                   f"fruto: calibre = mediana de las vistas ecuatoriales, color = media, defecto = máximo de las 5 vistas) y {len(COLS_C)} variables de dominio "
                   "(croma, matiz, amarillez, distancia a los umbrales de calibre, entre otras). Las transformaciones con estado se ajustan solo con train dentro de un Pipeline."))
    B.append(("h2", "4.1 Validación de la extracción"))
    if val and "mae" in val:
        B.append(("p", f"Calibre de la imagen frente al calibrador (n = {val['n']}): MAE {val['mae']:.2f} mm, RMSE {val['rmse']:.2f} mm, sesgo {val['sesgo']:+.2f} mm."
                       + (f" Defecto de la imagen frente a la severidad del panel: Spearman = {val['rho']:.2f}." if "rho" in val else "")))
    else:
        B.append(("nota", "No hay calibre en mm para validar: falta data/raw/calibracion_sesiones.csv o calibre_mm_ref."))
    if "validacion" in F:
        B.append(("fig", F["validacion"], "Figura 3. Validación de las variables extraídas contra el estándar de referencia."))
    B.append(("h2", "4.2 Partición congelada"))
    if c["part"] is not None:
        pt = pd.crosstab(c["frutos"].merge(c["part"], on="id_fruto")["cat"], c["part"].merge(c["frutos"], on="id_fruto")["subset"])
        pt = pt.reindex([x for x in eda.CATEGORIAS if x in pt.index])[["train", "val", "test"]].reset_index().rename(columns={"cat": "Categoría", "train": "Train", "val": "Val", "test": "Test"})
        B.append(("table", pt, f"Frutos por subconjunto (partición por fruto, huella {huella_particion(c['part'])})."))
    B.append(("h2", "4.3 Ablaciones de Feature Engineering"))
    if c["fe"] is not None:
        fe = c["fe"][~c["fe"].config.str.startswith("CONTROL")].copy()
        met_fab, nom_fab = ("f1_pos", "CV F1 Fábrica") if "cv_f1_pos_media" in fe else ("recall_pos", "CV recall Fábrica")
        tab = pd.DataFrame({"Configuración": fe.config, "#var": fe.n_features_out, "CV F1 macro": [pm(r, "f1_macro") for _, r in fe.iterrows()],
                            nom_fab: [pm(r, met_fab) for _, r in fe.iterrows()],
                            "Val F1 macro": fe.val_f1_macro.map(f3), "Ajuste (s)": fe.cv_fit_s_media.round(2)})
        B.append(("table", tab, "Ablación A (originales), B (transformaciones), C (dominio), S (colinealidad): media ± desviación en CV sobre train."))
        if "fe" in F:
            B.append(("fig", F["fe"], "Figura 4. F1 macro por configuración de Feature Engineering."))
        ctrl = c["fe"][c["fe"].config.str.startswith("CONTROL")]
        if len(ctrl):
            lin = []
            for _, r in ctrl.iterrows():
                m_ = r.config.split("|")[1]
                b = c["fe"][c["fe"].config == f"A|{m_}"].iloc[0]
                lin.append(f"{m_}: A = {pm(b, 'f1_macro')}; A + lote = {pm(r, 'f1_macro')} (Δ {r.cv_f1_macro_media - b.cv_f1_macro_media:+.3f}).")
            B.append(("p", "Control de fuga (añadir `lote` como entrada): " + " ".join(lin) +
                      " Una mejora clara sobre la desviación indicaría que el contexto predice la categoría."))
    else:
        B.append(("nota", "Las ablaciones requieren al menos 2 categorías con datos: se necesitan más categorías para ejecutarlas." if e["n_clases"] < 2
                  else "Aún no se ejecutaron las ablaciones (o la última corrida es de otros datos): python src/fe_experimentos.py"))
    B.append(("h2", "4.4 Importancia de variables"))
    imp = c["imp"]
    if imp is None:
        B.append(("nota", "Aún no se calculó la importancia de variables (requiere >= 2 categorías): python src/importancia.py"))
    else:
        m0 = list(imp["perm"])[0]
        B.append(("p", "Se combinan cuatro métodos que fallan de forma distinta: permutación fuera de muestra (en cada pliegue de la validación cruzada sobre train se mide cuánto cae el F1 macro "
                       "al barajar una variable), permutación agrupada por dimensión (Calibre, Forma, Color, Defecto, Textura), impureza de Random Forest e información mutua. "
                       "Se añade una columna aleatoria de control: una variable solo se considera informativa si supera claramente su importancia. No se usa el conjunto de prueba."))
        dt = imp["dim_tabla"][m0]
        clases = [x for x in dt.columns if x not in ("macro", "suma_individual", "redundancia")]
        top_dim = dt["macro"].idxmax()
        por_clase = "; ".join(f"{k}: {dt[k].idxmax()} ({dt[k].max():.2f})" for k in clases if dt[k].notna().any())
        ac = imp["acuerdo"].to_numpy(); ac = ac[~np.eye(len(ac), dtype=bool)]
        sup = {m: (int(r["supera_ruido"].sum()), len(r) - 1) for m, r in imp["perm"].items() if "supera_ruido" in r}
        B.append(("bullets", [f"Dimensión que más sostiene el F1 macro ({m0}): {top_dim} (caída de {dt['macro'].max():.2f}).",
                              f"Dimensión que más sostiene cada clase (caída de F1): {por_clase}.",
                              "Variables individuales que se distinguen del ruido (superan el umbral del control y ayudan en >= 60 % de los pliegues): " + ", ".join(f"{m} {a} de {b}" for m, (a, b) in sup.items()) + ".",
                              f"Acuerdo entre métodos (correlación de Spearman de los rangos, promedio): {np.nanmean(ac):.2f}."]))
        if "imp_dim" in F:
            B.append(("fig", F["imp_dim"], "Figura 5. Caída de F1 (macro y por clase) al permutar juntas todas las variables de una dimensión."))
        if "imp_perm" in F:
            B.append(("fig", F["imp_perm"], "Figura 6. Importancia por permutación fuera de muestra por variable (barras = media ± desviación; línea = umbral del control de ruido)."))
        cons = imp["consenso"].head(10)
        flags = {v: all(bool(r.loc[v, "supera_ruido"]) if "supera_ruido" in r and v in r.index else False for r in imp["perm"].values()) for v in cons.index}
        tab = pd.DataFrame({"Variable": cons.index, "Dimensión": cons["dimension"].values,
                            **{f"Perm. {m}": cons[f"perm_{m}"].round(3).values for m in imp["perm"]},
                            "Impureza RF": cons["impureza_rf"].round(3).values, "Info. mutua": cons["info_mutua"].round(3).values,
                            "Rango medio": cons["rango_medio"].round(1).values, "Supera ruido": ["sí" if flags[v] else "no" for v in cons.index]})
        B.append(("table", tab, "Diez variables mejor posicionadas por consenso de rangos (menor rango medio = más importante)."))
        red = dt["redundancia"].dropna()
        B.append(("p", "Lectura: la importancia por variable subestima la de las variables redundantes (p. ej. el calibre en píxeles y en milímetros, o el área: cada una se compensa con las demás); "
                       "por eso se reporta también la importancia agrupada. Un cociente de redundancia (grupo / suma de individuales) mayor que 1 indica información compartida entre variables de la dimensión"
                       + (": " + ", ".join(f"{k} {v:.1f}" for k, v in red.items()) + "." if len(red) else ".")
                       + " Una importancia alta no implica causalidad ni que la variable sea necesaria si hay otra equivalente."
                       + (" Con los datos actuales ninguna variable individual se distingue con claridad del ruido en algún modelo: las estimaciones por variable son inestables y la lectura agrupada por dimensión es la más confiable."
                          if any(a == 0 for a, _ in sup.values()) else "")))
    B.append(("h2", "4.5 Limitaciones conocidas"))
    B.append(("bullets", ["El color no está corregido con la carta de color entre sesiones (preprocesamiento pendiente): el análisis de deriva por lote/fecha mide el efecto.",
                          "La escala px→mm depende de calibracion_sesiones.csv; sin ella no hay calibre en mm.",
                          "Los umbrales de color y defecto de las variables de dominio son valores de ejemplo hasta fijarlos con el panel en el piloto.",
                          "La segmentación clásica asume fondo uniforme; con la carta de color cerca del fruto conviene usar las máscaras del modelo de segmentación."]))

    # ------------------------------------------------------------------ baseline
    B.append(("h1", "5. Baseline"))
    B.append(("p", "Referencia mínima antes de cualquier modelo complejo: clasificadores simples sobre las variables originales (A) con imputación por mediana y escalado. "
                   "Modelos: referencia por clase mayoritaria (dummy), kNN (k = 5) y Naive Bayes gaussiano. Misma partición congelada y mismos pliegues que Feature Engineering; "
                   "la validación cruzada usa solo train y el conjunto de prueba no se toca."))
    if c["bl"] is None:
        B.append(("nota", "El baseline y las variantes requieren al menos 2 categorías con datos (hoy hay " + str(e["n_clases"]) + "): "
                  "complete las categorías y cuotas faltantes y luego ejecute python src/modelo_baseline.py." if e["n_clases"] < 2
                  else "Aún no se ejecutó el baseline (o la última corrida es de otros datos): python src/modelo_baseline.py"))
        return B
    bs = c["bl"][c["bl"].variante == "BASE"]
    tab = pd.DataFrame({"Modelo": bs.modelo, "CV F1 macro": [pm(r, "f1_macro") for _, r in bs.iterrows()], "CV F1 Fábrica": [pm(r, "f1_pos") for _, r in bs.iterrows()],
                        "CV recall Fábrica": [pm(r, "recall_pos") for _, r in bs.iterrows()], "CV PR-AUC Fábrica": [pm(r, "prauc_pos") for _, r in bs.iterrows()],
                        "Val F1 macro": bs.val_f1_macro.map(f3), "Latencia (ms/fruto)": bs.latencia_total_ms_fruto.round(0).astype(int)})
    B.append(("table", tab, "Baseline: media ± desviación en validación cruzada (train) y F1 macro en validación."))
    if "confusion" in F:
        B.append(("fig", F["confusion"], "Figura 7. Matriz de confusión en validación del mejor baseline."))
    B.append(("p", "Métrica central: F1 macro (hipótesis de la tesis), acompañada del F1, recall y PR-AUC de la clase Fábrica, que es la crítica: un falso negativo envía un fruto defectuoso a una categoría comercial superior."))

    # ------------------------------------------------------------------ variantes
    B.append(("h1", "6. Variantes de datos"))
    B.append(("p", "Se cambia el dato, no el modelo, y una sola cosa por vez. V1: número de imágenes por fruto (1, 2, 4 ecuatoriales y 5); las features se re-agregan usando solo esas vistas "
                   "en todos los subconjuntos. V2: balanceo de clases en train (sobremuestreo y submuestreo aleatorios, aplicados solo dentro del ajuste para no contaminar validación)."))
    cmp_ = c["cmp"][c["cmp"].metrica == "f1_macro"].set_index(["modelo", "variante"])
    cmp_p = c["cmp"][c["cmp"].metrica == "f1_pos"].set_index(["modelo", "variante"])
    filas = []
    for r in c["bl"][c["bl"].modelo != "dummy"].sort_values(["modelo", "config"]).itertuples():
        rr = c["bl"][c["bl"].config == r.config].iloc[0]
        if r.variante == "BASE":
            filas.append((r.modelo, "BASE", pm(rr, "f1_macro"), "-", "-", "referencia", pm(rr, "f1_pos"), "-", int(round(r.latencia_total_ms_fruto))))
        else:
            a, b = cmp_.loc[(r.modelo, r.variante)], cmp_p.loc[(r.modelo, r.variante)]
            filas.append((r.modelo, r.variante, pm(rr, "f1_macro"), f"{a.dif_media:+.3f}", f"{a.p_holm:.3f}", a.veredicto, pm(rr, "f1_pos"), f"{b.dif_media:+.3f}", int(round(r.latencia_total_ms_fruto))))
    B.append(("table", pd.DataFrame(filas, columns=["Modelo", "Variante", "CV F1 macro", "Δ F1", "p (Holm)", "Veredicto", "CV F1 Fábrica", "Δ F1 Fáb.", "Lat. (ms)"]),
              "Variantes frente al baseline del mismo modelo. Δ = diferencia media pareada por pliegue; p = prueba t corregida para CV repetida con corrección de Holm."))
    if "variantes" in F:
        B.append(("fig", F["variantes"], "Figura 8. F1 macro por variante (gris = baseline, azul = V1 vistas, naranja = V2 balanceo; línea punteada = referencia dummy)."))
    B.append(("h2", "6.1 Número de vistas: precisión frente a costo"))
    if "vistas" in F:
        B.append(("fig", F["vistas"], "Figura 9. F1 macro y latencia de extracción según las imágenes usadas por fruto."))
    v1, v5 = c["bl"][(c["bl"].variante == "V1_1vista")], c["bl"][(c["bl"].variante == "BASE") & (c["bl"].modelo != "dummy")]
    if len(v1) and len(v5):
        B.append(("p", f"Usar 1 vista en lugar de 5 reduce la latencia de extracción de {v5.latencia_total_ms_fruto.mean():.0f} a {v1.latencia_total_ms_fruto.mean():.0f} ms por fruto "
                       "(medida en esta máquina; solo la extracción de features). Si la precisión no cae de forma demostrable, menos vistas simplifican la captura; si cae, se justifica el protocolo multi-vista."))
    B.append(("h2", "6.2 Balanceo de clases"))
    B.append(("p", "El sobremuestreo duplica frutos minoritarios (solo en train); el submuestreo descarta frutos de las clases mayoritarias y reduce el entrenamiento a la cantidad de la clase más pequeña. "
                   "Con clases de pocos frutos, ambos son inestables: priorizar la recolección dirigida antes que depender del remuestreo."))
    B.append(("h2", "6.3 Cómo leer la comparación"))
    B.append(("p", "Una diferencia solo se considera mejora si la prueba t corregida, con Holm, da p < 0,05. Los pliegues y semillas no son independientes, por lo que la prueba es aproximada y conservadora; "
                   "con pocos frutos, diferencias de 0,05 a 0,10 en F1 son ruido. Para la comparación final entre sistemas, usar McNemar o bootstrap pareado sobre el conjunto de prueba."))

    # ------------------------------------------------------------------ conclusiones
    B.append(("h1", "7. Conclusiones y siguientes pasos"))
    k = c["cmp"][c["cmp"].metrica == "f1_macro"].copy()
    top = k.reindex(k.dif_media.abs().sort_values(ascending=False).index).head(3)
    B.append(("bullets", ["Lectura automática (hechos a validar): " + ("; ".join(f"{r.variante} ({r.modelo}) Δ F1 macro {r.dif_media:+.3f}, {r.veredicto}" for r in top.itertuples()) if len(top) else "sin comparaciones") + ".",
                          "Prioridad de datos: completar las categorías y cuotas faltantes antes de interpretar diferencias entre variantes.",
                          "Implementar la corrección de color con la carta y el cálculo de mm_por_px por sesión.",
                          "Entrenar el baseline profundo (MVCNN) y el pipeline multitarea sobre la misma partición para la comparación final."]))
    B.append(("nota", "Interpretación del equipo: [completar con qué variante conviene conservar, por qué, y qué se hará con los hallazgos del EDA].") )

    # ------------------------------------------------------------------ reproducibilidad
    B.append(("h1", "8. Reproducibilidad"))
    rep = [("Huella de la partición", huella_particion(c["part"]) if c["part"] is not None else "-"),
           ("Corrida de Feature Engineering", c["fe_meta"].get("run_id", "-")), ("Corrida de baseline", c["bl_meta"].get("run_id", "-")),
           ("Corrida de importancia", c["imp"]["meta"].get("run_id", "-") if c["imp"] else "-"),
           ("Configuración (hash)", f"FE {c['fe_meta'].get('config_hash', '-')} | baseline {c['bl_meta'].get('config_hash', '-')}"),
           ("Versiones", f"Python {c['bl_meta'].get('python', '-')}, scikit-learn {c['bl_meta'].get('sklearn', '-')}, pandas {c['bl_meta'].get('pandas', '-')}"),
           ("Semillas / pliegues", f"3 semillas x {c['bl_meta'].get('n_splits', '-')} pliegues"),
           ("Comandos", "ingesta.py → particion.py → extraer_features.py → fe_experimentos.py → modelo_baseline.py → informe.py")]
    B.append(("table", pd.DataFrame(rep, columns=["Elemento", "Valor"]), "Elementos para repetir el análisis."))
    return B


# =================================================================================== render Markdown
def a_markdown(B: list, titulo: str, figrel: str) -> str:
    out = [f"# {titulo}\n"]
    for b in B:
        t = b[0]
        if t == "h1": out.append(f"\n## {b[1]}\n")
        elif t == "h2": out.append(f"\n### {b[1]}\n")
        elif t in ("p", "nota"): out.append(("> " if t == "nota" else "") + b[1] + "\n")
        elif t == "banner": out.append(f"> **{b[1]}**\n")
        elif t == "bullets": out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif t == "fig": out.append(f"![{b[2]}]({figrel}/{Path(b[1]).name})\n*{b[2]}*\n")
        elif t == "table":
            df = b[1].astype(str)
            out.append(f"*{b[2]}*\n\n| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns))
            out += ["| " + " | ".join(x.replace("|", "\\|") for x in r) + " |" for r in df.values]
            out.append("")
    return "\n".join(out)


# =================================================================================== render Word
def a_docx(B: list, titulo: str, ruta: Path, autores: str, matriz: dict | None = None):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2); sec.top_margin = sec.bottom_margin = Cm(2)
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    for n, tam in (("Heading 1", 15), ("Heading 2", 12)):
        h = doc.styles[n]; h.font.name = "Calibri"; h.font.size = Pt(tam); h.font.bold = True
        h.font.color.rgb = RGBColor(0x7A, 0x1F, 0x2B); h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

    def sombrear(p, fill):
        shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
        p._p.get_or_add_pPr().append(shd)

    def celda_fill(cell, fill):
        tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill); tcPr.append(shd)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Maestría en Inteligencia Artificial · Proyecto de Investigación II"); r.font.size = Pt(9.5); r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(titulo); r.bold = True; r.font.size = Pt(19); r.font.color.rgb = RGBColor(0x7A, 0x1F, 0x2B)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Clasificación comercial post-cosecha del limón con visión por computadora").italic = True
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"{autores}  ·  {date.today().strftime('%d/%m/%Y')}").font.size = Pt(9.5)

    for b in B:
        t = b[0]
        if t == "h1": doc.add_heading(b[1], 1)
        elif t == "h2": doc.add_heading(b[1], 2)
        elif t == "p": doc.add_paragraph(b[1])
        elif t == "bullets":
            for x in b[1]: doc.add_paragraph(x, style="List Bullet")
        elif t in ("nota", "banner"):
            p = doc.add_paragraph(); r = p.add_run(b[1]); r.bold = t == "banner"; r.italic = t == "nota"
            sombrear(p, "FDE8E8" if t == "banner" else "FFF6E0")
        elif t == "fig":
            ancho_fig = 9.5 if "confusion" in str(b[1]) else 12.5 if ("cuotas" in str(b[1]) or "calibre" in str(b[1])) else 15.0
            doc.add_picture(str(b[1]), width=Cm(ancho_fig))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(b[2]); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in cap.runs: r.italic = True; r.font.size = Pt(9)
        elif t == "table":
            df = b[1].astype(str)
            cap = doc.add_paragraph(b[2]); [setattr(r.font, "size", Pt(9)) or setattr(r, "italic", True) for r in cap.runs]
            tb = doc.add_table(rows=1, cols=len(df.columns)); tb.style = "Table Grid"; tb.autofit = False
            peso = np.clip([max([len(str(x)) for x in df[c_]] + [len(c_) * .8]) for c_ in df.columns], 9, 40).astype(float)
            ancho = 17.0 * peso / peso.sum()
            for j_, c_ in enumerate(df.columns):
                if c_ in ("Variante", "Configuración"): peso[j_] = max(peso[j_], 18)
            ancho = 17.0 * peso / peso.sum()
            fs = 7.5 if len(df.columns) >= 8 else 8
            for j, c_ in enumerate(df.columns):
                cell = tb.rows[0].cells[j]; cell.text = ""; run = cell.paragraphs[0].add_run(c_)
                run.bold = True; run.font.size = Pt(fs); run.font.color.rgb = RGBColor(255, 255, 255); celda_fill(cell, "7A1F2B")
            for fila in df.values:
                cells = tb.add_row().cells
                for j, v in enumerate(fila):
                    cells[j].text = ""; cells[j].paragraphs[0].add_run(v).font.size = Pt(fs)
            for row in tb.rows:
                for j, cell in enumerate(row.cells): cell.width = Cm(ancho[j])
            doc.add_paragraph()
    if matriz:                                   # Anexo A: matriz de consistencia en una seccion horizontal
        from docx.enum.section import WD_ORIENT, WD_SECTION
        from src import matriz_consistencia as mc
        s2 = doc.add_section(WD_SECTION.NEW_PAGE)
        s2.orientation = WD_ORIENT.LANDSCAPE
        s2.page_width, s2.page_height = Cm(29.7), Cm(21.0)
        s2.left_margin = s2.right_margin = s2.top_margin = s2.bottom_margin = Cm(1.4)
        mc.construir_en(doc, matriz, con_titulo=False)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(ruta))


# =================================================================================== principal
def generar(procesados: Path, salida: Path, autores: str = "Angel Joel Alvarado Crisanto · Javier Fernandez Purizaca") -> dict:
    c = cargar(procesados)
    e = calcular_eda(c)
    val = validar_extraccion(c)
    previo = matplotlib.get_backend()
    plt.switch_backend("Agg")                 # sin ventana; se restaura al terminar (no rompe los notebooks)
    try:
        F = hacer_figuras(c, e, val, salida / "figuras")
    finally:
        plt.switch_backend(previo)
    demo = "demo" in str(procesados).lower()
    B = construir(c, e, val, F, demo)
    titulo = "Informe de avance: EDA, Feature Engineering y Baseline con variantes de datos"
    import yaml
    from src import matriz_consistencia as mc
    ym = ROOT / "docs" / "matriz_consistencia.yaml"
    matriz = yaml.safe_load(ym.read_text(encoding="utf-8")) if ym.exists() else None
    md = salida / "Informe_EDA_FE_Baseline.md"
    md.write_text(a_markdown(B, titulo, "figuras") + ("\n" + mc.a_markdown(matriz, anexo=True) if matriz else ""), encoding="utf-8")
    dx = salida / "Informe_EDA_FE_Baseline.docx"
    a_docx(B, titulo, dx, autores, matriz)
    return dict(docx=dx, md=md, figuras=F)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--salida", type=Path, default=ROOT / "informe")
    a = ap.parse_args()
    r = generar(a.procesados, a.salida)
    print(f"Informe generado:\n  {r['docx']}\n  {r['md']}\n  figuras: {len(r['figuras'])}")
