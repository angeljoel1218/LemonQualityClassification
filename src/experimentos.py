"""
experimentos.py
----------------
Orquesta el ANÁLISIS de las ablaciones y de los contrastes del Plan de tesis v4 a partir de los artefactos que dejan las etapas
de cómputo pesado (inferir_hibrido.py y entrenar_directo.py). Aquí no se entrena nada: se agrega, se compara y se contrasta.

    E1  confirmatorio (H2)   mismo modelo, 1 vista (0°) vs 5 vistas agregadas, sobre la prueba (recall de Fábrica, McNemar)
    E2  confirmatorio (H3)   directo vs híbrido (F1 macro, bootstrap pareado estratificado B = 5 000), sobre la prueba
    E3  exploratorio         sin corrección vs brillo vs brillo+color (solo VALIDACIÓN): F1 de sobremadurez y F1 del par Primera-Segunda
    E4  exploratorio         balanceo del clasificador directo (ninguno | pesos | focal | aumento | combinación), solo VALIDACIÓN
    E5  exploratorio         perturbaciones leves de luminancia (solo VALIDACIÓN): ΔF1 y % de frutos que conservan la categoría
    (E1b  entrenar con varias vistas vs una sola: exploratorio adicional si alcanza el presupuesto; no se calcula aquí)

Convención de carpetas (las crean los scripts de cómputo):
    logs/inferencia/<nombre>/vistas_{val,test}.csv          híbrido (una fila por vista)     nombres: base, e3_brillo, e3_brillo_color,
                                                            e5_luz_mas, e5_luz_menos, e5_gamma_alto, e5_gamma_bajo, final (test)
    logs/directo/<nombre>/{pred_val.csv, pred_test.csv, resumen.json}                        nombres: directo_ninguno, directo_pesos, ...

Subcomandos:
    python src/experimentos.py congelar --directo directo_ninguno [--preproc ninguno]   # congela modelos (ANTES de la prueba)
    python src/experimentos.py analizar [--hibrido final --directo directo_ninguno]     # escribe logs/resultados/<corrida>/
La PRUEBA solo se usa si existen su huella registrada (src/protocolo.py) y los umbrales y modelos congelados.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import atributos as at  # noqa: E402
from src import config as C  # noqa: E402
from src import contrastes as ct  # noqa: E402
from src import estadistica as est  # noqa: E402
from src import protocolo  # noqa: E402
from src import regla_comercial as rc  # noqa: E402
from src.eda import normalizar_categorias  # noqa: E402
from src.ingesta import leer_csv  # noqa: E402
from src.particion import cargar_meta, cargar_particion, huella_particion  # noqa: E402

DIR_INF = ROOT / "logs" / "inferencia"
DIR_DIR = ROOT / "logs" / "directo"
DIR_RES = ROOT / "logs" / "resultados"
VARIANTES_E3 = {"ninguno": "base", "brillo": "e3_brillo", "brillo_color": "e3_brillo_color"}
PERTURBACIONES = ["luz_mas", "luz_menos", "gamma_alto", "gamma_bajo"]
BALANCEOS = ["ninguno", "pesos", "focal", "aumento", "combinacion"]


# ------------------------------------------------------------------------------------ contexto
def cargar_contexto(procesados: Path = ROOT / "data" / "processed", raw: Path = ROOT / "data" / "raw") -> pd.DataFrame:
    """Una fila por fruto: id_fruto, y (categoría de referencia), calibre_mm_ref, color_panel, lote, fecha, subset, reservado."""
    et = leer_csv(raw / "etiquetas.csv", numericas=["calibre_mm_ref", "severidad_defecto_panel"])
    et["y"] = normalizar_categorias(et["categoria_ref"])
    part = cargar_particion(procesados)
    if "reservado" not in part:
        part["reservado"] = 0
    cols = [c for c in ("id_fruto", "y", "calibre_mm_ref", "color_panel", "severidad_defecto_panel", "lote", "fecha") if c in et]
    return et[cols].merge(part, on="id_fruto")


def _leer_vistas(nombre: str, subset: str):
    ruta = DIR_INF / nombre / f"vistas_{subset}.csv"
    return pd.read_csv(ruta, dtype={"id_fruto": str, "vista": str, "lote": str, "fecha": str}) if ruta.exists() else None


def _leer_directo(nombre: str, archivo: str):
    ruta = DIR_DIR / nombre / archivo
    return pd.read_csv(ruta, dtype={"id_fruto": str}) if ruta.exists() else None


def _f1_ic(y, p, B):
    r = est.ic_estratificado(y, p, "f1_macro", B)
    return dict(f1_macro=r["estimado"], ic=(r["ic_inf"], r["ic_sup"]))


# ------------------------------------------------------------------------------------ bloques de análisis (puros)
def clasificar_hibrido(vistas: pd.DataFrame, ctx: pd.DataFrame, u: rc.UmbralesRegla, vistas_usadas=None) -> pd.DataFrame:
    """Frutos con atributos agregados, categoría de la regla, categoría de referencia y calibre medido."""
    t = at.clasificar_tabla(vistas, u, vistas_usadas)
    return t.merge(ctx[["id_fruto", "y", "calibre_mm_ref", "color_panel", "reservado"]], on="id_fruto")


def e1_multivista(vistas_test: pd.DataFrame, ctx: pd.DataFrame, u, B=None) -> dict:
    """H2: 1 vista (0°) vs 5 vistas con el MISMO modelo sobre los mismos frutos."""
    h5, h1 = clasificar_hibrido(vistas_test, ctx, u), clasificar_hibrido(vistas_test, ctx, u, ["0"])
    m = h5.merge(h1[["id_fruto", "pred"]], on="id_fruto", suffixes=("_5", "_1"))
    r = ct.contraste_h2(m["y"], m["pred_5"], m["pred_1"], B=B)
    r["tipos_defecto_sensibilidad"] = None
    return r


def e3_preprocesamiento(vistas_por_variante: dict, ctx: pd.DataFrame, u, B=None) -> list:
    """Exploratorio (validación): F1 macro, F1 de sobremadurez ('amarillo' del panel) y F1 del par Primera–Segunda por variante."""
    filas = []
    for var, v in vistas_por_variante.items():
        t = clasificar_hibrido(v, ctx, u)
        f1 = est.ic_estratificado(t["y"], t["pred"], "f1_macro", B)
        am = (t["color_panel"].astype(str).str.lower() == "amarillo") if t["color_panel"].notna().any() else None
        f1_color = float(est.f1_clase(np.where(am, "amarillo", "otro"), np.where(t["p_amarillo"] >= u.p_am, "amarillo", "otro"), "amarillo")) if am is not None else np.nan
        filas.append(dict(variante=var, n=int(len(t)), f1_macro=f1["estimado"], ic_inf=f1["ic_inf"], ic_sup=f1["ic_sup"], f1_madurez=f1_color,
                          f1_primera_segunda=est.f1_par(t["y"], t["pred"], ("Primera", "Segunda"))))
    d = pd.DataFrame(filas)
    if "ninguno" in set(d["variante"]):
        b = d[d["variante"] == "ninguno"].iloc[0]
        d["delta_f1_macro"] = d["f1_macro"] - b["f1_macro"]
        d["delta_f1_madurez_pp"] = 100 * (d["f1_madurez"] - b["f1_madurez"])
        d["delta_f1_par_pp"] = 100 * (d["f1_primera_segunda"] - b["f1_primera_segunda"])
    return d.to_dict("records")


def e4_balanceo(preds: dict, B=None) -> list:
    """Exploratorio (validación): clasificador directo por estrategia de balanceo (F1 macro, F1 de Tercera y de Fábrica; Δ vs 'ninguno')."""
    filas = []
    for b, p in preds.items():
        f1 = est.ic_estratificado(p["y"], p["pred"], "f1_macro", B)
        filas.append(dict(balanceo=b, n=int(len(p)), f1_macro=f1["estimado"], ic_inf=f1["ic_inf"], ic_sup=f1["ic_sup"],
                          f1_tercera=est.f1_clase(p["y"], p["pred"], "Tercera"), f1_fabrica=est.f1_clase(p["y"], p["pred"], "Fabrica")))
    d = pd.DataFrame(filas)
    if "ninguno" in set(d["balanceo"]):
        b0 = d[d["balanceo"] == "ninguno"].iloc[0]
        for c in ("f1_macro", "f1_tercera", "f1_fabrica"):
            d[f"delta_{c}_pp"] = 100 * (d[c] - b0[c])
    return d.to_dict("records")


def e5_perturbaciones(control: pd.DataFrame, perturbadas: dict, ctx: pd.DataFrame, u, B=None) -> list:
    """Exploratorio (validación): ΔF1 macro y proporción de frutos que conservan la categoría bajo perturbaciones leves de luminancia."""
    c0 = clasificar_hibrido(control, ctx, u)
    f0 = est.ic_estratificado(c0["y"], c0["pred"], "f1_macro", B)["estimado"]
    filas = []
    for nombre, v in perturbadas.items():
        t = clasificar_hibrido(v, ctx, u).merge(c0[["id_fruto", "pred"]], on="id_fruto", suffixes=("", "_ctrl"))
        f1 = est.ic_estratificado(t["y"], t["pred"], "f1_macro", B)
        filas.append(dict(perturbacion=nombre, n=int(len(t)), f1_macro=f1["estimado"], delta_f1=f1["estimado"] - f0,
                          conserva_categoria=float((t["pred"] == t["pred_ctrl"]).mean())))
    return filas


def sobreajuste_directo(resumen: dict) -> dict:
    """Brecha entrenamiento-validación de la semilla mediana (pérdida y F1) y variabilidad entre semillas."""
    cur = pd.DataFrame(resumen.get("curvas_semilla_mediana", []))
    if cur.empty:
        return dict(disponible=False)
    k = int(cur["loss_val"].idxmin())
    return dict(disponible=True, epocas=int(len(cur)), mejor_epoca=int(cur.loc[k, "epoca"]), loss_train=float(cur.loc[k, "loss_train"]),
                loss_val=float(cur.loc[k, "loss_val"]), brecha_loss=float(cur.loc[k, "loss_val"] - cur.loc[k, "loss_train"]),
                f1_val_mejor_epoca=float(cur.loc[k, "f1_val"]), f1_val_semillas=[s["val_f1_macro"] for s in resumen.get("semillas", [])],
                f1_val_std=resumen.get("val_f1_macro_std"))


# ------------------------------------------------------------------------------------ principal
def analizar(procesados: Path = ROOT / "data" / "processed", raw: Path = ROOT / "data" / "raw", hibrido_test: str = "final",
             directo: str = "directo_ninguno", B: int | None = None, verbose: bool = True) -> dict:
    B = int(B or C.EST["bootstrap_B"])
    ctx = cargar_contexto(procesados, raw)
    part = cargar_particion(procesados)
    meta = cargar_meta(procesados)
    u = rc.cargar_umbrales()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dict(run_id=run_id, fecha=datetime.now().isoformat(timespec="seconds"), huella_particion=huella_particion(part),
               particion_provisional=bool(meta.get("provisional", True)), config_hash=C.hash_config(),
               umbrales=dict(theta_def=u.theta_def, h_am=u.h_am, p_am=u.p_am, congelados=rc.RUTA_UMBRALES.exists()),
               n_por_subconjunto=ctx["subset"].value_counts().to_dict(), B=B)
    val = ctx[ctx["subset"] == "val"]

    # ---- desarrollo (validación): E3, E4, E5, base
    des = {}
    vb = _leer_vistas("base", "val")
    if vb is not None:
        t = clasificar_hibrido(vb, val, u)
        des["hibrido_base_val"] = dict(n=int(len(t)), **_f1_ic(t["y"], t["pred"], B), **{k: v for k, v in
                                       ct.contraste_h1(t["d_est_mm"], t["calibre_mm_ref"], t["y"], B).items() if k in ("mae", "rmse", "adyacente")})
    e3 = {k: _leer_vistas(n, "val") for k, n in VARIANTES_E3.items()}
    e3 = {k: v for k, v in e3.items() if v is not None}
    des["E3"] = e3_preprocesamiento(e3, val, u, B) if len(e3) >= 1 else []
    e4 = {b: _leer_directo(f"directo_{b}", "pred_val.csv") for b in BALANCEOS}
    e4 = {b: p for b, p in e4.items() if p is not None}
    des["E4"] = e4_balanceo(e4, B) if e4 else []
    e5 = {p: _leer_vistas(f"e5_{p}", "val") for p in PERTURBACIONES}
    e5 = {p: v for p, v in e5.items() if v is not None}
    des["E5"] = e5_perturbaciones(vb, e5, val, u, B) if (vb is not None and e5) else []
    res_dir = DIR_DIR / directo / "resumen.json"
    des["directo"] = json.loads(res_dir.read_text(encoding="utf-8")) if res_dir.exists() else None
    if des["directo"]:
        des["sobreajuste"] = sobreajuste_directo(des["directo"])
        des["directo"] = {k: v for k, v in des["directo"].items() if k != "curvas_semilla_mediana"}
    out["desarrollo"] = des

    # ---- prueba (UNA vez; con huella verificada)
    prueba = dict(disponible=False)
    vt = _leer_vistas(hibrido_test, "test")
    if vt is not None:
        chk = protocolo.verificar_artefacto(f"hibrido_{hibrido_test}_test", DIR_INF / hibrido_test / "vistas_test.csv")
        test = ctx[ctx["subset"] == "test"]
        h5 = clasificar_hibrido(vt, test, u)
        h1 = clasificar_hibrido(vt, test, u, ["0"])
        dt = _leer_directo(directo, "pred_test.csv")
        m = h5.merge(h1[["id_fruto", "pred"]], on="id_fruto", suffixes=("", "_1"))
        pred_d = None
        if dt is not None:
            m = m.merge(dt[["id_fruto", "pred"]].rename(columns={"pred": "pred_directo"}), on="id_fruto", how="left")
            m = m.dropna(subset=["pred_directo"])
            pred_d = m["pred_directo"].to_numpy()
        ops = None
        try:
            from src import operarios as op
            od = op.cargar_operarios()
            ops = op.resumen(od, m["id_fruto"], m["y"].to_numpy())
        except (SystemExit, FileNotFoundError, KeyError):
            ops = None
        preds_ops = ops["preds"] if ops and ops["suficientes"] else {}
        c = ct.evaluar(m["y"].to_numpy(), m["pred"].to_numpy(), m["pred_1"].to_numpy(), pred_d, m["d_est_mm"].to_numpy(), m["calibre_mm_ref"].to_numpy(),
                       preds_ops, B)
        prev = None
        meta_prev = ROOT / "data" / "processed" / "prevalencia_natural.json"
        if meta_prev.exists():
            prev = est.metricas_ponderadas(m["y"], m["pred"], json.loads(meta_prev.read_text(encoding="utf-8")))
        labs = C.CATEGORIAS
        cm = est.matriz_confusion_idx(est._a_indices(m["y"], labs), est._a_indices(m["pred"], labs), len(labs))
        prueba = dict(disponible=True, n=int(len(m)), huella=chk, contrastes=c, por_categoria=ct.por_categoria(m["y"], m["pred"]).to_dict("records"),
                      matriz_confusion=dict(labels=labs + ["otro"], matriz=cm.tolist()), prevalencia_natural=prev,
                      kappa_ponderado_calibre=est.kappa_ponderado_calibre(m["y"], m["pred"]),
                      lote_reservado=ct.por_lote_reservado(m["y"], m["pred"], m["id_fruto"], m.loc[m["reservado"] == 1, "id_fruto"]),
                      operarios=None if ops is None else {k: v for k, v in ops.items() if k != "preds"},
                      latencia_hibrido=_latencia(vt), latencia_directo_ms=float(dt["t_ms"].median()) if dt is not None and "t_ms" in dt else None,
                      errores=m.loc[m["y"] != m["pred"], ["id_fruto", "y", "pred", "d_est_mm", "calibre_mm_ref", "severidad_max", "p_amarillo", "motivo_fabrica"]]
                      .head(40).to_dict("records"))
    out["prueba"] = prueba

    d = DIR_RES / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "resultados.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=_json), encoding="utf-8")
    (DIR_RES / "ultimo.json").write_text(json.dumps(dict(run_id=run_id), indent=2), encoding="utf-8")
    if verbose:
        print(f"Resultados -> {d / 'resultados.json'} (prueba {'disponible' if prueba['disponible'] else 'NO disponible'})")
    return out


def _latencia(vt: pd.DataFrame) -> dict:
    if "t_total_ms" not in vt:
        return dict(mediana_ms=None, p95_ms=None)
    t = vt.groupby("id_fruto")["t_total_ms"].sum()
    return dict(mediana_ms=float(t.median()), p95_ms=float(t.quantile(.95)), cumple=bool(t.median() <= C.EST["latencia_ms_max"]))


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return str(o)


def cargar_ultimo() -> dict | None:
    p = DIR_RES / "ultimo.json"
    if not p.exists():
        return None
    r = DIR_RES / json.loads(p.read_text(encoding="utf-8"))["run_id"] / "resultados.json"
    return json.loads(r.read_text(encoding="utf-8")) if r.exists() else None


def congelar(directo: str, preproc: str = "ninguno", forzar: bool = False) -> dict:
    """Congela los modelos que se usarán en la prueba: segmentación elegida (fruto y defectos), clasificador directo (semilla mediana)."""
    from src import segmentacion as sg
    info = dict(preproc=preproc)
    for tarea in ("fruto", "defectos"):
        p = sg.DIR_SEG / tarea / "modelo_elegido.json"
        if p.exists():
            e = json.loads(p.read_text(encoding="utf-8"))["elegido"]
            info[f"segmentacion_{tarea}"] = dict(familia=e["familia"], pesos=e["pesos"], nombre=e["nombre"])
    if "segmentacion_fruto" not in info:
        raise SystemExit("Falta el modelo de segmentación del fruto elegido (python src/segmentacion.py elegir --tarea fruto).")
    r = DIR_DIR / directo / "resumen.json"
    if not r.exists():
        raise SystemExit(f"Falta {r}: entrene el clasificador directo (python src/entrenar_directo.py buscar).")
    rd = json.loads(r.read_text(encoding="utf-8"))
    info["directo"] = dict(nombre=directo, pesos=rd["pesos_elegidos"], semilla_mediana=rd["semilla_mediana"], hp=rd["hp"])
    info["umbrales_huella"] = rc.cargar_umbrales(exigir_congelados=True).__dict__ and json.loads(rc.RUTA_UMBRALES.read_text(encoding="utf-8")).get("huella")
    return protocolo.congelar_modelos(info, forzar)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["analizar", "congelar"])
    ap.add_argument("--hibrido", default="final")
    ap.add_argument("--directo", default="directo_ninguno")
    ap.add_argument("--preproc", default="ninguno")
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--B", type=int, default=None)
    a = ap.parse_args()
    if a.accion == "congelar":
        print(json.dumps(congelar(a.directo, a.preproc), indent=2, default=str))
    else:
        analizar(a.procesados, a.raw, a.hibrido, a.directo, a.B)
