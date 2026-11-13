"""
modelo_baseline.py
------------------
BASELINE + VARIANTES DE DATOS (Semana 3 del curso).

Baseline (clasico, sobre las variables originales "A" de Feature Engineering):
    dummy  siempre la clase mayoritaria (piso de desempeno)
    knn    k = 5                     gnb  Naive Bayes gaussiano

Variantes de DATOS (se cambia el dato, no el modelo; una cosa por vez):
    V1  numero de vistas por fruto: 1 vista (0), 2 vistas (0+180), 4 ecuatoriales (sin pedunculo), 5 (baseline)
        -> se re-agregan las features de cada fruto usando SOLO esas vistas (todos los subconjuntos).
    V2  balanceo de clases en TRAIN: sobremuestreo y submuestreo (solo dentro de fit; ver src/remuestreo.py)

Protocolo (identico a Feature Engineering):
    * misma particion congelada, mismos frutos y mismos pliegues/semillas -> comparaciones PAREADAS
    * CV estratificada solo sobre train + evaluacion en val. TEST no se toca.
    * Comparacion variante vs baseline: prueba t corregida para CV repetida (Nadeau & Bengio, 2003)
      con correccion de Holm para multiples comparaciones.
    * Se registra configuracion, metricas, costo (ajuste) y latencia (extraccion + prediccion).

Metricas (seccion 4.6.4 de la tesis): F1 macro (central), exactitud balanceada, kappa de Cohen frente a la referencia,
recall / F1 / PR-AUC de Fabrica y F1 por clase (Tercera y Fabrica: H5). En VALIDACION se guardan ademas las predicciones por
fruto (`predicciones_val.csv`) para el bootstrap pareado (B = 1000) y la prueba de McNemar de la tesis.

Salidas:  logs/baseline_runs/<corrida>/  y  logs/metrics_baseline.txt  (+ logs/baseline_historial.csv)

Uso:
    python src/modelo_baseline.py
    python src/modelo_baseline.py --procesados data/demo/processed
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import fe_experimentos as fx  # noqa: E402
from src.estadistica import holm, kappa_cohen, t_corregida  # noqa: E402,F401  (re-exportadas: las usan las pruebas)
from src.extraer_features import agregar_por_fruto, cargar_cfg  # noqa: E402
from src.fe_columnas import columnas_para  # noqa: E402
from src.fe_pipeline import construir_pipeline  # noqa: E402
from src.particion import huella_particion  # noqa: E402
from src.remuestreo import indices_remuestreo  # noqa: E402

LOGS = ROOT / "logs"
METRICAS = ["f1_macro", "f1_pos", "bal_acc", "recall_pos", "prauc_pos"]
NOMBRES_REMUESTREO = {"sobre": "V2_sobremuestreo", "sub": "V2_submuestreo"}


def cargar_cfg_baseline(ruta: Path = ROOT / "configs" / "baseline.yaml") -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------------------ metricas
def scorers(pos: str, clases=None) -> dict:
    """Metricas del baseline: las de Feature Engineering + F1 de la clase positiva + kappa de Cohen
    (+ F1 por clase si se pasa `clases`; Fabrica ya esta en `f1_pos`)."""
    sc = fx.hacer_scorers(pos)

    def f1_pos(est, X, y):
        yb = np.asarray(y) == pos
        return f1_score(yb, np.asarray(est.predict(X)) == pos, zero_division=0) if yb.any() else np.nan

    def kappa(est, X, y):
        return kappa_cohen(y, est.predict(X))

    def f1_de(c):
        def f(est, X, y):
            yb = np.asarray(y) == c
            return f1_score(yb, np.asarray(est.predict(X)) == c, zero_division=0) if yb.any() else np.nan
        return f

    sc["f1_pos"] = f1_pos
    sc["kappa"] = kappa
    for c in (clases or []):
        if c != pos:
            sc[f"f1_{c}"] = f1_de(c)
    return sc


# ------------------------------------------------------------------------------------ estadistica
# t_corregida y holm viven en src/estadistica.py (se re-exportan arriba para no romper el uso historico).


# ------------------------------------------------------------------------------------ datos
def datos_con_vistas(df_base: pd.DataFrame, vistas_df: pd.DataFrame, vistas, umbral: float) -> pd.DataFrame:
    """Mismos frutos y mismo orden que df_base, pero con features re-agregadas usando solo `vistas`."""
    f = agregar_por_fruto(vistas_df[vistas_df["vista"].isin(vistas)], umbral)
    cols = columnas_para({"A"}) + ["t_extraccion_ms"]
    return df_base[["id_fruto", "lote", "fecha", "y", "subset"]].merge(f[["id_fruto"] + cols],
                                                                       on="id_fruto", how="left")


def lista_experimentos(cfg: dict) -> list:
    variantes = [dict(variante="BASE", familia="BASE", datos="base", vistas=None, remuestreo=None)]
    for nom, vs in cfg["variantes"]["vistas"].items():
        variantes.append(dict(variante=f"V1_{nom}", familia="V1_vistas", datos=nom, vistas=vs, remuestreo=None))
    for m in cfg["variantes"]["remuestreo"]:
        variantes.append(dict(variante=NOMBRES_REMUESTREO[m], familia="V2_balanceo", datos="base",
                              vistas=None, remuestreo=m))
    exp = []
    for modelo in cfg["modelos"]:
        for v in variantes:
            if modelo == "dummy" and v["variante"] != "BASE":
                continue                                   # el dummy solo sirve como referencia
            exp.append({**v, "modelo": modelo, "config": f"{v['variante']}|{modelo}"})
    return exp


# ------------------------------------------------------------------------------------ evaluacion
def evaluar(exp: dict, tr: pd.DataFrame, va: pd.DataFrame, cfg: dict, cfg_pipe: dict, n_splits: int):
    cols = [c for c in columnas_para({"A"}) if tr[c].notna().any()]
    Xtr, ytr = tr[cols], tr["y"].to_numpy()
    sc = scorers(cfg["positivo"], sorted(set(ytr) | set(va["y"])))
    filas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for s in cfg["semillas"]:
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=s)
            pipe = construir_pipeline(cols, "A", exp["modelo"], s, cfg_pipe, True, None, exp["remuestreo"])
            r = cross_validate(pipe, Xtr, ytr, cv=skf, scoring=sc, error_score="raise")
            tam = [len(te) for _, te in skf.split(Xtr, ytr)]
            for k in range(n_splits):
                filas.append(dict(config=exp["config"], semilla=s, pliegue=k, n_test_fold=tam[k],
                                  n_train_fold=len(Xtr) - tam[k], fit_s=r["fit_time"][k],
                                  pred_ms=r["score_time"][k] / tam[k] * 1000,
                                  **{m: r[f"test_{m}"][k] for m in sc}))
        pipe = construir_pipeline(cols, "A", exp["modelo"], cfg["semillas"][0], cfg_pipe, True, None, exp["remuestreo"])
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        yva = va["y"].to_numpy()
        val = {m: f(pipe, va[cols], yva) for m, f in sc.items()}
        etiquetas = sorted(set(ytr) | set(yva))
        pred_va = pipe.predict(va[cols])
        cm = confusion_matrix(yva, pred_va, labels=etiquetas)
    cv = pd.DataFrame(filas)
    fila = dict(config=exp["config"], variante=exp["variante"], familia=exp["familia"], modelo=exp["modelo"],
                n_features=len(cols), n_train=len(Xtr),
                n_train_efectivo=int(len(indices_remuestreo(ytr, exp["remuestreo"]))) if exp["remuestreo"] else len(Xtr))
    for m in sc:
        fila[f"cv_{m}_media"], fila[f"cv_{m}_std"], fila[f"val_{m}"] = cv[m].mean(), cv[m].std(), val[m]
    fila.update(fit_s_train=fit_s, cv_fit_s_media=cv["fit_s"].mean(), pred_ms_fruto=cv["pred_ms"].mean(),
                extraccion_ms_fruto=float(tr["t_extraccion_ms"].mean()))
    fila["latencia_total_ms_fruto"] = fila["pred_ms_fruto"] + fila["extraccion_ms_fruto"]
    return cv, fila, dict(labels=etiquetas, matriz=cm.tolist(), pred_val=[str(x) for x in pred_va])


def comparar(cv_all: pd.DataFrame, res: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cada variante contra BASE (mismo modelo), con diferencias pareadas por (semilla, pliegue)."""
    filas = []
    for _, r in res[res.variante != "BASE"].iterrows():
        a = cv_all[cv_all.config == f"BASE|{r.modelo}"].sort_values(["semilla", "pliegue"])
        b = cv_all[cv_all.config == r.config].sort_values(["semilla", "pliegue"])
        for m in [m for m in ("f1_macro", "f1_pos", "kappa", "f1_Tercera") if m in a.columns and m in b.columns]:
            d = b[m].to_numpy() - a[m].to_numpy()
            ok = np.isfinite(d)
            t, p = t_corregida(d[ok], a["n_train_fold"].mean(), a["n_test_fold"].mean())
            filas.append(dict(modelo=r.modelo, variante=r.variante, familia=r.familia, metrica=m,
                              dif_media=float(np.mean(d[ok])), dif_std=float(np.std(d[ok], ddof=1)),
                              pliegues_mejoran_pct=float(100 * np.mean(d[ok] > 0)), t=t, p=p))
    c = pd.DataFrame(filas)
    if c.empty:
        return c
    c["p_holm"] = np.nan
    for m, idx in c.groupby("metrica").groups.items():
        c.loc[idx, "p_holm"] = holm(c.loc[idx, "p"])
    alfa = cfg.get("alfa", 0.05)
    c["veredicto"] = np.where(c.p_holm < alfa, np.where(c.dif_media > 0, "mejora", "empeora"), "sin evidencia")
    return c


# ------------------------------------------------------------------------------------ tablas / registro
def _ms(r, m):
    return f"{r[f'cv_{m}_media']:.3f} ± {r[f'cv_{m}_std']:.3f}"


def tabla_baseline_md(res: pd.DataFrame) -> str:
    b = res[res.variante == "BASE"]
    t = "| Modelo | CV F1 macro | CV kappa | CV F1 Fábrica | CV recall Fábrica | CV PR-AUC Fábrica | VAL F1 macro | VAL kappa | VAL F1 Fábrica | ajuste (s) | latencia (ms/fruto) |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    for _, r in b.iterrows():
        t += (f"| {r.modelo} | {_ms(r, 'f1_macro')} | {_ms(r, 'kappa')} | {_ms(r, 'f1_pos')} | {_ms(r, 'recall_pos')} | {_ms(r, 'prauc_pos')} | "
              f"{r.val_f1_macro:.3f} | {r.val_kappa:.3f} | {r.val_f1_pos:.3f} | {r.cv_fit_s_media:.2f} | {r.latencia_total_ms_fruto:.0f} |\n")
    return t


def tabla_variantes_md(res: pd.DataFrame, comp: pd.DataFrame) -> str:
    t = "| Modelo | Variante | CV F1 macro | Δ vs baseline | p (Holm) | Veredicto | CV F1 Fábrica | Δ F1 Fábrica | VAL F1 macro | latencia (ms/fruto) |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for _, r in res[res.modelo != "dummy"].sort_values(["modelo", "config"]).iterrows():
        if r.variante == "BASE":
            t += f"| {r.modelo} | BASE | {_ms(r, 'f1_macro')} | - | - | referencia | {_ms(r, 'f1_pos')} | - | {r.val_f1_macro:.3f} | {r.latencia_total_ms_fruto:.0f} |\n"
            continue
        g = comp[(comp.modelo == r.modelo) & (comp.variante == r.variante)].set_index("metrica")
        a, b = g.loc["f1_macro"], g.loc["f1_pos"]
        t += (f"| {r.modelo} | {r.variante} | {_ms(r, 'f1_macro')} | {a.dif_media:+.3f} | {a.p_holm:.3f} | {a.veredicto} | "
              f"{_ms(r, 'f1_pos')} | {b.dif_media:+.3f} | {r.val_f1_macro:.3f} | {r.latencia_total_ms_fruto:.0f} |\n")
    return t


def registrar(run_id, cfg, cfg_fe, res, cv_all, comp, cms, huella_part, extra):
    d = LOGS / "baseline_runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    extra = dict(extra)
    ids_val, y_val = extra.pop("ids_val", None), extra.pop("y_val", None)
    meta = dict(run_id=run_id, python=platform.python_version(), sklearn=sklearn.__version__,
                pandas=pd.__version__, numpy=np.__version__, huella_particion=huella_part,
                config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12], **extra)
    extra = dict(extra, ids_val=ids_val, y_val=y_val)
    (d / "config.json").write_text(json.dumps({"meta": meta, "cfg": cfg}, indent=2, ensure_ascii=False), encoding="utf-8")
    res.to_csv(d / "resultados.csv", index=False)
    cv_all.to_csv(d / "cv_pliegues.csv", index=False)
    comp.to_csv(d / "comparaciones.csv", index=False)
    cms_json = {k: {kk: vv for kk, vv in v.items() if kk != "pred_val"} for k, v in cms.items()}
    (d / "matrices_confusion_val.json").write_text(json.dumps(cms_json, indent=2), encoding="utf-8")
    if extra.get("y_val") is not None:
        pv = pd.DataFrame({"id_fruto": extra["ids_val"], "y": extra["y_val"]})
        for k, v in cms.items():
            pv[k] = v["pred_val"]
        pv.to_csv(d / "predicciones_val.csv", index=False)
    (d / "tabla_baseline.md").write_text(tabla_baseline_md(res), encoding="utf-8")
    (d / "tabla_variantes.md").write_text(tabla_variantes_md(res, comp), encoding="utf-8")
    hist = LOGS / "baseline_historial.csv"
    nuevo = res.assign(run_id=run_id, huella_particion=huella_part, config_hash=meta["config_hash"])
    if hist.exists():
        nuevo = pd.concat([pd.read_csv(hist), nuevo], ignore_index=True)
    nuevo.to_csv(hist, index=False)
    txt = [f"BASELINE Y VARIANTES DE DATOS - corrida {run_id}",
           f"Huella de particion: {huella_part} | config: {meta['config_hash']} | metrica central: {cfg['metrica_central']}",
           f"Frutos train/val/test: {extra['n_train']}/{extra['n_val']}/{extra['n_test']} | CV: {extra['n_splits']} pliegues x {len(cfg['semillas'])} semillas",
           "", "== BASELINE ==", tabla_baseline_md(res), "== VARIANTES DE DATOS (vs baseline del mismo modelo) ==",
           tabla_variantes_md(res, comp), "Δ = diferencia media pareada de F1; p = prueba t corregida (Nadeau & Bengio) con Holm."]
    (LOGS / "metrics_baseline.txt").write_text("\n".join(txt), encoding="utf-8")
    return d


def ejecutar(procesados: Path, cfg: dict | None = None, verbose: bool = True):
    cfg = cfg or cargar_cfg_baseline()
    cfg_fe = cargar_cfg()
    cfg_pipe = {**cfg_fe, "modelo_params": {"knn_k": cfg["knn_k"]}}
    df = fx.cargar_datos(procesados, cfg_fe)
    tr, va, te = fx.dividir(df)
    cuenta = tr["y"].value_counts()
    if len(cuenta) < 2 or cuenta.min() < 2:
        raise SystemExit(f"Se necesitan >=2 clases con >=2 frutos en train. Ahora: {cuenta.to_dict()}")
    n_splits = int(min(cfg["cv_folds"], cuenta.min()))
    vistas_df = pd.read_csv(procesados / "features_vistas.csv",
                            dtype={"id_fruto": str, "lote": str, "fecha": str, "vista": str})
    umbral = cfg_fe["dominio"].get("umbral_vista_defecto", 0.02)
    if verbose:
        print(f"Frutos train/val/test: {len(tr)}/{len(va)}/{len(te)} | CV: {n_splits} pliegues x {len(cfg['semillas'])} semillas")
        print(f"Train por categoria: {cuenta.to_dict()}")
    cache, filas, cvs, cms = {}, [], [], {}
    for exp in lista_experimentos(cfg):
        if exp["datos"] not in cache:
            d = df if exp["datos"] == "base" else datos_con_vistas(df, vistas_df, exp["vistas"], umbral)
            cache[exp["datos"]] = tuple(d[d["subset"] == s].reset_index(drop=True) for s in ("train", "val"))
        t_tr, t_va = cache[exp["datos"]]
        t0 = time.perf_counter()
        cv, fila, cm = evaluar(exp, t_tr, t_va, cfg, cfg_pipe, n_splits)
        filas.append(fila); cvs.append(cv); cms[exp["config"]] = cm
        if verbose:
            print(f"  {exp['config']:<28} CV F1 macro={fila['cv_f1_macro_media']:.3f}+-{fila['cv_f1_macro_std']:.3f} "
                  f"| F1 Fabrica={fila['cv_f1_pos_media']:.3f} | VAL={fila['val_f1_macro']:.3f} ({time.perf_counter()-t0:.1f}s)")
    res, cv_all = pd.DataFrame(filas), pd.concat(cvs, ignore_index=True)
    comp = comparar(cv_all, res, cfg)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    huella = huella_particion(fx.cargar_particion(procesados))
    d = registrar(run_id, cfg, cfg_fe, res, cv_all, comp, cms, huella,
                  dict(n_splits=n_splits, n_train=len(tr), n_val=len(va), n_test=len(te),
                       ids_val=va["id_fruto"].tolist(), y_val=va["y"].tolist()))
    if verbose:
        print(f"Registro guardado en {d}\n")
    return res, comp, run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    a = ap.parse_args()
    res, comp, _ = ejecutar(a.procesados)
    print("BASELINE\n" + tabla_baseline_md(res))
    print("VARIANTES DE DATOS\n" + tabla_variantes_md(res, comp))
