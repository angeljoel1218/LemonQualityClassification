"""
fe_experimentos.py
------------------
Ablaciones de Feature Engineering ("cambia una cosa por vez"):

    REF|dummy        referencia: siempre la clase mayoritaria
    A|modelo         variables originales
    A+B|modelo       + transformaciones (winsor, log1p, banderas, ángulos cíclicos)
    A+C|modelo       + features de dominio
    A+B+C|modelo     + ambas
    A+B+C+S|modelo   + filtro de colinealidad
    A_sin_escala     efecto del escalado (kNN / regresión logística)
    CONTROL_FUGA     (opcional) añade `lote` como entrada: si mejora mucho => atajo de contexto

Protocolo (control experimental):
    * Partición congelada (src/particion.py). Todo se ajusta SOLO con train.
    * Validación cruzada estratificada SOLO sobre train (semillas de configs/fe.yaml) + evaluación en val.
    * TEST se evalúa UNA sola vez, al final, con --evaluar-test (queda registrado y se bloquea repetirlo).
    * Métricas: F1 macro, exactitud balanceada, recall y PR-AUC de la clase positiva (Fabrica).
    * Se registra configuración + métrica + costo (tiempo de ajuste) + latencia (extracción + predicción).

Uso:
    python src/fe_experimentos.py --procesados data/demo/processed
    python src/fe_experimentos.py --control-fuga
    python src/fe_experimentos.py --evaluar-test "A+B+C|rf"
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda import normalizar_categorias  # noqa: E402
from src.extraer_features import cargar_cfg  # noqa: E402
from src.fe_columnas import columnas_para, crear_features_dominio  # noqa: E402
from src.fe_pipeline import construir_pipeline  # noqa: E402
from src.firma import firma_por_subconjunto  # noqa: E402
from src.particion import cargar_particion, huella_particion  # noqa: E402

LOGS = ROOT / "logs"
METRICAS = ["f1_macro", "bal_acc", "recall_pos", "prauc_pos"]


# ----------------------------------------------------------------------------- datos
def cargar_datos(procesados: Path, cfg: dict) -> pd.DataFrame:
    kw = {"dtype": {"id_fruto": str, "lote": str, "fecha": str}}
    f = pd.read_csv(procesados / "features_frutos.csv", **kw)
    fr = pd.read_csv(procesados / "indice_frutos.csv", **kw)
    extra = [c for c in ("categoria_ref", "calibre_mm_ref", "color_panel", "severidad_defecto_panel")
             if c in fr.columns]
    df = (f.merge(fr[["id_fruto"] + extra], on="id_fruto")
           .merge(cargar_particion(procesados), on="id_fruto"))
    df = crear_features_dominio(df, cfg["dominio"])            # sin estado: seguro antes de partir
    df["y"] = normalizar_categorias(df["categoria_ref"])
    return df[df["frac_seg_ok"] >= 0.8].reset_index(drop=True)


def dividir(df: pd.DataFrame):
    return (df[df["subset"] == s].reset_index(drop=True) for s in ("train", "val", "test"))


# ----------------------------------------------------------------------------- métricas
def hacer_scorers(pos: str):
    def f1m(est, X, y):
        return f1_score(y, est.predict(X), average="macro", zero_division=0)

    def bacc(est, X, y):
        return balanced_accuracy_score(y, est.predict(X))

    def rec(est, X, y):
        m = np.asarray(y) == pos
        return float((est.predict(X)[m] == pos).mean()) if m.any() else np.nan

    def pr(est, X, y):
        yb = np.asarray(y) == pos
        if not yb.any() or pos not in list(est.classes_) or not hasattr(est, "predict_proba"):
            return np.nan
        return float(average_precision_score(yb, est.predict_proba(X)[:, list(est.classes_).index(pos)]))

    return {"f1_macro": f1m, "bal_acc": bacc, "recall_pos": rec, "prauc_pos": pr}


# ----------------------------------------------------------------------------- configuraciones
def lista_configs(cfg: dict, control_fuga: bool = False) -> list:
    out = [dict(nombre="REF|dummy", modelo="dummy", conj="A", escalar=True, fuga=None)]
    for m in cfg["modelos"]:
        for c in cfg["conjuntos"]:
            out.append(dict(nombre=f"{c}|{m}", modelo=m, conj=c, escalar=True, fuga=None))
    for m in [x for x in cfg["modelos"] if x in ("knn", "logreg")]:
        out.append(dict(nombre=f"A_sin_escala|{m}", modelo=m, conj="A", escalar=False, fuga=None))
    if control_fuga:
        for m in [x for x in cfg["modelos"] if x in ("logreg", "rf")]:
            out.append(dict(nombre=f"CONTROL_FUGA_lote|{m}", modelo=m, conj="A", escalar=True, fuga="onehot_lote"))
    return out


def _columnas(conf: dict, cfg: dict, df_train: pd.DataFrame):
    cols = columnas_para(set(conf["conj"].split("+")), cfg.get("usar_calidad_como_feature", False))
    return [c for c in cols if df_train[c].notna().any()]       # fuera columnas totalmente vacías


# ----------------------------------------------------------------------------- evaluación
def evaluar_config(conf: dict, tr: pd.DataFrame, va: pd.DataFrame, cfg: dict, n_splits: int):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        fila, cm = _evaluar_config(conf, tr, va, cfg, n_splits)
    fila["avisos_no_convergencia"] = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    return fila, cm


def _evaluar_config(conf: dict, tr: pd.DataFrame, va: pd.DataFrame, cfg: dict, n_splits: int):
    cols = _columnas(conf, cfg, tr)
    entrada = cols + (["lote"] if conf["fuga"] else [])
    Xtr, ytr = tr[entrada], tr["y"].to_numpy()
    sc = hacer_scorers(cfg["positivo"])
    filas = []
    for s in cfg["semillas"]:
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=s)
        pipe = construir_pipeline(cols, conf["conj"], conf["modelo"], s, cfg, conf["escalar"], conf["fuga"])
        r = cross_validate(pipe, Xtr, ytr, cv=skf, scoring=sc, error_score="raise")
        tam = [len(te) for _, te in skf.split(Xtr, ytr)]
        for k in range(n_splits):
            filas.append(dict(semilla=s, pliegue=k, fit_s=r["fit_time"][k],
                              pred_ms=r["score_time"][k] / tam[k] * 1000,
                              **{m: r[f"test_{m}"][k] for m in sc}))
    cv = pd.DataFrame(filas)

    pipe = construir_pipeline(cols, conf["conj"], conf["modelo"], cfg["semillas"][0], cfg, conf["escalar"], conf["fuga"])
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    yva = va["y"].to_numpy()
    val = {m: f(pipe, va[entrada], yva) for m, f in sc.items()}
    etiquetas = sorted(set(ytr) | set(yva))
    cm = confusion_matrix(yva, pipe.predict(va[entrada]), labels=etiquetas)
    n_out = len(pipe[:-1].get_feature_names_out())

    fila = dict(config=conf["nombre"], modelo=conf["modelo"], conjuntos=conf["conj"],
                escalado=conf["escalar"], control_fuga=conf["fuga"] or "", n_features_in=len(cols),
                n_features_out=n_out, n_folds=len(cv))
    for m in sc:
        fila[f"cv_{m}_media"], fila[f"cv_{m}_std"] = cv[m].mean(), cv[m].std()
        fila[f"val_{m}"] = val[m]
    fila.update(fit_s_train=fit_s, cv_fit_s_media=cv["fit_s"].mean(), pred_ms_fruto=cv["pred_ms"].mean(),
                extraccion_ms_fruto=float(tr["t_extraccion_ms"].mean()))
    fila["latencia_total_ms_fruto"] = fila["pred_ms_fruto"] + fila["extraccion_ms_fruto"]
    return fila, dict(labels=etiquetas, matriz=cm.tolist())


def tabla_markdown(res: pd.DataFrame) -> str:
    cab = "| Configuración | #feat | CV F1 macro | CV bal. acc | CV recall Fab | CV PR-AUC Fab | VAL F1 macro | ajuste (s) | latencia (ms/fruto) |\n"
    cab += "|---|---|---|---|---|---|---|---|---|\n"
    ms = lambda r, m: f"{getattr(r, f'cv_{m}_media'):.3f} ± {getattr(r, f'cv_{m}_std'):.3f}"
    filas = [f"| {r.config.replace(chr(124), chr(92) + chr(124))} | {r.n_features_out} | {ms(r, 'f1_macro')} | {ms(r, 'bal_acc')} | {ms(r, 'recall_pos')} | "
             f"{ms(r, 'prauc_pos')} | {r.val_f1_macro:.3f} | {r.cv_fit_s_media:.2f} | {r.latencia_total_ms_fruto:.0f} |"
             for r in res.itertuples()]
    return cab + "\n".join(filas) + "\n"


def registrar(run_id, cfg, res, cms, huella_part, huella_firma, extra=None):
    d = LOGS / "fe_runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(run_id=run_id, python=platform.python_version(), sklearn=sklearn.__version__,
                pandas=pd.__version__, numpy=np.__version__, huella_particion=huella_part,
                huella_firma_datos=huella_firma, config_hash=hashlib.sha256(
                    json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12], **(extra or {}))
    (d / "config.json").write_text(json.dumps({"meta": meta, "cfg": cfg}, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "matrices_confusion_val.json").write_text(json.dumps(cms, indent=2), encoding="utf-8")
    res.to_csv(d / "resultados.csv", index=False)
    (d / "tabla_final.md").write_text(tabla_markdown(res), encoding="utf-8")
    hist = LOGS / "experimentos_fe.csv"
    nuevo = res.assign(run_id=run_id, huella_particion=huella_part, config_hash=meta["config_hash"])
    if hist.exists():
        nuevo = pd.concat([pd.read_csv(hist), nuevo], ignore_index=True)
    nuevo.to_csv(hist, index=False)
    return d


def ejecutar(procesados: Path, cfg: dict | None = None, control_fuga: bool = False, verbose: bool = True):
    cfg = cfg or cargar_cfg()
    df = cargar_datos(procesados, cfg)
    tr, va, te = dividir(df)
    cuenta = tr["y"].value_counts()
    if len(cuenta) < 2 or cuenta.min() < 2:
        raise SystemExit(f"Se necesitan >=2 clases con >=2 frutos en train. Ahora: {cuenta.to_dict()}")
    n_splits = int(min(cfg["cv_folds"], cuenta.min()))
    cols_firma = columnas_para({"A", "C"})
    _, huella_firma = firma_por_subconjunto(df, cols_firma)
    huella_part = huella_particion(df[["id_fruto", "subset"]])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if verbose:
        print(f"Frutos train/val/test: {len(tr)}/{len(va)}/{len(te)} | CV: {n_splits} pliegues x {len(cfg['semillas'])} semillas")
    filas, cms = [], {}
    for conf in lista_configs(cfg, control_fuga):
        t0 = time.perf_counter()
        fila, cm = evaluar_config(conf, tr, va, cfg, n_splits)
        filas.append(fila); cms[conf["nombre"]] = cm
        if verbose:
            print(f"  {conf['nombre']:<28} CV F1={fila['cv_f1_macro_media']:.3f}+-{fila['cv_f1_macro_std']:.3f}  "
                  f"VAL F1={fila['val_f1_macro']:.3f}  ({time.perf_counter()-t0:.1f}s)")
    res = pd.DataFrame(filas)
    d = registrar(run_id, cfg, res, cms, huella_part, huella_firma, dict(n_splits=n_splits))
    if verbose:
        print(f"Registro guardado en {d}")
    return res, run_id, cfg


def evaluar_test(procesados: Path, config: str, cfg: dict | None = None, forzar: bool = False):
    """Evalúa UNA configuración en TEST (ajustada con train+val). Solo se permite una vez."""
    cfg = cfg or cargar_cfg()
    marca = LOGS / "test_evaluado.json"
    if marca.exists() and not forzar:
        raise SystemExit(f"TEST ya fue evaluado ({marca.read_text(encoding='utf-8')[:120]}...). "
                         "Repetirlo invalida la evaluación final. Use --forzar-test solo si lo justifica.")
    df = cargar_datos(procesados, cfg)
    tr, va, te = dividir(df)
    conf = next((c for c in lista_configs(cfg, True) if c["nombre"] == config), None)
    if conf is None:
        raise SystemExit(f"Configuración desconocida: {config}")
    trva = pd.concat([tr, va], ignore_index=True)
    cols = _columnas(conf, cfg, trva)
    entrada = cols + (["lote"] if conf["fuga"] else [])
    pipe = construir_pipeline(cols, conf["conj"], conf["modelo"], cfg["semillas"][0], cfg, conf["escalar"], conf["fuga"])
    pipe.fit(trva[entrada], trva["y"].to_numpy())
    sc = hacer_scorers(cfg["positivo"])
    yte = te["y"].to_numpy()
    out = {m: f(pipe, te[entrada], yte) for m, f in sc.items()}
    etiquetas = sorted(set(trva["y"]) | set(yte))
    cm = confusion_matrix(yte, pipe.predict(te[entrada]), labels=etiquetas)
    LOGS.mkdir(exist_ok=True)
    marca.write_text(json.dumps(dict(config=config, fecha=datetime.now().isoformat(timespec="seconds"),
                                     metricas=out, labels=etiquetas, matriz=cm.tolist()), indent=2), encoding="utf-8")
    return out, etiquetas, cm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--control-fuga", action="store_true")
    ap.add_argument("--evaluar-test", metavar="CONFIG")
    ap.add_argument("--forzar-test", action="store_true")
    a = ap.parse_args()
    if a.evaluar_test:
        met, lab, cm = evaluar_test(a.procesados, a.evaluar_test, forzar=a.forzar_test)
        print("TEST (una sola vez):", {k: round(v, 3) for k, v in met.items()})
        print(pd.DataFrame(cm, index=lab, columns=lab))
    else:
        res, run_id, _ = ejecutar(a.procesados, control_fuga=a.control_fuga)
        print("\n" + tabla_markdown(res))
