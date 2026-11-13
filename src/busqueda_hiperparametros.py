"""
busqueda_hiperparametros.py
---------------------------
Busqueda ACOTADA de hiperparametros con PRESUPUESTO DE COMPUTO FIJO por modelo (Tabla 4.2 y seccion 4.6.1 de la tesis).

Diseno (para que el resultado sea valido y comparable con el baseline):
  * MISMOS frutos, MISMA particion congelada, MISMOS pliegues y semillas que `modelo_baseline.py` -> comparaciones
    PAREADAS (prueba t corregida de Nadeau & Bengio con correccion de Holm).
  * La busqueda y la eleccion del mejor candidato usan SOLO train (validacion cruzada estratificada).
    La validacion (VAL) se calcula para cada candidato como control de sobreajuste a la busqueda; TEST no se toca.
  * Cada modelo recibe el MISMO numero de candidatos (muestreo aleatorio sin reemplazo del espacio definido en
    `configs/hiperparametros.yaml`). El nivel de balanceo se busca junto con los hiperparametros, en el orden de la
    Tabla 3.4: ninguno / pesos de clase / sobremuestreo / submuestreo (siempre DENTRO de fit, solo train).
  * Se registran configuracion, metricas (F1 macro, exactitud balanceada, kappa, F1 y recall de Fabrica, F1 de Tercera),
    costo (tiempo de ajuste) y latencia de prediccion.

Salidas:  logs/hiperparametros_runs/<corrida>/  y  logs/hiperparametros_historial.csv

Uso:
    python src/busqueda_hiperparametros.py
    python src/busqueda_hiperparametros.py --presupuesto 10 --modelos knn logreg
"""
from __future__ import annotations

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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import estadistica as est  # noqa: E402
from src import fe_experimentos as fx  # noqa: E402
from src import modelo_baseline as mb  # noqa: E402
from src.extraer_features import cargar_cfg  # noqa: E402
from src.fe_columnas import columnas_para  # noqa: E402
from src.fe_pipeline import construir_pipeline  # noqa: E402
from src.particion import huella_particion  # noqa: E402
from src.remuestreo import ConRemuestreo  # noqa: E402

LOGS = ROOT / "logs"
SOPORTA_PESOS = {"logreg", "rf"}


def cargar_cfg_hp(ruta: Path = ROOT / "configs" / "hiperparametros.yaml") -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------------------ espacio y muestreo
def _muestrear_valor(spec, rng):
    if isinstance(spec, dict):
        if spec.get("tipo") == "log":
            return float(np.exp(rng.uniform(np.log(spec["min"]), np.log(spec["max"]))))
        raise ValueError(f"Especificacion no soportada: {spec}")
    return spec[int(rng.integers(len(spec)))]


def candidatos(modelo: str, cfg: dict, n: int, semilla: int) -> list:
    """Hasta `n` candidatos DISTINTOS (hiperparametros + nivel de balanceo) para un modelo.
    El espacio discreto pequeno se agota sin repetir; el continuo se muestrea en escala logaritmica."""
    rng = np.random.default_rng(semilla + sum(map(ord, modelo)))
    niveles = [b for b in cfg["balanceo"] if b != "pesos" or modelo in SOPORTA_PESOS]
    espacio = cfg["espacio"][modelo]
    vistos, out, intentos = set(), [], 0
    while len(out) < n and intentos < n * 200:
        intentos += 1
        c = {k: _muestrear_valor(v, rng) for k, v in espacio.items()}
        c["balanceo"] = niveles[int(rng.integers(len(niveles)))]
        clave = json.dumps(c, sort_keys=True, default=str)
        if clave not in vistos:
            vistos.add(clave)
            out.append(c)
    return out


def construir(modelo: str, cand: dict, cols, conj: str, semilla: int, cfg_pipe: dict):
    """Pipeline sin fuga con los hiperparametros del candidato. El balanceo por remuestreo envuelve al clasificador."""
    bal = cand["balanceo"]
    remuestreo = bal if bal in ("sobre", "sub") else None
    pipe = construir_pipeline(cols, conj, modelo, semilla, cfg_pipe, True, None, remuestreo)
    clf = pipe.named_steps["clf"]
    interno = clf.estimator if isinstance(clf, ConRemuestreo) else clf
    prefijo = "clf__estimator__" if isinstance(clf, ConRemuestreo) else "clf__"
    params = {k: v for k, v in cand.items() if k != "balanceo"}
    if modelo in SOPORTA_PESOS:
        params["class_weight"] = "balanced" if bal == "pesos" else None
    interno.set_params(**{k: v for k, v in params.items()})        # valida los nombres
    pipe.set_params(**{prefijo + k: v for k, v in params.items()})
    return pipe


# ------------------------------------------------------------------------------------ evaluacion de un candidato
def evaluar_candidato(modelo, cand, tr, va, cols, cfg, cfg_pipe, n_splits, sc):
    Xtr, ytr, yva = tr[cols], tr["y"].to_numpy(), va["y"].to_numpy()
    filas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for s in cfg["semillas"]:
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=s)
            pipe = construir(modelo, cand, cols, cfg["conjunto_features"], s, cfg_pipe)
            r = cross_validate(pipe, Xtr, ytr, cv=skf, scoring=sc, error_score="raise")
            for k in range(n_splits):
                filas.append(dict(semilla=s, pliegue=k, fit_s=r["fit_time"][k],
                                  **{m: r[f"test_{m}"][k] for m in sc}))
        pipe = construir(modelo, cand, cols, cfg["conjunto_features"], cfg["semillas"][0], cfg_pipe)
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = pipe.predict(va[cols])
        pred_ms = (time.perf_counter() - t0) / len(va) * 1000
        val = {m: f(pipe, va[cols], yva) for m, f in sc.items()}
        labels = sorted(set(ytr) | set(yva))
        cm = confusion_matrix(yva, pred, labels=labels)
    cv = pd.DataFrame(filas)
    fila = {}
    for m in sc:
        fila[f"cv_{m}_media"], fila[f"cv_{m}_std"], fila[f"val_{m}"] = cv[m].mean(), cv[m].std(), val[m]
    fila.update(fit_s_train=fit_s, cv_fit_s_media=cv["fit_s"].mean(), pred_ms_fruto=pred_ms)
    return cv, fila, dict(labels=labels, matriz=cm.tolist())


# ------------------------------------------------------------------------------------ comparacion con el baseline
def comparar_con_baseline(cv_mejores: pd.DataFrame, cv_baseline: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Mejor candidato de cada modelo vs. el baseline (hiperparametros por defecto) del MISMO modelo, por pliegue."""
    filas = []
    for modelo, g in cv_mejores.groupby("modelo"):
        a = cv_baseline[cv_baseline.config == f"BASE|{modelo}"].sort_values(["semilla", "pliegue"])
        b = g.sort_values(["semilla", "pliegue"])
        if a.empty or len(a) != len(b):
            continue
        for m in ("f1_macro", "f1_pos", "kappa"):
            d = b[m].to_numpy() - a[m].to_numpy()
            ok = np.isfinite(d)
            t, p = est.t_corregida(d[ok], a["n_train_fold"].mean(), a["n_test_fold"].mean())
            filas.append(dict(modelo=modelo, metrica=m, base_media=float(a[m].mean()), ajustado_media=float(b[m].mean()),
                              dif_media=float(np.mean(d[ok])), pliegues_mejoran_pct=float(100 * np.mean(d[ok] > 0)),
                              t=t, p=p))
    c = pd.DataFrame(filas)
    if c.empty:
        return c
    c["p_holm"] = np.nan
    for m, idx in c.groupby("metrica").groups.items():
        c.loc[idx, "p_holm"] = est.holm(c.loc[idx, "p"])
    alfa = cfg.get("alfa", 0.05)
    c["veredicto"] = np.where(c.p_holm < alfa, np.where(c.dif_media > 0, "mejora", "empeora"), "sin evidencia")
    return c


# ------------------------------------------------------------------------------------ registro
def registrar(run_id, cfg, trials, mejores, cv_mejores, comp, cms, huella, extra):
    d = LOGS / "hiperparametros_runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(run_id=run_id, python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__,
                numpy=np.__version__, huella_particion=huella,
                config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12], **extra)
    (d / "config.json").write_text(json.dumps({"meta": meta, "cfg": cfg}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    trials.to_csv(d / "trials.csv", index=False)
    mejores.to_csv(d / "mejores.csv", index=False)
    cv_mejores.to_csv(d / "cv_pliegues_mejores.csv", index=False)
    comp.to_csv(d / "comparacion_vs_baseline.csv", index=False)
    (d / "matrices_confusion_val.json").write_text(json.dumps(cms, indent=2), encoding="utf-8")
    hist = LOGS / "hiperparametros_historial.csv"
    nuevo = mejores.assign(run_id=run_id, huella_particion=huella, config_hash=meta["config_hash"])
    if hist.exists():
        nuevo = pd.concat([pd.read_csv(hist), nuevo], ignore_index=True)
    nuevo.to_csv(hist, index=False)
    lineas = [f"BUSQUEDA DE HIPERPARAMETROS - corrida {run_id}",
              f"Huella de particion: {huella} | config: {meta['config_hash']} | presupuesto: {cfg['presupuesto_por_modelo']} candidatos por modelo",
              f"Frutos train/val: {extra['n_train']}/{extra['n_val']} | CV: {extra['n_splits']} pliegues x {len(cfg['semillas'])} semillas | metrica de seleccion: {cfg['metrica_seleccion']}",
              "", "== MEJOR CANDIDATO POR MODELO ==",
              mejores[["modelo", "balanceo", "cv_f1_macro_media", "cv_f1_macro_std", "cv_f1_pos_media", "cv_kappa_media", "val_f1_macro", "val_kappa", "params"]].round(3).to_string(index=False),
              "", "== AJUSTADO vs BASELINE (mismo modelo; prueba t corregida + Holm) ==",
              comp.round(3).to_string(index=False) if len(comp) else "(sin baseline comparable para esta particion)"]
    (LOGS / "metrics_hiperparametros.txt").write_text("\n".join(lineas), encoding="utf-8")
    return d


# ------------------------------------------------------------------------------------ orquestacion
def ejecutar(procesados: Path, cfg: dict | None = None, modelos: list | None = None, presupuesto: int | None = None,
             verbose: bool = True):
    cfg = dict(cfg or cargar_cfg_hp())
    if presupuesto:
        cfg["presupuesto_por_modelo"] = int(presupuesto)
    if modelos:
        cfg["modelos"] = list(modelos)
    cfg_fe = cargar_cfg()
    cfg_pipe = {**cfg_fe, "modelo_params": {}}
    df = fx.cargar_datos(procesados, cfg_fe)
    tr, va, te = fx.dividir(df)
    cuenta = tr["y"].value_counts()
    if len(cuenta) < 2 or cuenta.min() < 2:
        raise SystemExit(f"Se necesitan >=2 clases con >=2 frutos en train. Ahora: {cuenta.to_dict()}")
    n_splits = int(min(cfg["cv_folds"], cuenta.min()))
    conj = cfg["conjunto_features"]
    cols = [c for c in columnas_para(set(conj.split("+"))) if tr[c].notna().any()]
    clases = sorted(set(df["y"]))
    sc = mb.scorers(cfg["positivo"], clases)
    if verbose:
        print(f"Frutos train/val/test: {len(tr)}/{len(va)}/{len(te)} | CV {n_splits}x{len(cfg['semillas'])} | "
              f"{len(cols)} variables ({conj}) | presupuesto {cfg['presupuesto_por_modelo']} por modelo")
    trials, cvs, cms = [], [], {}
    for modelo in cfg["modelos"]:
        t_modelo = time.perf_counter()
        for i, cand in enumerate(candidatos(modelo, cfg, cfg["presupuesto_por_modelo"], cfg["semilla_muestreo"])):
            cv, fila, cm = evaluar_candidato(modelo, cand, tr, va, cols, cfg, cfg_pipe, n_splits, sc)
            ident = f"{modelo}#{i:02d}"
            fila.update(candidato=ident, modelo=modelo, iteracion=i, balanceo=cand["balanceo"],
                        params=json.dumps({k: v for k, v in cand.items() if k != "balanceo"}, default=str),
                        **{f"p_{k}": v for k, v in cand.items() if k != "balanceo"})
            trials.append(fila)
            cvs.append(cv.assign(candidato=ident, modelo=modelo))
            cms[ident] = cm
        if verbose:
            t = pd.DataFrame([x for x in trials if x["modelo"] == modelo])
            metrica = cfg["metrica_seleccion"]
            mejor = t[f"cv_{metrica}_media"].max()
            print(f"  {modelo:<7} {len(t)} candidatos  mejor CV {metrica} = "
                  f"{mejor:.3f}  ({time.perf_counter() - t_modelo:.0f}s)")
    trials = pd.DataFrame(trials)
    col = f"cv_{cfg['metrica_seleccion']}_media"
    mejores = (trials.sort_values(col, ascending=False).groupby("modelo", as_index=False).head(1)
               .sort_values(col, ascending=False).reset_index(drop=True))
    cv_all = pd.concat(cvs, ignore_index=True)
    cv_mejores = cv_all[cv_all.candidato.isin(mejores.candidato)].copy()
    # baseline del mismo protocolo (hiperparametros por defecto) para el pareo
    cv_base = _cv_baseline(tr, cfg, cfg_pipe, cols, n_splits, sc, cfg["modelos"])
    comp = comparar_con_baseline(cv_mejores, cv_base, cfg)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    huella = huella_particion(fx.cargar_particion(procesados))
    d = registrar(run_id, cfg, trials, mejores, cv_mejores, comp, {k: cms[k] for k in mejores.candidato}, huella,
                  dict(n_splits=n_splits, n_train=len(tr), n_val=len(va), n_test=len(te)))
    base_tbl = cv_base.groupby("config")[[m for m in sc]].mean().reset_index()
    base_tbl.to_csv(d / "baseline_defecto_cv.csv", index=False)
    cv_base.to_csv(d / "cv_pliegues_baseline_defecto.csv", index=False)
    if verbose:
        print(f"Registro guardado en {d}")
    return dict(trials=trials, mejores=mejores, cv_mejores=cv_mejores, comp=comp, cv_base=cv_base, run_id=run_id,
                cfg=cfg, cols=cols, dir=d)


def _cv_baseline(tr, cfg, cfg_pipe, cols, n_splits, sc, modelos) -> pd.DataFrame:
    """CV del modelo con hiperparametros por DEFECTO del baseline (k = 5, etc.), con exactamente los mismos pliegues."""
    Xtr, ytr, filas = tr[cols], tr["y"].to_numpy(), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for modelo in modelos:
            for s in cfg["semillas"]:
                skf = StratifiedKFold(n_splits, shuffle=True, random_state=s)
                pipe = construir_pipeline(cols, cfg["conjunto_features"], modelo, s, {**cfg_pipe, "modelo_params": {"knn_k": 5}},
                                          True, None, None)
                if modelo in SOPORTA_PESOS:                       # "defecto" = sin ponderar, como en el baseline clasico
                    pipe.set_params(clf__class_weight=None)
                r = cross_validate(pipe, Xtr, ytr, cv=skf, scoring=sc, error_score="raise")
                tam = [len(te) for _, te in skf.split(Xtr, ytr)]
                for k in range(n_splits):
                    filas.append(dict(config=f"BASE|{modelo}", semilla=s, pliegue=k, n_test_fold=tam[k],
                                      n_train_fold=len(Xtr) - tam[k], **{m: r[f"test_{m}"][k] for m in sc}))
    return pd.DataFrame(filas)


def cargar_run(d: Path) -> dict:
    d = Path(d)
    o = {"dir": d, "cfg": json.loads((d / "config.json").read_text(encoding="utf-8"))}
    for k, f in (("trials", "trials.csv"), ("mejores", "mejores.csv"), ("cv_mejores", "cv_pliegues_mejores.csv"),
                 ("comp", "comparacion_vs_baseline.csv"), ("cv_base", "cv_pliegues_baseline_defecto.csv")):
        o[k] = pd.read_csv(d / f) if (d / f).exists() else pd.DataFrame()
    o["cms"] = json.loads((d / "matrices_confusion_val.json").read_text(encoding="utf-8"))
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--presupuesto", type=int, default=None)
    ap.add_argument("--modelos", nargs="*", default=None)
    a = ap.parse_args()
    r = ejecutar(a.procesados, modelos=a.modelos, presupuesto=a.presupuesto)
    print(r["mejores"][["modelo", "balanceo", "cv_f1_macro_media", "cv_f1_macro_std", "val_f1_macro", "params"]].round(3).to_string(index=False))
    print(r["comp"].round(3).to_string(index=False))
