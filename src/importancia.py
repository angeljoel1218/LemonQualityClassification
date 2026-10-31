"""
importancia.py
--------------
IMPORTANCIA DE VARIABLES, con cuatro métodos que fallan de forma distinta (si coinciden, la conclusión es robusta):

  1. Permutación fuera de muestra (OOF): en cada pliegue de la CV (solo train) se ajusta el pipeline con el pliegue de
     entrenamiento y se mide cuánto cae el F1 macro al barajar una variable en el pliegue retenido. Se promedia sobre
     pliegues y semillas -> media ± desviación y % de pliegues en que la variable ayuda.
  2. Permutación AGRUPADA por dimensión (Calibre, Forma, Color, Defecto, Textura): se barajan juntas todas las variables de
     un grupo. Evita subestimar variables redundantes (p. ej. diámetro en px y en mm: barajar una sola no daña porque la
     otra la sustituye). También por clase: ¿el calibre sostiene Extra..Tercera y el defecto/color sostienen Fábrica?
  3. Impureza de Random Forest (rápida, pero sesgada hacia variables continuas y correlacionadas; solo comparación).
  4. Información mutua con la categoría (no depende del modelo).

Control de ruido: se añade una columna aleatoria `ruido_control`. Toda variable que no supere claramente su importancia
no se distingue del azar.

Se calcula SOLO con train. El conjunto de prueba (test) no se usa.

Uso:
    python src/importancia.py
    python src/importancia.py --procesados data/demo/processed
"""
import argparse
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import fe_experimentos as fx, seleccion  # noqa: E402
from src.extraer_features import cargar_cfg  # noqa: E402
from src.fe_columnas import columnas_para, dimension_de  # noqa: E402
from src.fe_pipeline import construir_pipeline  # noqa: E402
from src.particion import cargar_particion, huella_particion  # noqa: E402

LOGS = ROOT / "logs"
RUIDO = "ruido_control"
DEFAULT = dict(modelos=["rf", "logreg"], conjunto="A+C", n_repeats=5, top_k=10, ruido_control=True)


def cfg_importancia(cfg: dict) -> dict:
    return {**DEFAULT, **cfg.get("importancia", {})}


# ------------------------------------------------------------------------------------ utilidades
def fuente(nombre: str) -> str:
    """Variable original a partir del nombre de una columna transformada (p. ej. 'hue_deg_sin' -> 'hue_deg')."""
    n = nombre.replace("missingindicator_", "")
    for suf in ("_atipico", "_sin", "_cos"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def columnas_importancia(tr: pd.DataFrame, conjunto: str, ruido: bool):
    letras = set(conjunto.split("+"))
    cols = [c for c in columnas_para(letras) if tr[c].notna().any()]
    return cols + ([RUIDO] if ruido else [])


def con_ruido(df: pd.DataFrame, semilla: int = 12345) -> pd.DataFrame:
    d = df.copy()
    d[RUIDO] = np.random.default_rng(semilla).normal(size=len(d))
    return d


# ------------------------------------------------------------------------------------ 1. permutación OOF
def permutacion_oof(tr, cols, conjunto, modelo, cfg, semillas, n_splits, n_repeats=5) -> pd.DataFrame:
    X, y = tr[cols], tr["y"].to_numpy()
    filas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in semillas:
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=s)
            for k, (i_tr, i_te) in enumerate(skf.split(X, y)):
                pipe = construir_pipeline(cols, conjunto, modelo, s, cfg).fit(X.iloc[i_tr], y[i_tr])
                r = permutation_importance(pipe, X.iloc[i_te], y[i_te], scoring="f1_macro",
                                           n_repeats=n_repeats, random_state=s * 100 + k)
                filas += [(s, k, c, m) for c, m in zip(cols, r.importances_mean)]
    return pd.DataFrame(filas, columns=["semilla", "pliegue", "variable", "caida_f1"])


def resumen_permutacion(largo: pd.DataFrame) -> pd.DataFrame:
    g = largo.groupby("variable")["caida_f1"]
    r = pd.DataFrame({"media": g.mean(), "std": g.std(), "pct_pliegues_ayuda": g.apply(lambda x: 100 * (x > 0).mean())})
    r["dimension"] = [dimension_de(v) for v in r.index]
    if RUIDO in r.index:
        umbral = r.loc[RUIDO, "media"] + 2 * (0 if np.isnan(r.loc[RUIDO, "std"]) else r.loc[RUIDO, "std"])
        # criterio doble: supera al ruido Y ayuda en la mayoria de los pliegues (con RF el ruido casi no varia y,
        # solo con el umbral, casi cualquier variable "ganaria")
        r["supera_ruido"] = (r["media"] > umbral) & (r["pct_pliegues_ayuda"] >= 60) & (r.index != RUIDO)
        r["umbral_ruido"] = umbral
    return r.sort_values("media", ascending=False)


def estabilidad_topk(largo: pd.DataFrame, k: int = 10) -> pd.Series:
    """% de (semilla, pliegue) en que la variable está entre las k más importantes (con aporte > 0)."""
    t = largo.copy()
    t["rango"] = t.groupby(["semilla", "pliegue"])["caida_f1"].rank(ascending=False, method="first")
    t["top"] = (t["rango"] <= k) & (t["caida_f1"] > 0)
    return (100 * t.groupby("variable")["top"].mean()).rename(f"pct_top{k}")


# ------------------------------------------------------------------------------------ 2. permutación agrupada
def permutacion_dimensiones_oof(tr, cols, conjunto, modelo, cfg, semillas, n_splits, n_repeats=5) -> pd.DataFrame:
    X, y = tr[cols], tr["y"].to_numpy()
    clases = sorted(set(y))
    grupos = {}
    for c in cols:
        grupos.setdefault(dimension_de(c), []).append(c)
    filas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in semillas:
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=s)
            for k, (i_tr, i_te) in enumerate(skf.split(X, y)):
                pipe = construir_pipeline(cols, conjunto, modelo, s, cfg).fit(X.iloc[i_tr], y[i_tr])
                Xte, yte = X.iloc[i_te], y[i_te]
                pred = pipe.predict(Xte)
                base_m = f1_score(yte, pred, average="macro", zero_division=0)
                base_c = f1_score(yte, pred, labels=clases, average=None, zero_division=0)
                presente = np.array([(yte == c).any() for c in clases])
                rng = np.random.default_rng(s * 100 + k)
                for dim, cs in grupos.items():
                    m_l, c_l = [], []
                    for _ in range(n_repeats):
                        Xp = Xte.copy()
                        Xp[cs] = Xte[cs].to_numpy()[rng.permutation(len(Xte))]       # se barajan JUNTAS
                        pp = pipe.predict(Xp)
                        m_l.append(f1_score(yte, pp, average="macro", zero_division=0))
                        c_l.append(f1_score(yte, pp, labels=clases, average=None, zero_division=0))
                    filas.append((s, k, dim, "macro", base_m - np.mean(m_l)))
                    dc = base_c - np.mean(c_l, axis=0)
                    for c, d, ok in zip(clases, dc, presente):
                        filas.append((s, k, dim, c, d if ok else np.nan))
    return pd.DataFrame(filas, columns=["semilla", "pliegue", "dimension", "clase", "caida_f1"])


def tabla_dimensiones(dim_largo: pd.DataFrame, perm_resumen: pd.DataFrame) -> pd.DataFrame:
    """Caída media de F1 por dimensión x clase, y comparación con la suma de las caídas individuales (redundancia)."""
    t = dim_largo.groupby(["dimension", "clase"])["caida_f1"].mean().unstack("clase")
    suma = perm_resumen.groupby("dimension")["media"].sum().clip(lower=0)
    t["suma_individual"] = suma.reindex(t.index)
    t["redundancia"] = np.where(t["suma_individual"] > 1e-9, t["macro"] / t["suma_individual"], np.nan)
    return t.sort_values("macro", ascending=False)


# ------------------------------------------------------------------------------------ 3 y 4. impureza y MI
def importancia_impureza(tr, cols, conjunto, cfg, semillas) -> pd.Series:
    acum = pd.Series(0.0, index=cols)
    for s in semillas:
        pipe = construir_pipeline(cols, conjunto, "rf", s, cfg).fit(tr[cols], tr["y"].to_numpy())
        imp = pd.Series(pipe.named_steps["clf"].feature_importances_,
                        index=[fuente(n) for n in pipe[:-1].get_feature_names_out()])
        acum = acum.add(imp.groupby(level=0).sum(), fill_value=0.0)
    return (acum / len(semillas)).reindex(cols).fillna(0.0).rename("impureza").sort_values(ascending=False)


def consenso(perm: dict, impureza: pd.Series, mi: pd.Series):
    t = pd.DataFrame({f"perm_{m}": r["media"] for m, r in perm.items()})
    t["impureza_rf"], t["info_mutua"] = impureza, mi
    rangos = t.rank(ascending=False, method="average")
    t["rango_medio"] = rangos.mean(axis=1)
    t["dimension"] = [dimension_de(v) for v in t.index]
    return t.sort_values("rango_medio"), rangos.corr(method="spearman")


# ------------------------------------------------------------------------------------ principal
def ejecutar(procesados: Path, cfg: dict | None = None, verbose: bool = True) -> dict:
    cfg = cfg or cargar_cfg()
    ic = cfg_importancia(cfg)
    df = fx.cargar_datos(procesados, cfg)
    tr = next(iter(fx.dividir(df)))
    cuenta = tr["y"].value_counts()
    if len(cuenta) < 2 or cuenta.min() < 2:
        raise SystemExit(f"Se necesitan >=2 clases con >=2 frutos en train. Ahora: {cuenta.to_dict()}")
    n_splits = int(min(cfg["cv_folds"], cuenta.min()))
    if ic["ruido_control"]:
        tr = con_ruido(tr)
    cols = columnas_importancia(tr, ic["conjunto"], ic["ruido_control"])
    if verbose:
        print(f"Train: {len(tr)} frutos | {len(cols)} variables (incl. control de ruido: {ic['ruido_control']}) | "
              f"CV {n_splits} pliegues x {len(cfg['semillas'])} semillas | conjunto {ic['conjunto']}")
    perm, largos, dims = {}, {}, {}
    for m in ic["modelos"]:
        largos[m] = permutacion_oof(tr, cols, ic["conjunto"], m, cfg, cfg["semillas"], n_splits, ic["n_repeats"])
        perm[m] = resumen_permutacion(largos[m])
        perm[m][f"pct_top{ic['top_k']}"] = estabilidad_topk(largos[m], ic["top_k"])
        dims[m] = permutacion_dimensiones_oof(tr, cols, ic["conjunto"], m, cfg, cfg["semillas"], n_splits, ic["n_repeats"])
        if verbose:
            top = perm[m].head(3)
            print(f"  {m}: top3 = " + ", ".join(f"{i} ({r.media:.3f})" for i, r in top.iterrows()))
    imp = importancia_impureza(tr, cols, ic["conjunto"], cfg, cfg["semillas"])
    mi = seleccion.ranking_mi(tr[cols], tr["y"]).rename("info_mutua")
    cons, acuerdo = consenso(perm, imp, mi)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dict(run_id=run_id, cfg=ic, cols=cols, perm=perm, largos=largos, dims=dims, impureza=imp, mi=mi,
               consenso=cons, acuerdo=acuerdo, n_splits=n_splits, n_train=len(tr),
               dim_tabla={m: tabla_dimensiones(dims[m], perm[m]) for m in perm})
    out["procesados"] = procesados
    out["dir"] = registrar(out, cfg)
    if verbose:
        print(f"Registro guardado en {out['dir']}")
    return out


def registrar(o: dict, cfg: dict) -> Path:
    d = LOGS / "importancia_runs" / o["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    for m in o["perm"]:
        o["perm"][m].to_csv(d / f"perm_{m}.csv")
        o["largos"][m].to_csv(d / f"perm_{m}_pliegues.csv", index=False)
        o["dim_tabla"][m].to_csv(d / f"dimensiones_{m}.csv")
    o["impureza"].to_csv(d / "impureza_rf.csv"); o["mi"].to_csv(d / "info_mutua.csv")
    o["consenso"].to_csv(d / "consenso.csv"); o["acuerdo"].to_csv(d / "acuerdo_metodos.csv")
    meta = dict(run_id=o["run_id"], python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__,
                huella_particion=huella_particion(cargar_particion(o.get("procesados", ROOT / "data" / "processed"))),
                config_hash=hashlib.sha256(json.dumps(o["cfg"], sort_keys=True).encode()).hexdigest()[:12],
                n_splits=o["n_splits"], n_train=o["n_train"])
    (d / "config.json").write_text(json.dumps({"meta": meta, "cfg": o["cfg"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


# ------------------------------------------------------------------------------------ figuras
def fig_permutacion(res: pd.DataFrame, titulo: str, top: int = 15):
    import matplotlib.pyplot as plt
    col = {"Calibre": "#2e7d32", "Forma": "#7cb342", "Color": "#fbc02d", "Defecto": "#7a1f2b",
           "Textura": "#888888", "Control (ruido)": "#000000", "Otra": "#bbbbbb"}
    r = res.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(r) + 1.2))
    ax.barh(r.index, r["media"], xerr=r["std"], color=[col.get(d, "#999") for d in r["dimension"]], capsize=2)
    if "umbral_ruido" in res:
        ax.axvline(res["umbral_ruido"].iloc[0], color="k", ls="--", lw=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for d, c in col.items() if d in set(r["dimension"])]
    ax.legend(handles + ([plt.Line2D([0], [0], color="k", ls="--")] if "umbral_ruido" in res else []),
              [d for d in col if d in set(r["dimension"])] + (["umbral de ruido"] if "umbral_ruido" in res else []),
              fontsize=8, loc="lower right")
    ax.set_xlabel("caída de F1 macro al permutar (fuera de muestra)"); ax.set_title(titulo, fontsize=10); ax.grid(alpha=.25)
    fig.tight_layout()
    return fig


def fig_dimensiones(tabla: pd.DataFrame, titulo: str):
    import matplotlib.pyplot as plt
    clases = [c for c in tabla.columns if c not in ("macro", "suma_individual", "redundancia")]
    M = tabla[["macro"] + clases].to_numpy(float)
    fig, ax = plt.subplots(figsize=(1.0 * (len(clases) + 1) + 2.2, 0.55 * len(tabla) + 1.4))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=max(np.nanmax(M), 1e-6), aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(["macro"] + clases, rotation=30, ha="right")
    ax.set_yticks(range(len(tabla))); ax.set_yticklabels(tabla.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > np.nanmax(M) / 2 else "black")
    ax.set_title(titulo, fontsize=10); ax.grid(False); fig.colorbar(im, ax=ax, shrink=.8, label="caída de F1")
    fig.tight_layout()
    return fig


def cargar_run(d: Path) -> dict:
    """Relee una corrida guardada (la usa el informe)."""
    meta = json.loads((d / "config.json").read_text(encoding="utf-8"))
    o = dict(meta=meta["meta"], cfg=meta["cfg"], dir=d, perm={}, dim_tabla={})
    for f in sorted(d.glob("perm_*.csv")):
        if f.stem.endswith("_pliegues"):
            continue
        o["perm"][f.stem.replace("perm_", "")] = pd.read_csv(f, index_col=0)
    for f in d.glob("dimensiones_*.csv"):
        o["dim_tabla"][f.stem.replace("dimensiones_", "")] = pd.read_csv(f, index_col=0)
    orden = [m for m in o["cfg"].get("modelos", []) if m in o["perm"]]       # el primero de la configuracion es el principal
    o["perm"] = {m: o["perm"][m] for m in orden + [x for x in o["perm"] if x not in orden]}
    o["dim_tabla"] = {m: o["dim_tabla"][m] for m in o["perm"] if m in o["dim_tabla"]}
    o["consenso"] = pd.read_csv(d / "consenso.csv", index_col=0)
    o["acuerdo"] = pd.read_csv(d / "acuerdo_metodos.csv", index_col=0)
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    a = ap.parse_args()
    r = ejecutar(a.procesados)
    for m, p in r["perm"].items():
        print(f"\n== Permutación OOF ({m}) ==")
        print(p.head(10)[["media", "std", "pct_pliegues_ayuda", "dimension"]].round(3).to_string())
    print("\n== Por dimensión (rf) ==\n" + r["dim_tabla"][list(r["dim_tabla"])[0]].round(3).to_string())
