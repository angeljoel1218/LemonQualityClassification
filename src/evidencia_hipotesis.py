"""
evidencia_hipotesis.py
----------------------
Calcula, con las features clasicas ya extraidas, las medidas que la tesis pide para las hipotesis que HOY se pueden
acotar. Son SUSTITUTOS (no el modelo profundo): sirven para dejar listo el analisis y fijar la linea base.

    H3 / E4   defectos que pasan sin detectarse: 1 vista ecuatorial (0 grados) frente a las 5 vistas (McNemar pareado)
    H4 / E1   linea base SIN preprocesamiento: F1 de color/madurez y F1 del par Primera-Segunda
    H5 / E3   F1 de Tercera y Fabrica con y sin balanceo (sobre los resultados del baseline)

Todo se calcula con frutos de train+val SIN usar TEST. La referencia de "Fabrica por defecto" sale del panel
(`severidad_defecto_panel >= theta_def_panel`), nunca de las imagenes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import estadistica as est

VISTAS_EQ = ["0", "90", "180", "270"]
VISTAS_TODAS = VISTAS_EQ + ["pedunculo"]


# ------------------------------------------------------------------------------------ H3 / E4
def _ref_fabrica_defecto(frutos: pd.DataFrame, theta_panel: float) -> pd.DataFrame:
    """Frutos de Fabrica POR DEFECTO segun el panel: categoria de referencia Fabrica y severidad >= theta_panel."""
    from src.eda import normalizar_categorias
    cat = frutos["y"] if "y" in frutos else normalizar_categorias(frutos["categoria_ref"])
    return frutos[(cat == "Fabrica") & (frutos["severidad_defecto_panel"] >= theta_panel)][["id_fruto"]]


def e4_defectos_omitidos(vistas: pd.DataFrame, frutos: pd.DataFrame, theta_ratio: float = 0.02,
                         theta_panel: float = 0.5, vista_unica: str = "0", subconjuntos=None) -> dict:
    """Proporcion de frutos de Fabrica POR DEFECTO que el detector NO marca, con 1 vista frente a 5 vistas.

    * Referencia: fruto de categoria Fabrica con `severidad_defecto_panel >= theta_panel` (Fabrica por defecto segun el panel).
    * Detector (regla "basta una vista", Tabla 3.3): marca el fruto si ALGUNA de las vistas usadas tiene
      `defecto_ratio >= theta_ratio` (fraccion de superficie con defecto extraida de la imagen).
    * Se compara `vista_unica` (0 grados, la unica "frontal" que define el protocolo) con las cinco vistas.
    * McNemar pareado sobre los mismos frutos; reduccion = omitidos(1 vista) - omitidos(5 vistas) en puntos porcentuales.
    `frutos` debe traer id_fruto, severidad_defecto_panel y, opcionalmente, subset para filtrar.
    """
    f = frutos.copy()
    if subconjuntos is not None and "subset" in f:
        f = f[f["subset"].isin(subconjuntos)]
    ref = _ref_fabrica_defecto(f, theta_panel)
    v = vistas[vistas["id_fruto"].isin(ref["id_fruto"])]
    v = v[v["vista"].isin(VISTAS_TODAS) & v["seg_ok"].fillna(False).astype(bool)]
    marca = lambda vs: v[v["vista"].isin(vs)].groupby("id_fruto")["defecto_ratio"].max().reindex(ref["id_fruto"]).fillna(0.0) >= theta_ratio
    det1, det5 = marca([vista_unica]), marca(VISTAS_TODAS)
    n = len(ref)
    om1, om5 = int((~det1).sum()), int((~det5).sum())
    mc = est.mcnemar(det5.to_numpy(), det1.to_numpy())          # "acierto" = detectado; A = 5 vistas, B = 1 vista
    lo1, hi1 = est.wilson(om1, n)
    lo5, hi5 = est.wilson(om5, n)
    return dict(n_fabrica_defecto=n, omitidos_1vista=om1, omitidos_5vistas=om5,
                pct_omitidos_1vista=100 * om1 / max(n, 1), pct_omitidos_5vistas=100 * om5 / max(n, 1),
                ic_1vista=(100 * lo1, 100 * hi1), ic_5vistas=(100 * lo5, 100 * hi5),
                reduccion_pp=100 * (om1 - om5) / max(n, 1), mcnemar=mc, theta_ratio=theta_ratio, theta_panel=theta_panel,
                vista_unica=vista_unica, meta_pp=15.0, cumple=bool(100 * (om1 - om5) / max(n, 1) >= 15.0 and mc["p"] < 0.05))


def e4_barrido_umbrales(vistas, frutos, umbrales=(0.005, 0.01, 0.02, 0.03, 0.05, 0.10), **kw) -> pd.DataFrame:
    """E4 para varios umbrales de fraccion de defecto: muestra cuanto depende la conclusion del umbral."""
    filas = []
    for t in umbrales:
        r = e4_defectos_omitidos(vistas, frutos, theta_ratio=t, **kw)
        filas.append(dict(theta_ratio=t, omitidos_1vista_pct=r["pct_omitidos_1vista"], omitidos_5vistas_pct=r["pct_omitidos_5vistas"],
                          reduccion_pp=r["reduccion_pp"], p_mcnemar=r["mcnemar"]["p"]))
    return pd.DataFrame(filas)


def e4_por_vista(vistas, frutos, theta_ratio=0.02, theta_panel=0.5, subconjuntos=None) -> pd.DataFrame:
    """Que vista detecta mas defectos de Fabrica (aporte marginal de cada una)."""
    f = frutos[frutos["subset"].isin(subconjuntos)] if subconjuntos is not None and "subset" in frutos else frutos
    ref = _ref_fabrica_defecto(f, theta_panel)["id_fruto"]
    v = vistas[vistas["id_fruto"].isin(ref) & vistas["seg_ok"].fillna(False).astype(bool)]
    filas = []
    for vs in VISTAS_TODAS:
        det = v[v["vista"] == vs].groupby("id_fruto")["defecto_ratio"].max().reindex(ref).fillna(0) >= theta_ratio
        filas.append(dict(vista=vs, detectados=int(det.sum()), n=len(ref), pct_detectados=100 * det.mean()))
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------ H4 / E1
def h4_linea_base(df: pd.DataFrame, cols, semillas=(0, 1, 2), n_splits: int = 5) -> dict:
    """F1 de color/madurez (clase del panel a partir de las features de COLOR) y F1 del par Primera-Segunda,
    SIN preprocesamiento de brillo/color. Es la linea base contra la que E1 debe sumar +5 y +3 puntos.
    Se calcula con predicciones fuera de muestra (CV) sobre `df` (pasar solo train)."""
    out = {}
    d = df.dropna(subset=["color_panel"]).copy()
    cols_color = [c for c in cols if c in ("L_media", "a_media", "b_media", "L_std", "a_std", "b_std", "L_p10", "L_p90",
                                           "entropia_L", "disp_b_vistas")]
    f1c = []
    for s in semillas:
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          RandomForestClassifier(100, class_weight="balanced", random_state=s, n_jobs=1))
        skf = StratifiedKFold(min(n_splits, d["color_panel"].value_counts().min()), shuffle=True, random_state=s)
        p = cross_val_predict(m, d[cols_color], d["color_panel"], cv=skf)
        f1c.append(f1_score(d["color_panel"], p, average="macro", zero_division=0))
    out["f1_color"] = float(np.mean(f1c))
    out["f1_color_std"] = float(np.std(f1c, ddof=1)) if len(f1c) > 1 else 0.0
    out["n_color"] = int(len(d))
    out["clases_color"] = d["color_panel"].value_counts().to_dict()
    par = df[df["y"].isin(["Primera", "Segunda"])].copy()
    f1p = []
    for s in semillas:
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          RandomForestClassifier(100, class_weight="balanced", random_state=s, n_jobs=1))
        skf = StratifiedKFold(min(n_splits, par["y"].value_counts().min()), shuffle=True, random_state=s)
        p = cross_val_predict(m, par[cols], par["y"], cv=skf)
        f1p.append(f1_score(par["y"], p, average="macro", zero_division=0))
    out["f1_primera_segunda"] = float(np.mean(f1p))
    out["f1_primera_segunda_std"] = float(np.std(f1p, ddof=1)) if len(f1p) > 1 else 0.0
    out["n_par"] = int(len(par))
    return out


# ------------------------------------------------------------------------------------ H5 / E3
def h5_por_clase(res: pd.DataFrame, cmp_: pd.DataFrame | None = None) -> pd.DataFrame:
    """F1 en CV de Tercera y Fabrica en el baseline y en cada variante de balanceo (V2), con el cambio frente al BASE."""
    cols = [c for c in ("cv_f1_Tercera_media", "cv_f1_pos_media") if c in res.columns]
    if not cols:
        return pd.DataFrame()
    r = res[res["modelo"] != "dummy"][["modelo", "variante"] + cols].copy()
    r = r.rename(columns={"cv_f1_Tercera_media": "F1 Tercera", "cv_f1_pos_media": "F1 Fabrica"})
    base = r[r["variante"] == "BASE"].set_index("modelo")
    for c in ("F1 Tercera", "F1 Fabrica"):
        if c in r:
            r[f"Δ {c}"] = r.apply(lambda x: x[c] - base.loc[x["modelo"], c], axis=1)
    return r[r["variante"].isin(["BASE", "V2_sobremuestreo", "V2_submuestreo"])].reset_index(drop=True)


# ------------------------------------------------------------------------------------ control de atajo por sesion
def atajo_sesion(df: pd.DataFrame, cols, semillas=(0, 1, 2), n_splits: int = 5) -> dict:
    """Cuanto se puede "adivinar" sin mirar el fruto.

    1) Clasificador de SESION (lote x fecha): predice la categoria mas frecuente de la sesion (ajustado con train) y se
       evalua en val. Si su F1 macro se acerca al de los modelos, las metricas pueden estar inflando el contexto.
    2) Identificabilidad de la sesion: exactitud en CV de un kNN que predice la SESION a partir de las features (A).
       Cerca de 1 = las features contienen la identidad de la sesion (escala de pixeles, iluminacion).
    Con la categoria repartida de forma balanceada entre sesiones, (1) debe parecerse al dummy (~0,07).
    """
    from sklearn.neighbors import KNeighborsClassifier
    d = df.copy()
    d["sesion"] = d["lote"].astype(str) + "|" + d["fecha"].astype(str)
    tr, va = d[d["subset"] == "train"], d[d["subset"] == "val"]
    mayoritaria = tr.groupby("sesion")["y"].agg(lambda s: s.value_counts().index[0])
    pred = va["sesion"].map(mayoritaria).fillna(tr["y"].value_counts().index[0])
    m = est.metricas_tesis(va["y"].to_numpy(), pred.to_numpy())
    cols = [c for c in cols if d[c].notna().any()]
    accs = []
    for s in semillas:
        pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), KNeighborsClassifier(5))
        skf = StratifiedKFold(min(n_splits, tr["sesion"].value_counts().min()), shuffle=True, random_state=s)
        p = cross_val_predict(pipe, tr[cols], tr["sesion"], cv=skf)
        accs.append(float((p == tr["sesion"].to_numpy()).mean()))
    return dict(f1_macro_por_sesion=m["f1_macro"], kappa_por_sesion=m["kappa"], recall_fabrica_por_sesion=m["recall_pos"],
                exactitud_sesion_desde_features=float(np.mean(accs)), n_sesiones=int(d["sesion"].nunique()),
                pureza_sesion=float(tr.groupby("sesion")["y"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean()))
