"""
panel.py
--------
Fiabilidad del panel de agrónomos y construcción de la categoría de referencia (Plan de tesis v4, secciones 4.6.2, 4.6.7 y 4.7.4).

Entrada: `data/raw/panel_evaluaciones.csv`, una fila por fruto x evaluador x ronda (plantilla en docs/plantillas/):
    id_fruto, evaluador, ronda, color, severidad_defecto | severidad_nivel, tipo_defecto, [categoria], [rol], fecha_evaluacion
      color               verde | verde_amarillo | amarillo        ('amarillo' = sobremadurez según la rúbrica)
      severidad_nivel     nivel ordinal 0-3 (Tabla 4.2). Si solo hay `severidad_defecto` (escala histórica 0-1) se convierte con
                          `configs/proyecto.yaml: rubrica.severidad_panel_a_nivel`.
      categoria           (opcional) categoría comercial asignada por el evaluador; si falta se DERIVA con la regla jerárquica
                          a partir de su color, su severidad y el calibre medido (calibre_mm_ref de etiquetas.csv).
      rol                 (opcional) 'panel' | 'adjudicador' (cuarto evaluador que dirime si no hay mayoría)
      ronda               1 = evaluación principal; 2 = repetición ciega para la repetibilidad intra-evaluador (>= 1 semana después)

Salidas (funciones):
    acuerdo_panel        κ de Fleiss (fruto = unidad), AC1 de Gwet, acuerdo por categoría y κ de Cohen por pares
    repetibilidad        κ de Cohen entre la ronda 1 y la 2 de cada evaluador (submuestra de 50 frutos)
    adjudicar            categoría de referencia: mayoría simple; sin mayoría decide el cuarto evaluador
    criterio_piloto      ¿κ de Fleiss >= 0,75 y repetibilidad >= 0,80? (elegibilidad metodológica, no hipótesis del sistema)
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src import estadistica as est
from src import regla_comercial as rc
from src.ingesta import leer_csv

RUTA_PANEL = C.ROOT / "data" / "raw" / "panel_evaluaciones.csv"


def cargar_panel(ruta: Path = RUTA_PANEL) -> pd.DataFrame:
    df = leer_csv(ruta, numericas=["severidad_defecto", "severidad_nivel", "ronda"])
    falt = [c for c in ("id_fruto", "evaluador") if c not in df.columns]
    if falt:
        raise SystemExit(f"panel_evaluaciones.csv: faltan columnas {falt}")
    if "ronda" not in df:
        df["ronda"] = 1
    df["ronda"] = df["ronda"].fillna(1).astype(int)
    if "rol" not in df:
        df["rol"] = "panel"
    return df


def nivel_severidad(df: pd.DataFrame) -> pd.Series:
    if "severidad_nivel" in df and df["severidad_nivel"].notna().any():
        return df["severidad_nivel"].astype(float)
    if "severidad_defecto" in df:
        return df["severidad_defecto"].map(rc.nivel_desde_panel)
    return pd.Series(np.nan, index=df.index)


def categoria_por_evaluador(panel: pd.DataFrame, etiquetas: pd.DataFrame, u: rc.UmbralesRegla | None = None) -> pd.DataFrame:
    """Añade `cat_eval`: la categoría que asigna cada evaluador (la declarada o la derivada con la regla y el calibre medido)."""
    u = u or rc.cargar_umbrales()
    panel = panel.copy()
    if "rol" not in panel:
        panel["rol"] = "panel"
    if "ronda" not in panel:
        panel["ronda"] = 1
    p = panel.merge(etiquetas[["id_fruto", "calibre_mm_ref"]], on="id_fruto", how="left")
    p["nivel"] = nivel_severidad(p)
    p["cat_eval"] = pd.Series([None] * len(p), index=p.index, dtype=object)
    if "categoria" in p and p["categoria"].notna().any():
        from src.eda import normalizar_categorias
        ok = p["categoria"].notna()
        p.loc[ok, "cat_eval"] = normalizar_categorias(p.loc[ok, "categoria"]).astype(object)
    falta = p["cat_eval"].isna()
    if falta.any():
        amarillo = p["color"].astype(str).str.lower().eq("amarillo") if "color" in p else pd.Series(False, index=p.index)
        sev = p["nivel"].fillna(0)
        cat = rc.categoria_por_calibre_vec(p["calibre_mm_ref"].to_numpy(float))
        fab = (sev >= u.theta_def) | amarillo
        cat[fab.to_numpy()] = "Fabrica"
        p.loc[falta, "cat_eval"] = pd.Series(cat, index=p.index, dtype=object)[falta]
    return p


def matriz_calificaciones(p: pd.DataFrame, ronda: int = 1, evaluadores=None, solo_completos: bool = True):
    """(ids, calif N x R, evaluadores): los R evaluadores del panel que calificaron TODOS los frutos de la ronda."""
    q = p[(p["ronda"] == ronda) & (p["rol"].fillna("panel") != "adjudicador")]
    w = q.pivot_table(index="id_fruto", columns="evaluador", values="cat_eval", aggfunc="first")
    if evaluadores is None:
        cobertura = w.notna().mean().sort_values(ascending=False)
        evaluadores = list(cobertura[cobertura >= 0.8].index) or list(cobertura.index[:3])
    w = w[evaluadores]
    if solo_completos:
        w = w.dropna()
    return w.index.to_numpy(), w.to_numpy(object), list(evaluadores)


def acuerdo_panel(p: pd.DataFrame, ids=None, ronda: int = 1) -> dict:
    """κ de Fleiss y AC1 con el fruto como unidad de análisis, R evaluadores independientes y cinco categorías nominales."""
    idx, calif, evs = matriz_calificaciones(p, ronda)
    if ids is not None:
        m = np.isin(idx, list(ids))
        idx, calif = idx[m], calif[m]
    if len(idx) < 2 or calif.shape[1] < 2:
        return dict(n_frutos=int(len(idx)), n_evaluadores=int(calif.shape[1]) if calif.ndim == 2 else 0, kappa_fleiss=np.nan, ac1=np.nan)
    pares = {f"{a} vs {b}": float(est.kappa_cohen(calif[:, i], calif[:, j]))
             for (i, a), (j, b) in itertools.combinations(list(enumerate(evs)), 2)}
    acc_bruto = float(np.mean([len(set(f)) == 1 for f in calif]))
    return dict(n_frutos=int(len(idx)), n_evaluadores=int(calif.shape[1]), evaluadores=evs,
                kappa_fleiss=est.kappa_fleiss(calif), ac1=est.ac1_gwet(calif), acuerdo_unanime=acc_bruto,
                por_categoria=est.acuerdo_por_categoria(calif), kappa_pares=pares)


def repetibilidad(p: pd.DataFrame) -> dict:
    """κ de Cohen entre la ronda 1 y la 2 de cada evaluador, sobre los frutos repetidos (submuestra de 50)."""
    r1, r2 = p[p["ronda"] == 1], p[p["ronda"] == 2]
    out = {}
    for e in sorted(set(r2["evaluador"])):
        a = r1[r1["evaluador"] == e].set_index("id_fruto")["cat_eval"]
        b = r2[r2["evaluador"] == e].set_index("id_fruto")["cat_eval"]
        comun = a.index.intersection(b.index)
        if len(comun) >= 2:
            out[e] = dict(n=int(len(comun)), kappa=float(est.kappa_cohen(a[comun], b[comun])),
                          acuerdo=float((a[comun] == b[comun]).mean()))
    ks = [v["kappa"] for v in out.values() if np.isfinite(v["kappa"])]
    return dict(por_evaluador=out, kappa_medio=float(np.mean(ks)) if ks else np.nan, kappa_min=float(np.min(ks)) if ks else np.nan)


def adjudicar(p: pd.DataFrame) -> pd.DataFrame:
    """Categoría de referencia por fruto con la regla del plan: mayoría simple de los tres evaluadores; si no hay mayoría, decide
    el cuarto evaluador (rol 'adjudicador'). Devuelve id_fruto, categoria_panel, via (unanime | mayoria | adjudicador | sin_decision)."""
    q = p[p["ronda"] == 1]
    filas = []
    for fid, g in q.groupby("id_fruto"):
        votos = g[g["rol"].fillna("panel") != "adjudicador"]["cat_eval"].dropna().tolist()
        adj = g[g["rol"].fillna("panel") == "adjudicador"]["cat_eval"].dropna().tolist()
        cuenta = pd.Series(votos).value_counts()
        if len(cuenta) == 0:
            filas.append((fid, np.nan, "sin_decision"))
        elif len(cuenta) == 1:
            filas.append((fid, cuenta.index[0], "unanime"))
        elif cuenta.iloc[0] > len(votos) / 2:
            filas.append((fid, cuenta.index[0], "mayoria"))
        elif adj:
            filas.append((fid, adj[0], "adjudicador"))
        else:
            filas.append((fid, np.nan, "sin_decision"))
    return pd.DataFrame(filas, columns=["id_fruto", "categoria_panel", "via"])


def criterio_piloto(p: pd.DataFrame, ids_piloto=None) -> dict:
    """κ de Fleiss >= 0,75 y repetibilidad >= 0,80 sobre el piloto (condición (a) de la sección 4.6.7)."""
    a = acuerdo_panel(p, ids_piloto)
    r = repetibilidad(p if ids_piloto is None else p[p["id_fruto"].isin(list(ids_piloto))])
    pm = C.EST["panel"]
    cumple_k = bool(np.isfinite(a["kappa_fleiss"]) and a["kappa_fleiss"] >= pm["kappa_fleiss_min"])
    cumple_r = bool(np.isfinite(r["kappa_medio"]) and r["kappa_medio"] >= pm["repetibilidad_min"])
    return dict(acuerdo=a, repetibilidad=r, cumple_kappa=cumple_k, cumple_repetibilidad=cumple_r, cumple=cumple_k and cumple_r,
                accion="" if (cumple_k and cumple_r) else "Reajustar la rúbrica y reentrenar al panel antes de autorizar la recolección masiva.")
