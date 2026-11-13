"""
contrastes.py
-------------
Contrastes CONFIRMATORIOS del Plan de tesis v4 (secciones 3.3 y 4.7.4) sobre el conjunto de prueba congelado:

    HG   (sola, α = 0,05)   límite inferior del IC95 % (bootstrap estratificado por categoría, B = 5 000) del F1 macro >= 0,85
    familia {H1, H2, H3, H4} con corrección de Holm escalonada (m = 4):
    H1   calibre:   MAE <= 0,50 mm, RMSE <= 0,70 mm (límites superiores de sus IC) y exactitud adyacente >= 0,98;
                    p(H1) = el mayor de los p de sus criterios
    H2   multi-vista: recall de Fábrica con 5 vistas - con 1 vista (0°) >= 15 pp y límite inferior del IC95 % > 0; McNemar exacto
    H3   híbrido vs directo: límite inferior del IC95 % de la diferencia de F1 macro > 0 (relevancia práctica si >= 0,05)
    H4   no inferioridad: κ_sis - κ_op >= -0,05 (límite inferior del IC95 %, bootstrap pareado por fruto)

Una hipótesis se ACEPTA solo si Holm la rechaza (p <= alfa/(m - posición)) Y cumple los criterios puntuales de su fila.
Los IC bilaterales del 95 % se reportan como descriptivos. Si falta evidencia (p. ej. no hay operarios) la hipótesis queda
'sin evidencia' y NUNCA se completa con valores supuestos.

Entradas (todas alineadas por id_fruto, solo frutos de la PRUEBA):
    y                    categoría de referencia (estándar)                    pred_h5 / pred_h1   híbrido con 5 vistas / con 1 vista (0°)
    pred_directo         clasificador directo (semilla mediana)                d_est_mm, calibre_ref_mm   calibre estimado y medido
    preds_operarios      {operario: categorías}                                (opcional) lote_reservado: ids del lote/fecha reservado
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import calibracion as cb
from src import config as C
from src import estadistica as est

EST = C.EST


def _limpio(d: dict) -> dict:
    """Quita arrays (réplicas) para poder serializar a JSON."""
    return {k: (_limpio(v) if isinstance(v, dict) else v) for k, v in d.items() if k != "reps"}


def contraste_general(y, pred, B=None, semilla=None) -> dict:
    ic = est.ic_estratificado(y, pred, "f1_macro", B, semilla)
    kap = est.ic_estratificado(y, pred, "kappa", B, semilla)
    rec = est.ic_estratificado(y, pred, "recall_fabrica", B, semilla)
    ba = est.ic_estratificado(y, pred, "bal_acc", B, semilla)
    minimo = EST["hg_f1_macro_min"]
    return dict(f1_macro=ic["estimado"], ic_inf=ic["ic_inf"], ic_sup=ic["ic_sup"], umbral=minimo, acepta=bool(ic["ic_inf"] >= minimo),
                p=float((np.sum(ic["reps"] < minimo) + 1) / (len(ic["reps"]) + 1)), kappa=kap["estimado"], kappa_ic=(kap["ic_inf"], kap["ic_sup"]),
                recall_fabrica=rec["estimado"], recall_fabrica_ic=(rec["ic_inf"], rec["ic_sup"]), exactitud_balanceada=ba["estimado"],
                n=ic["n"], B=ic["B"])


def contraste_h1(d_est_mm, calibre_ref_mm, y=None, B=None) -> dict:
    r = cb.evaluar_h1(d_est_mm, calibre_ref_mm, y, B)
    if r.get("n", 0) == 0:
        return dict(n=0, estado="sin evidencia", p=np.nan)
    r = _limpio(r)
    r["p"] = r["p_h1"]
    return r


def contraste_h2(y, pred_h5, pred_h1, ids=None, B=None, semilla=None) -> dict:
    """Recall de Fábrica con 5 vistas menos recall con la vista frontal (0°), sobre los MISMOS frutos de Fábrica de la prueba."""
    y, a, b = np.asarray(y, object), np.asarray(pred_h5, object), np.asarray(pred_h1, object)
    m = y == C.POSITIVO
    n = int(m.sum())
    if n == 0:
        return dict(n_fabrica=0, estado="sin evidencia", p=np.nan)
    d5, d1 = (a[m] == C.POSITIVO), (b[m] == C.POSITIVO)            # "acierto" = Fábrica detectada
    mc = est.mcnemar(d5, d1)
    rng = np.random.default_rng(EST["semilla"] if semilla is None else semilla)
    B = int(B or EST["bootstrap_B"])
    idx = rng.integers(0, n, (B, n))
    reps = d5[idx].mean(1) - d1[idx].mean(1)
    lo, hi = float(np.quantile(reps, .025)), float(np.quantile(reps, .975))
    dif = float(d5.mean() - d1.mean())
    meta = EST["h2"]["incremento_pp_min"] / 100.0
    return dict(n_fabrica=n, recall_5vistas=float(d5.mean()), recall_1vista=float(d1.mean()), diferencia=dif, diferencia_pp=100 * dif,
                ic_inf=lo, ic_sup=hi, discordantes=dict(solo_5=mc["b"], solo_1=mc["c"]), p=float(mc["p_unilateral"]), p_bilateral=float(mc["p"]),
                prueba=mc["prueba"], umbral_pp=EST["h2"]["incremento_pp_min"], criterio_puntual=bool(dif >= meta), criterio_ic=bool(lo > 0))


def contraste_h3(y, pred_h, pred_d, B=None, semilla=None) -> dict:
    d = est.dif_estratificada(y, pred_h, pred_d, "f1_macro", B, semilla)
    f1h = est.ic_estratificado(y, pred_h, "f1_macro", B, semilla)
    f1d = est.ic_estratificado(y, pred_d, "f1_macro", B, semilla)
    rel = EST["h3"]["diferencia_relevante"]
    return dict(f1_hibrido=f1h["estimado"], f1_directo=f1d["estimado"], diferencia=d["estimado"], ic_inf=d["ic_inf"], ic_sup=d["ic_sup"],
                p=d["p_unilateral"], excluye_cero=bool(d["ic_inf"] > 0), relevancia_practica=bool(d["estimado"] >= rel), diferencia_relevante=rel, B=d["B"])


def contraste_h4(y, pred_sis, preds_operarios: dict, B=None, semilla=None) -> dict:
    if not preds_operarios:
        return dict(estado="sin evidencia", n_operarios=0, p=np.nan)
    r = est.dif_kappa_vs_operarios(y, pred_sis, preds_operarios, EST["delta_kappa"], B, semilla)
    r = _limpio(r)
    r.update(n_operarios=len(preds_operarios), p=r["p_no_inferioridad"], criterio_puntual=True)
    return r


def familia_holm(res: dict) -> dict:
    """Aplica Holm a {H1..H4} con los p disponibles y decide la aceptación de cada una. Las que no tienen evidencia quedan fuera
    de la familia (m = nº de hipótesis evaluables) y se declaran 'sin evidencia'."""
    p = {k: v["p"] for k, v in res.items() if k in ("H1", "H2", "H3", "H4") and np.isfinite(v.get("p", np.nan))}
    h = est.holm_escalonado(p) if p else {}
    out = {}
    for k in ("H1", "H2", "H3", "H4"):
        v = res.get(k, {})
        if k not in h:
            out[k] = dict(estado="sin evidencia", acepta=None)
            continue
        puntual = {"H1": bool(v.get("cumple", False)), "H2": bool(v.get("criterio_puntual") and v.get("criterio_ic")),
                   "H3": bool(v.get("excluye_cero")), "H4": bool(v.get("no_inferior", False))}[k]
        out[k] = dict(holm=h[k], criterios_puntuales=puntual, acepta=bool(h[k]["rechaza"] and puntual),
                      estado="se acepta" if (h[k]["rechaza"] and puntual) else "no se acepta")
    return out


def evaluar(y, pred_h5, pred_h1=None, pred_directo=None, d_est_mm=None, calibre_ref_mm=None, preds_operarios=None, B=None) -> dict:
    """Todos los contrastes sobre la prueba. Lo que no se puede calcular por falta de entradas se reporta como 'sin evidencia'."""
    res = dict(n=int(len(y)), B=int(B or EST["bootstrap_B"]))
    res["HG"] = contraste_general(y, pred_h5, B)
    res["H1"] = contraste_h1(d_est_mm, calibre_ref_mm, y, B) if d_est_mm is not None else dict(estado="sin evidencia", p=np.nan)
    res["H2"] = contraste_h2(y, pred_h5, pred_h1, B=B) if pred_h1 is not None else dict(estado="sin evidencia", p=np.nan)
    res["H3"] = contraste_h3(y, pred_h5, pred_directo, B) if pred_directo is not None else dict(estado="sin evidencia", p=np.nan)
    res["H4"] = contraste_h4(y, pred_h5, preds_operarios or {}, B)
    res["decision"] = familia_holm(res)
    return res


def por_lote_reservado(y, pred, ids, ids_reservado, minimo: int | None = None) -> dict:
    """Análisis SECUNDARIO por lote/fecha reservado: restringido a las categorías con >= `minimo` frutos (20), con IC de Wilson;
    sin contraste de hipótesis. No forma parte de la garantía del diseño principal."""
    minimo = int(minimo or EST["umbral_categorias_lote_reservado"])
    y, pred, ids = np.asarray(y, object), np.asarray(pred, object), np.asarray(ids)
    m = np.isin(ids, list(ids_reservado))
    if not m.any():
        return dict(disponible=False, motivo="no hay lote ni fecha reservados")
    filas = {}
    for c in C.CATEGORIAS:
        mc = m & (y == c)
        if mc.sum() >= minimo:
            k = int((pred[mc] == c).sum())
            lo, hi = est.wilson(k, int(mc.sum()))
            filas[c] = dict(n=int(mc.sum()), recall=k / mc.sum(), wilson=(lo, hi))
    return dict(disponible=True, n_frutos=int(m.sum()), categorias=filas, categorias_omitidas=[c for c in C.CATEGORIAS if c not in filas])


def por_categoria(y, pred) -> pd.DataFrame:
    """Precisión, recall y F1 por categoría (para los fallos y la interpretación)."""
    y, pred = np.asarray(y, object), np.asarray(pred, object)
    filas = []
    for c in C.CATEGORIAS:
        tp = int(((y == c) & (pred == c)).sum()); fp = int(((y != c) & (pred == c)).sum()); fn = int(((y == c) & (pred != c)).sum())
        pr, rc_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        filas.append(dict(categoria=c, n=int((y == c).sum()), precision=pr, recall=rc_, f1=2 * pr * rc_ / max(pr + rc_, 1e-12)))
    return pd.DataFrame(filas)
