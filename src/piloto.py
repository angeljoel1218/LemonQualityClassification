"""
piloto.py
---------
Prueba piloto de validación del protocolo (Plan de tesis v4, sección 4.6.7). Se ejecuta sobre ~100 frutos (columna `piloto` = 1 en
`data/raw/etiquetas.csv`, que NO entran a la muestra definitiva) ANTES de la recolección definitiva y debe confirmar:

    (a) κ de Fleiss >= 0,75 y repetibilidad intra-evaluador >= 0,80 en el panel
    (b) MAE del calibre dentro del umbral de H1 (<= 0,50 mm) y repetibilidad del vernier documentada
    (c) que la captura multi-vista es sostenible en el tiempo disponible por fruto (planificación: 3 min por fruto)
Solo si se cumplen las tres se autoriza escalar; si no, se ajustan la rúbrica, el instrumental o el procedimiento y se repite.

También estima los parámetros de planificación del tamaño de muestra: MAE del calibre y su σ (H1), proporción de pares discordantes ψ
(H2), correlación ρ entre sistema y operario (H4), prevalencia comercial natural (`data/processed/prevalencia_natural.json`, que usa
la evaluación ponderada) y los tiempos de captura y anotación. Además ofrece la calibración/congelamiento de los umbrales de la rúbrica.

Entradas opcionales (lo que falte se reporta como pendiente, nunca se inventa):
    data/raw/panel_evaluaciones.csv        data/raw/piloto_tiempos.csv  (id_fruto, captura_s, vernier_1_mm, vernier_2_mm)
    data/processed/features_frutos.csv     (calibre estimado d_est_mm de los frutos del piloto)

Uso:
    python src/piloto.py evaluar
    python src/piloto.py calibrar --vistas data/processed/features_vistas.csv        # rejilla de (theta_def, p_am) sobre el piloto
    python src/piloto.py congelar --theta-def 2 --p-am 0.30 [--h-am 103]             # congela los umbrales (no se cambian después)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import atributos as at  # noqa: E402
from src import calibracion as cb  # noqa: E402
from src import config as C  # noqa: E402
from src import panel as pn  # noqa: E402
from src import potencia as pw  # noqa: E402
from src import regla_comercial as rc  # noqa: E402
from src.eda import normalizar_categorias  # noqa: E402
from src.ingesta import leer_csv  # noqa: E402

MIN_POR_FRUTO_S = 180.0          # valor de planificación: 3 min por fruto (cinco vistas y medición con vernier)


def frutos_piloto(raw: Path) -> pd.DataFrame:
    et = leer_csv(raw / "etiquetas.csv", numericas=["calibre_mm_ref", "piloto"])
    if "piloto" not in et or not (et["piloto"].fillna(0) > 0).any():
        return et.iloc[0:0]
    return et[et["piloto"].fillna(0) > 0].copy()


def evaluar(raw: Path = ROOT / "data" / "raw", procesados: Path = ROOT / "data" / "processed") -> dict:
    pil = frutos_piloto(raw)
    out = dict(n_frutos_piloto=int(len(pil)), objetivo=C.CFG["piloto_frutos"])
    if pil.empty:
        out["estado"] = "sin frutos de piloto: marque `piloto` = 1 en etiquetas.csv"
        return out
    ids = list(pil["id_fruto"])
    pil["y"] = normalizar_categorias(pil["categoria_ref"])
    # (a) panel
    ruta = raw / "panel_evaluaciones.csv"
    if ruta.exists():
        p = pn.categoria_por_evaluador(pn.cargar_panel(ruta), pil)
        out["a_panel"] = pn.criterio_piloto(p, ids)
    else:
        out["a_panel"] = dict(estado="sin panel_evaluaciones.csv")
    # (b) calibre
    ff = procesados / "features_frutos.csv"
    if ff.exists():
        f = pd.read_csv(ff, dtype={"id_fruto": str})
        m = pil.merge(f[["id_fruto", "d_est_mm"]], on="id_fruto") if "d_est_mm" in f else pil.iloc[0:0]
        if len(m) and m["d_est_mm"].notna().any():
            h1 = cb.evaluar_h1(m["d_est_mm"], m["calibre_mm_ref"], m["y"], B=2000)
            out["b_calibre"] = dict(n=h1["n"], mae=h1["mae"], mae_ic=h1["mae_ic"], rmse=h1["rmse"], sd_abs_error=float(np.abs(m["d_est_mm"] - m["calibre_mm_ref"]).std(ddof=1)),
                                    cumple=bool(h1["mae_ic"][1] <= C.EST["h1"]["mae_max"]))
        else:
            out["b_calibre"] = dict(estado="sin calibre en mm (falta calibracion_sesiones.csv)")
    else:
        out["b_calibre"] = dict(estado="sin features_frutos.csv")
    tiempos = raw / "piloto_tiempos.csv"
    if tiempos.exists():
        t = leer_csv(tiempos, numericas=["captura_s", "vernier_1_mm", "vernier_2_mm"])
        if {"vernier_1_mm", "vernier_2_mm"} <= set(t.columns):
            d = (t["vernier_1_mm"] - t["vernier_2_mm"]).dropna()
            out["b_vernier"] = dict(n=int(len(d)), dif_abs_media_mm=float(d.abs().mean()), sd_dif_mm=float(d.std(ddof=1)) if len(d) > 1 else np.nan)
        if "captura_s" in t:
            med = float(t["captura_s"].median())
            out["c_captura"] = dict(mediana_s=med, p95_s=float(t["captura_s"].quantile(.95)), planificado_s=MIN_POR_FRUTO_S, cumple=bool(med <= MIN_POR_FRUTO_S))
    else:
        out["c_captura"] = dict(estado="sin piloto_tiempos.csv")
    # parámetros de planificación
    prev = pil["y"].value_counts(normalize=True).to_dict()
    out["prevalencia_natural"] = {c: float(prev.get(c, 0.0)) for c in C.CATEGORIAS}
    s = out.get("b_calibre", {}).get("sd_abs_error")
    if s and np.isfinite(s) and out["b_calibre"].get("mae") is not None:
        out["n_h1_con_piloto"] = pw.n_h1(out["b_calibre"]["mae"], s) if out["b_calibre"]["mae"] < C.EST["h1"]["mae_max"] else None
    conds = [out["a_panel"].get("cumple"), out.get("b_calibre", {}).get("cumple"), out.get("c_captura", {}).get("cumple")]
    out["autoriza_escalar"] = bool(all(c is True for c in conds))
    out["pendientes"] = [n for n, c in zip(("a (panel)", "b (calibre)", "c (captura)"), conds) if c is None]
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "piloto.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if out["n_frutos_piloto"] >= 20:
        (procesados / "prevalencia_natural.json").write_text(json.dumps(out["prevalencia_natural"], indent=2), encoding="utf-8")
    return out


def calibrar(vistas_csv: Path, raw: Path = ROOT / "data" / "raw") -> pd.DataFrame:
    """Rejilla de (theta_def, p_am) sobre los frutos del piloto (S y P_am agregados desde las vistas)."""
    pil = frutos_piloto(raw)
    if pil.empty:
        raise SystemExit("No hay frutos de piloto: agregue la columna `piloto` (1 = piloto) en data/raw/etiquetas.csv.")
    v = pd.read_csv(vistas_csv, dtype={"id_fruto": str, "vista": str})
    faltan = [c for c in ("seg_ok", "defecto_ratio", "hue_hist") if c not in v.columns]   # ancho_mm no se necesita para θ_def y p_am
    if faltan:
        raise SystemExit(f"{vistas_csv} no tiene {faltan}: vuelva a ejecutar `python src/extraer_features.py` (versión v4).")
    v = v[v["id_fruto"].isin(pil["id_fruto"])]
    if v.empty:
        raise SystemExit("Ninguno de los frutos de piloto tiene vistas en el archivo de features.")
    t = at.agregar_tabla(v).merge(pil[["id_fruto", "categoria_ref"]], on="id_fruto")
    y = normalizar_categorias(t["categoria_ref"]) == C.POSITIVO
    return rc.calibrar_umbrales_v4(t, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["evaluar", "calibrar", "congelar"])
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--vistas", type=Path)
    ap.add_argument("--theta-def", type=int)
    ap.add_argument("--p-am", type=float)
    ap.add_argument("--h-am", type=float)
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    if a.accion == "evaluar":
        print(json.dumps(evaluar(a.raw, a.procesados), indent=2, ensure_ascii=False, default=str))
    elif a.accion == "calibrar":
        g = calibrar(a.vistas or a.procesados / "features_vistas.csv", a.raw)
        print(g.sort_values("f1", ascending=False).head(10).to_string(index=False))
        print("Mejor:", rc.elegir_umbrales(g))
    else:
        base = rc.UmbralesRegla()
        u = rc.UmbralesRegla(theta_def=a.theta_def if a.theta_def is not None else base.theta_def, p_am=a.p_am if a.p_am is not None else base.p_am,
                             h_am=a.h_am if a.h_am is not None else base.h_am).validar()
        print(json.dumps(rc.congelar_umbrales(u, origen="piloto", forzar=a.forzar), indent=2, default=str))


if __name__ == "__main__":
    main()
