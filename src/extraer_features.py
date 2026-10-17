"""
extraer_features.py
-------------------
Convierte las imágenes en una tabla de features:
    data/processed/features_vistas.csv   una fila por imagen
    data/processed/features_frutos.csv   una fila por fruto (agregación de sus 5 vistas)

La agregación sigue la regla del flujo propuesto:
    calibre -> mediana de las 4 vistas ecuatoriales      color -> media (superficie sana)
    defecto -> "basta una vista" (máximo sobre las 5 vistas, incluido el pedúnculo)

Esta extracción NO aprende nada de los datos (no hay medias/desviaciones calculadas con el dataset),
por eso puede hacerse antes de partir sin fuga. Las transformaciones con estado (escalado,
winsorización, etc.) se ajustan después, solo con train (src/fe_pipeline.py).

Escala px->mm: se lee de <raw>/calibracion_sesiones.csv (columnas lote, fecha, mm_por_px).
Sin ese archivo las features en mm quedan vacías (NaN) y se avisa.

Uso:
    python src/extraer_features.py
    python src/extraer_features.py --raw data/demo/raw --procesados data/demo/processed --jobs 4
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # permite `python src/extraer_features.py`
    sys.path.insert(0, str(ROOT))

from src.features_imagen import PARAMS, features_vista  # noqa: E402
from src.ingesta import leer_etiquetas

EQ = {"0", "90", "180", "270"}


def cargar_cfg(ruta: Path = ROOT / "configs" / "fe.yaml") -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_calibracion(raw: Path) -> dict:
    ruta = raw / "calibracion_sesiones.csv"
    if not ruta.exists():
        print(f"[AVISO] No existe {ruta}: las features en mm quedarán vacías (solo px).")
        return {}
    c = leer_etiquetas(ruta, numericas=["mm_por_px"])
    return {(str(r.lote), str(r.fecha)): r.mm_por_px for r in c.itertuples()}


def _procesar(fila: dict, mm_por_px, params: dict) -> dict:
    ruta = Path(fila["ruta"])
    ruta = ruta if ruta.is_absolute() else ROOT / ruta
    img = cv2.imread(str(ruta))
    base = {k: fila[k] for k in ("id_fruto", "lote", "fecha", "vista")}
    if img is None:
        return {**base, "seg_ok": False, "legible": False}
    return {**base, "legible": True, **features_vista(img, mm_por_px=mm_por_px, p=params)}


def extraer_vistas(df_img: pd.DataFrame, calib: dict, params: dict, n_jobs: int = 1) -> pd.DataFrame:
    filas = df_img.to_dict("records")
    mm = [calib.get((str(f["lote"]), str(f["fecha"]))) for f in filas]
    if n_jobs == 1:
        res = [_procesar(f, m, params) for f, m in zip(filas, mm)]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(_procesar)(f, m, params) for f, m in zip(filas, mm))
    return pd.DataFrame(res)


# --------------------------------------------------------------------------- agregación por fruto
def _cv(x):
    x = pd.Series(x).dropna()
    return float(x.std(ddof=0) / x.mean()) if len(x) > 1 and x.mean() != 0 else np.nan


def agregar_fruto(g: pd.DataFrame, umbral_vista_defecto: float) -> pd.Series:
    ok = g[g["seg_ok"].fillna(False).astype(bool)]
    eq = ok[ok["vista"].isin(EQ)]
    ped = ok[ok["vista"] == "pedunculo"]
    r = {}
    mm = lambda s: s.median() if s.notna().any() else np.nan
    # --- geometría (calibre = mediana de las vistas ecuatoriales)
    r["diam_ecu_px"] = eq["eje_menor_px"].median()
    r["diam_ecu_mm"] = mm(eq["eje_menor_mm"]) if "eje_menor_mm" in eq else np.nan
    r["long_polar_px"] = eq["eje_mayor_px"].median()
    r["long_polar_mm"] = mm(eq["eje_mayor_mm"]) if "eje_mayor_mm" in eq else np.nan
    r["aspecto"] = eq["aspecto"].median()
    r["area_px"] = eq["area_px"].median()
    r["circularidad"] = eq["circularidad"].mean()
    r["solidez"] = eq["solidez"].mean()
    r["excentricidad"] = eq["excentricidad"].mean()
    r["asimetria_cv_diam"] = _cv(eq["eje_menor_px"])
    r["ped_diam_px"] = ped["diam_equiv_px"].mean() if len(ped) else np.nan
    r["ped_circularidad"] = ped["circularidad"].mean() if len(ped) else np.nan
    r["ped_area_ratio"] = (ped["area_px"].mean() / r["area_px"]) if len(ped) and r["area_px"] else np.nan
    # --- color (media de las vistas ecuatoriales, superficie sana)
    for c in ("L_media", "a_media", "b_media", "L_std", "a_std", "b_std", "L_p10", "L_p90", "entropia_L"):
        r[c] = eq[c].mean()
    r["disp_b_vistas"] = eq["b_media"].std(ddof=0) if len(eq) > 1 else np.nan
    # --- defecto ("basta una vista": máximo sobre las 5 vistas)
    r["defecto_max"] = ok["defecto_ratio"].max() if len(ok) else np.nan
    r["defecto_media"] = ok["defecto_ratio"].mean() if len(ok) else np.nan
    r["n_vistas_defecto"] = int((ok["defecto_ratio"] >= umbral_vista_defecto).sum()) if len(ok) else np.nan
    r["mancha_max"] = ok["mancha_max_ratio"].max() if len(ok) else np.nan
    r["n_manchas_total"] = ok["n_manchas"].sum() if len(ok) else np.nan
    # --- textura
    r["lap_var"] = eq["lap_var"].mean()
    r["grad_media"] = eq["grad_media"].mean()
    # --- metadatos de calidad (no son features por defecto)
    r["n_vistas_validas"] = len(ok)
    r["frac_seg_ok"] = len(ok) / len(g)
    r["nitidez"] = g["nitidez"].mean()
    r["brillo_fondo"] = g["brillo_fondo"].mean()
    r["t_extraccion_ms"] = g["t_ms"].sum()
    return pd.Series(r)


def agregar_por_fruto(vistas: pd.DataFrame, umbral_vista_defecto: float = 0.02) -> pd.DataFrame:
    meta = vistas.groupby("id_fruto")[["lote", "fecha"]].first()
    agg = vistas.groupby("id_fruto").apply(agregar_fruto, umbral_vista_defecto=umbral_vista_defecto,
                                           include_groups=False)
    return meta.join(agg).reset_index()


def main(raw: Path, procesados: Path, n_jobs: int):
    cfg = cargar_cfg()
    params = {**PARAMS, **cfg.get("extraccion", {})}
    df_img = pd.read_csv(procesados / "indice_imagenes.csv", dtype=str)
    t0 = time.perf_counter()
    vistas = extraer_vistas(df_img, cargar_calibracion(raw), params, n_jobs)
    vistas.to_csv(procesados / "features_vistas.csv", index=False)
    frutos = agregar_por_fruto(vistas, cfg["dominio"].get("umbral_vista_defecto", 0.02))
    frutos.to_csv(procesados / "features_frutos.csv", index=False)
    mal = int((~vistas["seg_ok"].fillna(False).astype(bool)).sum())
    print(f"Vistas procesadas: {len(vistas)} (segmentación fallida: {mal})")
    print(f"Frutos: {len(frutos)} | tiempo total: {time.perf_counter()-t0:.1f} s")
    print(f"Guardado en {procesados}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    main(a.raw, a.procesados, a.jobs)
