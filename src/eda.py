"""
eda.py
------
Funciones reutilizables para el análisis exploratorio (las usa notebooks/EDA_basico.ipynb).

Todas trabajan sobre los índices que genera src/ingesta.py.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.regla_comercial import CALIBRE_EXTRA, CALIBRE_PRIMERA, CALIBRE_SEGUNDA

ROOT = Path(__file__).resolve().parents[1]
CATEGORIAS = ["Extra", "Primera", "Segunda", "Tercera", "Fabrica"]
CUOTAS_MIN = {"Extra": 250, "Primera": 250, "Segunda": 250, "Tercera": 200, "Fabrica": 200}
MIN_LOTES, MIN_FECHAS = 3, 4


# ----------------------------------------------------------------- calidad de etiquetas
def normalizar_categorias(s: pd.Series) -> pd.Series:
    """Quita tildes/espacios/mayúsculas para que 'Fábrica' == 'fabrica'."""
    return (s.astype(str).str.strip().str.lower()
            .str.replace("á", "a").str.replace("é", "e").str.replace("í", "i")
            .str.replace("ó", "o").str.replace("ú", "u").str.capitalize())


def resumen_calidad_etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    """Nulos, duplicados y valores inválidos en la tabla de frutos."""
    out = {
        "frutos_totales": len(df),
        "id_fruto_duplicados": int(df["id_fruto"].duplicated().sum()),
        "sin_categoria_ref": int(df["categoria_ref"].isna().sum()),
        "categoria_invalida": int((~df["categoria_ref"].isna() &
                                   ~normalizar_categorias(df["categoria_ref"]).isin(CATEGORIAS)).sum()),
        "sin_calibre_ref": int(df["calibre_mm_ref"].isna().sum()),
        "calibre_fuera_de_rango(25-60mm)": int(((df["calibre_mm_ref"] < 25) | (df["calibre_mm_ref"] > 60)).sum()),
        "frutos_sin_imagenes": int((df["origen"] == "solo_etiqueta").sum()),
        "frutos_sin_etiqueta": int((df["origen"] == "solo_imagenes").sum()),
        "capturas_incompletas": int((~df["completo"]).sum()),
    }
    return pd.Series(out, name="conteo").to_frame()


# ----------------------------------------------------------------- balance y cobertura
def tabla_cuotas(df: pd.DataFrame) -> pd.DataFrame:
    """Frutos por categoría vs cuota mínima objetivo."""
    cat = normalizar_categorias(df["categoria_ref"])
    t = cat.value_counts().reindex(CATEGORIAS).fillna(0).astype(int).to_frame("frutos")
    t["cuota_min"] = pd.Series(CUOTAS_MIN)
    t["faltan"] = (t["cuota_min"] - t["frutos"]).clip(lower=0)
    t["cumple"] = t["faltan"] == 0
    t["%"] = (100 * t["frutos"] / max(t["frutos"].sum(), 1)).round(1)
    return t


def cobertura_lote_fecha(df: pd.DataFrame) -> dict:
    cat = normalizar_categorias(df["categoria_ref"])
    return {
        "n_lotes": df["lote"].nunique(),
        "n_fechas": df["fecha"].nunique(),
        "cumple_lotes": df["lote"].nunique() >= MIN_LOTES,
        "cumple_fechas": df["fecha"].nunique() >= MIN_FECHAS,
        "cat_x_lote": pd.crosstab(cat, df["lote"]).reindex(CATEGORIAS),
        "cat_x_fecha": pd.crosstab(cat, df["fecha"]).reindex(CATEGORIAS),
    }


def asociacion_categoria_metadato(df: pd.DataFrame, col: str) -> float:
    """V de Cramér entre categoría y lote/fecha. Cerca de 1 = riesgo de atajo (el modelo podría
    'adivinar' la categoría por el lote/fecha en vez de por el fruto)."""
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(normalizar_categorias(df["categoria_ref"]), df[col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    return float(np.sqrt((chi2 / n) / (min(ct.shape) - 1)))


# ----------------------------------------------------------------- calibre vs categoría
def categoria_por_calibre(calibre_mm: float) -> str:
    if calibre_mm >= CALIBRE_EXTRA:
        return "Extra"
    if calibre_mm >= CALIBRE_PRIMERA:
        return "Primera"
    if calibre_mm >= CALIBRE_SEGUNDA:
        return "Segunda"
    return "Tercera"


def inconsistencias_calibre_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """Frutos NO-Fábrica cuya categoría de referencia no coincide con la que daría su calibre.
    (Fábrica se decide por defecto/color, así que no se contrasta con el calibre.)"""
    d = df.dropna(subset=["calibre_mm_ref", "categoria_ref"]).copy()
    d["cat"] = normalizar_categorias(d["categoria_ref"])
    d = d[d["cat"] != "Fabrica"]
    d["cat_segun_calibre"] = d["calibre_mm_ref"].apply(categoria_por_calibre)
    return d[d["cat"] != d["cat_segun_calibre"]][
        ["id_fruto", "lote", "fecha", "calibre_mm_ref", "cat", "cat_segun_calibre"]]


def fronterizos(df: pd.DataFrame, margen_mm: float = 0.5) -> pd.DataFrame:
    """Frutos a ≤ margen_mm de un umbral de calibre: son los más difíciles y los más sensibles
    al error de medición (relevante para el MAE objetivo de 1,0 mm)."""
    umbrales = np.array([CALIBRE_SEGUNDA, CALIBRE_PRIMERA, CALIBRE_EXTRA])
    d = df.dropna(subset=["calibre_mm_ref"]).copy()
    d["dist_umbral"] = d["calibre_mm_ref"].apply(lambda c: np.abs(umbrales - c).min())
    return d[d["dist_umbral"] <= margen_mm]


# ----------------------------------------------------------------- calidad de imagen
def metricas_imagen(ruta: Path) -> dict:
    import cv2
    img = cv2.imread(str(ruta))
    if img is None:
        return {"legible": False}
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return {
        "legible": True,
        "alto": img.shape[0], "ancho": img.shape[1],
        "nitidez": float(cv2.Laplacian(gris, cv2.CV_64F).var()),   # bajo = borrosa
        "brillo": float(gris.mean()),
        "saturacion": float(hsv[..., 1].mean()),
        "pct_sobreexpuesto": float((gris > 250).mean() * 100),
        "pct_subexpuesto": float((gris < 5).mean() * 100),
        "md5": hashlib.md5(img.tobytes()).hexdigest(),
    }


def calcular_metricas_imagenes(df_img: pd.DataFrame, max_imgs: int | None = None) -> pd.DataFrame:
    d = df_img if max_imgs is None else df_img.sample(min(max_imgs, len(df_img)), random_state=0)
    filas = []
    for _, r in d.iterrows():
        ruta = Path(r["ruta"])
        ruta = ruta if ruta.is_absolute() else ROOT / ruta
        filas.append({**r.to_dict(), **metricas_imagen(ruta)})
    out = pd.DataFrame(filas)
    out["duplicada"] = out["md5"].duplicated(keep=False) & out["legible"]
    return out


def marcar_atipicas(m: pd.DataFrame, col: str, k: float = 3.0) -> pd.Series:
    """Outliers por regla IQR (k=3: solo valores claramente extremos)."""
    q1, q3 = m[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return (m[col] < q1 - k * iqr) | (m[col] > q3 + k * iqr)


# ----------------------------------------------------------------- vista previa de partición
def vista_previa_particion(df: pd.DataFrame, semilla: int = 0) -> pd.DataFrame:
    """VISTA PREVIA (no definitiva): 70/15/15 por fruto, estratificada por categoría.
    La partición oficial se genera una sola vez y se congela como artefacto versionado."""
    rng = np.random.default_rng(semilla)
    d = df.dropna(subset=["categoria_ref"]).copy()
    d["cat"] = normalizar_categorias(d["categoria_ref"])
    d["subset"] = ""
    for _, idx in d.groupby("cat").groups.items():
        idx = np.array(list(idx)); rng.shuffle(idx)
        n = len(idx); n_tr, n_va = int(round(.70 * n)), int(round(.15 * n))
        d.loc[idx[:n_tr], "subset"] = "train"
        d.loc[idx[n_tr:n_tr + n_va], "subset"] = "val"
        d.loc[idx[n_tr + n_va:], "subset"] = "test"
    return d
