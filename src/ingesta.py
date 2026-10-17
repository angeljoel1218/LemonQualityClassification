"""
ingesta.py
----------
Construye los índices del dataset a partir de las imágenes crudas y las etiquetas.

Estructura esperada en --raw (por defecto data/raw/):
    imagenes/<lote>_<fecha>_<id_fruto>_<vista>.jpg
        lote      p. ej. L1            (sin guiones bajos)
        fecha     p. ej. 20260915      (AAAAMMDD)
        id_fruto  p. ej. F0001
        vista     0 | 90 | 180 | 270 | pedunculo
    etiquetas.csv   (una fila por fruto)
        id_fruto, lote, fecha, calibre_mm_ref, categoria_ref
        [opcionales: color_panel, severidad_defecto_panel]

Salidas en --salida (por defecto data/processed/):
    indice_imagenes.csv   una fila por imagen
    indice_frutos.csv     una fila por fruto (vistas presentes/faltantes + etiquetas)
    logs/capturas_incompletas.csv   frutos con vistas faltantes

Uso:
    python src/ingesta.py
    python src/ingesta.py --raw data/demo/raw --salida data/demo/processed
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VISTAS_ESPERADAS = ["0", "90", "180", "270", "pedunculo"]
EXTENSIONES = {".jpg", ".jpeg", ".png"}


COLS_NUMERICAS = ["calibre_mm_ref", "severidad_defecto_panel"]
COLS_REQUERIDAS = ["id_fruto", "lote", "fecha", "calibre_mm_ref", "categoria_ref"]


def leer_etiquetas(path: Path) -> pd.DataFrame:
    """Lee etiquetas.csv tolerando lo que hace Excel en configuración regional peruana/española:
    separador ';' en vez de ',', BOM de UTF-8, decimales con coma y espacios sobrantes."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        primera = f.readline()
    sep = ";" if primera.count(";") > primera.count(",") else ","
    df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)
    df = df.dropna(how="all")
    df = df[df["id_fruto"].notna()] if "id_fruto" in df.columns else df
    faltan = [c for c in COLS_REQUERIDAS if c not in df.columns]
    if faltan:
        raise SystemExit(f"etiquetas.csv: faltan columnas {faltan}. Columnas leídas: {list(df.columns)} "
                         f"(separador detectado: '{sep}')")
    for c in COLS_NUMERICAS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
    malas = df[~df["fecha"].fillna("").str.fullmatch(r"\d{8}")]
    if len(malas):
        print(f"[AVISO] {len(malas)} filas con `fecha` que no es AAAAMMDD (¿Excel la convirtió?), p. ej.: "
              f"{malas['fecha'].head(3).tolist()}")
    print(f"etiquetas.csv leído: {len(df)} filas (separador '{sep}')")
    return df


def indexar_imagenes(imagenes_dir: Path) -> pd.DataFrame:
    """Una fila por imagen, parseando lote/fecha/id_fruto/vista del nombre de archivo."""
    filas, invalidos = [], []
    for ruta in sorted(imagenes_dir.glob("*")):
        if ruta.suffix.lower() not in EXTENSIONES:
            continue
        partes = ruta.stem.split("_")
        if len(partes) != 4 or partes[3] not in VISTAS_ESPERADAS:
            invalidos.append(ruta.name)
            continue
        lote, fecha, id_fruto, vista = partes
        filas.append({"id_fruto": id_fruto, "lote": lote, "fecha": fecha,
                      "vista": vista, "ruta": str(ruta.relative_to(ROOT)) if ROOT in ruta.parents else str(ruta)})
    if invalidos:
        print(f"[AVISO] {len(invalidos)} archivos con nombre fuera de convención, p. ej.: {invalidos[:3]}")
    return pd.DataFrame(filas, columns=["id_fruto", "lote", "fecha", "vista", "ruta"])


def indexar_frutos(df_img: pd.DataFrame, etiquetas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por fruto: vistas presentes/faltantes + etiquetas del estándar de referencia."""
    g = df_img.groupby("id_fruto")["vista"].apply(lambda s: sorted(set(s)))
    df = g.rename("vistas_presentes").reset_index()
    df["n_vistas"] = df["vistas_presentes"].apply(len)
    df["vistas_faltantes"] = df["vistas_presentes"].apply(
        lambda v: ";".join(x for x in VISTAS_ESPERADAS if x not in v))
    df["vistas_presentes"] = df["vistas_presentes"].apply(";".join)
    df["completo"] = df["n_vistas"] == len(VISTAS_ESPERADAS)
    # frutos con etiqueta pero sin imágenes también deben aparecer
    df = etiquetas.merge(df, on="id_fruto", how="outer", suffixes=("", "_img"), indicator="origen")
    df["origen"] = df["origen"].map({"both": "ok", "left_only": "solo_etiqueta", "right_only": "solo_imagenes"})
    df["n_vistas"] = df["n_vistas"].fillna(0).astype(int)
    df["completo"] = df["completo"].fillna(False).astype(bool)
    return df


def main(raw: Path, salida: Path):
    imagenes_dir = raw / "imagenes"
    etiquetas_path = raw / "etiquetas.csv"
    if not imagenes_dir.exists() or not etiquetas_path.exists():
        raise SystemExit(f"Faltan {imagenes_dir} y/o {etiquetas_path}. Ver docstring de este archivo.")

    salida.mkdir(parents=True, exist_ok=True)
    df_img = indexar_imagenes(imagenes_dir)
    etiquetas = leer_etiquetas(etiquetas_path)
    df_fr = indexar_frutos(df_img, etiquetas)

    df_img.to_csv(salida / "indice_imagenes.csv", index=False)
    df_fr.to_csv(salida / "indice_frutos.csv", index=False)

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    incompletos = df_fr[~df_fr["completo"]][["id_fruto", "n_vistas", "vistas_faltantes", "origen"]]
    incompletos.to_csv(logs / "capturas_incompletas.csv", index=False)

    print(f"Imágenes indexadas: {len(df_img)}")
    print(f"Frutos: {len(df_fr)}  (completos: {int(df_fr['completo'].sum())}, incompletos: {len(incompletos)})")
    print(f"Índices guardados en {salida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--salida", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()
    main(args.raw, args.salida)
