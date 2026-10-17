"""
particion.py
------------
Partición train / val / test CONGELADA (regla del curso: "divide primero, transforma después").

- Unidad de partición: el FRUTO (las 5 vistas de un fruto van siempre al mismo subconjunto).
- Proporción 70/15/15, estratificada por categoría (y por lote cuando hay suficientes frutos),
  de modo que cada subconjunto conserve todas las categorías y lotes.
- Determinista (semilla fija) y se guarda UNA SOLA VEZ en data/processed/particion.csv.
  Si el archivo ya existe NO se regenera, salvo que se use --forzar (y entonces se invalidan
  los resultados anteriores).

Uso:
    python src/particion.py
    python src/particion.py --procesados data/demo/processed
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROPORCIONES = (0.70, 0.15, 0.15)
SEMILLA_PARTICION = 42
MIN_POR_ESTRATO = 3


def _normalizar(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")):
        t = t.str.replace(a, b, regex=False)
    return t.str.capitalize()


def asignar_subconjuntos(df: pd.DataFrame, semilla: int = SEMILLA_PARTICION,
                         proporciones=PROPORCIONES) -> pd.DataFrame:
    """Devuelve df[['id_fruto','subset']] con subset en {train, val, test}."""
    d = df.dropna(subset=["categoria_ref"]).copy()
    d["cat"] = _normalizar(d["categoria_ref"])
    d["lote"] = d["lote"].astype(str)

    # Estrato fino (categoría|lote); los estratos con pocos frutos se funden en el de categoría.
    d["estrato"] = d["cat"] + "|" + d["lote"]
    pequenos = d["estrato"].map(d["estrato"].value_counts()) < MIN_POR_ESTRATO
    d.loc[pequenos, "estrato"] = d.loc[pequenos, "cat"]

    rng = np.random.default_rng(semilla)
    d["subset"] = ""
    for _, idx in d.groupby("estrato").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_va = int(round(proporciones[1] * n))
        n_te = int(round(proporciones[2] * n))
        if n >= MIN_POR_ESTRATO:                 # garantiza >=1 en val y test
            n_va, n_te = max(n_va, 1), max(n_te, 1)
        n_tr = n - n_va - n_te
        d.loc[idx[:n_tr], "subset"] = "train"
        d.loc[idx[n_tr:n_tr + n_va], "subset"] = "val"
        d.loc[idx[n_tr + n_va:], "subset"] = "test"
    return d[["id_fruto", "subset"]]


def verificar_particion(df: pd.DataFrame, part: pd.DataFrame) -> list:
    """Devuelve la lista de problemas encontrados (vacía si todo está bien)."""
    m = df.merge(part, on="id_fruto")
    m["cat"] = _normalizar(m["categoria_ref"])
    problemas = []
    if part["id_fruto"].duplicated().any():
        problemas.append("Hay frutos repetidos en la partición (fuga entre subconjuntos).")
    for col in ("cat", "lote", "fecha"):
        tabla = pd.crosstab(m[col], m["subset"])
        for sub in ("train", "val", "test"):
            if sub not in tabla.columns:
                problemas.append(f"El subconjunto '{sub}' está vacío.")
                continue
            faltan = tabla.index[tabla[sub] == 0].tolist()
            if faltan:
                problemas.append(f"'{col}': {faltan} no aparece(n) en {sub}.")
    return problemas


def cargar_particion(procesados: Path = ROOT / "data" / "processed") -> pd.DataFrame:
    ruta = Path(procesados) / "particion.csv"
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}. Ejecuta primero: python src/particion.py")
    return pd.read_csv(ruta, dtype={"id_fruto": str})


def huella_particion(part: pd.DataFrame) -> str:
    """Hash corto de la partición, para registrarlo en cada experimento."""
    txt = "|".join(f"{a}:{b}" for a, b in part.sort_values("id_fruto").itertuples(index=False))
    return hashlib.sha256(txt.encode()).hexdigest()[:12]


def main(procesados: Path, forzar: bool):
    ruta = procesados / "particion.csv"
    if ruta.exists() and not forzar:
        part = pd.read_csv(ruta, dtype={"id_fruto": str})
        print(f"La partición ya existe y está congelada ({ruta}). Huella: {huella_particion(part)}")
        print("Para regenerarla (invalida resultados previos) usa --forzar.")
        return
    frutos = pd.read_csv(procesados / "indice_frutos.csv",
                         dtype={"id_fruto": str, "lote": str, "fecha": str})
    frutos = frutos[frutos["completo"]]                      # solo capturas completas
    part = asignar_subconjuntos(frutos)
    problemas = verificar_particion(frutos, part)
    part.to_csv(ruta, index=False)
    print(part["subset"].value_counts().to_string())
    print(f"Partición guardada en {ruta}. Huella: {huella_particion(part)}")
    for p in problemas:
        print("[AVISO]", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    main(a.procesados, a.forzar)
