"""
ingesta.py
----------
Carga las imágenes crudas de data/raw/ y las agrupa por fruto.

Convención de nombres esperada por sesión de captura:
    <lote>_<fecha>_<id_fruto>_<vista>.jpg
    vista ∈ {0, 90, 180, 270, pedunculo}

Responsabilidades:
    1. Recorrer data/raw/ y construir un índice {id_fruto: {vista: ruta_imagen}}.
    2. Verificar que cada fruto tenga sus 5 vistas; si falta alguna, registrar el
       fruto en logs/capturas_incompletas.csv (ver protocolo de capturas incompletas).
    3. Adjuntar metadatos de sesión (lote, fecha) sin usarlos como entrada al modelo.
    4. Guardar el índice resultante en data/processed/indice_frutos.csv.
"""

import csv
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VISTAS_ESPERADAS = {"0", "90", "180", "270", "pedunculo"}


def construir_indice(raw_dir: Path = RAW_DIR) -> dict:
    """Agrupa las imágenes crudas por ID de fruto. TODO: implementar parseo real."""
    indice = {}
    # TODO: recorrer raw_dir, parsear nombre de archivo y poblar `indice`.
    return indice


def registrar_capturas_incompletas(indice: dict, log_path: Path = Path("logs/capturas_incompletas.csv")):
    """Marca frutos con vistas faltantes, según el protocolo definido en el entregable de Semana 2."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_fruto", "vistas_faltantes"])
        for id_fruto, vistas in indice.items():
            faltantes = VISTAS_ESPERADAS - set(vistas.keys())
            if faltantes:
                writer.writerow([id_fruto, ";".join(sorted(faltantes))])


if __name__ == "__main__":
    indice = construir_indice()
    registrar_capturas_incompletas(indice)
    print(f"Frutos indexados: {len(indice)}")
