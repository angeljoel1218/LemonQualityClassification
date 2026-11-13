"""
config.py
---------
Lee `configs/proyecto.yaml` (parámetros del Plan de tesis v4) y expone constantes y utilidades comunes.
El resto de los módulos importan de aquí en vez de repetir números (intervalos de calibre, cuotas, B, δ, etc.).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUTA_PROYECTO = ROOT / "configs" / "proyecto.yaml"
VISTAS = ["0", "90", "180", "270", "pedunculo"]
VISTAS_EQ = ["0", "90", "180", "270"]


@lru_cache(maxsize=4)
def _leer(ruta: str) -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar(ruta: Path | str | None = None) -> dict:
    """Configuración del proyecto (copia: se puede modificar sin afectar a los demás)."""
    import copy
    return copy.deepcopy(_leer(str(ruta or RUTA_PROYECTO)))


def guardar_copia(destino: Path, cfg: dict | None = None):
    """Guarda una copia de la configuración usada en una corrida (trazabilidad)."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(cfg or cargar(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def hash_config(cfg: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(cfg or cargar(), sort_keys=True, default=str).encode()).hexdigest()[:12]


CFG = cargar()
CATEGORIAS = list(CFG["categorias"])
CATEGORIAS_CALIBRE = list(CFG["categorias_calibre"])           # Tercera < Segunda < Primera < Extra
POSITIVO = "Fabrica"
CALIBRE = CFG["calibre_mm"]
CUOTAS = CFG["cuotas"]
EST = CFG["estadistica"]
RUBRICA = CFG["rubrica"]

# Cortes de calibre (compatibilidad con el código anterior)
CALIBRE_SEGUNDA = float(CALIBRE["tercera_max"])    # 35,0: límite Tercera/Segunda
CALIBRE_PRIMERA = float(CALIBRE["segunda_max"])    # 37,0: límite Segunda/Primera
CALIBRE_EXTRA = float(CALIBRE["primera_max"])      # 39,0: límite Primera/Extra
CALIBRE_MAXIMO = float(CALIBRE["extra_max"])       # 42,0: techo del alcance

CUOTAS_VALIDAS = {c: int(v["validos"]) for c, v in CUOTAS.items()}
TOTAL_VALIDOS = sum(CUOTAS_VALIDAS.values())       # 1 500
TOTAL_TRAIN = sum(int(v["train"]) for v in CUOTAS.values())   # 880
TOTAL_VAL = sum(int(v["val"]) for v in CUOTAS.values())       # 220
TOTAL_TEST = sum(int(v["test"]) for v in CUOTAS.values())     # 400
