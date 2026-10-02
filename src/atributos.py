"""
atributos.py
-------------
A partir de las máscaras por vista (segmentacion.py), estima los atributos físicos
del fruto y los agrega a nivel de fruto (no de vista):

    - Calibre (mm): a partir de la máscara del fruto y el factor de escala de la sesión.
    - Color / madurez (CIELAB L*, a*, b*): promedio sobre la superficie sana.
    - Defecto: área, tipo y severidad por vista.

Reglas de agregación entre vistas (definidas en el flujo propuesto):
    - Calibre -> mediana entre las vistas ecuatoriales.
    - Color   -> media entre las vistas ecuatoriales (solo superficie sana).
    - Defecto -> "basta una vista con severidad >= umbral" para marcar el fruto.
"""

from dataclasses import dataclass
from typing import List
import statistics


@dataclass
class AtributosFruto:
    calibre_mm: float
    color_lab: tuple          # (L, a, b)
    severidad_defecto_max: float


def calibre_desde_mascara(mascara_fruto, factor_escala_mm_por_px) -> float:
    """Calcula el calibre (mm) a partir del diámetro ecuatorial en la máscara."""
    raise NotImplementedError


def color_cielab_superficie_sana(imagen, mascara_fruto, mascaras_defecto):
    """Calcula L*, a*, b* promedio excluyendo las regiones con defecto."""
    raise NotImplementedError


def agregar_atributos_por_fruto(atributos_por_vista: List[dict]) -> AtributosFruto:
    """Agrega calibre (mediana), color (media) y defecto (máximo) entre vistas."""
    calibres = [v["calibre_mm"] for v in atributos_por_vista]
    severidades = [v["severidad_defecto"] for v in atributos_por_vista]
    # TODO: promediar color CIELAB componente a componente.
    return AtributosFruto(
        calibre_mm=statistics.median(calibres),
        color_lab=(0.0, 0.0, 0.0),
        severidad_defecto_max=max(severidades) if severidades else 0.0,
    )
