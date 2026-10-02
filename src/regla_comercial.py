"""
regla_comercial.py
-------------------
Regla jerárquica (determinística, no aprendida) que asigna la categoría comercial
final a partir de los atributos agregados del fruto (atributos.py).

Orden de evaluación (de mayor a menor prioridad):
    1. ¿Algún defecto con severidad >= theta_def en cualquier vista?      -> Fábrica
    2. ¿Color/madurez en clase "amarillo" (umbral theta_col en CIELAB)?   -> Fábrica
    3. ¿Calibre >= 39.0 mm?                                              -> Extra
    4. ¿37.0 <= calibre < 39.0 mm?                                       -> Primera
    5. ¿35.0 <= calibre < 37.0 mm?                                       -> Segunda
    6. calibre < 35.0 mm                                                 -> Tercera

Los umbrales theta_def y theta_col se fijan en el piloto junto con el panel de
agrónomos y se guardan versionados en configs/.
"""

from dataclasses import dataclass

CALIBRE_EXTRA = 39.0
CALIBRE_PRIMERA = 37.0
CALIBRE_SEGUNDA = 35.0


@dataclass
class UmbralesRegla:
    theta_def: float
    theta_col: float


def clasificar(calibre_mm: float, color_lab: tuple, severidad_defecto_max: float,
               umbrales: UmbralesRegla) -> str:
    """Aplica el árbol de decisión jerárquico y devuelve la categoría final."""
    if severidad_defecto_max >= umbrales.theta_def:
        return "Fabrica"
    if _es_color_amarillo(color_lab, umbrales.theta_col):
        return "Fabrica"
    if calibre_mm >= CALIBRE_EXTRA:
        return "Extra"
    if calibre_mm >= CALIBRE_PRIMERA:
        return "Primera"
    if calibre_mm >= CALIBRE_SEGUNDA:
        return "Segunda"
    return "Tercera"


def _es_color_amarillo(color_lab: tuple, theta_col: float) -> bool:
    """TODO: implementar el umbral real en el espacio CIELAB definido en el piloto."""
    raise NotImplementedError
