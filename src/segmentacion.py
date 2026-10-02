"""
segmentacion.py
----------------
Segmentación de instancias por vista: máscara del fruto y máscaras de regiones con
defecto. Arquitectura a elegir en el piloto entre Mask R-CNN y YOLOv8-seg, según
velocidad de inferencia medida (criterio de decisión: se elige la opción más rápida
que no pierda calidad de máscara -IoU/Dice- más allá de un margen acordado).

Entrada:  imagen preprocesada de una vista (data/processed/...)
Salida:   máscara del fruto (bool array) + lista de máscaras de defecto con su severidad
"""

from dataclasses import dataclass
from typing import List


@dataclass
class MascaraDefecto:
    tipo: str
    severidad: float   # 0.0 - 1.0
    area_px: int


@dataclass
class ResultadoSegmentacion:
    mascara_fruto: "np.ndarray"
    defectos: List[MascaraDefecto]


def segmentar_vista(imagen, modelo) -> ResultadoSegmentacion:
    """Ejecuta el modelo de segmentación sobre una vista y devuelve máscaras + defectos."""
    raise NotImplementedError


def cargar_modelo_segmentacion(backbone: str = "resnet50", variante: str = "mask_rcnn"):
    """Carga el modelo de segmentación elegido en el piloto (mask_rcnn | yolov8_seg)."""
    raise NotImplementedError
