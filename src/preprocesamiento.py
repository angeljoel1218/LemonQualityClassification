"""
preprocesamiento.py
--------------------
Corrige cada imagen cruda usando los objetos de referencia incluidos en el encuadre:
    - Carta de color  -> balance de blancos y corrección de color (CIELAB comparable
      entre sesiones).
    - Patrón de tamaño conocido -> factor de calibración píxel -> milímetro de la sesión.

También corrige distorsión de lente y normaliza brillo.

Entrada:  data/raw/<lote>_<fecha>_<id_fruto>_<vista>.jpg
Salida:   data/processed/<lote>_<fecha>_<id_fruto>_<vista>.jpg
          data/processed/calibracion_sesiones.csv  (factor px->mm y matriz de color por sesión)
"""

from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def detectar_carta_de_color(imagen):
    """Ubica los parches de la carta de color en la imagen. TODO: implementar detección."""
    raise NotImplementedError


def calcular_correccion_color(parches_detectados, valores_certificados):
    """Calcula la transformación de color a aplicar a la imagen completa."""
    raise NotImplementedError


def calcular_factor_escala(patron_tamano_conocido):
    """Devuelve el factor píxel -> milímetro para la sesión."""
    raise NotImplementedError


def procesar_imagen(ruta_imagen: Path, correccion_color, factor_escala) -> "np.ndarray":
    """Aplica corrección de distorsión, color y normalización de brillo a una imagen."""
    raise NotImplementedError


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: iterar por sesión de captura, calcular correccion_color y factor_escala,
    # y aplicar procesar_imagen() a cada vista del fruto.
    print("Preprocesamiento pendiente de implementar.")
