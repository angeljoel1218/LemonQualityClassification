"""
api_inferencia.py
-------------------
Servicio de inferencia de laboratorio: recibe las 5 vistas de un fruto nuevo (ya
capturadas con el protocolo: light box, carta de color, patron de escala) y devuelve
la categoria final junto con los atributos y la evidencia por vista.

Flujo (usa los modulos ya definidos en este mismo paquete):
    preprocesamiento -> segmentacion -> atributos -> regla_comercial

No entrena nada: carga pesos ya entrenados (checkpoints en logs/) y solo ejecuta
el forward pass. Corre en una estacion de trabajo local (PC con GPU), no en la nube
y no en microcontroladores tipo Arduino.
"""

import argparse

from . import preprocesamiento, segmentacion, atributos, regla_comercial


def clasificar_fruto(fruto_id: str) -> dict:
    """Ejecuta el pipeline completo sobre las 5 vistas de un fruto y devuelve el resultado."""
    # TODO:
    # 1. Cargar y preprocesar las 5 vistas del fruto `fruto_id`.
    # 2. Segmentar cada vista (segmentacion.segmentar_vista).
    # 3. Calcular atributos por vista y agregarlos (atributos.agregar_atributos_por_fruto).
    # 4. Aplicar la regla comercial (regla_comercial.clasificar) con los umbrales del piloto.
    # 5. Devolver categoria + atributos + evidencia (mascaras/recortes) por vista.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fruto_id", required=True)
    args = parser.parse_args()
    resultado = clasificar_fruto(args.fruto_id)
    print(resultado)
