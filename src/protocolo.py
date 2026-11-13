"""
protocolo.py
------------
Garantías del protocolo experimental del Plan de tesis v4 (secciones 4.5, 4.7.1 y "Roles y separación de funciones"):

  * La PRUEBA se evalúa UNA sola vez con la configuración congelada: `registrar_artefacto` guarda la huella SHA-256 de cada
    archivo de predicciones de la prueba ANTES de integrar las etiquetas del panel y las de los operarios, y se niega a
    sobrescribirlo (salvo `forzar=True`, que queda anotado).
  * `verificar_artefacto` comprueba después que el archivo no cambió (los resultados se calculan sobre esas predicciones).
  * Los umbrales de la regla comercial y los modelos seleccionados se congelan antes (`regla_comercial.congelar_umbrales`,
    `congelar_modelos`) y las evaluaciones de la prueba exigen que existan.

Registro: logs/prueba/registro_prueba.json
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

from src import config as C

RUTA_REGISTRO = C.ROOT / "logs" / "prueba" / "registro_prueba.json"
RUTA_MODELOS = C.ROOT / "configs" / "modelos_congelados.json"


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _leer() -> dict:
    return json.loads(RUTA_REGISTRO.read_text(encoding="utf-8")) if RUTA_REGISTRO.exists() else {}


def prueba_ya_generada(nombre: str) -> bool:
    return nombre in _leer()


def registrar_artefacto(nombre: str, ruta: Path, meta: dict | None = None, forzar: bool = False) -> dict:
    """Registra la huella del archivo de predicciones de la prueba. Una segunda vez falla (la prueba se evalúa una sola vez)."""
    reg = _leer()
    if nombre in reg and not forzar:
        raise SystemExit(f"La prueba '{nombre}' ya fue evaluada el {reg[nombre]['fecha']} (sha256 {reg[nombre]['sha256'][:12]}…). "
                         "Repetirla invalida la evaluación final; use forzar=True solo si lo justifica (queda registrado).")
    previo = reg.get(nombre, {}).get("reemplazos", [])
    if nombre in reg:
        previo = previo + [dict(sha256=reg[nombre]["sha256"], fecha=reg[nombre]["fecha"])]
    reg[nombre] = dict(archivo=str(Path(ruta).relative_to(C.ROOT)) if C.ROOT in Path(ruta).resolve().parents else str(ruta),
                       sha256=sha256(Path(ruta)), fecha=datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
                       config=C.hash_config(), reemplazos=previo, **(meta or {}))
    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    RUTA_REGISTRO.write_text(json.dumps(reg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return reg[nombre]


def verificar_artefacto(nombre: str, ruta: Path | None = None) -> dict:
    """¿El archivo actual coincide con la huella registrada? Devuelve dict(ok, esperado, actual, fecha)."""
    reg = _leer().get(nombre)
    if reg is None:
        return dict(ok=False, motivo="no registrado")
    ruta = Path(ruta) if ruta else (C.ROOT / reg["archivo"])
    if not ruta.exists():
        return dict(ok=False, motivo=f"falta {ruta}")
    act = sha256(ruta)
    return dict(ok=act == reg["sha256"], esperado=reg["sha256"], actual=act, fecha=reg["fecha"])


def congelar_modelos(info: dict, forzar: bool = False) -> dict:
    """Guarda qué modelos se usarán en la prueba (segmentación elegida, clasificador directo, semilla, preprocesamiento)."""
    if RUTA_MODELOS.exists() and not forzar:
        raise SystemExit(f"Los modelos ya están congelados en {RUTA_MODELOS}.")
    d = dict(info, fecha=datetime.now().isoformat(timespec="seconds"))
    d["huella"] = hashlib.sha256(json.dumps(info, sort_keys=True, default=str).encode()).hexdigest()[:12]
    RUTA_MODELOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_MODELOS.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return d


def cargar_modelos_congelados(exigir: bool = True) -> dict:
    if RUTA_MODELOS.exists():
        return json.loads(RUTA_MODELOS.read_text(encoding="utf-8"))
    if exigir:
        raise SystemExit("Faltan los modelos congelados (configs/modelos_congelados.json): seleccione y congele los modelos con "
                         "`python src/experimentos.py congelar` ANTES de evaluar la prueba.")
    return {}


def exigir_listo_para_prueba():
    """La prueba solo se evalúa con umbrales y modelos congelados."""
    from src import regla_comercial as rc
    if not rc.RUTA_UMBRALES.exists():
        rc.cargar_umbrales(exigir_congelados=True)
    cargar_modelos_congelados(True)
