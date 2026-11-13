"""
api_inferencia.py
-------------------
Servicio de inferencia de LABORATORIO (Plan de tesis v4, secciones 4.7.1 y 4.7.2): recibe las 5 vistas de un fruto capturadas con el
protocolo y devuelve un JSON con la categoría asignada, los valores numéricos estimados (calibre d_est, severidad S, P_am) y las
máscaras de evidencia por vista para auditoría. No entrena nada: carga los modelos congelados. Corre en una estación de trabajo con
GPU; no es un sistema de línea (sin faja, sin separación física: fuera del alcance de la tesis).

    API REST (FastAPI):      uvicorn src.api_inferencia:app --port 8000
    POST /clasificar         multipart: archivos `v0`, `v90`, `v180`, `v270`, `pedunculo` + campo `mm_por_px` (o `lote` y `fecha`)
    GET  /salud              estado del servicio y de los modelos

    Línea de comandos:       python src/api_inferencia.py --fruto F0001 --raw data/raw
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import atributos as at  # noqa: E402
from src import config as C  # noqa: E402
from src import regla_comercial as rc  # noqa: E402
from src import segmentacion as sg  # noqa: E402

_PIPE = {}


def _pipeline():
    if "p" not in _PIPE:
        from src.inferir_hibrido import cargar_pipeline
        _PIPE["p"] = cargar_pipeline()
    return _PIPE["p"]


def clasificar_vistas(imagenes: dict, mm_por_px: float, pipeline=None, con_evidencia: bool = True) -> dict:
    """imagenes: {vista: imagen BGR} con las 5 vistas de C.VISTAS. Devuelve el JSON de respuesta (dict serializable)."""
    faltan = [v for v in C.VISTAS if v not in imagenes]
    if faltan:
        raise ValueError(f"Faltan vistas: {faltan}")
    pipe = pipeline or _pipeline()
    t0 = time.perf_counter()
    filas, evidencia = [], {}
    for v in C.VISTAS:
        f = pipe.procesar_vista(imagenes[v], mm_por_px, devolver_mascaras=con_evidencia)
        m, md = f.pop("_mascara_fruto", None), f.pop("_mascaras_defecto", None)
        f["vista"] = v
        f["id_fruto"] = "consulta"
        filas.append(f)
        if con_evidencia and m is not None:
            evidencia[v] = dict(fruto=sg.mascara_a_poligono_txt(m, 0, max_inst=1),
                                defectos=[dict(tipo=t, poligonos=sg.mascara_a_poligono_txt(mk, 0)) for t, mk in (md or [])])
    import pandas as pd
    df = pd.DataFrame(filas)
    t = at.clasificar_tabla(df, pipe.u).iloc[0]
    return dict(categoria=str(t["pred"]), d_est_mm=_f(t["d_est_mm"]), d_mediana4_mm=_f(t["d_med4_mm"]), severidad_max=_f(t["severidad_max"]),
                p_amarillo=_f(t["p_amarillo"]), motivo_fabrica=str(t["motivo_fabrica"]),
                fuera_de_alcance=bool(t["pred"] == rc.FUERA), umbrales=dict(theta_def=pipe.u.theta_def, p_am=pipe.u.p_am, h_am=pipe.u.h_am),
                por_vista=[dict(vista=r["vista"], ancho_mm=_f(r.get("ancho_mm")), A_def=_f(r.get("defecto_ratio")), tipos_defecto=r.get("tipos_defecto", ""),
                                p_amarillo=_f(r.get("p_amarillo")), ok=bool(r.get("seg_ok"))) for r in filas],
                latencia_ms=(time.perf_counter() - t0) * 1000.0, evidencia=evidencia if con_evidencia else None)


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def crear_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    app = FastAPI(title="Clasificación comercial de limón sutil (laboratorio)", version="1.0")

    @app.get("/salud")
    def salud():
        from src import protocolo
        return dict(estado="ok", modelos_congelados=protocolo.RUTA_MODELOS.exists(), umbrales_congelados=rc.RUTA_UMBRALES.exists(),
                    categorias=C.CATEGORIAS, vistas=C.VISTAS)

    @app.post("/clasificar")
    async def clasificar(v0: UploadFile = File(...), v90: UploadFile = File(...), v180: UploadFile = File(...), v270: UploadFile = File(...),
                         pedunculo: UploadFile = File(...), mm_por_px: float | None = Form(None), lote: str | None = Form(None),
                         fecha: str | None = Form(None), evidencia: bool = Form(True)):
        imgs = {}
        for nombre, up in (("0", v0), ("90", v90), ("180", v180), ("270", v270), ("pedunculo", pedunculo)):
            im = cv2.imdecode(np.frombuffer(await up.read(), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise HTTPException(400, f"imagen ilegible: {nombre}")
            imgs[nombre] = im
        if mm_por_px is None:
            from src.extraer_features import cargar_calibracion
            mm_por_px = cargar_calibracion(ROOT / "data" / "raw").get((str(lote), str(fecha)))
        if not mm_por_px:
            raise HTTPException(400, "Falta la escala: envíe `mm_por_px` o `lote`+`fecha` con calibracion_sesiones.csv")
        return clasificar_vistas(imgs, float(mm_por_px), con_evidencia=evidencia)

    return app


try:                                   # `uvicorn src.api_inferencia:app`
    app = crear_app()
except ImportError:                    # fastapi no instalado: las funciones de clasificación siguen disponibles
    app = None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fruto", required=True)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    a = ap.parse_args()
    from src.extraer_features import cargar_calibracion
    carpeta = a.raw / "imagenes"
    arch = {v: next(iter(carpeta.glob(f"*_{a.fruto}_{v}.*")), None) for v in C.VISTAS}
    if any(p is None for p in arch.values()):
        raise SystemExit(f"No se encontraron las 5 vistas de {a.fruto} en {carpeta}")
    lote, fecha = next(iter(arch.values())).stem.split("_")[:2]
    mm = cargar_calibracion(a.raw).get((lote, fecha))
    if not mm:
        raise SystemExit("Falta la escala de la sesión (data/raw/calibracion_sesiones.csv)")
    import json
    r = clasificar_vistas({v: cv2.imread(str(p)) for v, p in arch.items()}, mm, con_evidencia=False)
    print(json.dumps(r, indent=2, ensure_ascii=False))
