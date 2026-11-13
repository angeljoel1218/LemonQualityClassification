"""
inferir_hibrido.py
-------------------
Fase ONLINE del sistema híbrido (Plan de tesis v4, sección 4.7.1 B): para cada vista de cada fruto

    1. (opcional) perturbación de luminancia (E5) y preprocesamiento: ninguno | brillo | brillo_color (E3)
    2. segmentación neuronal congelada: máscara del fruto (YOLOv8-seg o Mask R-CNN) y máscaras de defecto con su tipo
    3. estimación ANALÍTICA de atributos por vista (ancho ecuatorial en px y mm con la escala de la sesión, A_def, P_am)
y escribe una fila por vista en `logs/inferencia/<nombre>/vistas_<subset>.csv`. La agregación a fruto, la regla jerárquica y las
comparaciones (E1, E2...) se hacen después con `atributos.py`/`regla_comercial.py`/`experimentos.py` sobre ese CSV, de modo que
UNA sola corrida sobre la prueba sirve para evaluar 1 vista frente a 5 (E1) sin volver a tocar los modelos.

Protección de la prueba: el subconjunto `test` exige umbrales y modelos congelados y se genera UNA sola vez; el archivo se registra
con su huella SHA-256 (src/protocolo.py) antes de integrar las etiquetas del panel y de los operarios.

Requiere los pesos entrenados (src/segmentacion.py). Uso:
    python src/inferir_hibrido.py --subset val --nombre base
    python src/inferir_hibrido.py --subset val --preproc brillo --nombre e3_brillo
    python src/inferir_hibrido.py --subset val --perturbacion luz_mas --nombre e5_luz_mas
    python src/inferir_hibrido.py --subset test --nombre final            # una sola vez, con modelos congelados
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import atributos as at  # noqa: E402
from src import config as C  # noqa: E402
from src import preprocesamiento as pp  # noqa: E402
from src import protocolo  # noqa: E402
from src import regla_comercial as rc  # noqa: E402
from src import segmentacion as sg  # noqa: E402
from src.extraer_features import cargar_calibracion  # noqa: E402
from src.features_imagen import PARAMS, features_vista  # noqa: E402
from src.particion import cargar_particion, huella_particion  # noqa: E402

DIR_INF = ROOT / "logs" / "inferencia"


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class PipelineHibrido:
    """Segmentación + atributos analíticos de UNA vista. `modelo_defectos=None` usa el detector clásico de defectos."""

    def __init__(self, modelo_fruto, modelo_defectos=None, umbrales: rc.UmbralesRegla | None = None):
        self.fruto, self.defectos = modelo_fruto, modelo_defectos
        self.u = umbrales or rc.cargar_umbrales()

    def procesar_vista(self, img_bgr: np.ndarray, mm_por_px: float | None = None, devolver_mascaras: bool = False) -> dict:
        _sync()
        t0 = time.perf_counter()
        det = self.fruto.predecir(img_bgr)
        mask = sg.instancia_fruto([d["mascara"] for d in det], [d["score"] for d in det])
        mdef = None
        if self.defectos is not None and mask is not None:
            nombres = sg.CLASES["defectos"]
            mdef = [(nombres[d["clase"]] if d["clase"] < len(nombres) else "mancha", d["mascara"]) for d in self.defectos.predecir(img_bgr)]
        _sync()
        t_seg = (time.perf_counter() - t0) * 1000.0
        f = features_vista(img_bgr, mask=mask, mm_por_px=mm_por_px, p=PARAMS, mascaras_defecto=mdef, h_am=self.u.h_am)
        f["t_seg_ms"] = t_seg
        f["t_total_ms"] = t_seg + f.get("t_ms", 0.0)
        if devolver_mascaras:                      # evidencia para auditoría (no se guarda en los CSV)
            f["_mascara_fruto"], f["_mascaras_defecto"] = mask, mdef
        return f


def cargar_pipeline(cfg_modelos: dict | None = None, device=None) -> PipelineHibrido:
    """Carga los modelos de `configs/modelos_congelados.json` (o de los `modelo_elegido.json` de cada tarea)."""
    cfg = cfg_modelos or protocolo.cargar_modelos_congelados(exigir=False)
    def _modelo(tarea):
        d = cfg.get(f"segmentacion_{tarea}")
        if d is None:
            ruta = sg.DIR_SEG / tarea / "modelo_elegido.json"
            if not ruta.exists():
                return None
            import json
            e = json.loads(ruta.read_text(encoding="utf-8"))["elegido"]
            d = dict(familia=e["familia"], pesos=e["pesos"])
        return sg.cargar_modelo(d["familia"], d["pesos"], tarea, device)
    fruto = _modelo("fruto")
    if fruto is None:
        raise SystemExit("No hay modelo de segmentación del fruto: entrene y seleccione con `python src/segmentacion.py entrenar|elegir`.")
    return PipelineHibrido(fruto, _modelo("defectos"))


def inferir_subconjunto(subset: str, nombre: str, preproc: str = "ninguno", perturbacion: str = "ninguna", procesados: Path = ROOT / "data" / "processed",
                        raw: Path = ROOT / "data" / "raw", max_frutos: int | None = None, device=None, pipeline: PipelineHibrido | None = None) -> Path:
    if subset == "test":
        protocolo.exigir_listo_para_prueba()
        if protocolo.prueba_ya_generada(f"hibrido_{nombre}_test"):
            raise SystemExit("La prueba ya fue evaluada con este nombre (una sola vez). Ver logs/prueba/registro_prueba.json.")
    pipe = pipeline or cargar_pipeline(device=device)
    part = cargar_particion(procesados).set_index("id_fruto")["subset"]
    idx = pd.read_csv(procesados / "indice_imagenes.csv", dtype=str)
    ids = [i for i in sorted(idx["id_fruto"].unique()) if part.get(i) == subset][:max_frutos]
    idx = idx[idx["id_fruto"].isin(ids)]
    calib, sesiones, camara = cargar_calibracion(raw), pp.cargar_sesiones(), pp.cargar_camara()
    filas = []
    for r in idx.itertuples():
        ruta = Path(r.ruta)
        img = cv2.imread(str(ruta if ruta.is_absolute() else ROOT / ruta))
        base = dict(id_fruto=r.id_fruto, lote=r.lote, fecha=r.fecha, vista=r.vista, subset=subset)
        if img is None:
            filas.append({**base, "seg_ok": False, "legible": False})
            continue
        img = pp.aplicar_perturbacion(img, perturbacion)               # E5: primero la perturbación del capturador...
        img = pp.aplicar(img, preproc, (r.lote, r.fecha), sesiones, camara)   # ...y luego el preprocesamiento (E3) que debe compensarla
        filas.append({**base, "legible": True, **pipe.procesar_vista(img, calib.get((str(r.lote), str(r.fecha))))})
    out = DIR_INF / nombre
    out.mkdir(parents=True, exist_ok=True)
    ruta_csv = out / f"vistas_{subset}.csv"
    pd.DataFrame(filas).to_csv(ruta_csv, index=False)
    meta = dict(nombre=nombre, subset=subset, preproc=preproc, perturbacion=perturbacion, n_frutos=len(ids),
                huella_particion=huella_particion(cargar_particion(procesados)), umbrales=pipe.u.__dict__ if hasattr(pipe.u, "__dict__") else {})
    (out / f"meta_{subset}.json").write_text(__import__("json").dumps(meta, indent=2, default=str), encoding="utf-8")
    if subset == "test":
        protocolo.registrar_artefacto(f"hibrido_{nombre}_test", ruta_csv, meta)
    print(f"{len(filas)} vistas de {len(ids)} frutos -> {ruta_csv}")
    return ruta_csv


def latencia_por_fruto(vistas: pd.DataFrame) -> dict:
    """Latencia por fruto = suma de las 5 vistas (segmentación + atributos). Mediana y percentil 95 (requisito <= 200 ms)."""
    t = vistas.groupby("id_fruto")["t_total_ms"].sum()
    return dict(mediana_ms=float(t.median()), p95_ms=float(t.quantile(.95)), n=int(len(t)))


def mae_calibre_validacion(pipeline: PipelineHibrido, procesados: Path = ROOT / "data" / "processed", raw: Path = ROOT / "data" / "raw",
                           max_frutos: int = 80) -> float:
    """MAE (mm) del calibre d_est frente al vernier en validación con el modelo dado (para la regla de selección del plan)."""
    from src.ingesta import leer_csv
    et = leer_csv(raw / "etiquetas.csv", numericas=["calibre_mm_ref"]).set_index("id_fruto")["calibre_mm_ref"]
    tmp = inferir_subconjunto("val", f"_sel_{int(time.time())}", procesados=procesados, raw=raw, max_frutos=max_frutos, pipeline=pipeline)
    v = pd.read_csv(tmp, dtype={"id_fruto": str, "vista": str})
    a = at.agregar_tabla(v)
    err = (a["d_est_mm"] - a["id_fruto"].map(et)).abs()
    return float(err.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["train", "val", "test"], default="val")
    ap.add_argument("--nombre", default="base")
    ap.add_argument("--preproc", choices=list(pp.VARIANTES), default="ninguno")
    ap.add_argument("--perturbacion", choices=["ninguna"] + list(pp.PERTURBACIONES_E5), default="ninguna")
    ap.add_argument("--max-frutos", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    inferir_subconjunto(a.subset, a.nombre, a.preproc, a.perturbacion, max_frutos=a.max_frutos, device=a.device)
