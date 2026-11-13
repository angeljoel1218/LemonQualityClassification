"""
anotacion.py
-------------
Apoyo a la anotación de contornos y máscaras de defecto (Plan de tesis v4, sección 4.6.2 "Delimitación de defectos y contornos"):

  * CONTORNOS del fruto en las 7 500 imágenes de forma SEMIAUTOMÁTICA: `preanotar_contornos` genera la preanotación (segmentación
    clásica de respaldo o, si ya existe, un modelo YOLOv8-seg/Mask R-CNN) en formato YOLO-seg; luego una persona corrige
    (CVAT / Roboflow / LabelMe importan y exportan ese formato). ≈ 0,5 min por imagen con preanotación (≈ 63 h en total).
  * MÁSCARAS DE DEFECTO solo en una SUBMUESTRA de 500 frutos (2 500 imágenes): 400 del conjunto de desarrollo (80 por categoría)
    y 100 de la prueba (20 por categoría). `submuestra_defectos` la elige de forma reproducible y estratificada por sesión.
    Las máscaras de los 100 frutos de prueba se usan SOLO para las métricas secundarias de segmentación de defectos.
  * CONTROL DE CALIDAD: segunda anotación independiente, por el otro tesista, del 10 % de los frutos anotados (50 frutos);
    `iou_entre_anotadores` calcula el IoU medio (criterio mínimo referencial 0,70, que se confirma en el piloto) y
    `muestra_control_calidad` elige los frutos. Además, un agrónomo revisa una muestra de 25 máscaras.
  * `estimar_horas` calcula las horas de anotación para el cronograma.

Formato de etiquetas: data/anotaciones/{fruto,defectos}/<stem>.txt, con `clase x1 y1 x2 y2 ...` normalizado (clase 0 = fruto;
defectos: índice en configs/proyecto.yaml -> segmentacion.clases_defecto).

Uso:
    python src/anotacion.py preanotar     [--modelo pesos.pt --familia yolo]
    python src/anotacion.py submuestra
    python src/anotacion.py control-calidad
    python src/anotacion.py iou --dir-a data/anotaciones/fruto --dir-b data/anotaciones_2/fruto
    python src/anotacion.py horas
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src import segmentacion as sg  # noqa: E402
from src.features_imagen import PARAMS, _fondo, a_lab, segmentar_clasico  # noqa: E402

ANOT = C.CFG["anotacion"]


def preanotar_contornos(procesados: Path = ROOT / "data" / "processed", salida: Path = sg.DIR_ANOT / "fruto", modelo=None,
                        sobrescribir: bool = False, max_imgs: int | None = None) -> int:
    """Escribe una etiqueta YOLO-seg del contorno del fruto por imagen (no sobrescribe las ya corregidas salvo `sobrescribir`)."""
    salida.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(procesados / "indice_imagenes.csv", dtype=str)
    n = 0
    for r in idx.itertuples():
        stem = Path(r.ruta).stem
        dest = salida / f"{stem}.txt"
        if dest.exists() and not sobrescribir:
            continue
        ruta = Path(r.ruta)
        img = cv2.imread(str(ruta if ruta.is_absolute() else ROOT / ruta))
        if img is None:
            continue
        if modelo is not None:
            det = modelo.predecir(img)
            mask = sg.instancia_fruto([d["mascara"] for d in det], [d["score"] for d in det])
        else:
            lab = a_lab(img)
            mask = segmentar_clasico(lab, _fondo(lab, PARAMS["borde"]), PARAMS)
        if mask is None:
            continue
        lineas = sg.mascara_a_poligono_txt(mask, 0, max_inst=1)
        if lineas:
            dest.write_text("\n".join(lineas), encoding="utf-8")
            n += 1
        if max_imgs and n >= max_imgs:
            break
    print(f"Preanotación: {n} etiquetas nuevas en {salida} (corríjalas con la herramienta de anotación).")
    return n


def submuestra_defectos(frutos: pd.DataFrame, particion: pd.DataFrame, semilla: int | None = None) -> pd.DataFrame:
    """500 frutos: `defectos_desarrollo_por_categoria` (80) de train+val y `defectos_prueba_por_categoria` (20) de test, por
    categoría y estratificados por sesión (lote x fecha). Devuelve id_fruto, categoria, subset, grupo ('desarrollo' | 'prueba')."""
    from src.eda import normalizar_categorias
    semilla = C.CFG["semilla_particion"] if semilla is None else semilla
    rng = np.random.default_rng(semilla)
    d = frutos.merge(particion[["id_fruto", "subset"]], on="id_fruto")
    d["cat"] = normalizar_categorias(d["categoria_ref"])
    filas, avisos = [], []
    for grupo, subs, n in (("desarrollo", ("train", "val"), ANOT["defectos_desarrollo_por_categoria"]),
                           ("prueba", ("test",), ANOT["defectos_prueba_por_categoria"])):
        for c in C.CATEGORIAS:
            g = d[(d["cat"] == c) & d["subset"].isin(subs)]
            if len(g) < n:
                avisos.append(f"{grupo}/{c}: hay {len(g)} frutos < {n} requeridos; se toman todos")
            sel = []
            ses = g.groupby(["lote", "fecha"]).groups                       # round-robin por sesión: reparte lotes y fechas
            colas = {k: list(rng.permutation(np.array(sorted(v)))) for k, v in ses.items()}
            claves = sorted(colas)
            while len(sel) < min(n, len(g)) and any(colas.values()):
                for k in claves:
                    if colas[k] and len(sel) < min(n, len(g)):
                        sel.append(colas[k].pop())
            for i in sel:
                filas.append(dict(id_fruto=g.loc[i, "id_fruto"], categoria=c, subset=g.loc[i, "subset"], grupo=grupo))
    out = pd.DataFrame(filas)
    out.attrs["avisos"] = avisos
    return out


def muestra_control_calidad(ids_anotados, fraccion: float | None = None, semilla: int = 0) -> list:
    """Frutos para la segunda anotación independiente (10 % de los anotados; 50 si son 500)."""
    fraccion = ANOT["control_calidad_fraccion"] if fraccion is None else fraccion
    ids = sorted(ids_anotados)
    n = max(1, int(round(fraccion * len(ids))))
    return sorted(np.random.default_rng(semilla).choice(ids, n, replace=False).tolist())


def iou_entre_anotadores(dir_a: Path, dir_b: Path, ids_stems=None, forma=(480, 640)) -> dict:
    """IoU medio entre dos anotadores (misma etiqueta .txt en dos carpetas). Compara la UNIÓN de polígonos de cada imagen (por clase
    si hay varias). Criterio mínimo referencial: 0,70."""
    h, w = forma
    a_, b_ = sorted(Path(dir_a).glob("*.txt")), {p.name for p in Path(dir_b).glob("*.txt")}
    ious = []
    for pa in a_:
        if pa.name not in b_ or (ids_stems is not None and pa.stem not in ids_stems and not any(s in pa.stem for s in ids_stems)):
            continue
        A = sg.poligonos_desde_txt(pa, h, w)
        B = sg.poligonos_desde_txt(Path(dir_b) / pa.name, h, w)
        for c in sorted({c for c, _ in A} | {c for c, _ in B}):
            ious.append(sg.iou_mascaras(sg.mascara_desde_poligonos(A, h, w, c), sg.mascara_desde_poligonos(B, h, w, c)))
    ious = [x for x in ious if np.isfinite(x)]
    minimo = C.EST["segmentacion"]["iou_anotadores_min"]
    return dict(n=len(ious), iou_medio=float(np.mean(ious)) if ious else float("nan"), iou_p10=float(np.percentile(ious, 10)) if ious else float("nan"),
                criterio_minimo=minimo, cumple=bool(ious and np.mean(ious) >= minimo))


def estimar_horas(n_imagenes: int = 7500, n_frutos_defectos: int = 500, tesistas: int = 2, min_contorno: float = 0.5, min_defecto_fruto: float = 5.0) -> dict:
    """Horas de anotación del plan: contornos (~0,5 min por imagen) y máscaras de defecto (~5 min por fruto)."""
    h_c, h_d = n_imagenes * min_contorno / 60, n_frutos_defectos * min_defecto_fruto / 60
    return dict(horas_contornos=h_c, horas_defectos=h_d, horas_total=h_c + h_d, horas_por_tesista=(h_c + h_d) / tesistas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["preanotar", "submuestra", "control-calidad", "iou", "horas"])
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--modelo", default=None, help="pesos para preanotar con un modelo entrenado")
    ap.add_argument("--familia", choices=["yolo", "maskrcnn"], default="yolo")
    ap.add_argument("--dir-a", type=Path)
    ap.add_argument("--dir-b", type=Path)
    a = ap.parse_args()
    if a.accion == "preanotar":
        m = sg.cargar_modelo(a.familia, a.modelo, "fruto") if a.modelo else None
        preanotar_contornos(a.procesados, modelo=m)
    elif a.accion == "submuestra":
        from src.ingesta import leer_csv
        from src.particion import cargar_particion
        s = submuestra_defectos(leer_csv(a.raw / "etiquetas.csv"), cargar_particion(a.procesados))
        ruta = sg.DIR_ANOT / "submuestra_defectos.csv"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(ruta, index=False)
        print(s.groupby(["grupo", "categoria"]).size().unstack(fill_value=0).to_string())
        for x in s.attrs.get("avisos", []):
            print("[AVISO]", x)
        print(f"-> {ruta}")
    elif a.accion == "control-calidad":
        s = pd.read_csv(sg.DIR_ANOT / "submuestra_defectos.csv")
        ids = muestra_control_calidad(s["id_fruto"])
        ruta = sg.DIR_ANOT / "control_calidad_frutos.csv"
        pd.DataFrame({"id_fruto": ids}).to_csv(ruta, index=False)
        print(f"{len(ids)} frutos para la segunda anotación -> {ruta}")
    elif a.accion == "iou":
        print(iou_entre_anotadores(a.dir_a, a.dir_b))
    else:
        print(estimar_horas())


if __name__ == "__main__":
    main()
