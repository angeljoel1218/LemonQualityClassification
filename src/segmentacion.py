"""
segmentacion.py
----------------
Segmentación de instancias por vista (Plan de tesis v4, secciones 4.6.2 y 4.7.1). El aprendizaje profundo SOLO segmenta:
el calibre, el color y la categoría se obtienen después con procedimientos analíticos (atributos.py) y la regla jerárquica.

Modelos (decisión del plan):
    principal      YOLOv8-seg (Ultralytics): menor costo de entrenamiento e inferencia.        hasta 8 configuraciones (búsqueda
                   aleatoria) y 3 semillas para la configuración elegida.
    alternativo    Mask R-CNN con ResNet-50 (torchvision): ÚNICO candidato alternativo.        hasta 3 configuraciones, 1 semilla.
Selección (antes de las ablaciones y SOLO con validación): mayor IoU medio de la máscara del fruto y, a igualdad práctica
(diferencia de MAE de calibre < 0,05 mm), el de menor costo. Una vez elegido se congela y se usa en E1-E5.

Dos tareas, dos modelos de la misma familia (decisión de implementación, ver README):
    'fruto'      una clase: contorno del fruto. Se entrena con las 7 500 imágenes (anotación semiautomática).
    'defectos'   6 clases (mancha, rayadura, cicatriz, moho, dano_mecanico, deformacion). Se entrena SOLO con la submuestra de
                 500 frutos con máscaras de defecto (2 500 imágenes): entrenar con una sola red etiquetas faltantes en el resto
                 de las imágenes las trataría como "sin defecto". La categoría Fábrica la asigna el panel por fruto, no depende
                 de esas máscaras (4.6.2), y las máscaras de la prueba solo alimentan las métricas secundarias de defectos.

Etiquetas: formato YOLO-seg (una línea por instancia: `clase x1 y1 x2 y2 ...` normalizadas) en
    data/anotaciones/fruto/<stem>.txt      data/anotaciones/defectos/<stem>.txt        (stem = nombre de la imagen sin extensión)
Se pueden corregir con cualquier herramienta que exporte/importe ese formato (CVAT, Roboflow, LabelMe -> YOLO-seg).

Requiere GPU, `ultralytics` y `torch`/`torchvision` (se importan solo cuando se entrena/predice). Las funciones numpy
(IoU, Dice, polígonos <-> máscaras, selección) no los necesitan y están probadas con pytest.

Uso:
    python src/segmentacion.py preparar  --tarea fruto
    python src/segmentacion.py entrenar  --tarea fruto --familia yolo --configs 8
    python src/segmentacion.py entrenar  --tarea fruto --familia maskrcnn --configs 3
    python src/segmentacion.py elegir    --tarea fruto            # selección con VALIDACIÓN
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

SEG = C.CFG["segmentacion"]
BUS = C.CFG["busqueda"]
CLASES = {"fruto": list(SEG["clases_fruto"]), "defectos": list(SEG["clases_defecto"])}
DIR_ANOT = ROOT / "data" / "anotaciones"
DIR_YOLO = ROOT / "data" / "yolo"
DIR_SEG = ROOT / "logs" / "segmentacion"


# =====================================================================================================================
# numpy: polígonos, máscaras, métricas (sin torch)
# =====================================================================================================================
def poligonos_desde_txt(ruta: Path, h: int, w: int):
    """Lee un .txt YOLO-seg -> lista de (clase_idx, poligono Nx2 en píxeles)."""
    out = []
    ruta = Path(ruta)
    if not ruta.exists():
        return out
    for ln in ruta.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 7:
            continue
        pts = np.array(p[1:], float).reshape(-1, 2) * np.array([w, h])
        out.append((int(p[0]), pts))
    return out


def mascara_desde_poligonos(poligonos, h: int, w: int, clase: int | None = None) -> np.ndarray:
    """Máscara 0/255 de la UNIÓN de los polígonos (de una clase o de todas)."""
    m = np.zeros((h, w), np.uint8)
    for c, pts in poligonos:
        if clase is None or c == clase:
            cv2.fillPoly(m, [pts.round().astype(np.int32)], 255)
    return m


def mascara_a_poligono_txt(mascara: np.ndarray, clase: int = 0, eps_frac: float = 0.002, max_inst: int = 20) -> list:
    """Máscara 0/255 -> líneas YOLO-seg (un contorno externo por componente, simplificado y normalizado)."""
    h, w = mascara.shape[:2]
    cnts, _ = cv2.findContours((mascara > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lineas = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:max_inst]:
        if cv2.contourArea(c) < 20:
            continue
        a = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(a) < 3:
            continue
        lineas.append(f"{clase} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in a))
    return lineas


def iou_mascaras(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def dice_mascaras(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def metricas_mascaras(pred, ref) -> dict:
    """IoU y Dice medios de listas de máscaras emparejadas (una por imagen). Una predicción vacía cuenta como IoU = 0."""
    iou = [iou_mascaras(p, r) if p is not None else 0.0 for p, r in zip(pred, ref)]
    dice = [dice_mascaras(p, r) if p is not None else 0.0 for p, r in zip(pred, ref)]
    return dict(n=len(iou), iou_medio=float(np.nanmean(iou)) if iou else np.nan, dice_medio=float(np.nanmean(dice)) if dice else np.nan,
                iou_p10=float(np.nanpercentile(iou, 10)) if iou else np.nan)


def instancia_fruto(mascaras: list, scores: list | None = None, shape=None):
    """Elige LA máscara del fruto entre las instancias predichas: la de mayor puntaje×área (descarta la carta y el patrón)."""
    if not mascaras:
        return None
    area = np.array([(m > 0).sum() for m in mascaras], float)
    sc = np.ones(len(mascaras)) if scores is None else np.asarray(scores, float)
    k = int(np.argmax(area * sc))
    return (np.asarray(mascaras[k]) > 0).astype(np.uint8) * 255


# =====================================================================================================================
# preparación del dataset YOLO
# =====================================================================================================================
def preparar_dataset(tarea: str, procesados: Path = ROOT / "data" / "processed", preproc: str = "ninguno") -> Path:
    """Crea data/yolo/<tarea>/{images,labels}/{train,val} con las imágenes que TIENEN etiqueta, según la partición congelada
    (la PRUEBA no se copia: no interviene en el entrenamiento ni en la selección). Devuelve la ruta de data.yaml."""
    from src import preprocesamiento as pp
    from src.particion import cargar_particion
    part = cargar_particion(procesados).set_index("id_fruto")["subset"]
    idx = pd.read_csv(procesados / "indice_imagenes.csv", dtype=str)
    dest = DIR_YOLO / tarea
    if dest.exists():
        shutil.rmtree(dest)
    n = {"train": 0, "val": 0}
    sesiones, camara = pp.cargar_sesiones(), pp.cargar_camara()
    for r in idx.itertuples():
        sub = part.get(r.id_fruto)
        if sub not in n:
            continue
        stem = Path(r.ruta).stem
        lab = DIR_ANOT / tarea / f"{stem}.txt"
        if not lab.exists():
            continue
        ruta = Path(r.ruta)
        ruta = ruta if ruta.is_absolute() else ROOT / ruta
        (dest / "images" / sub).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / sub).mkdir(parents=True, exist_ok=True)
        if preproc == "ninguno" and camara is None:
            shutil.copy2(ruta, dest / "images" / sub / ruta.name)
        else:
            img = pp.aplicar(cv2.imread(str(ruta)), preproc, (r.lote, r.fecha), sesiones, camara)
            cv2.imwrite(str(dest / "images" / sub / ruta.name), img)
        shutil.copy2(lab, dest / "labels" / sub / lab.name)
        n[sub] += 1
    if n["train"] == 0 or n["val"] == 0:
        raise SystemExit(f"Dataset '{tarea}' incompleto (train={n['train']}, val={n['val']}). Faltan etiquetas en {DIR_ANOT / tarea} "
                         "(ver src/anotacion.py).")
    yml = dest / "data.yaml"
    nombres = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASES[tarea]))
    yml.write_text(f"path: {dest.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{nombres}\n", encoding="utf-8")
    print(f"Dataset {tarea}: {n} imágenes -> {yml}")
    return yml


# =====================================================================================================================
# búsqueda aleatoria (presupuesto fijo por configuración)
# =====================================================================================================================
def espacio_yolo(n: int, semilla: int | None = None) -> list:
    semilla = BUS["semilla_busqueda"] if semilla is None else semilla
    rng = np.random.default_rng(semilla)
    out = []
    for _ in range(n):
        out.append(dict(lr0=float(10 ** rng.uniform(-4, -2)), weight_decay=float(10 ** rng.uniform(-5, -2)),
                        batch=int(rng.choice([8, 16])), imgsz=int(SEG["imgsz"]),
                        hsv_v=float(rng.choice([0.2, 0.4])), fliplr=0.5, flipud=float(rng.choice([0.0, 0.5])),
                        epochs=100, patience=int(C.CFG["directo"]["early_stopping"]["paciencia"])))
    return out

def espacio_maskrcnn(n: int, semilla: int | None = None) -> list:
    semilla = BUS["semilla_busqueda"] if semilla is None else semilla
    rng = np.random.default_rng(semilla)
    return [dict(lr=float(10 ** rng.uniform(-5, -3)), weight_decay=float(10 ** rng.uniform(-5, -2)), batch=int(rng.choice([2, 4])),
                 epochs=40, patience=int(C.CFG["directo"]["early_stopping"]["paciencia"])) for _ in range(n)]


# =====================================================================================================================
# YOLOv8-seg
# =====================================================================================================================
def entrenar_yolo(tarea: str, hp: dict, semilla: int, nombre: str, data_yaml: Path | None = None, device=None) -> dict:
    """Entrena YOLOv8-seg con AdamW + decaimiento coseno + parada temprana (la monitorea sobre la validación)."""
    from ultralytics import YOLO, settings
    settings.update({"mlflow": False})
    data_yaml = data_yaml or (DIR_YOLO / tarea / "data.yaml")
    modelo = YOLO(str(SEG["yolo_pesos"]))
    t0 = time.perf_counter()
    r = modelo.train(data=str(data_yaml), epochs=hp["epochs"], patience=hp["patience"], imgsz=hp["imgsz"], batch=hp["batch"],
                     lr0=hp["lr0"], weight_decay=hp["weight_decay"], optimizer="AdamW", cos_lr=True, seed=semilla,
                     hsv_v=hp.get("hsv_v", 0.4), fliplr=hp.get("fliplr", 0.5), flipud=hp.get("flipud", 0.0), deterministic=True,
                     project=str(DIR_SEG / tarea / "yolo"), name=nombre, exist_ok=True, verbose=False, plots=False, device=device)
    seg = getattr(r, "seg", None)
    return dict(familia="yolo", nombre=nombre, semilla=semilla, pesos=str(Path(r.save_dir) / "weights" / "best.pt"),
                tiempo_entrenamiento_s=time.perf_counter() - t0, map50_mascara=float(getattr(seg, "map50", np.nan)),
                map_mascara=float(getattr(seg, "map", np.nan)), hp=hp)


class ModeloYolo:
    """Envoltura de inferencia: `predecir(img_bgr) -> lista de dict(clase, mascara 0/255, score)` a la resolución original."""
    familia = "yolo"

    def __init__(self, pesos: str, conf: float = 0.25, device=None):
        from ultralytics import YOLO
        self.m, self.conf, self.device = YOLO(pesos), conf, device

    def predecir(self, img_bgr: np.ndarray) -> list:
        r = self.m.predict(img_bgr, imgsz=SEG["imgsz"], conf=self.conf, verbose=False, retina_masks=True, device=self.device)[0]
        if r.masks is None:
            return []
        ms = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8) * 255
        cls = r.boxes.cls.cpu().numpy().astype(int)
        sc = r.boxes.conf.cpu().numpy()
        return [dict(clase=int(c), mascara=cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                     if m.shape[:2] != img_bgr.shape[:2] else m, score=float(s)) for m, c, s in zip(ms, cls, sc)]


# =====================================================================================================================
# Mask R-CNN (torchvision), único candidato alternativo
# =====================================================================================================================
def _construir_maskrcnn(n_clases: int):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, n_clases + 1)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, n_clases + 1)
    return m


class _DatasetSeg:
    """Dataset torch de imágenes + polígonos YOLO-seg (para Mask R-CNN)."""
    def __init__(self, carpeta_images: Path, carpeta_labels: Path, aumento: bool = False):
        self.imgs = sorted(Path(carpeta_images).glob("*.*"))
        self.labels, self.aumento = Path(carpeta_labels), aumento

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        import torch
        ruta = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(str(ruta)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        pols = poligonos_desde_txt(self.labels / f"{ruta.stem}.txt", h, w)
        masks = np.stack([mascara_desde_poligonos([p], h, w) > 0 for p in pols]) if pols else np.zeros((0, h, w), bool)
        if self.aumento and np.random.rand() < 0.5:
            img, masks = img[:, ::-1].copy(), masks[:, :, ::-1].copy()
        boxes = []
        for m in masks:
            ys, xs = np.where(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [0, 0, 1, 1])
        t = dict(boxes=torch.as_tensor(np.array(boxes, np.float32).reshape(-1, 4)),
                 labels=torch.as_tensor([c + 1 for c, _ in pols], dtype=torch.int64),
                 masks=torch.as_tensor(masks.astype(np.uint8)))
        return torch.as_tensor(img.transpose(2, 0, 1) / 255.0, dtype=torch.float32), t


def entrenar_maskrcnn(tarea: str, hp: dict, semilla: int, nombre: str, device=None) -> dict:
    """Entrena Mask R-CNN (ResNet-50 FPN) con AdamW + coseno + parada temprana sobre la pérdida de validación."""
    import torch
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = DIR_YOLO / tarea
    dl = lambda sub, aum: torch.utils.data.DataLoader(_DatasetSeg(base / "images" / sub, base / "labels" / sub, aum), batch_size=hp["batch"],
                                                      shuffle=aum, collate_fn=lambda b: tuple(zip(*b)), num_workers=2)
    tr, va = dl("train", True), dl("val", False)
    modelo = _construir_maskrcnn(len(CLASES[tarea])).to(dev)
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=hp["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    carpeta = DIR_SEG / tarea / "maskrcnn" / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    mejor, sin_mejora, t0 = np.inf, 0, time.perf_counter()
    for ep in range(hp["epochs"]):
        modelo.train()
        for imgs, tg in tr:
            imgs = [i.to(dev) for i in imgs]
            tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
            loss = sum(modelo(imgs, tg).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        vl = 0.0
        with torch.no_grad():                      # pérdida de validación (modo train sin gradientes: BN congelado en torchvision)
            for imgs, tg in va:
                imgs = [i.to(dev) for i in imgs]
                tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
                vl += float(sum(modelo(imgs, tg).values())) * len(imgs)
        vl /= max(len(va.dataset), 1)
        if vl < mejor - 1e-4:
            mejor, sin_mejora = vl, 0
            torch.save(modelo.state_dict(), carpeta / "best.pt")
        else:
            sin_mejora += 1
            if sin_mejora >= hp["patience"]:
                break
    return dict(familia="maskrcnn", nombre=nombre, semilla=semilla, pesos=str(carpeta / "best.pt"), val_loss=float(mejor),
                tiempo_entrenamiento_s=time.perf_counter() - t0, epocas=ep + 1, hp=hp, tarea=tarea)


class ModeloMaskRCNN:
    familia = "maskrcnn"

    def __init__(self, pesos: str, tarea: str = "fruto", conf: float = 0.5, device=None):
        import torch
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = _construir_maskrcnn(len(CLASES[tarea]))
        self.m.load_state_dict(torch.load(pesos, map_location=self.dev))
        self.m.to(self.dev).eval()
        self.conf = conf

    def predecir(self, img_bgr: np.ndarray) -> list:
        t = self.torch.as_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0, dtype=self.torch.float32).to(self.dev)
        with self.torch.no_grad():
            o = self.m([t])[0]
        out = []
        for m, c, s in zip(o["masks"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()):
            if s >= self.conf:
                out.append(dict(clase=int(c) - 1, mascara=((m[0] > 0.5).astype(np.uint8) * 255), score=float(s)))
        return out


def cargar_modelo(familia: str, pesos: str, tarea: str = "fruto", device=None):
    return ModeloYolo(pesos, device=device) if familia == "yolo" else ModeloMaskRCNN(pesos, tarea, device=device)


# =====================================================================================================================
# evaluación y selección con VALIDACIÓN
# =====================================================================================================================
def evaluar_en_validacion(modelo, tarea: str = "fruto", max_imgs: int | None = None) -> dict:
    """IoU/Dice medios de la máscara del fruto en las imágenes de validación (data/yolo/<tarea>/images/val) y latencia por imagen."""
    base = DIR_YOLO / tarea
    imgs = sorted((base / "images" / "val").glob("*.*"))[:max_imgs]
    pred, ref, t = [], [], []
    for ruta in imgs:
        img = cv2.imread(str(ruta))
        h, w = img.shape[:2]
        gt = mascara_desde_poligonos(poligonos_desde_txt(base / "labels" / "val" / f"{ruta.stem}.txt", h, w), h, w)
        t0 = time.perf_counter()
        det = modelo.predecir(img)
        t.append((time.perf_counter() - t0) * 1000)
        pred.append(instancia_fruto([d["mascara"] for d in det], [d["score"] for d in det]))
        ref.append(gt)
    m = metricas_mascaras(pred, ref)
    m.update(latencia_ms_mediana=float(np.median(t)) if t else np.nan, latencia_ms_p95=float(np.percentile(t, 95)) if t else np.nan)
    return m


def seleccionar_modelo(candidatos: pd.DataFrame, margen_mae_mm: float | None = None) -> dict:
    """Regla del plan: mayor IoU medio de la máscara del fruto en validación; a igualdad práctica (diferencia de MAE de calibre
    < 0,05 mm respecto del mejor) se elige el de MENOR costo (latencia). Columnas: nombre, iou_medio, mae_calibre_mm,
    latencia_ms_mediana, tiempo_entrenamiento_s."""
    margen = BUS["margen_mae_seleccion_mm"] if margen_mae_mm is None else margen_mae_mm
    d = candidatos.dropna(subset=["iou_medio"]).sort_values("iou_medio", ascending=False)
    if d.empty:
        raise ValueError("Sin candidatos evaluados")
    mejor = d.iloc[0]
    if "mae_calibre_mm" in d and d["mae_calibre_mm"].notna().any():
        iguales = d[(d["mae_calibre_mm"] - mejor["mae_calibre_mm"]).abs() < margen]
        if len(iguales) > 1:
            costo = "latencia_ms_mediana" if "latencia_ms_mediana" in iguales else "tiempo_entrenamiento_s"
            elegido = iguales.sort_values(costo).iloc[0]
            return dict(elegido=elegido.to_dict(), criterio="igualdad practica de MAE (< %.2f mm): menor costo" % margen)
    return dict(elegido=mejor.to_dict(), criterio="mayor IoU medio de la mascara del fruto")


def ejecutar_busqueda(tarea: str, familia: str, n_configs: int, semillas: list, device=None, procesados: Path = ROOT / "data" / "processed") -> pd.DataFrame:
    """Entrena `n_configs` configuraciones (1 semilla c/u; el plan fija 3 semillas SOLO para la configuración elegida) y registra
    el IoU de validación. Presupuesto fijo por familia (8 para el principal, 3 para el alternativo)."""
    preparar_dataset(tarea, procesados)
    hps = espacio_yolo(n_configs) if familia == "yolo" else espacio_maskrcnn(n_configs)
    filas = []
    for i, hp in enumerate(hps):
        nombre = f"{familia}_{tarea}_cfg{i}_s{semillas[0]}"
        fn = entrenar_yolo if familia == "yolo" else entrenar_maskrcnn
        res = fn(tarea, hp, semillas[0], nombre, device=device)
        m = evaluar_en_validacion(cargar_modelo(familia, res["pesos"], tarea, device), tarea)
        filas.append({**{k: v for k, v in res.items() if k != "hp"}, **m, "hp": json.dumps(hp)})
        pd.DataFrame(filas).to_csv(DIR_SEG / tarea / f"trials_{familia}.csv", index=False)
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["preparar", "entrenar", "elegir"])
    ap.add_argument("--tarea", choices=["fruto", "defectos"], default="fruto")
    ap.add_argument("--familia", choices=["yolo", "maskrcnn"], default="yolo")
    ap.add_argument("--configs", type=int, default=None)
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    (DIR_SEG / a.tarea).mkdir(parents=True, exist_ok=True)
    if a.accion == "preparar":
        preparar_dataset(a.tarea, a.procesados)
    elif a.accion == "entrenar":
        n = a.configs or (BUS["configs_modelo_principal"] if a.familia == "yolo" else BUS["configs_candidato_alternativo"])
        df = ejecutar_busqueda(a.tarea, a.familia, n, BUS["semillas_entrenamiento"], a.device, a.procesados)
        print(df[["nombre", "iou_medio", "dice_medio", "latencia_ms_mediana"]].to_string(index=False))
    else:
        tr = [pd.read_csv(p) for p in sorted((DIR_SEG / a.tarea).glob("trials_*.csv"))]
        if not tr:
            raise SystemExit("No hay corridas: ejecute `entrenar` primero.")
        sel = seleccionar_modelo(pd.concat(tr, ignore_index=True))
        ruta = DIR_SEG / a.tarea / "modelo_elegido.json"
        ruta.write_text(json.dumps(dict(sel, fecha=datetime.now().isoformat(timespec="seconds")), indent=2, default=str), encoding="utf-8")
        print(f"Modelo elegido ({sel['criterio']}): {sel['elegido']['nombre']} -> {ruta}")


if __name__ == "__main__":
    main()
