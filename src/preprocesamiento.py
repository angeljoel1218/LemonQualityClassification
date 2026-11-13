"""
preprocesamiento.py
--------------------
Corrección de las imágenes crudas con los objetos de referencia del encuadre (Plan de tesis v4, secciones 4.6.1, 4.6.3 y
ablación E3). Antes era un esqueleto; ahora implementa:

  Geometría / escala
    calibrar_camara_damero(rutas, patron, cuadro_mm)      calibración intrínseca (Zhang, 2000) con tablero de damero
    corregir_distorsion(img, K, dist)                      corrección de la distorsión de lente
    escala_desde_damero(img, patron, cuadro_mm)            mm/px del plano del patrón (promedio de la distancia entre
                                                           esquinas contiguas) + diagnóstico de inclinación
    escalas_de_sesiones(raw)                               escribe data/raw/calibracion_sesiones.csv desde las imágenes
                                                           del patrón de cada sesión (`patrones/patron_<lote>_<fecha>.jpg`)
    verificar_escala(...)                                  error de la escala con un objeto de dimensión conocida

  Color / brillo (variantes de la ablación E3)
    'ninguno'       sin corrección
    'brillo'        corrección de brillo: ganancia sobre L* para igualar el fondo al nivel de referencia (+ viñeteo opcional)
    'brillo_color'  brillo + corrección de color con la carta (parches) por sesión

  Perturbaciones (ablación E5): `perturbar_luminancia(img, factor, gamma)`.

La corrección de color usa los parches de la carta por sesión descritos en `configs/sesiones.yaml` (ROI en píxeles y valores
CIELAB de referencia de la carta); sin esa información la variante 'brillo_color' degrada a 'brillo' y lo avisa.

Control de fuga: todos los parámetros se estiman con la propia imagen/sesión (fondo, carta), nunca con las etiquetas ni con
otros frutos.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src import config as C

ROOT = C.ROOT
RUTA_SESIONES = ROOT / "configs" / "sesiones.yaml"
RUTA_CAMARA = ROOT / "data" / "raw" / "camara_intrinseca.json"
VARIANTES = ("ninguno", "brillo", "brillo_color")
L_FONDO_REF = 85.0          # L* de referencia del fondo del light box tras la corrección de brillo


# ------------------------------------------------------------------------------------ cámara y escala
def calibrar_camara_damero(rutas, patron=(9, 6), cuadro_mm: float = 25.0, guardar: Path | None = RUTA_CAMARA) -> dict:
    """Calibración intrínseca con tablero de damero (Zhang, 2000). `patron` = (esquinas internas por fila, por columna).
    Devuelve K, dist y el error de reproyección RMS (píxeles). Requiere >= 8 imágenes con el tablero completo y poses distintas."""
    objp = np.zeros((patron[0] * patron[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patron[0], 0:patron[1]].T.reshape(-1, 2) * cuadro_mm
    obj, img_pts, tam = [], [], None
    for r in rutas:
        im = cv2.imread(str(r), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        tam = im.shape[::-1]
        ok, c = cv2.findChessboardCorners(im, patron, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            c = cv2.cornerSubPix(im, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            obj.append(objp)
            img_pts.append(c)
    if len(obj) < 8:
        raise ValueError(f"Se detectó el tablero en {len(obj)} imágenes; se necesitan >= 8 (poses distintas).")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img_pts, tam, None, None)
    res = dict(K=K.tolist(), dist=dist.ravel().tolist(), rms_px=float(rms), n_imagenes=len(obj), imagen=list(tam),
               patron=list(patron), cuadro_mm=cuadro_mm)
    if guardar:
        Path(guardar).parent.mkdir(parents=True, exist_ok=True)
        Path(guardar).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def cargar_camara(ruta: Path = RUTA_CAMARA):
    if Path(ruta).exists():
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return np.array(d["K"]), np.array(d["dist"])
    return None


def corregir_distorsion(img: np.ndarray, K, dist) -> np.ndarray:
    h, w = img.shape[:2]
    nueva, _ = cv2.getOptimalNewCameraMatrix(np.asarray(K), np.asarray(dist), (w, h), 0)
    return cv2.undistort(img, np.asarray(K), np.asarray(dist), None, nueva)


def escala_desde_damero(img: np.ndarray, patron=(9, 6), cuadro_mm: float = 25.0) -> dict:
    """mm/px del plano del tablero (distancia media entre esquinas contiguas). `anisotropia` = escala horizontal / vertical
    (≈ 1 si el eje óptico es perpendicular al plano); alejarse de 1 indica inclinación (error de paralaje/orientación)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ok, c = cv2.findChessboardCorners(g, patron, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return dict(ok=False, mm_por_px=np.nan)
    c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    c = c.reshape(patron[1], patron[0], 2)
    dh = np.linalg.norm(np.diff(c, axis=1), axis=2).mean()      # esquinas contiguas en una fila
    dv = np.linalg.norm(np.diff(c, axis=0), axis=2).mean()      # esquinas contiguas en una columna
    px_por_cuadro = (dh + dv) / 2.0
    return dict(ok=True, mm_por_px=float(cuadro_mm / px_por_cuadro), anisotropia=float(dh / dv), px_por_cuadro=float(px_por_cuadro))


def verificar_escala(mm_por_px: float, medida_px: float, dimension_conocida_mm: float) -> dict:
    """Error de la escala al medir un objeto de dimensión conocida (al inicio y al final de cada sesión)."""
    est = mm_por_px * medida_px
    return dict(estimado_mm=float(est), real_mm=float(dimension_conocida_mm), error_mm=float(est - dimension_conocida_mm),
                error_pct=float(100 * (est - dimension_conocida_mm) / dimension_conocida_mm))


def escalas_de_sesiones(raw: Path, patron=(9, 6), cuadro_mm: float = 25.0, salida: Path | None = None) -> pd.DataFrame:
    """Para cada `patron_<lote>_<fecha>.jpg` de <raw>/patrones/ calcula mm/px y escribe calibracion_sesiones.csv.
    Si existe la calibración intrínseca de la cámara, corrige antes la distorsión."""
    raw = Path(raw)
    cam = cargar_camara(raw / "camara_intrinseca.json")
    filas = []
    for ruta in sorted((raw / "patrones").glob("patron_*_*.*")):
        _, lote, fecha = ruta.stem.split("_", 2)
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        if cam is not None:
            img = corregir_distorsion(img, *cam)
        e = escala_desde_damero(img, patron, cuadro_mm)
        filas.append(dict(lote=lote, fecha=fecha, mm_por_px=e["mm_por_px"], anisotropia=e.get("anisotropia", np.nan), detectado=e["ok"]))
    df = pd.DataFrame(filas, columns=["lote", "fecha", "mm_por_px", "anisotropia", "detectado"])
    df[["lote", "fecha", "mm_por_px"]].dropna().to_csv(salida or (raw / "calibracion_sesiones.csv"), index=False)
    return df


# ------------------------------------------------------------------------------------ parámetros por sesión
def cargar_sesiones(ruta: Path = RUTA_SESIONES) -> dict:
    """configs/sesiones.yaml: {'<lote>_<fecha>': {'carta': {'parches': [[x,y,w,h], ...], 'lab_ref': [[L,a,b], ...]}}}."""
    if Path(ruta).exists():
        return yaml.safe_load(Path(ruta).read_text(encoding="utf-8")) or {}
    return {}


# ------------------------------------------------------------------------------------ brillo
def _lab(img):  # BGR uint8 -> Lab float32
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _bgr(lab):  # Lab float32 -> BGR uint8
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR) * 255.0, 0, 255).astype(np.uint8)


def nivel_fondo(img: np.ndarray, borde: int = 12) -> float:
    lab = _lab(img)
    b = np.concatenate([lab[:borde, :, 0].ravel(), lab[-borde:, :, 0].ravel(), lab[:, :borde, 0].ravel(), lab[:, -borde:, 0].ravel()])
    return float(np.median(b))


def correccion_brillo(img: np.ndarray, L_ref: float = L_FONDO_REF, flat_field: np.ndarray | None = None) -> np.ndarray:
    """Iguala el nivel del fondo del light box a `L_ref` con una ganancia sobre L* (+ viñeteo si se da `flat_field`, una
    imagen del fondo vacío de la sesión). No toca a* ni b*: solo el brillo."""
    lab = _lab(img)
    if flat_field is not None:
        ff = _lab(cv2.resize(flat_field, (img.shape[1], img.shape[0])))[..., 0]
        ff = np.clip(ff / max(float(np.median(ff)), 1e-3), 0.5, 1.5)
        lab[..., 0] = lab[..., 0] / ff
    L_fondo = nivel_fondo(_bgr(lab))
    g = L_ref / max(L_fondo, 1.0)
    lab[..., 0] = np.clip(lab[..., 0] * g, 0, 100)
    return _bgr(lab)


# ------------------------------------------------------------------------------------ color con carta
def _medias_parches(img, parches):
    out = []
    for x, y, w, h in parches:
        r = img[int(y):int(y + h), int(x):int(x + w)].reshape(-1, 3).astype(float)
        out.append(r.mean(axis=0) if len(r) else np.full(3, np.nan))
    return np.array(out)


def balance_blancos(img: np.ndarray, roi_gris) -> np.ndarray:
    """Balance de blancos con un parche neutro (x, y, w, h): iguala los tres canales a su media preservando el brillo."""
    m = _medias_parches(img, [roi_gris])[0]                      # medias B, G, R del parche gris
    g = m.mean() / np.maximum(m, 1e-3)
    return np.clip(img.astype(float) * g[None, None, :], 0, 255).astype(np.uint8)


def correccion_color_carta(img: np.ndarray, parches, lab_ref) -> np.ndarray:
    """Ajusta una transformación afín 3x4 (mínimos cuadrados) de los parches medidos (CIELAB) a los valores certificados de la
    carta y la aplica a toda la imagen. Requiere >= 4 parches (idealmente 24)."""
    parches = list(parches)
    if len(parches) < 4:
        raise ValueError("La corrección de color necesita >= 4 parches de la carta.")
    med = _medias_parches(img, parches)
    med_lab = _lab(np.clip(med, 0, 255).reshape(-1, 1, 3).astype(np.uint8)).reshape(-1, 3)
    A = np.c_[med_lab, np.ones(len(med_lab))]
    W, *_ = np.linalg.lstsq(A, np.asarray(lab_ref, float), rcond=None)       # (4, 3)
    lab = _lab(img)
    flat = np.c_[lab.reshape(-1, 3), np.ones(lab.shape[0] * lab.shape[1])] @ W
    out = flat.reshape(lab.shape)
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    return _bgr(out)


# ------------------------------------------------------------------------------------ interfaz
_AVISADO = set()


def aplicar(img: np.ndarray, variante: str = "ninguno", sesion=None, sesiones: dict | None = None, camara=None) -> np.ndarray:
    """Aplica la variante de preprocesamiento a una imagen. `sesion` = (lote, fecha) o '<lote>_<fecha>'."""
    if variante not in VARIANTES:
        raise ValueError(f"variante desconocida: {variante}. Opciones: {VARIANTES}")
    if camara is not None:
        img = corregir_distorsion(img, *camara)
    if variante == "ninguno":
        return img
    out = correccion_brillo(img)
    if variante == "brillo_color":
        clave = sesion if isinstance(sesion, str) else "_".join(map(str, sesion or ()))
        carta = ((sesiones if sesiones is not None else cargar_sesiones()).get(clave) or {}).get("carta")
        if carta and carta.get("parches") and carta.get("lab_ref"):
            out = correccion_color_carta(out, carta["parches"], carta["lab_ref"])
        elif clave not in _AVISADO:
            _AVISADO.add(clave)
            print(f"[AVISO] sin carta de color para la sesión '{clave}' en configs/sesiones.yaml: 'brillo_color' = 'brillo' en esa sesión.")
    return out


def perturbar_luminancia(img: np.ndarray, factor: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Perturbación leve de luminancia para E5: L* -> 100 * (L*/100)**gamma * factor."""
    lab = _lab(img)
    L = np.clip(lab[..., 0] / 100.0, 0, 1) ** gamma
    lab[..., 0] = np.clip(100.0 * L * factor, 0, 100)
    return _bgr(lab)


PERTURBACIONES_E5 = {"luz_mas": dict(factor=1.10), "luz_menos": dict(factor=0.90),
                     "gamma_alto": dict(gamma=1.15), "gamma_bajo": dict(gamma=0.87)}


def aplicar_perturbacion(img: np.ndarray, nombre: str | None) -> np.ndarray:
    return img if nombre in (None, "", "ninguna") else perturbar_luminancia(img, **PERTURBACIONES_E5[nombre])
