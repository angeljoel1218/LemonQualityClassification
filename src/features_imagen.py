"""
features_imagen.py
------------------
Extracción de features de UNA vista (una imagen). Sin estado: no aprende nada de los datos,
así que no hay riesgo de fuga (la parte con estado vive en transformadores.py y se ajusta con train).

Flujo por vista:
    1. Máscara del fruto: o bien la entrega el modelo de segmentación (src/segmentacion.py) o se
       calcula con una segmentación clásica de respaldo (distancia de color al fondo + Otsu).
    2. Geometría (px y, si hay escala, mm): ejes, área, circularidad, solidez, excentricidad.
    3. Color en CIELAB real (L* 0-100, a*, b*) sobre la superficie SANA (sin defectos ni borde).
    4. Defectos: regiones cuyo color se aleja del color mediano del fruto (más oscuras, no brillos).
    5. Textura: varianza del laplaciano, gradiente medio, entropía de L*.

Los parámetros se leen de configs/fe.yaml (sección `extraccion`) y quedan registrados.
"""
import time

import cv2
import numpy as np

PARAMS = dict(
    borde=12,              # píxeles del marco usado para estimar el color del fondo
    de_min=8.0,            # diferencia de color mínima (Delta E) fondo-fruto para segmentar
    min_frac_area=0.01,    # el fruto debe ocupar al menos 1 % de la imagen
    erosion_frac=0.04,     # erosión del borde (fracción del diámetro) para evitar el contorno
    defecto_k=2.5,         # sensibilidad (en desviaciones robustas) del detector de defectos
    defecto_de_min=12.0,   # Delta E mínimo respecto al color mediano para ser defecto
    defecto_min_frac=0.002,  # tamaño mínimo de una mancha (fracción de la superficie)
    claro_tol=5.0,         # tolerancia de L* por encima de la mediana (excluye brillos especulares)
)


def a_lab(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> CIELAB real en float32 (L* 0..100, a*/b* aprox. -128..127)."""
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _fondo(lab: np.ndarray, b: int) -> np.ndarray:
    borde = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                            lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    return np.median(borde, axis=0)


def segmentar_clasico(lab: np.ndarray, fondo: np.ndarray, p: dict = PARAMS):
    """Segmentación clásica de respaldo para fondo uniforme (light box). Devuelve máscara 0/255 o None.
    Si hay carta de color / patrón de escala en el encuadre se elige el objeto más grande y centrado."""
    h, w = lab.shape[:2]
    de = np.linalg.norm(lab - fondo, axis=2)
    de8 = cv2.GaussianBlur(np.clip(de * 4, 0, 255).astype(np.uint8), (5, 5), 0)
    t, _ = cv2.threshold(de8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(t, p["de_min"] * 4)
    m = ((de8 > t).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mejor, mejor_score = None, 0.0
    diag = float(np.hypot(h, w))
    for c in cnts:
        area = cv2.contourArea(c)
        if area < p["min_frac_area"] * h * w:
            continue
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        d = np.hypot(cx - w / 2, cy - h / 2)
        score = area * np.exp(-(d / (0.35 * diag)) ** 2)
        if score > mejor_score:
            mejor, mejor_score = c, score
    if mejor is None:
        return None
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, [mejor], -1, 255, thickness=-1)
    return out


def features_geometricas(mask: np.ndarray) -> dict:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    harea = cv2.contourArea(cv2.convexHull(c))
    (w_, h_) = cv2.minAreaRect(c)[1]
    mayor, menor = float(max(w_, h_)), float(min(w_, h_))
    return dict(
        area_px=float(area), perim_px=float(per),
        solidez=float(area / harea) if harea > 0 else np.nan,
        circularidad=float(min(1.0, 4 * np.pi * area / per ** 2)) if per > 0 else np.nan,
        eje_mayor_px=mayor, eje_menor_px=menor,
        diam_equiv_px=float(np.sqrt(4 * area / np.pi)),
        excentricidad=float(np.sqrt(max(0.0, 1 - (menor / mayor) ** 2))) if mayor > 0 else np.nan,
        aspecto=float(mayor / menor) if menor > 0 else np.nan,
    )


def features_color_defecto_textura(img_bgr, lab, mask, diam_px, p: dict = PARAMS) -> dict:
    k = max(3, int(p["erosion_frac"] * diam_px))
    core = cv2.erode(mask, np.ones((k, k), np.uint8))
    if int((core > 0).sum()) < 200:
        core = mask
    sel = core > 0
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    med = np.array([np.median(L[sel]), np.median(a[sel]), np.median(b[sel])])
    de = np.sqrt((L - med[0]) ** 2 + (a - med[1]) ** 2 + (b - med[2]) ** 2)
    de_in = de[sel]
    mad = 1.4826 * np.median(np.abs(de_in - np.median(de_in)))
    thr = max(np.median(de_in) + p["defecto_k"] * mad, p["defecto_de_min"])
    cand = (sel & (de > thr) & (L < med[0] + p["claro_tol"])).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    area_core = float(sel.sum())
    minpx = max(5, p["defecto_min_frac"] * area_core)
    defecto = np.zeros_like(cand)
    areas = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= minpx:
            defecto[lbl == i] = 255
            areas.append(float(st[i, cv2.CC_STAT_AREA]))
    sana = sel & (cv2.dilate(defecto, np.ones((5, 5), np.uint8)) == 0)
    if int(sana.sum()) < 50:
        sana = sel
    Ls, as_, bs = L[sana], a[sana], b[sana]
    hist, _ = np.histogram(Ls, bins=32, range=(0, 100))
    pr = hist / max(hist.sum(), 1)
    pr = pr[pr > 0]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx, gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return dict(
        L_media=float(Ls.mean()), a_media=float(as_.mean()), b_media=float(bs.mean()),
        L_std=float(Ls.std()), a_std=float(as_.std()), b_std=float(bs.std()),
        L_p10=float(np.percentile(Ls, 10)), L_p90=float(np.percentile(Ls, 90)),
        entropia_L=float(-(pr * np.log2(pr)).sum()),
        defecto_ratio=float(sum(areas) / area_core) if area_core else 0.0,
        n_manchas=len(areas),
        mancha_max_ratio=float(max(areas) / area_core) if areas and area_core else 0.0,
        lap_var=float(cv2.Laplacian(gray, cv2.CV_64F)[sana].var()),
        grad_media=float(np.hypot(gx, gy)[sana].mean()),
    )


def features_vista(img_bgr: np.ndarray, mask=None, mm_por_px=None, p: dict = PARAMS) -> dict:
    """Todas las features de una vista. `mask` (0/255) permite usar la segmentación del modelo
    profundo; si es None se usa la segmentación clásica de respaldo."""
    t0 = time.perf_counter()
    lab = a_lab(img_bgr)
    fondo = _fondo(lab, p["borde"])
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out = dict(seg_ok=False, brillo_fondo=float(fondo[0]),
               nitidez=float(cv2.Laplacian(gris, cv2.CV_64F).var()))
    if mask is None:
        mask = segmentar_clasico(lab, fondo, p)
    if mask is not None and int((mask > 0).sum()) >= p["min_frac_area"] * mask.size:
        geo = features_geometricas(mask)
        out.update(geo)
        out.update(features_color_defecto_textura(img_bgr, lab, mask, geo["diam_equiv_px"], p))
        if mm_por_px and np.isfinite(mm_por_px):
            out["eje_menor_mm"] = geo["eje_menor_px"] * mm_por_px
            out["eje_mayor_mm"] = geo["eje_mayor_px"] * mm_por_px
        out["seg_ok"] = True
    out["t_ms"] = (time.perf_counter() - t0) * 1000.0
    return out
