"""
generar_datos_demo.py
---------------------
Genera un dataset SINTÉTICO (elipses que imitan limones) solo para probar el pipeline y el
notebook de EDA antes de tener datos reales. NO sirve para entrenar ni para sacar conclusiones.

Incluye, a propósito, problemas típicos que el EDA debe detectar:
  - clases desbalanceadas (Tercera y Fábrica con pocos frutos)
  - algunos frutos con vistas faltantes
  - 2 etiquetas inconsistentes con el calibre
  - imágenes borrosas y una imagen duplicada
  - el lote L3 con iluminación más brillante (riesgo de atajo)

Uso:  python scripts/generar_datos_demo.py            (escribe en data/demo/raw)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VISTAS = ["0", "90", "180", "270", "pedunculo"]
PX_POR_MM = 5.5


def dibujar(calibre_mm, color, defecto, brillo, rng, vista):
    img = np.full((480, 640, 3), 215, np.uint8)
    img = np.clip(img.astype(float) * brillo, 0, 255).astype(np.uint8)
    d = calibre_mm * PX_POR_MM * (1 + rng.normal(0, .02))
    ejes = (int(d / 2), int(d / 2 * (0.82 if vista != "pedunculo" else 0.95)))
    bgr = {"verde": (60, 170, 90), "verde_amarillo": (60, 200, 170), "amarillo": (40, 220, 235)}[color]
    cv2.ellipse(img, (320, 240), ejes, int(rng.integers(0, 30)), 0, 360, bgr, -1)
    if defecto > 0:
        for _ in range(int(2 + 6 * defecto)):
            c = (320 + int(rng.normal(0, ejes[0] * .4)), 240 + int(rng.normal(0, ejes[1] * .4)))
            cv2.circle(img, c, int(6 + 18 * defecto), (30, 50, 90), -1)
    ruido = rng.normal(0, 4, img.shape)
    return np.clip(img + ruido, 0, 255).astype(np.uint8)


def main(salida: Path, n: int, semilla: int):
    rng = np.random.default_rng(semilla)
    (salida / "imagenes").mkdir(parents=True, exist_ok=True)
    proporciones = {"Extra": .25, "Primera": .27, "Segunda": .23, "Tercera": .13, "Fabrica": .12}
    rangos = {"Extra": (39.2, 45), "Primera": (37.1, 38.9), "Segunda": (35.1, 36.9), "Tercera": (29, 34.9)}
    lotes, fechas = ["L1", "L2", "L3"], ["20260901", "20260905", "20260910", "20260915"]
    brillo_lote = {"L1": 1.0, "L2": 0.97, "L3": 1.12}

    filas, k = [], 0
    for cat, p in proporciones.items():
        for _ in range(int(round(n * p))):
            k += 1
            idf = f"F{k:04d}"
            lote, fecha = rng.choice(lotes), rng.choice(fechas)
            if cat == "Fabrica":
                calibre = rng.uniform(32, 44)
                defecto, color = (rng.uniform(.5, 1), "verde") if rng.random() < .6 else (rng.uniform(0, .1), "amarillo")
            else:
                calibre = rng.uniform(*rangos[cat])
                defecto, color = rng.uniform(0, .12), rng.choice(["verde", "verde_amarillo"], p=[.7, .3])
            filas.append(dict(id_fruto=idf, lote=lote, fecha=fecha, calibre_mm_ref=round(calibre, 1),
                              categoria_ref=cat, color_panel=color, severidad_defecto_panel=round(defecto, 2)))
    et = pd.DataFrame(filas)
    # 2 etiquetas inconsistentes a propósito
    for i in et[et.categoria_ref == "Primera"].index[:2]:
        et.loc[i, "calibre_mm_ref"] = 33.0
    et = et.sample(frac=1, random_state=semilla).reset_index(drop=True)
    et.to_csv(salida / "etiquetas.csv", index=False)

    faltantes = set(et.id_fruto.sample(4, random_state=1))
    borrosos = set(et.id_fruto.sample(4, random_state=2))
    primera_ruta = None
    for _, r in et.iterrows():
        for v in VISTAS:
            if r.id_fruto in faltantes and v == "180":
                continue
            img = dibujar(r.calibre_mm_ref, r.color_panel, r.severidad_defecto_panel,
                          brillo_lote[r.lote], rng, v)
            if r.id_fruto in borrosos and v == "90":
                img = cv2.GaussianBlur(img, (0, 0), 9)
            ruta = salida / "imagenes" / f"{r.lote}_{r.fecha}_{r.id_fruto}_{v}.jpg"
            cv2.imwrite(str(ruta), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
            primera_ruta = primera_ruta or ruta
    # 1 imagen duplicada (copia exacta de otra, bajo otro fruto)
    ult = et.iloc[-1]
    dup = salida / "imagenes" / f"{ult.lote}_{ult.fecha}_{ult.id_fruto}_0.jpg"
    dup.write_bytes(primera_ruta.read_bytes())
    print(f"Demo generado en {salida}: {len(et)} frutos, {len(list((salida/'imagenes').glob('*.jpg')))} imágenes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=ROOT / "data" / "demo" / "raw")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    main(a.salida, a.n, a.semilla)
