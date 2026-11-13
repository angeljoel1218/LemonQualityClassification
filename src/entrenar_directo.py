"""
entrenar_directo.py
--------------------
Entrenamiento y evaluación del clasificador DIRECTO (ResNet-50, 5 vistas, promedio de logits; src/modelo_directo.py), con el mismo
protocolo que el sistema híbrido (Plan de tesis v4, secciones 4.7.1 y 7.6):

    * Mismas 5 imágenes por fruto, mismas particiones y mismo presupuesto de ajuste: hasta 8 configuraciones (búsqueda aleatoria,
      1 semilla c/u), selección por F1 macro en VALIDACIÓN y 3 semillas para la configuración elegida.
    * AdamW, decaimiento de la tasa de aprendizaje por coseno, parada temprana sobre la pérdida de validación (paciencia 5).
    * Para el contraste confirmatorio (H3) se usa el modelo de la semilla con F1 macro MEDIANO en validación, elegido ANTES de
      evaluar la prueba; las demás semillas se reportan solo como variabilidad del entrenamiento.
    * No recibe el calibre medido con vernier ni etiquetas intermedias del panel.
    * Ablación E4 (balanceo, solo con validación): ninguno | pesos | focal | aumento (dirigido a Tercera y Fábrica) | combinacion.
    * Normalización de entrada: media/desviación de ImageNet (constantes del backbone preentrenado; NO se ajustan con tus datos).

Salida: logs/directo/<nombre>/{trials.csv, pred_val_seed<k>.csv, pred_val.csv, resumen.json, mejor.pt} y, con `predecir --subset test`
(una sola vez, con modelos congelados), pred_test.csv registrado con su huella SHA-256.

Requiere GPU y torch/torchvision (se importan al entrenar). Las funciones de selección (búsqueda, semilla mediana, pesos) son
numpy y están probadas con pytest.

Uso:
    python src/entrenar_directo.py buscar   --nombre directo_ninguno --balanceo ninguno
    python src/entrenar_directo.py buscar   --nombre directo_focal   --balanceo focal
    python src/entrenar_directo.py predecir --nombre directo_ninguno --subset val
    python src/entrenar_directo.py predecir --nombre directo_ninguno --subset test      # una sola vez (modelos congelados)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

DIR_DIRECTO = ROOT / "logs" / "directo"
BALANCEOS = ["ninguno", "pesos", "focal", "aumento", "combinacion"]
MINORITARIAS = ["Tercera", "Fabrica"]
MEDIA, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
DIR = C.CFG["directo"]
BUS = C.CFG["busqueda"]


# =====================================================================================================================
# lógica pura (sin torch)
# =====================================================================================================================
def espacio_directo(n: int, balanceo: str = "ninguno", semilla: int | None = None) -> list:
    """`n` configuraciones por búsqueda aleatoria (mismo presupuesto para TODAS las variantes de balanceo). Los hiperparámetros de base
    (lr, weight decay, batch, paciencia) salen de un generador PROPIO y son IDÉNTICOS en todas las variantes: así E4 cambia solo el
    balanceo (comparación justa). Los parámetros del balanceo salen de otro generador."""
    semilla = BUS["semilla_busqueda"] if semilla is None else semilla
    rng = np.random.default_rng(semilla)
    rng_b = np.random.default_rng(semilla + 7919)
    out = []
    for _ in range(n):
        hp = dict(lr=float(10 ** rng.uniform(-5, -3)), weight_decay=float(10 ** rng.uniform(-5, -1)), batch_size=int(rng.choice([8, 16])),
                  paciencia=int(rng.choice([5, 8])), epocas=50, balanceo=balanceo, gamma=0.0, alpha=None, aumento="ninguno", pesos=False)
        aum_base = str(rng.choice(["ninguno", "geometrico"]))                # aumento de base, el mismo en todas las variantes
        g, a = float(rng_b.choice([1.0, 2.0, 3.0])), float(rng_b.choice([0.25, 0.5, 0.75]))
        hp["aumento"] = aum_base
        if balanceo in ("pesos", "combinacion"):
            hp["pesos"] = True
        if balanceo in ("focal", "combinacion"):
            hp["gamma"], hp["alpha"] = g, a
        if balanceo in ("aumento", "combinacion"):
            hp["aumento"] = "dirigido"
        out.append(hp)
    return out


def pesos_de_clase(y, clases=None) -> np.ndarray:
    """Pesos inversamente proporcionales a la frecuencia en TRAIN (normalizados a media 1)."""
    clases = list(clases or C.CATEGORIAS)
    n = np.array([(np.asarray(y) == c).sum() for c in clases], float)
    w = np.where(n > 0, n.sum() / (len(clases) * np.maximum(n, 1)), 0.0)
    return (w / w[w > 0].mean()).astype(np.float32)


def elegir_mejor_config(trials: pd.DataFrame) -> int:
    """Índice de la configuración con mayor F1 macro de validación (desempate: menor pérdida de validación)."""
    t = trials.sort_values(["val_f1_macro", "val_loss"], ascending=[False, True])
    return int(t.index[0])


def elegir_semilla_mediana(res: pd.DataFrame) -> int:
    """Semilla cuyo F1 macro de validación es el MEDIANO (con número par de semillas, la inferior de las dos centrales)."""
    r = res.sort_values("val_f1_macro").reset_index(drop=True)
    return int(r.loc[(len(r) - 1) // 2, "semilla"])


# =====================================================================================================================
# datos
# =====================================================================================================================
def tabla_frutos(subset: str, procesados: Path = ROOT / "data" / "processed", raw: Path = ROOT / "data" / "raw") -> pd.DataFrame:
    """id_fruto, y, lote, fecha, rutas (5, en el orden de C.VISTAS) de los frutos COMPLETOS del subconjunto."""
    from src.eda import normalizar_categorias
    from src.ingesta import leer_csv
    from src.particion import cargar_particion
    part = cargar_particion(procesados)
    idx = pd.read_csv(procesados / "indice_imagenes.csv", dtype=str)
    et = leer_csv(raw / "etiquetas.csv")[["id_fruto", "categoria_ref"]]
    et["y"] = normalizar_categorias(et["categoria_ref"])
    ids = part[part["subset"] == subset]["id_fruto"]
    filas = []
    for fid, g in idx[idx["id_fruto"].isin(ids)].groupby("id_fruto"):
        r = g.set_index("vista")["ruta"].to_dict()
        if all(v in r for v in C.VISTAS):
            filas.append(dict(id_fruto=fid, lote=g["lote"].iloc[0], fecha=g["fecha"].iloc[0], rutas=[r[v] for v in C.VISTAS]))
    return pd.DataFrame(filas).merge(et[["id_fruto", "y"]], on="id_fruto")


def a_cuadrado(img: np.ndarray, lado: int) -> np.ndarray:
    """Relleno a cuadrado (sin deformar el fruto) y redimensionado."""
    h, w = img.shape[:2]
    m = max(h, w)
    out = np.full((m, m, 3), np.median(np.concatenate([img[0], img[-1]]), axis=0), np.uint8)
    out[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img
    return cv2.resize(out, (lado, lado), interpolation=cv2.INTER_AREA)


def _aumentar(img: np.ndarray, color: bool, rng) -> np.ndarray:
    if rng.random() < 0.5:
        img = img[:, ::-1]
    ang = rng.uniform(-10, 10)
    h, w = img.shape[:2]
    img = cv2.warpAffine(np.ascontiguousarray(img), cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0), (w, h), borderMode=cv2.BORDER_REPLICATE)
    if color:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.92, 1.08) + rng.uniform(-8, 8), 0, 255).astype(np.uint8)
    return img


def cargar_vistas(rutas: list, lado: int, preproc: str, perturbacion: str, lote: str, fecha: str, aumento: str, es_minoritaria: bool, rng,
                  sesiones=None, camara=None) -> np.ndarray:
    """(5, 3, lado, lado) float32 normalizado. Perturbación (E5) -> preprocesamiento (E3) -> aumento (solo train)."""
    from src import preprocesamiento as pp
    out = []
    for r in rutas:
        ruta = Path(r)
        img = cv2.imread(str(ruta if ruta.is_absolute() else ROOT / ruta))
        img = pp.aplicar_perturbacion(img, perturbacion)
        img = pp.aplicar(img, preproc, (lote, fecha), sesiones, camara)
        img = a_cuadrado(img, lado)
        if aumento in ("geometrico", "geometrico_color_suave") or (aumento == "dirigido" and es_minoritaria):
            img = _aumentar(img, aumento == "geometrico_color_suave" or aumento == "dirigido", rng)
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEDIA) / STD
        out.append(x.transpose(2, 0, 1))
    return np.stack(out)


# =====================================================================================================================
# entrenamiento / predicción (torch)
# =====================================================================================================================
def _dispositivo(device):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _loader(df, hp, entrenamiento: bool, preproc: str, perturbacion: str, semilla: int, workers: int = 2):
    import torch
    from src import preprocesamiento as pp
    sesiones, camara = pp.cargar_sesiones(), pp.cargar_camara()
    clases = list(C.CATEGORIAS)
    lado = int(DIR["imgsz"])

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            r = df.iloc[i]
            rng = np.random.default_rng(semilla * 100003 + i + (int(time.time() * 1000) % 997 if entrenamiento else 0))
            aum = hp["aumento"] if entrenamiento else "ninguno"
            x = cargar_vistas(r["rutas"], lado, preproc, perturbacion, r["lote"], r["fecha"], aum, r["y"] in MINORITARIAS, rng, sesiones, camara)
            return torch.from_numpy(x), clases.index(r["y"]), i

    return torch.utils.data.DataLoader(DS(), batch_size=hp["batch_size"], shuffle=entrenamiento, num_workers=workers, drop_last=False)


def entrenar(hp: dict, semilla: int, nombre: str, procesados: Path = ROOT / "data" / "processed", preproc: str = "ninguno",
             device=None, workers: int = 2) -> dict:
    """Entrena UNA configuración con UNA semilla. Devuelve val_f1_macro, val_loss, curvas y la ruta de los pesos."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score
    from src.modelo_directo import DirectoCincoVistas, FocalLoss
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    dev = _dispositivo(device)
    tr, va = tabla_frutos("train", procesados), tabla_frutos("val", procesados)
    clases = list(C.CATEGORIAS)
    w = torch.tensor(pesos_de_clase(tr["y"], clases)) if hp["pesos"] else None
    crit = FocalLoss(hp["gamma"], hp["alpha"], w.to(dev) if w is not None else None).to(dev) if (hp["gamma"] > 0) else \
        torch.nn.CrossEntropyLoss(weight=w.to(dev) if w is not None else None)
    modelo = DirectoCincoVistas().to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epocas"])
    dl_tr, dl_va = _loader(tr, hp, True, preproc, "ninguna", semilla, workers), _loader(va, hp, False, preproc, "ninguna", semilla, workers)
    carpeta = DIR_DIRECTO / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    mejor, sin_mejora, curvas, t0 = np.inf, 0, [], time.perf_counter()
    pesos = carpeta / f"pesos_s{semilla}.pt"
    for ep in range(hp["epocas"]):
        modelo.train()
        lt, n = 0.0, 0
        for x, y, _ in dl_tr:
            x, y = x.to(dev), y.to(dev)
            loss = crit(modelo(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lt += float(loss) * len(y)
            n += len(y)
        sch.step()
        modelo.eval()
        lv, nv, yt, yp = 0.0, 0, [], []
        with torch.no_grad():
            for x, y, _ in dl_va:
                lo = modelo(x.to(dev))
                lv += float(F.cross_entropy(lo, y.to(dev), reduction="sum"))        # pérdida de validación SIN pesos (comparable)
                nv += len(y)
                yt += y.tolist()
                yp += lo.argmax(1).cpu().tolist()
        lv /= max(nv, 1)
        curvas.append(dict(epoca=ep + 1, loss_train=lt / max(n, 1), loss_val=lv, f1_val=float(f1_score(yt, yp, average="macro", zero_division=0))))
        if lv < mejor - 1e-4:
            mejor, sin_mejora = lv, 0
            torch.save(modelo.state_dict(), pesos)
        else:
            sin_mejora += 1
            if sin_mejora >= hp["paciencia"]:
                break
    modelo.load_state_dict(torch.load(pesos, map_location=dev))
    pv = predecir(pesos, "val", procesados, preproc, "ninguna", device=dev, modelo=modelo, hp=hp, salida=carpeta / f"pred_val_seed{semilla}.csv")
    f1 = float(f1_score(pv["y"], pv["pred"], average="macro", zero_division=0))
    return dict(semilla=semilla, val_f1_macro=f1, val_loss=float(mejor), epocas=len(curvas), tiempo_s=time.perf_counter() - t0,
                pesos=str(pesos), curvas=curvas, hp=hp)


def predecir(pesos, subset: str, procesados: Path = ROOT / "data" / "processed", preproc: str = "ninguno", perturbacion: str = "ninguna",
             device=None, modelo=None, hp: dict | None = None, salida: Path | None = None, workers: int = 2) -> pd.DataFrame:
    """Predicciones por fruto (id_fruto, y, pred, logit_<cat>, t_ms). Con subset='test' exige modelos congelados y se genera UNA vez."""
    import torch
    from src import protocolo
    from src.modelo_directo import DirectoCincoVistas
    if subset == "test":
        protocolo.exigir_listo_para_prueba()
    dev = _dispositivo(device)
    if modelo is None:
        modelo = DirectoCincoVistas(preentrenado=False).to(dev)
        modelo.load_state_dict(torch.load(pesos, map_location=dev))
    modelo.eval()
    df = tabla_frutos(subset, procesados)
    hp = hp or dict(batch_size=8, aumento="ninguno")
    dl = _loader(df, hp, False, preproc, perturbacion, 0, workers)
    clases, filas = list(C.CATEGORIAS), []
    with torch.no_grad():
        for x, y, i in dl:
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            lo = modelo(x.to(dev)).cpu().numpy()
            if dev == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / len(i)
            for k, j in enumerate(i.tolist()):
                filas.append(dict(id_fruto=df.iloc[j]["id_fruto"], y=df.iloc[j]["y"], pred=clases[int(lo[k].argmax())], t_ms=ms,
                                  **{f"logit_{c}": float(lo[k][m]) for m, c in enumerate(clases)}))
    out = pd.DataFrame(filas)
    if salida is not None:
        out.to_csv(salida, index=False)
        if subset == "test":
            protocolo.registrar_artefacto(f"directo_{Path(salida).parent.name}_test", salida, dict(subset="test"))
    return out


def buscar(nombre: str, balanceo: str, n_configs: int, semillas: list, procesados: Path = ROOT / "data" / "processed",
           preproc: str = "ninguno", device=None, workers: int = 2) -> dict:
    """Búsqueda aleatoria (1 semilla por configuración) + 3 semillas para la elegida + semilla mediana. Todo con VALIDACIÓN."""
    hps = espacio_directo(n_configs, balanceo)
    carpeta = DIR_DIRECTO / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    filas, curvas_cfg = [], {}
    for i, hp in enumerate(hps):
        r = entrenar(hp, semillas[0], f"{nombre}/cfg{i}", procesados, preproc, device, workers)
        filas.append(dict(cfg=i, val_f1_macro=r["val_f1_macro"], val_loss=r["val_loss"], epocas=r["epocas"], tiempo_s=r["tiempo_s"],
                          **{(("epocas_max" if k == "epocas" else k)): v for k, v in hp.items() if k != "balanceo"}))
        curvas_cfg[i] = r["curvas"]
        pd.DataFrame(filas).to_csv(carpeta / "trials.csv", index=False)
    trials = pd.DataFrame(filas)
    k = elegir_mejor_config(trials)
    hp = hps[k]
    res = [entrenar(hp, s, nombre, procesados, preproc, device, workers) for s in semillas]
    df_s = pd.DataFrame([{k_: v for k_, v in r.items() if k_ not in ("curvas", "hp")} for r in res])
    semilla_med = elegir_semilla_mediana(df_s)
    elegido = next(r for r in res if r["semilla"] == semilla_med)
    pv = pd.read_csv(carpeta / f"pred_val_seed{semilla_med}.csv")
    pv.to_csv(carpeta / "pred_val.csv", index=False)
    resumen = dict(nombre=nombre, balanceo=balanceo, preproc=preproc, n_configs=n_configs, config_elegida=k, hp=hp,
                   semillas=df_s.to_dict("records"), semilla_mediana=semilla_med, pesos_elegidos=elegido["pesos"],
                   curvas_semilla_mediana=elegido["curvas"], val_f1_macro_media=float(df_s["val_f1_macro"].mean()),
                   val_f1_macro_std=float(df_s["val_f1_macro"].std(ddof=1)) if len(df_s) > 1 else 0.0,
                   tiempo_total_s=float(df_s["tiempo_s"].sum() + trials["tiempo_s"].sum()), fecha=pd.Timestamp.now().isoformat(timespec="seconds"))
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return resumen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["buscar", "predecir"])
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--balanceo", choices=BALANCEOS, default="ninguno")
    ap.add_argument("--configs", type=int, default=BUS["configs_clasificador_directo"])
    ap.add_argument("--subset", choices=["train", "val", "test"], default="val")
    ap.add_argument("--preproc", default="ninguno")
    ap.add_argument("--perturbacion", default="ninguna")
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.accion == "buscar":
        r = buscar(a.nombre, a.balanceo, a.configs, BUS["semillas_entrenamiento"], a.procesados, a.preproc, a.device, a.workers)
        print(f"Elegida cfg {r['config_elegida']}; F1 macro val {r['val_f1_macro_media']:.3f} ± {r['val_f1_macro_std']:.3f}; semilla mediana {r['semilla_mediana']}")
    else:
        res = json.loads((DIR_DIRECTO / a.nombre / "resumen.json").read_text(encoding="utf-8"))
        salida = DIR_DIRECTO / a.nombre / (f"pred_{a.subset}.csv" if a.perturbacion == "ninguna" else f"pred_{a.subset}_{a.perturbacion}.csv")
        p = predecir(res["pesos_elegidos"], a.subset, a.procesados, a.preproc, a.perturbacion, a.device, hp=res["hp"], salida=salida, workers=a.workers)
        print(f"{len(p)} frutos -> {salida}")


if __name__ == "__main__":
    main()
