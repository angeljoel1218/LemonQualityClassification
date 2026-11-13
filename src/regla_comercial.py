"""
regla_comercial.py
-------------------
Regla jerárquica (determinista, no aprendida) del Plan de tesis v4 (secciones 4.5 y 4.6.2, Tabla 4.2) que asigna la
categoría comercial final a partir de los atributos AGREGADOS de un fruto (atributos.py).

Orden de evaluación (de mayor a menor prioridad):
    1. ¿Severidad S >= theta_def en AL MENOS UNA vista?                      -> Fabrica   ("basta una vista")
    2. ¿Proporción de superficie sana amarilla P_am >= p_am (sobremadurez)?  -> Fabrica
    3. d_est > 42,0 mm                                                       -> Fuera_de_alcance (se registra y se excluye)
    4. d_est < 35,0 mm                                                       -> Tercera
       35,0 <= d_est < 37,0 mm                                               -> Segunda
       37,0 <= d_est < 39,0 mm                                               -> Primera
       39,0 <= d_est <= 42,0 mm                                              -> Extra

Operacionalización (Tabla 4.2):
    * Extensión del defecto A_def: área de la UNIÓN de las máscaras de defecto de la vista (sin doble conteo) dividida por
      el área visible del fruto; el valor del fruto es el máximo entre vistas.
    * Nivel por área: 0 si A < a1; 1 si a1 <= A < a2; 2 si a2 <= A < a3; 3 si A >= a3 (a1 < a2 < a3 se fijan en el piloto).
    * Severidad S (0-3): máximo entre el nivel por área y el nivel mínimo que la rúbrica asigna al TIPO de defecto, por vista;
      el fruto toma el máximo entre vistas ("un tipo grave eleva la severidad aunque su área sea pequeña").
    * Color: P_am = proporción de píxeles de la superficie SANA con matiz CIELAB h*_ab <= h_am (tras normalizar con la carta).
      L*, a* y b* medios se reportan solo como descriptivos.

Los valores de a1, a2, a3, theta_def, h_am y p_am vienen de `configs/proyecto.yaml` (bloque `rubrica`, provisional) y se fijan
con el panel en el PILOTO; después de iniciada la recolección definitiva no se modifican (ver `congelar_umbrales`).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

CALIBRE_EXTRA = C.CALIBRE_EXTRA          # 39,0
CALIBRE_PRIMERA = C.CALIBRE_PRIMERA      # 37,0
CALIBRE_SEGUNDA = C.CALIBRE_SEGUNDA      # 35,0
CALIBRE_MAXIMO = C.CALIBRE_MAXIMO        # 42,0
CATEGORIAS = list(C.CATEGORIAS)
FUERA = "Fuera_de_alcance"
RUTA_UMBRALES = C.ROOT / "configs" / "umbrales_congelados.json"


@dataclass
class UmbralesRegla:
    """Umbrales de la regla (Tabla 4.2). Los valores por defecto son los de trabajo de `configs/proyecto.yaml`."""
    theta_def: int = int(C.RUBRICA["theta_def"])              # S >= theta_def -> defecto relevante
    h_am: float = float(C.RUBRICA["h_am"])                    # matiz (grados) por debajo del cual un píxel sano es amarillo
    p_am: float = float(C.RUBRICA["p_am"])                    # P_am >= p_am -> sobremadurez
    puntos_corte_area: tuple = tuple(C.RUBRICA["puntos_corte_area"])    # a1 < a2 < a3
    nivel_minimo_por_tipo: dict = field(default_factory=lambda: dict(C.RUBRICA["nivel_minimo_por_tipo"]))

    def validar(self):
        a = list(self.puntos_corte_area)
        if not (len(a) == 3 and a[0] < a[1] < a[2]):
            raise ValueError(f"puntos_corte_area debe cumplir a1 < a2 < a3: {a}")
        if self.theta_def not in (0, 1, 2, 3):
            raise ValueError("theta_def es un nivel ordinal 0-3")
        return self


def hue_cielab(color_lab: tuple) -> float:
    """Matiz h*_ab (grados, 0-360) a partir de (L*, a*, b*)."""
    _, a, b = color_lab
    return math.degrees(math.atan2(b, a)) % 360.0


# ------------------------------------------------------------------------------------ calibre
def categoria_por_calibre(d_mm: float) -> str:
    """Categoría por calibre con los intervalos semiabiertos del plan. NaN -> None; > 42,0 mm -> 'Fuera_de_alcance'."""
    if d_mm is None or not np.isfinite(d_mm):
        return None
    if d_mm > CALIBRE_MAXIMO:
        return FUERA
    if d_mm >= CALIBRE_EXTRA:
        return "Extra"
    if d_mm >= CALIBRE_PRIMERA:
        return "Primera"
    if d_mm >= CALIBRE_SEGUNDA:
        return "Segunda"
    return "Tercera"


def categoria_por_calibre_vec(d_mm) -> np.ndarray:
    d = np.asarray(d_mm, float)
    out = np.where(d >= CALIBRE_EXTRA, "Extra", np.where(d >= CALIBRE_PRIMERA, "Primera",
                   np.where(d >= CALIBRE_SEGUNDA, "Segunda", "Tercera"))).astype(object)
    out[d > CALIBRE_MAXIMO] = FUERA
    out[~np.isfinite(d)] = None
    return out


def indice_calibre(cat) -> float:
    """k = 1 (Tercera) .. 4 (Extra); NaN para Fábrica/otros (no tienen calibre ordenado)."""
    try:
        return float(C.CATEGORIAS_CALIBRE.index(cat) + 1)
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------------------------ severidad (Tabla 4.2)
def nivel_por_area(area, cortes=None):
    """Nivel ordinal 0-3 a partir de la fracción de área afectada (acepta escalares o arrays)."""
    a1, a2, a3 = cortes or C.RUBRICA["puntos_corte_area"]
    a = np.asarray(area, float)
    n = (a >= a1).astype(int) + (a >= a2).astype(int) + (a >= a3).astype(int)
    return n if n.ndim else int(n)


def severidad_vista(area_union, tipos_presentes=(), u: UmbralesRegla | None = None) -> int:
    """S de UNA vista = máx(nivel por área, nivel mínimo del tipo de defecto más grave presente)."""
    u = u or UmbralesRegla()
    a = 0.0 if area_union is None or not np.isfinite(area_union) else float(area_union)
    s = int(nivel_por_area(a, u.puntos_corte_area))
    for t in tipos_presentes or ():
        s = max(s, int(u.nivel_minimo_por_tipo.get(t, 0)))
    return s


def nivel_desde_panel(sev, mapa=None) -> float:
    """Convierte la escala histórica 0-1 de `severidad_defecto_panel` (0; .25; .5; .75; 1) a nivel 0-3. NaN -> NaN.
    Si las etiquetas ya traen `severidad_nivel_panel` (0-3) se usa esa columna directamente."""
    mapa = {float(k): v for k, v in (mapa or C.RUBRICA["severidad_panel_a_nivel"]).items()}
    if sev is None or not np.isfinite(sev):
        return float("nan")
    claves = np.array(sorted(mapa))
    return float(mapa[float(claves[np.abs(claves - float(sev)).argmin()])])


# ------------------------------------------------------------------------------------ regla por fruto
def clasificar(d_est_mm: float, p_amarillo: float, severidad_max: float, umbrales: UmbralesRegla | None = None) -> str:
    """Aplica el árbol jerárquico a UN fruto y devuelve la categoría final."""
    u = umbrales or UmbralesRegla()
    if np.isfinite(severidad_max) and severidad_max >= u.theta_def:
        return "Fabrica"
    if np.isfinite(p_amarillo) and p_amarillo >= u.p_am:
        return "Fabrica"
    return categoria_por_calibre(d_est_mm)


def motivo_fabrica(severidad_max: float, p_amarillo: float, u: UmbralesRegla | None = None) -> str:
    """Explicación trazable de por qué un fruto es Fábrica (la ventaja de la arquitectura híbrida es la explicabilidad)."""
    u = u or UmbralesRegla()
    m = []
    if np.isfinite(severidad_max) and severidad_max >= u.theta_def:
        m.append(f"defecto: S={int(severidad_max)} >= theta_def={u.theta_def}")
    if np.isfinite(p_amarillo) and p_amarillo >= u.p_am:
        m.append(f"sobremadurez: P_am={p_amarillo:.2f} >= p_am={u.p_am:.2f}")
    return "; ".join(m)


def es_fabrica_v4(df: pd.DataFrame, u: UmbralesRegla, col_sev: str = "severidad_max", col_pam: str = "p_amarillo") -> np.ndarray:
    """Pasos 1 y 2 (Fábrica por defecto o por sobremadurez); no necesita el calibre."""
    s = df[col_sev].to_numpy(float)
    p = df[col_pam].to_numpy(float)
    return (np.nan_to_num(s, nan=-1) >= u.theta_def) | (np.nan_to_num(p, nan=-1) >= u.p_am)


def clasificar_df(df: pd.DataFrame, umbrales: UmbralesRegla | None = None, col_calibre: str = "d_est_mm",
                  col_sev: str = "severidad_max", col_pam: str = "p_amarillo") -> pd.Series:
    """Regla completa sobre una tabla de frutos (vectorizada). Si falta el calibre, el fruto no-Fábrica queda sin categoría."""
    u = umbrales or UmbralesRegla()
    fab = es_fabrica_v4(df, u, col_sev, col_pam)
    cat = categoria_por_calibre_vec(df[col_calibre].to_numpy(float))
    cat[fab] = "Fabrica"
    return pd.Series(cat, index=df.index, name="categoria_regla")


# ------------------------------------------------------------------------------------ calibración y congelamiento
def calibrar_umbrales_v4(df: pd.DataFrame, y_fabrica, rejilla_theta=(1, 2, 3), rejilla_p=None,
                         col_sev: str = "severidad_max", col_pam: str = "p_amarillo") -> pd.DataFrame:
    """Rejilla de (theta_def, p_am) y desempeño de la regla para detectar Fábrica (paso 4 de la fase offline, sección 4.7.1).

    `df` debe traer la severidad S del fruto y P_am. Se elige sobre piloto/validación (NUNCA sobre la prueba).
    Devuelve una fila por combinación con recall, precisión y F1 de Fábrica.
    """
    y = np.asarray(y_fabrica, bool)
    rejilla_p = np.arange(0.05, 0.81, 0.05) if rejilla_p is None else rejilla_p
    s = np.nan_to_num(df[col_sev].to_numpy(float), nan=-1)
    pam = np.nan_to_num(df[col_pam].to_numpy(float), nan=-1)
    filas = []
    for td in rejilla_theta:
        for pa in rejilla_p:
            pred = (s >= td) | (pam >= pa)
            tp = int((pred & y).sum())
            rec = tp / max(int(y.sum()), 1)
            pre = tp / max(int(pred.sum()), 1)
            f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
            filas.append(dict(theta_def=int(td), p_am=float(round(pa, 4)), recall=rec, precision=pre, f1=f1, n_pred=int(pred.sum())))
    return pd.DataFrame(filas)


def elegir_umbrales(grilla: pd.DataFrame, recall_min: float | None = None) -> dict:
    """Mejor fila por F1 (con recall mínimo opcional). Desempate: menor theta_def y menor p_am (regla más simple)."""
    g = grilla if recall_min is None else grilla[grilla["recall"] >= recall_min]
    if g.empty:
        g = grilla
    return g.sort_values(["f1", "theta_def", "p_am"], ascending=[False, True, True]).iloc[0].to_dict()


def congelar_umbrales(u: UmbralesRegla, ruta: Path = RUTA_UMBRALES, origen: str = "piloto", forzar: bool = False) -> dict:
    """Guarda los umbrales con fecha y huella. Después de congelarlos NO se pueden cambiar sin `forzar=True`
    (la regla se fija antes del entrenamiento y no cambia con los resultados parciales; sección 4.5)."""
    ruta = Path(ruta)
    if ruta.exists() and not forzar:
        raise SystemExit(f"Los umbrales ya están congelados en {ruta}. Cambiarlos invalida las ablaciones; "
                         "use forzar=True solo si lo justifica.")
    d = dict(asdict(u), origen=origen, fecha=datetime.now().isoformat(timespec="seconds"))
    d["huella"] = hashlib.sha256(json.dumps({k: v for k, v in d.items() if k != "fecha"}, sort_keys=True,
                                            default=str).encode()).hexdigest()[:12]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def cargar_umbrales(ruta: Path = RUTA_UMBRALES, exigir_congelados: bool = False) -> UmbralesRegla:
    """Umbrales congelados si existen; si no, los de trabajo (provisionales) del YAML."""
    ruta = Path(ruta)
    if ruta.exists():
        d = json.loads(ruta.read_text(encoding="utf-8"))
        return UmbralesRegla(theta_def=int(d["theta_def"]), h_am=float(d["h_am"]), p_am=float(d["p_am"]),
                             puntos_corte_area=tuple(d["puntos_corte_area"]),
                             nivel_minimo_por_tipo=dict(d["nivel_minimo_por_tipo"])).validar()
    if exigir_congelados:
        raise SystemExit("Faltan los umbrales congelados (configs/umbrales_congelados.json): fíjelos con el piloto "
                         "(python src/piloto.py --congelar) antes de evaluar la prueba.")
    return UmbralesRegla().validar()


# ------------------------------------------------------------------------------------ compatibilidad (features clásicas)
def es_fabrica_df(df: pd.DataFrame, theta_def: float, theta_col: float, col_defecto: str = "defecto_max",
                  col_a: str = "a_media", col_b: str = "b_media") -> np.ndarray:
    """Versión continua de los pasos 1-2 usada por el baseline clásico y los notebooks (fracción de superficie con defecto
    y matiz medio CIELAB). Se conserva para el análisis exploratorio; la regla del plan v4 es `clasificar_df`."""
    hue = np.degrees(np.arctan2(df[col_b].to_numpy(float), df[col_a].to_numpy(float))) % 360.0
    d = df[col_defecto].to_numpy(float)
    return (d >= theta_def) | (hue < theta_col)


def calibrar_umbrales(df: pd.DataFrame, y_fabrica, rejilla_def=None, rejilla_col=None,
                      col_defecto: str = "defecto_max") -> pd.DataFrame:
    """Rejilla continua de (theta_def, theta_col) para el baseline clásico (compatibilidad con los notebooks)."""
    y = np.asarray(y_fabrica, bool)
    rejilla_def = np.arange(0.00, 0.31, 0.01) if rejilla_def is None else rejilla_def
    rejilla_col = np.arange(80.0, 131.0, 2.5) if rejilla_col is None else rejilla_col
    filas = []
    for td in rejilla_def:
        for tc in rejilla_col:
            p = es_fabrica_df(df, td, tc, col_defecto)
            tp = int((p & y).sum())
            rec = tp / max(int(y.sum()), 1)
            pre = tp / max(int(p.sum()), 1)
            f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
            filas.append(dict(theta_def=float(td), theta_col=float(tc), recall=rec, precision=pre, f1=f1, n_pred=int(p.sum())))
    return pd.DataFrame(filas)
