"""
particion.py
------------
Partición train / val / test CONGELADA del Plan de tesis v4 (secciones 4.4 y 4.5; Tabla 4.1).

Reglas ("divide primero, transforma después"):
  * Unidad de partición: el FRUTO. Las 5 vistas de un fruto van siempre al mismo subconjunto (partición por grupos).
  * Asignación por cuotas (no proporcional 70/15/15): 880 train / 220 val / 400 test, y la PRUEBA está BALANCEADA (80 por
    categoría): Extra, Primera, Segunda 192/48/80 y Tercera, Fábrica 152/38/80 (de 320/320/320/270/270 frutos válidos).
  * Estratificada por categoría y por sesión (lote x fecha) para repartir lotes y fechas entre los subconjuntos.
  * Lote reservado (análisis secundario): si hay disponibilidad se reserva un lote completo para la prueba, con cuotas por
    categoría que no excedan las de la prueba (80); ninguno de sus frutos entra a train ni a val. Si no hay un lote
    completo se usa una fecha de captura completa y, si no es posible, el análisis se omite y se declara como limitación.
  * Los frutos del PILOTO (columna `piloto` = 1 en etiquetas.csv) y los de calibre > 42,0 mm NO entran a la muestra definitiva.
  * Si todavía no se alcanzaron las cuotas (p. ej. se está recolectando), la partición se calcula con las MISMAS proporciones
    de la Tabla 4.1 aplicadas a los frutos disponibles y queda marcada como PROVISIONAL en `particion_meta.json`. La prueba
    sigue balanceada (el mismo número de frutos por categoría). Con los datos completos, vuelva a generarla (--forzar)
    ANTES de entrenar: una partición provisional no sirve para la evaluación confirmatoria.
  * Determinista (semilla fija) y se guarda UNA SOLA VEZ en data/processed/particion.csv (+ particion_meta.json).

Uso:
    python src/particion.py
    python src/particion.py --lote-reservado auto        # (por defecto) busca un lote/fecha elegible
    python src/particion.py --lote-reservado L3 | --lote-reservado ninguno
    python src/particion.py --procesados data/demo/processed --forzar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

SEMILLA_PARTICION = int(C.CFG["semilla_particion"])
PROPORCIONES = (0.70, 0.15, 0.15)      # solo para el modo legado (proporciones explícitas)
ORDEN = ("test", "val", "train")


def _normalizar(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")):
        t = t.str.replace(a, b, regex=False)
    return t.str.capitalize()


# ------------------------------------------------------------------------------------ objetivos por categoría
def objetivos_por_categoria(n_disp: dict) -> tuple[dict, bool]:
    """{cat: {train, val, test}} y bandera `provisional`. Con las cuotas completas usa los valores de la Tabla 4.1; si no, aplica
    las mismas proporciones a los frutos disponibles y fija la MISMA cantidad de prueba por categoría (prueba balanceada)."""
    cat = [c for c in C.CATEGORIAS if n_disp.get(c, 0) > 0]
    completas = all(n_disp.get(c, 0) >= C.CUOTAS_VALIDAS[c] for c in C.CATEGORIAS)
    if completas:
        return {c: dict(train=int(C.CUOTAS[c]["train"]) + (n_disp[c] - C.CUOTAS_VALIDAS[c]),   # excedente -> train
                        val=int(C.CUOTAS[c]["val"]), test=int(C.CUOTAS[c]["test"])) for c in C.CATEGORIAS}, False
    # proporcional: prueba balanceada = mínimo entre categorías de la fracción de prueba (tope 80)
    ratio = {c: {k: C.CUOTAS[c][k] / C.CUOTAS_VALIDAS[c] for k in ("train", "val", "test")} for c in C.CATEGORIAS}
    m_test = min(int(round(n_disp[c] * ratio[c]["test"])) for c in cat) if cat else 0
    m_test = max(1, min(m_test, max(C.CUOTAS[c]["test"] for c in C.CATEGORIAS)))
    obj = {}
    for c in cat:
        n = n_disp[c]
        t = min(m_test, max(n - 2, 0)) if n >= 3 else 0
        v = int(round(n * ratio[c]["val"]))
        v = max(v, 1) if n >= 3 else 0
        v = min(v, max(n - t - 1, 0))
        obj[c] = dict(train=n - t - v, val=v, test=t)
    return obj, True


# ------------------------------------------------------------------------------------ reparto proporcional por sesión
def _reparto(sizes: dict, objetivo: dict, rng) -> dict:
    """Reparte los frutos de cada estrato (sesión) entre subconjuntos de modo que cada subconjunto reciba EXACTAMENTE su cuota
    y cada estrato aporte en proporción a su tamaño (método del mayor resto)."""
    restante = dict(sizes)
    out = {s: {k: 0 for k in ORDEN} for s in sizes}
    for k in ORDEN:
        T = int(objetivo.get(k, 0))
        total = sum(restante.values())
        if k == ORDEN[-1]:
            for s in sizes:
                out[s][k] = restante[s]
            break
        if total == 0 or T == 0:
            continue
        ideal = {s: restante[s] * T / total for s in sizes}
        base = {s: min(int(math.floor(ideal[s])), restante[s]) for s in sizes}
        faltan = T - sum(base.values())
        orden = sorted(sizes, key=lambda s: (-(ideal[s] - base[s]), rng.random()))
        vueltas = 0
        while faltan > 0 and vueltas < 10 * len(sizes) + T:
            s = orden[vueltas % len(orden)]
            if base[s] < restante[s]:
                base[s] += 1
                faltan -= 1
            vueltas += 1
        for s in sizes:
            out[s][k] = base[s]
            restante[s] -= base[s]
    return out


# ------------------------------------------------------------------------------------ lote reservado
def _conteos(d: pd.DataFrame, col: str, valor) -> dict:
    return d[d[col].astype(str) == str(valor)]["cat"].value_counts().to_dict()


def reservado_elegible(d: pd.DataFrame, col: str, valor, obj: dict) -> tuple[bool, str]:
    """¿Se puede reservar `col`=`valor` (lote o fecha) para la prueba sin vaciar train/val de ninguna categoría?"""
    R = _conteos(d, col, valor)
    if not R:
        return False, f"{col} '{valor}' no existe"
    for c, o in obj.items():
        r = R.get(c, 0)
        if r > o["test"]:
            return False, f"{c}: {r} frutos en {col} {valor} > cuota de prueba {o['test']}"
        if (d["cat"] == c).sum() - r < o["train"] + o["val"]:
            return False, f"{c}: quedarían menos frutos que train+val ({o['train'] + o['val']})"
    return True, "ok"


def proponer_reservado(d: pd.DataFrame, obj: dict):
    """Busca un lote completo y, si no, una fecha completa, que cumpla las condiciones. Prefiere el que cubre más categorías con
    >= `umbral_categorias_lote_reservado` frutos y, luego, el que aporta más frutos. Devuelve (col, valor) o None."""
    minimo = int(C.EST["umbral_categorias_lote_reservado"])
    for col in ("lote", "fecha"):
        cand = []
        for v in sorted(d[col].astype(str).unique()):
            ok, _ = reservado_elegible(d, col, v, obj)
            if ok:
                R = _conteos(d, col, v)
                cand.append((sum(1 for c in R if R[c] >= minimo), sum(R.values()), col, v))
        if cand:
            cand.sort(reverse=True)
            return cand[0][2], cand[0][3]
    return None


# ------------------------------------------------------------------------------------ partición
def particionar(df: pd.DataFrame, semilla: int = SEMILLA_PARTICION, reservado="auto", detalle: bool = False):
    """Devuelve df[['id_fruto','subset','reservado']] (y un dict con metadatos si detalle=True)."""
    d = df.dropna(subset=["categoria_ref"]).copy()
    d["cat"] = _normalizar(d["categoria_ref"])
    d["lote"] = d["lote"].astype(str)
    d["fecha"] = d["fecha"].astype(str)
    avisos, excluidos = [], {}
    if "piloto" in d:
        piloto = d["piloto"].fillna(0).astype(float) > 0
        excluidos["piloto"] = int(piloto.sum())
        d = d[~piloto]
    if "calibre_mm_ref" in d:
        grande = d["calibre_mm_ref"].astype(float) > C.CALIBRE_MAXIMO
        excluidos["calibre_>42mm"] = int(grande.sum())
        d = d[~grande]
    n_disp = d["cat"].value_counts().to_dict()
    obj, provisional = objetivos_por_categoria(n_disp)
    d = d[d["cat"].isin(obj)].copy()
    if provisional:
        avisos.append("PROVISIONAL: no se alcanzaron las cuotas de la Tabla 4.1; se aplicaron sus proporciones a los frutos disponibles. "
                      "Vuelva a generar la partición (--forzar) con la muestra completa antes de la evaluación confirmatoria.")
    falta = [c for c in C.CATEGORIAS if n_disp.get(c, 0) == 0]
    if falta:
        avisos.append(f"Sin frutos en: {falta}.")

    # lote / fecha reservado
    res = None
    if reservado in (None, "ninguno", "ninguna", False):
        res = None
    elif reservado == "auto":
        res = proponer_reservado(d, obj)
        if res is None:
            avisos.append("No hay un lote ni una fecha completos elegibles para reservar: el análisis por lote se omite y se declara como limitación.")
    else:
        col = "lote" if str(reservado) in set(d["lote"]) else "fecha"
        ok, why = reservado_elegible(d, col, reservado, obj)
        if not ok:
            raise SystemExit(f"No se puede reservar {col} '{reservado}': {why}")
        res = (col, str(reservado))

    rng = np.random.default_rng(semilla)
    d["subset"] = ""
    d["reservado"] = 0
    for c, o in obj.items():
        g = d[d["cat"] == c]
        ids_res = []
        if res is not None:
            m = g[res[0]].astype(str) == res[1]
            ids_res = g[m].index.tolist()
            d.loc[ids_res, ["subset", "reservado"]] = ["test", 1]
            g = g[~m]
        objetivo = dict(o, test=o["test"] - len(ids_res))
        sesiones = g.groupby(["lote", "fecha"]).groups
        sizes = {k: len(v) for k, v in sesiones.items()}
        rep = _reparto(sizes, objetivo, rng)
        for k, idx in sesiones.items():
            idx = np.array(sorted(idx))
            rng.shuffle(idx)
            pos = 0
            for sub in ORDEN:
                n = rep[k][sub]
                d.loc[idx[pos:pos + n], "subset"] = sub
                pos += n
    out = d[["id_fruto", "subset", "reservado"]].reset_index(drop=True)
    if not detalle:
        return out
    meta = dict(provisional=provisional, objetivos=obj, disponibles={k: int(v) for k, v in n_disp.items()},
                reservado=None if res is None else dict(tipo=res[0], valor=res[1]), excluidos=excluidos, avisos=avisos,
                semilla=semilla, fecha=datetime.now().isoformat(timespec="seconds"))
    return out, meta


def asignar_subconjuntos(df: pd.DataFrame, semilla: int = SEMILLA_PARTICION, proporciones=None, reservado=None) -> pd.DataFrame:
    """Devuelve df[['id_fruto','subset']] con subset en {train, val, test} (v4: por cuotas y prueba balanceada).
    `reservado=None` por defecto aquí (la ejecución oficial con `main` usa 'auto'). Con `proporciones` explícitas (p. ej.
    (0.70, 0.15, 0.15)) se usa el esquema proporcional LEGADO, solo para análisis exploratorios."""
    if proporciones is not None:
        return _asignar_proporcional(df, semilla, proporciones)
    return particionar(df, semilla, reservado=reservado)[["id_fruto", "subset"]]


def _asignar_proporcional(df, semilla, proporciones):
    d = df.dropna(subset=["categoria_ref"]).copy()
    d["cat"] = _normalizar(d["categoria_ref"])
    d["lote"] = d["lote"].astype(str)
    d["estrato"] = d["cat"] + "|" + d["lote"]
    pequenos = d["estrato"].map(d["estrato"].value_counts()) < 3
    d.loc[pequenos, "estrato"] = d.loc[pequenos, "cat"]
    rng = np.random.default_rng(semilla)
    d["subset"] = ""
    for _, idx in d.groupby("estrato").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_va, n_te = int(round(proporciones[1] * n)), int(round(proporciones[2] * n))
        if n >= 3:
            n_va, n_te = max(n_va, 1), max(n_te, 1)
        n_tr = n - n_va - n_te
        d.loc[idx[:n_tr], "subset"] = "train"
        d.loc[idx[n_tr:n_tr + n_va], "subset"] = "val"
        d.loc[idx[n_tr + n_va:], "subset"] = "test"
    return d[["id_fruto", "subset"]]


# ------------------------------------------------------------------------------------ verificación y utilidades
def verificar_particion(df: pd.DataFrame, part: pd.DataFrame) -> list:
    """Problemas encontrados (lista vacía si todo está bien). El lote/fecha reservado solo puede estar en test."""
    m = df.merge(part, on="id_fruto")
    m["cat"] = _normalizar(m["categoria_ref"])
    problemas = []
    if part["id_fruto"].duplicated().any():
        problemas.append("Hay frutos repetidos en la partición (fuga entre subconjuntos).")
    if "reservado" in m:
        m = m[m["reservado"].fillna(0) == 0]            # el reservado solo está en test por diseño
    for col in ("cat", "lote", "fecha"):
        tabla = pd.crosstab(m[col], m["subset"])
        for sub in ("train", "val", "test"):
            if sub not in tabla.columns:
                problemas.append(f"El subconjunto '{sub}' está vacío.")
                continue
            faltan = tabla.index[tabla[sub] == 0].tolist()
            if faltan:
                problemas.append(f"'{col}': {faltan} no aparece(n) en {sub}.")
    return problemas


def tabla_asignacion(df: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """Frutos por categoría y subconjunto (equivalente a la Tabla 4.1 con los datos reales)."""
    m = df.merge(part, on="id_fruto")
    m["cat"] = _normalizar(m["categoria_ref"])
    t = pd.crosstab(m["cat"], m["subset"]).reindex(C.CATEGORIAS).fillna(0).astype(int)
    for s in ("train", "val", "test"):
        if s not in t:
            t[s] = 0
    t = t[["train", "val", "test"]]
    t["total"] = t.sum(axis=1)
    return t


def cargar_particion(procesados: Path = ROOT / "data" / "processed") -> pd.DataFrame:
    ruta = Path(procesados) / "particion.csv"
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}. Ejecuta primero: python src/particion.py")
    return pd.read_csv(ruta, dtype={"id_fruto": str})


def cargar_meta(procesados: Path = ROOT / "data" / "processed") -> dict:
    ruta = Path(procesados) / "particion_meta.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def huella_particion(part: pd.DataFrame) -> str:
    """Hash corto de la partición, para registrarlo en cada experimento."""
    p = part[["id_fruto", "subset"]]
    txt = "|".join(f"{a}:{b}" for a, b in p.sort_values("id_fruto").itertuples(index=False))
    return hashlib.sha256(txt.encode()).hexdigest()[:12]


def main(procesados: Path, forzar: bool, reservado="auto", semilla: int = SEMILLA_PARTICION):
    ruta = procesados / "particion.csv"
    if ruta.exists() and not forzar:
        part = pd.read_csv(ruta, dtype={"id_fruto": str})
        print(f"La partición ya existe y está congelada ({ruta}). Huella: {huella_particion(part)}")
        print("Para regenerarla (invalida resultados previos) usa --forzar.")
        return
    frutos = pd.read_csv(procesados / "indice_frutos.csv", dtype={"id_fruto": str, "lote": str, "fecha": str})
    frutos = frutos[frutos["completo"]]                      # solo capturas completas
    part, meta = particionar(frutos, semilla, reservado, detalle=True)
    problemas = verificar_particion(frutos, part)
    part.to_csv(ruta, index=False)
    meta["huella"] = huella_particion(part)
    meta["problemas"] = problemas
    (procesados / "particion_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(tabla_asignacion(frutos, part).to_string())
    print(f"Partición guardada en {ruta}. Huella: {meta['huella']}" + ("  [PROVISIONAL]" if meta["provisional"] else ""))
    if meta["reservado"]:
        print(f"Reservado para la prueba: {meta['reservado']['tipo']} = {meta['reservado']['valor']}")
    for a in meta["avisos"] + problemas:
        print("[AVISO]", a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesados", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--lote-reservado", default="auto", help="auto | ninguno | <lote o fecha>")
    ap.add_argument("--semilla", type=int, default=SEMILLA_PARTICION)
    a = ap.parse_args()
    main(a.procesados, a.forzar, a.lote_reservado, a.semilla)
