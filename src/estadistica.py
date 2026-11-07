"""
estadistica.py
--------------
Herramientas estadisticas de la tesis (secciones 3.1.8, 3.1.9 y 4.6.4), en un solo lugar y sin dependencias
pesadas (numpy / scipy / sklearn):

    kappa_cohen            acuerdo por encima del azar entre dos "evaluadores" (modelo vs estandar, operario vs estandar)
    bootstrap_pareado      IC 95 % por remuestreo de FRUTOS (B = 1000 en la tesis) de una metrica o de la diferencia
                           de una metrica entre dos clasificadores evaluados sobre los mismos frutos
    mcnemar                prueba pareada de aciertos (exacta si hay pocos discordantes; si no, chi^2 con continuidad)
    wilson                 intervalo de Wilson para una proporcion
    holm                   correccion de Holm para una familia de contrastes
    no_inferioridad        limite inferior del IC de la diferencia frente a -delta (H7)
    verifica_umbral_ic     la regla de la hipotesis general: el LIMITE INFERIOR del IC95 % debe superar el umbral

Todo es determinista (semilla fija) para que los resultados se puedan reproducir.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score


# ------------------------------------------------------------------------------------ acuerdo
def kappa_cohen(y_ref, y_pred) -> float:
    """Kappa de Cohen (1960) sin ponderar. Con una sola clase en ambos vectores devuelve NaN."""
    y_ref, y_pred = np.asarray(y_ref), np.asarray(y_pred)
    if len(set(y_ref) | set(y_pred)) < 2:
        return float("nan")
    return float(cohen_kappa_score(y_ref, y_pred))


# ------------------------------------------------------------------------------------ metricas de la tesis
def metricas_tesis(y_ref, y_pred, positivo: str = "Fabrica") -> dict:
    """F1 macro, exactitud balanceada, kappa y recall/F1 de la clase critica (Fabrica)."""
    y_ref, y_pred = np.asarray(y_ref), np.asarray(y_pred)
    m = y_ref == positivo
    out = dict(
        f1_macro=float(f1_score(y_ref, y_pred, average="macro", zero_division=0)),
        bal_acc=float(np.mean([(y_pred[y_ref == c] == c).mean() for c in np.unique(y_ref)])),
        kappa=kappa_cohen(y_ref, y_pred),
        recall_pos=float((y_pred[m] == positivo).mean()) if m.any() else float("nan"),
        f1_pos=float(f1_score(m, y_pred == positivo, zero_division=0)) if m.any() else float("nan"),
    )
    return out


# ------------------------------------------------------------------------------------ bootstrap pareado
def bootstrap_pareado(y_ref, pred_a, pred_b=None, metrica=None, B: int = 1000, semilla: int = 0,
                      nivel: float = 0.95) -> dict:
    """IC por bootstrap pareado, remuestreando FRUTOS (unidad de analisis de la tesis).

    * Con `pred_b = None` estima el IC de `metrica(y_ref, pred_a)`.
    * Con `pred_b` estima el IC de la diferencia `metrica(a) - metrica(b)` sobre los MISMOS frutos remuestreados.
    `metrica` es una funcion (y_ref, y_pred) -> float; por defecto F1 macro.
    """
    y_ref, pred_a = np.asarray(y_ref), np.asarray(pred_a)
    pred_b = None if pred_b is None else np.asarray(pred_b)
    metrica = metrica or (lambda y, p: f1_score(y, p, average="macro", zero_division=0))
    n = len(y_ref)
    rng = np.random.default_rng(semilla)
    est = metrica(y_ref, pred_a) - (metrica(y_ref, pred_b) if pred_b is not None else 0.0)
    reps = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        va = metrica(y_ref[i], pred_a[i])
        reps[b] = va - (metrica(y_ref[i], pred_b[i]) if pred_b is not None else 0.0)
    reps = reps[np.isfinite(reps)]
    a = (1 - nivel) / 2
    lo, hi = (float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))) if len(reps) else (np.nan, np.nan)
    return dict(estimado=float(est), ic_inf=lo, ic_sup=hi, B=int(len(reps)), n=int(n),
                excluye_cero=bool(lo > 0 or hi < 0) if np.isfinite(lo) else False)


def verifica_umbral_ic(ic: dict, umbral: float) -> str:
    """Regla de la hipotesis general (4.6.4): el limite inferior del IC95 % debe satisfacer el umbral."""
    if not np.isfinite(ic.get("ic_inf", np.nan)):
        return "sin dato"
    return "cumple (IC inf >= umbral)" if ic["ic_inf"] >= umbral else (
        "estimado >= umbral pero IC inf < umbral" if ic["estimado"] >= umbral else "no cumple")


def no_inferioridad(ic_dif: dict, delta: float = 0.05) -> bool:
    """H7: el limite inferior del IC de (sistema - operario) debe quedar por encima de -delta."""
    return bool(ic_dif["ic_inf"] > -delta)


# ------------------------------------------------------------------------------------ McNemar
def mcnemar(acierto_a, acierto_b, exacta_hasta: int = 25) -> dict:
    """Prueba de McNemar sobre aciertos pareados (booleanos) de dos clasificadores en los mismos frutos.

    b = A acierta y B falla;  c = A falla y B acierta.  Usa la prueba binomial exacta cuando b + c <= `exacta_hasta`
    y chi^2 con correccion de continuidad en otro caso. Devuelve tambien la diferencia de exactitudes (A - B) y su
    IC95 % de Wald sobre los discordantes.
    """
    a, b_ = np.asarray(acierto_a, bool), np.asarray(acierto_b, bool)
    n = len(a)
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    disc = b + c
    if disc == 0:
        p, test = 1.0, "sin discordantes"
    elif disc <= exacta_hasta:
        p, test = float(stats.binomtest(min(b, c), disc, 0.5).pvalue), "exacta"
    else:
        chi = (abs(b - c) - 1) ** 2 / disc
        p, test = float(stats.chi2.sf(chi, 1)), "chi2 con continuidad"
    dif = (b - c) / n
    se = np.sqrt(max(b + c - (b - c) ** 2 / n, 0)) / n
    return dict(n=int(n), b=b, c=c, discordantes=disc, p=p, prueba=test, dif=float(dif),
                ic_inf=float(dif - 1.96 * se), ic_sup=float(dif + 1.96 * se))


# ------------------------------------------------------------------------------------ proporciones y multiples contrastes
def wilson(k: int, n: int, nivel: float = 0.95):
    """Intervalo de Wilson (1927) para una proporcion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - nivel) / 2)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    mitad = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(centro - mitad), float(centro + mitad))


def holm(p) -> np.ndarray:
    """Correccion de Holm (1979) para una familia de contrastes."""
    p = np.asarray(p, float)
    orden, ajust, corriente = np.argsort(p), np.empty(len(p)), 0.0
    for rango, i in enumerate(orden):
        corriente = max(corriente, (len(p) - rango) * p[i])
        ajust[i] = min(1.0, corriente)
    return ajust


def t_corregida(d, n_train: float, n_test: float):
    """Prueba t corregida para CV repetida (Nadeau & Bengio, 2003). d = diferencias pareadas por pliegue."""
    d = np.asarray(d, float)
    J, m = len(d), float(np.mean(d))
    v = float(np.var(d, ddof=1)) if J > 1 else 0.0
    if v == 0.0:
        return (0.0, 1.0) if m == 0 else (float(np.sign(m) * np.inf), 0.0)
    t = m / np.sqrt((1.0 / J + n_test / n_train) * v)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def f1_clase(y_ref, y_pred, clase: str) -> float:
    """F1 de una sola clase (p. ej. Tercera o Fabrica en H5)."""
    y_ref, y_pred = np.asarray(y_ref), np.asarray(y_pred)
    return float(f1_score(y_ref == clase, y_pred == clase, zero_division=0)) if (y_ref == clase).any() else float("nan")


def f1_par(y_ref, y_pred, clases=("Primera", "Segunda")) -> float:
    """F1 macro restringido a un par de clases (H4: separacion Primera-Segunda). Solo frutos cuya referencia esta en el par."""
    y_ref, y_pred = np.asarray(y_ref), np.asarray(y_pred)
    m = np.isin(y_ref, clases)
    return float(f1_score(y_ref[m], y_pred[m], labels=list(clases), average="macro", zero_division=0)) if m.any() else float("nan")


def recall_clase(y_ref, y_pred, clase: str) -> float:
    y_ref, y_pred = np.asarray(y_ref), np.asarray(y_pred)
    return float(recall_score(y_ref == clase, y_pred == clase, zero_division=0)) if (y_ref == clase).any() else float("nan")
