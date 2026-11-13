"""
estadistica.py
--------------
Herramientas estadisticas de la tesis (secciones 3.1.8, 3.1.9 y 4.6.4), en un solo lugar y sin dependencias
pesadas (numpy / scipy / sklearn):

    kappa_cohen            acuerdo por encima del azar entre dos "evaluadores" (modelo vs estandar, operario vs estandar)
    bootstrap_pareado      IC 95 % por remuestreo de FRUTOS (B = 5 000 en el plan v4; ver `ic_estratificado` para el estratificado) de una metrica o de la diferencia
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

from src.config import CATEGORIAS, CATEGORIAS_CALIBRE, EST


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
def bootstrap_pareado(y_ref, pred_a, pred_b=None, metrica=None, B: int | None = None, semilla: int = 0,
                      nivel: float = 0.95) -> dict:
    """IC por bootstrap pareado, remuestreando FRUTOS (unidad de analisis de la tesis).

    * Con `pred_b = None` estima el IC de `metrica(y_ref, pred_a)`.
    * Con `pred_b` estima el IC de la diferencia `metrica(a) - metrica(b)` sobre los MISMOS frutos remuestreados.
    `metrica` es una funcion (y_ref, y_pred) -> float; por defecto F1 macro.
    """
    B = int(B or EST["bootstrap_B"])
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
    p1 = float(stats.binom.sf(b - 1, disc, 0.5)) if disc and b > c else (1.0 if disc else 1.0)   # unilateral exacto: A mejor que B
    return dict(n=int(n), b=b, c=c, discordantes=disc, p=p, p_unilateral=p1, prueba=test, dif=float(dif),
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
    """F1 macro restringido a un par de clases (separacion Primera-Segunda). Solo frutos cuya referencia esta en el par.
    Las predicciones faltantes (None) o de otras categorias cuentan como error."""
    y_ref = np.asarray(y_ref, object)
    y_pred = np.asarray([p if p in clases else "otra" for p in y_pred], object)
    m = np.isin(y_ref, clases)
    if not m.any():
        return float("nan")
    return float(f1_score(y_ref[m].astype(str), y_pred[m].astype(str), labels=list(clases), average="macro", zero_division=0))


def recall_clase(y_ref, y_pred, clase: str) -> float:
    y_ref, y_pred = np.asarray(y_ref), np.asarray(y_pred)
    return float(recall_score(y_ref == clase, y_pred == clase, zero_division=0)) if (y_ref == clase).any() else float("nan")


# =====================================================================================================================
# PLAN DE TESIS v4 (sección 4.7.4): bootstrap ESTRATIFICADO por categoría con B = 5 000, familia de Holm {H1..H4},
# kappa de Fleiss, AC1 de Gwet, categoría adyacente y reponderación por prevalencia natural.
# =====================================================================================================================
def _a_indices(y, clases):
    """Etiquetas -> índices 0..K-1; todo lo que no esté en `clases` (p. ej. 'Fuera_de_alcance' o None) -> K ('otro')."""
    m = {c: i for i, c in enumerate(clases)}
    return np.array([m.get(v, len(clases)) for v in y], dtype=int)


def matriz_confusion_idx(y_idx, p_idx, K):
    """Matriz (K+1)x(K+1) (la última fila/columna es 'otro') a partir de índices; admite lotes: arrays (B, n) -> (B, K+1, K+1)."""
    y_idx, p_idx = np.asarray(y_idx), np.asarray(p_idx)
    k1 = K + 1
    if y_idx.ndim == 1:
        return np.bincount(y_idx * k1 + p_idx, minlength=k1 * k1).reshape(k1, k1)
    B = y_idx.shape[0]
    flat = (y_idx * k1 + p_idx) + (np.arange(B)[:, None] * k1 * k1)
    return np.bincount(flat.ravel(), minlength=B * k1 * k1).reshape(B, k1, k1)


def f1_macro_cm(cm, K):
    """F1 macro sobre las K clases (la columna 'otro' cuenta como error de la clase verdadera). cm: (..., K+1, K+1)."""
    cm = np.asarray(cm, float)
    tp = np.einsum("...ii->...i", cm)[..., :K]
    fn = cm[..., :K, :].sum(-1) - tp
    fp = cm[..., :K].sum(-2) - tp
    den = 2 * tp + fp + fn
    return (np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)).mean(-1)


def kappa_cm(cm):
    cm = np.asarray(cm, float)
    n = cm.sum((-1, -2))
    po = np.einsum("...ii->...", cm) / n
    pe = (cm.sum(-1) * cm.sum(-2)).sum(-1) / n ** 2
    return np.where(pe < 1, (po - pe) / np.maximum(1 - pe, 1e-12), np.nan)


def recall_cm(cm, k):
    cm = np.asarray(cm, float)
    fila = cm[..., k, :].sum(-1)
    return np.where(fila > 0, cm[..., k, k] / np.maximum(fila, 1), np.nan)


def acc_cm(cm):
    cm = np.asarray(cm, float)
    return np.einsum("...ii->...", cm) / cm.sum((-1, -2))


def bal_acc_cm(cm, K):
    cm = np.asarray(cm, float)
    fila = cm[..., :K, :].sum(-1)
    return (np.einsum("...ii->...i", cm)[..., :K] / np.maximum(fila, 1)).mean(-1)


METRICAS_CM = {
    "f1_macro": lambda cm, K, clases: f1_macro_cm(cm, K),
    "kappa": lambda cm, K, clases: kappa_cm(cm),
    "exactitud": lambda cm, K, clases: acc_cm(cm),
    "bal_acc": lambda cm, K, clases: bal_acc_cm(cm, K),
    "recall_fabrica": lambda cm, K, clases: recall_cm(cm, clases.index("Fabrica")),
}


def _indices_estratificados(y_idx, B, semilla):
    """(B, n) índices de remuestreo CON reposición dentro de cada categoría verdadera (se conserva el tamaño de cada estrato)."""
    rng = np.random.default_rng(semilla)
    n = len(y_idx)
    out = np.empty((B, n), dtype=int)
    pos = 0
    for k in np.unique(y_idx):
        idx = np.where(y_idx == k)[0]
        out[:, pos:pos + len(idx)] = idx[rng.integers(0, len(idx), (B, len(idx)))]
        pos += len(idx)
    return out


def _metrica_cm(nombre, cm, clases):
    return METRICAS_CM[nombre](cm, len(clases), list(clases))


def ic_estratificado(y_ref, pred, metrica: str = "f1_macro", B: int | None = None, semilla: int | None = None,
                     nivel: float = 0.95, clases=None, idx=None) -> dict:
    """IC por bootstrap ESTRATIFICADO por categoría verdadera (B = 5 000). `metrica`: f1_macro | kappa | exactitud | bal_acc |
    recall_fabrica. Devuelve estimado, ic_inf, ic_sup (percentiles) y las réplicas (para p-valores)."""
    clases = list(clases or CATEGORIAS)
    B = int(B or EST["bootstrap_B"])
    semilla = EST["semilla"] if semilla is None else semilla
    y, p = _a_indices(y_ref, clases), _a_indices(pred, clases)
    est = float(_metrica_cm(metrica, matriz_confusion_idx(y, p, len(clases)), clases))
    idx = _indices_estratificados(y, B, semilla) if idx is None else idx
    reps = np.asarray(_metrica_cm(metrica, matriz_confusion_idx(y[idx], p[idx], len(clases)), clases), float)
    reps = reps[np.isfinite(reps)]
    a = (1 - nivel) / 2
    return dict(estimado=est, ic_inf=float(np.quantile(reps, a)), ic_sup=float(np.quantile(reps, 1 - a)), B=int(len(reps)),
                n=int(len(y)), reps=reps)


def dif_estratificada(y_ref, pred_a, pred_b, metrica: str = "f1_macro", B: int | None = None, semilla: int | None = None,
                      nivel: float = 0.95, clases=None, idx=None) -> dict:
    """IC por bootstrap PAREADO estratificado de metrica(a) - metrica(b) sobre los MISMOS frutos remuestreados.
    `p_unilateral` = proporción de réplicas con diferencia <= 0 (evidencia de que a > b)."""
    clases = list(clases or CATEGORIAS)
    B = int(B or EST["bootstrap_B"])
    semilla = EST["semilla"] if semilla is None else semilla
    y, pa, pb = _a_indices(y_ref, clases), _a_indices(pred_a, clases), _a_indices(pred_b, clases)
    K = len(clases)
    est = float(_metrica_cm(metrica, matriz_confusion_idx(y, pa, K), clases) - _metrica_cm(metrica, matriz_confusion_idx(y, pb, K), clases))
    idx = _indices_estratificados(y, B, semilla) if idx is None else idx
    ra = np.asarray(_metrica_cm(metrica, matriz_confusion_idx(y[idx], pa[idx], K), clases), float)
    rb = np.asarray(_metrica_cm(metrica, matriz_confusion_idx(y[idx], pb[idx], K), clases), float)
    reps = ra - rb
    reps = reps[np.isfinite(reps)]
    a = (1 - nivel) / 2
    lo, hi = float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))
    return dict(estimado=est, ic_inf=lo, ic_sup=hi, B=int(len(reps)), n=int(len(y)), reps=reps,
                p_unilateral=float((np.sum(reps <= 0) + 1) / (len(reps) + 1)), excluye_cero=bool(lo > 0 or hi < 0))


def dif_kappa_vs_operarios(y_ref, pred_sistema, preds_operarios: dict, delta: float | None = None, B: int | None = None,
                           semilla: int | None = None, clases=None) -> dict:
    """H4: kappa_sis - promedio de los kappa de los operarios (ambos frente al estándar), con IC por bootstrap pareado
    estratificado que remuestrea FRUTOS y conserva las clasificaciones de cada evaluador. No inferioridad si el límite
    inferior del IC95 % >= -delta. `p_no_inferioridad` = proporción de réplicas con diferencia <= -delta (H0: peor que δ)."""
    clases = list(clases or CATEGORIAS)
    delta = EST["delta_kappa"] if delta is None else delta
    B = int(B or EST["bootstrap_B"])
    semilla = EST["semilla"] if semilla is None else semilla
    K = len(clases)
    y = _a_indices(y_ref, clases)
    idx = _indices_estratificados(y, B, semilla)
    ps = _a_indices(pred_sistema, clases)
    ks = lambda p: kappa_cm(matriz_confusion_idx(y, p, K))
    kb = lambda p: kappa_cm(matriz_confusion_idx(y[idx], p[idx], K))
    ops = {o: _a_indices(p, clases) for o, p in preds_operarios.items()}
    est_ops = {o: float(ks(p)) for o, p in ops.items()}
    est = float(ks(ps) - np.mean(list(est_ops.values())))
    reps = kb(ps) - np.mean([kb(p) for p in ops.values()], axis=0)
    reps = reps[np.isfinite(reps)]
    lo, hi = float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))
    por_op = {o: float(ks(ps) - k) for o, k in est_ops.items()}
    return dict(estimado=est, ic_inf=lo, ic_sup=hi, delta=delta, no_inferior=bool(lo >= -delta), B=int(len(reps)),
                kappa_sistema=float(ks(ps)), kappa_operarios=est_ops, kappa_operarios_media=float(np.mean(list(est_ops.values()))),
                dif_por_operario=por_op, p_no_inferioridad=float((np.sum(reps <= -delta) + 1) / (len(reps) + 1)), reps=reps)


# ------------------------------------------------------------------------------------ Holm escalonado (familia H1-H4)
def holm_escalonado(pvals: dict, alfa: float | None = None) -> dict:
    """Corrección de Holm: se ordenan los p y se comparan con alfa/m, alfa/(m-1), ..., alfa; al primer no rechazo se detiene
    el procedimiento. Devuelve {nombre: dict(p, umbral, rechaza, p_ajustado)}."""
    alfa = EST["alfa"] if alfa is None else alfa
    m = len(pvals)
    orden = sorted(pvals, key=lambda k: pvals[k])
    out, seguir = {}, True
    ajust = holm(np.array([pvals[k] for k in orden]))
    for i, k in enumerate(orden):
        umbral = alfa / (m - i)
        rech = bool(seguir and pvals[k] <= umbral)
        seguir = rech
        out[k] = dict(p=float(pvals[k]), umbral=float(umbral), rechaza=rech, p_ajustado=float(ajust[i]))
    return out


# ------------------------------------------------------------------------------------ acuerdo entre varios evaluadores
def _tabla_items(calif: np.ndarray, categorias):
    """calif: (N items, R evaluadores) con etiquetas -> n_ij (N, q) de conteos por categoría."""
    calif = np.asarray(calif, object)
    cat = list(categorias)
    n = np.zeros((calif.shape[0], len(cat)))
    for j, c in enumerate(cat):
        n[:, j] = (calif == c).sum(1)
    return n


def kappa_fleiss(calif, categorias=None) -> float:
    """Kappa de Fleiss (1971): fruto = unidad de análisis; R evaluadores por fruto; categorías nominales. calif: (N, R)."""
    calif = np.asarray(calif, object)
    cat = list(categorias or CATEGORIAS)
    n = _tabla_items(calif, cat)
    N, R = calif.shape
    if R < 2:
        return float("nan")
    Pi = ((n ** 2).sum(1) - R) / (R * (R - 1))
    pj = n.sum(0) / (N * R)
    Pe = float((pj ** 2).sum())
    return float((Pi.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def ac1_gwet(calif, categorias=None) -> float:
    """Coeficiente AC1 de Gwet (2008) para R evaluadores: robusto a la prevalencia (kappa baja aunque el acuerdo bruto sea alto)."""
    calif = np.asarray(calif, object)
    cat = list(categorias or CATEGORIAS)
    n = _tabla_items(calif, cat)
    N, R = calif.shape
    Pa = float((((n ** 2).sum(1) - R) / (R * (R - 1))).mean())
    pi = n.sum(0) / (N * R)
    q = len(cat)
    Pe = float((pi * (1 - pi)).sum() / (q - 1))
    return float((Pa - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def acuerdo_por_categoria(calif, categorias=None) -> dict:
    """Acuerdo observado por categoría: de los pares de evaluadores que asignan la categoría a un fruto, qué fracción coincide."""
    calif = np.asarray(calif, object)
    cat = list(categorias or CATEGORIAS)
    n = _tabla_items(calif, cat)
    R = calif.shape[1]
    out = {}
    for j, c in enumerate(cat):
        pares = (n[:, j] * (n[:, j] - 1)).sum()
        asign = (n[:, j] * (R - 1)).sum()
        out[c] = float(pares / asign) if asign > 0 else float("nan")
    return out


def kappa_ponderado_calibre(y_ref, y_pred) -> float:
    """Kappa ponderado cuadrático sobre las categorías por calibre (Tercera < Segunda < Primera < Extra); Fábrica no entra."""
    orden = {c: i for i, c in enumerate(CATEGORIAS_CALIBRE)}
    a, b = np.asarray(y_ref, object), np.asarray(y_pred, object)
    m = np.array([x in orden and y in orden for x, y in zip(a, b)])
    if m.sum() < 2:
        return float("nan")
    return float(cohen_kappa_score([orden[x] for x in a[m]], [orden[y] for y in b[m]], weights="quadratic"))


def exactitud_adyacente(y_ref, y_pred) -> float:
    """Proporción de frutos con |k_est - k_ref| <= 1 sobre las categorías por calibre (Fábrica y fuera de alcance no entran)."""
    orden = {c: i + 1 for i, c in enumerate(CATEGORIAS_CALIBRE)}
    a, b = np.asarray(y_ref, object), np.asarray(y_pred, object)
    m = np.array([x in orden and y in orden for x, y in zip(a, b)])
    if m.sum() == 0:
        return float("nan")
    return float(np.mean([abs(orden[x] - orden[y]) <= 1 for x, y in zip(a[m], b[m])]))


# ------------------------------------------------------------------------------------ prevalencia natural
def reponderar_matriz(cm, prevalencia: dict, clases=None) -> np.ndarray:
    """Re-pondera por categoría VERDADERA la matriz de confusión (filas) para que las filas sumen la prevalencia natural estimada
    en el piloto (la prueba es balanceada). Devuelve una matriz con 'conteos' equivalentes a n = 1 000 frutos."""
    clases = list(clases or CATEGORIAS)
    cm = np.asarray(cm, float)
    K = len(clases)
    w = np.array([prevalencia.get(c, 0.0) for c in clases], float)
    w = w / w.sum()
    out = cm.copy()
    for i in range(K):
        fila = cm[i].sum()
        out[i] = cm[i] / fila * w[i] * 1000.0 if fila > 0 else 0.0
    return out


def metricas_ponderadas(y_ref, pred, prevalencia: dict, clases=None) -> dict:
    """Exactitud, F1 macro, kappa y F1 por clase bajo la prevalencia comercial natural (reponderación de la matriz)."""
    clases = list(clases or CATEGORIAS)
    K = len(clases)
    cm = matriz_confusion_idx(_a_indices(y_ref, clases), _a_indices(pred, clases), K)
    w = reponderar_matriz(cm, prevalencia, clases)
    tp = np.diag(w)[:K]
    fp = w[:, :K].sum(0) - tp
    fn = w[:K].sum(1) - tp
    f1c = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    return dict(exactitud=float(acc_cm(w)), f1_macro=float(f1c.mean()), kappa=float(kappa_cm(w)),
                f1_por_clase={c: float(v) for c, v in zip(clases, f1c)}, precision_por_clase={c: float(tp[i] / max(tp[i] + fp[i], 1e-12)) for i, c in enumerate(clases)})


# ------------------------------------------------------------------------------------ calibre: IC bootstrap de MAE, RMSE y adyacente
def ic_calibre(est_mm, ref_mm, cat_ref=None, B: int | None = None, semilla: int | None = None) -> dict:
    """H1: MAE y RMSE con IC por bootstrap por fruto (estratificado por categoría si se pasa `cat_ref`) y exactitud en categoría adyacente
    (a partir de los calibres: categoría por calibre estimada vs medida). Los p-valores son la proporción de réplicas que NO cumplen."""
    from src.regla_comercial import categoria_por_calibre_vec
    B = int(B or EST["bootstrap_B"])
    semilla = EST["semilla"] if semilla is None else semilla
    e, r = np.asarray(est_mm, float), np.asarray(ref_mm, float)
    ok = np.isfinite(e) & np.isfinite(r)
    e, r = e[ok], r[ok]
    n = len(e)
    if n == 0:
        return dict(n=0)
    est_cat, ref_cat = categoria_por_calibre_vec(e), categoria_por_calibre_vec(r)
    orden = {c: i + 1 for i, c in enumerate(CATEGORIAS_CALIBRE)}
    ke = np.array([orden.get(c, np.nan) for c in est_cat], float)
    kr = np.array([orden.get(c, np.nan) for c in ref_cat], float)
    adj = (np.abs(ke - kr) <= 1).astype(float)
    exa = (est_cat == ref_cat).astype(float)
    err = e - r
    rng = np.random.default_rng(semilla)
    if cat_ref is not None:
        g = np.asarray(cat_ref, object)[ok]
        y = _a_indices(g, sorted(set(g)))
        idx = _indices_estratificados(y, B, semilla)
    else:
        idx = rng.integers(0, n, (B, n))
    mae_b, rmse_b = np.abs(err)[idx].mean(1), np.sqrt((err[idx] ** 2).mean(1))
    adj_b, exa_b = np.nanmean(adj[idx], 1), exa[idx].mean(1)
    q = lambda x: (float(np.quantile(x, .025)), float(np.quantile(x, .975)))
    return dict(n=int(n), mae=float(np.abs(err).mean()), mae_ic=q(mae_b), rmse=float(np.sqrt((err ** 2).mean())), rmse_ic=q(rmse_b),
                adyacente=float(np.nanmean(adj)), adyacente_ic=q(adj_b), exacta=float(exa.mean()), exacta_ic=q(exa_b),
                sesgo=float(err.mean()), sd_diferencias=float(err.std(ddof=1)) if n > 1 else float("nan"),
                p_mae=float((np.sum(mae_b >= EST["h1"]["mae_max"]) + 1) / (B + 1)),
                p_rmse=float((np.sum(rmse_b >= EST["h1"]["rmse_max"]) + 1) / (B + 1)),
                p_adyacente=float((np.sum(adj_b < EST["h1"]["adyacente_min"]) + 1) / (B + 1)))
