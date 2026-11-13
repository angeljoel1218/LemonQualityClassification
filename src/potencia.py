"""
potencia.py
-----------
Simulación Monte Carlo de potencia del Plan de tesis v4 (Anexo 2 / Anexo F del informe) y fórmulas de tamaño de muestra de
H1, H2 y H4 (sección 4.4). Reproduce el procedimiento DOCUMENTADO, con las mismas semillas, para que cualquiera pueda
repetirlo:

  * Hipótesis general: categorías en el orden Extra, Primera, Segunda, Tercera, Fábrica; 80 frutos por categoría (n = 400);
    cada categoría se clasifica bien con sensibilidad r y el error restante se reparte por igual entre las categorías vecinas
    (los extremos envían todo el error a su único vecino). 5 000 simulaciones (semilla 20261112); bootstrap estratificado por
    categoría verdadera con B = 5 000; límite inferior = percentil 2,5 %; acierta si el límite inferior >= 0,85.
  * H3: par (predicción del híbrido, predicción del directo) con cuatro resultados: ambos aciertan (0,80), solo el híbrido
    acierta (0,09), solo el directo (0,04), ambos fallan (0,07). Escenario 0,08: 0,78; 0,12; 0,04; 0,06. Errores a categorías
    vecinas; si ambos fallan, cada sistema elige un vecino de forma independiente. 2 000 simulaciones (semilla 20261113);
    acierta si el límite inferior del IC95 % de la diferencia de F1 macro es > 0.

Uso:
    python src/potencia.py                 # completo (puede tardar varios minutos)
    python src/potencia.py --rapido        # MC = 300, B = 500: verificación rápida
Salida: logs/potencia/anexo2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

K, N = 5, 80


def conf_matrix(r: float) -> np.ndarray:
    """Matriz de confusión verdadera (filas = verdadera) con sensibilidad r y error repartido entre vecinos."""
    M = np.zeros((K, K))
    for i in range(K):
        M[i, i] = r
        vec = [j for j in (i - 1, i + 1) if 0 <= j < K]
        for j in vec:
            M[i, j] = (1 - r) / len(vec)
    return M


def f1_macro(C: np.ndarray) -> np.ndarray:
    """C: (..., K, K) verdadera x predicha."""
    tp = np.einsum("...ii->...i", C).astype(float)
    fp = C.sum(-2) - tp
    fn = C.sum(-1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1)).mean(-1)


def potencia_hg(r: float, MC: int = 5000, B: int = 5000, semilla: int = 20261112, umbral: float = 0.85) -> dict:
    rng = np.random.default_rng(semilla)
    M = conf_matrix(r)
    ok, f1s = 0, []
    for _ in range(MC):
        C = np.stack([rng.multinomial(N, M[i]) for i in range(K)])
        f1s.append(float(f1_macro(C)))
        bs = np.stack([rng.multinomial(N, C[i] / N, size=B) for i in range(K)], axis=1)        # (B, K, K)
        ok += np.percentile(f1_macro(bs), 2.5) >= umbral
    return dict(escenario=f"Hipótesis general, r = {r:.2f}", f1_media=float(np.mean(f1s)), error_estandar=float(np.std(f1s, ddof=1)),
                potencia=ok / MC, MC=MC, B=B, semilla=semilla)


def _celdas_h3(p_ambos, p_solo_h, p_solo_d, p_fallan):
    """Probabilidad de cada celda (pred_hibrido, pred_directo) para una categoría verdadera i -> array (K, K, K) [i, ph, pd]."""
    P = np.zeros((K, K, K))
    for i in range(K):
        vec = [j for j in (i - 1, i + 1) if 0 <= j < K]
        err = {j: 1 / len(vec) for j in vec}
        P[i, i, i] += p_ambos
        for j, w in err.items():
            P[i, i, j] += p_solo_h * w                 # el híbrido acierta, el directo falla
            P[i, j, i] += p_solo_d * w                 # el directo acierta, el híbrido falla
            for j2, w2 in err.items():
                P[i, j, j2] += p_fallan * w * w2       # ambos fallan: cada uno elige un vecino de forma independiente
    return P


def potencia_h3(dif_exactitud: float = 0.05, MC: int = 2000, B: int = 5000, semilla: int = 20261113) -> dict:
    if abs(dif_exactitud - 0.05) < 1e-9:
        probs = (0.80, 0.09, 0.04, 0.07)
    elif abs(dif_exactitud - 0.08) < 1e-9:
        probs = (0.78, 0.12, 0.04, 0.06)
    else:
        solo_d = 0.04
        probs = (0.80, solo_d + dif_exactitud, solo_d, 1 - 0.80 - (solo_d + dif_exactitud) - solo_d)
    P = _celdas_h3(*probs)
    rng = np.random.default_rng(semilla)
    ok, difs = 0, []
    for _ in range(MC):
        cnt = np.stack([rng.multinomial(N, P[i].ravel()).reshape(K, K) for i in range(K)])       # (K, ph, pd)
        Ch, Cd = cnt.sum(2), cnt.sum(1)                                                          # verdadera x predicha
        difs.append(float(f1_macro(Ch) - f1_macro(Cd)))
        bs = np.stack([rng.multinomial(N, cnt[i].ravel() / N, size=B).reshape(B, K, K) for i in range(K)], axis=1)  # (B,K,ph,pd)
        d = f1_macro(bs.sum(3)) - f1_macro(bs.sum(2))
        ok += np.percentile(d, 2.5) > 0
    return dict(escenario=f"H3, diferencia de exactitud {dif_exactitud:.2f}", f1_media=float(np.mean(difs)), error_estandar=None,
                potencia=ok / MC, MC=MC, B=B, semilla=semilla)


# ------------------------------------------------------------------------------------ fórmulas de tamaño de muestra (sección 4.4)
def n_h1(mu: float, sigma: float, alfa: float = 0.05, pot: float = 0.80, mae_max: float = 0.50) -> int:
    """n = (z_(1-α) + z_(1-β))² σ² / (mae_max - μ)² (prueba unilateral)."""
    z = stats.norm.ppf(1 - alfa) + stats.norm.ppf(pot)
    return int(np.ceil(z ** 2 * sigma ** 2 / (mae_max - mu) ** 2))


def potencia_h2(n: int, psi: float, delta: float, umbral: float = 0.15) -> dict:
    """Potencia del criterio conjunto de H2 (incremento >= umbral y límite inferior > 0): ≈ Φ((Δ - umbral)/EE), EE = √((ψ - Δ²)/n)."""
    ee = float(np.sqrt((psi - delta ** 2) / n))
    return dict(EE=ee, potencia=float(stats.norm.cdf((delta - umbral) / ee)))


def n_h4(rho: float, delta: float = 0.05, p0: float = 0.88, pe: float = 0.20, alfa: float = 0.05, pot: float = 0.80) -> int:
    V = p0 * (1 - p0) / (1 - pe) ** 2
    s2 = 2 * (1 - rho) * V
    z = stats.norm.ppf(1 - alfa) + stats.norm.ppf(pot)
    return int(np.ceil(z ** 2 * s2 / delta ** 2))


def ejecutar(rapido: bool = False) -> dict:
    mc_hg, mc_h3, B = (300, 300, 500) if rapido else (5000, 2000, 5000)
    filas = [potencia_hg(r, mc_hg, B) for r in (0.88, 0.90, 0.92)] + [potencia_h3(d, mc_h3, B) for d in (0.05, 0.08)]
    out = dict(rapido=rapido, simulaciones=filas,
               formulas=dict(h1={"sigma_constante": {"mu_0.40": n_h1(0.40, 0.30), "mu_0.45": n_h1(0.45, 0.30)},
                                 "seminormal": {"mu_0.40": n_h1(0.40, 0.30), "mu_0.45": n_h1(0.45, 0.34)}},
                             h2={"psi_0.25": potencia_h2(80, 0.25, 0.20), "psi_0.30": potencia_h2(80, 0.30, 0.20)},
                             h4={"rho_0.70": n_h4(0.70), "rho_0.50": n_h4(0.50)}))
    d = ROOT / "logs" / "potencia"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("anexo2_rapido.json" if rapido else "anexo2.json")).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()
    r = ejecutar(a.rapido)
    for f in r["simulaciones"]:
        print(f"{f['escenario']:<38} F1/dif media {f['f1_media']:.3f}  EE {f['error_estandar'] if f['error_estandar'] is None else round(f['error_estandar'], 4)}  potencia {f['potencia']:.3f}")
    print(json.dumps(r["formulas"], indent=1))
