# Baseline y variantes de datos

## 1. Baseline

Referencia mínima antes de cualquier modelo complejo. Si un modelo sofisticado no supera con claridad a estos, el problema está en los datos o en las variables, no en el modelo.

| Modelo | Descripción |
|---|---|
| `dummy` | Predice siempre la clase mayoritaria (piso de desempeño; su F1 de Fábrica es 0) |
| `knn` | k = 5 vecinos, sobre variables escaladas |
| `gnb` | Naive Bayes gaussiano |

Entrada: las 30 variables originales (conjunto **A** de Feature Engineering) con imputación por mediana y escalado estándar (ajustados solo con train). No usa transformaciones B ni variables de dominio C.

## 2. Variantes de datos

Se cambia **el dato**, no el modelo, y una sola cosa por vez.

| Variante | Qué cambia | Dónde actúa |
|---|---|---|
| **V1 — vistas** | Cuántas de las 5 imágenes por fruto se usan: `1vista` (0°), `2vistas` (0° + 180°), `4ecuatoriales` (sin pedúnculo), `BASE` = 5 | Las features se re-agregan por fruto con solo esas vistas, en **todos** los subconjuntos (es una decisión de captura) |
| **V2 — balanceo** | `sobremuestreo`: duplica frutos minoritarios hasta igualar a la mayoritaria; `submuestreo`: descarta frutos mayoritarios hasta igualar a la minoritaria | Solo **train**, dentro de `fit` (`src/remuestreo.py`) |

**Por qué el balanceo va dentro de `fit`:** si se balancea todo el dataset antes de partir, copias de un mismo fruto quedan en train y en validación, y las métricas salen infladas. Dentro de `fit`, en validación cruzada solo se remuestrea el pliegue de entrenamiento (verificado en `tests/test_baseline.py`).

**Costo de V1:** menos vistas = captura y extracción más rápidas. El informe reporta la latencia de extracción por fruto (medida en la máquina donde se corre; no incluye el tiempo de captura física).

## 3. Protocolo

- Misma partición congelada, mismos frutos y mismos pliegues/semillas que Feature Engineering → comparaciones **pareadas**.
- CV estratificada (5 pliegues × 3 semillas) **solo sobre train** + evaluación en validación. **TEST no se toca.**
- Métrica central: **F1 macro**. Se reportan además F1, recall y PR-AUC de **Fábrica** (clase crítica: un falso negativo manda un fruto defectuoso a una categoría superior).
- Variante vs baseline del mismo modelo: diferencias pareadas por pliegue → **prueba t corregida para CV repetida** (Nadeau & Bengio, 2003) → **corrección de Holm** por métrica. Veredicto: `mejora` / `empeora` solo si p ajustado < 0,05; si no, `sin evidencia`.
- Limitación: pliegues y semillas no son independientes, así que la prueba es aproximada. Con decenas de frutos, diferencias de 0,05–0,10 en F1 son ruido. Para la comparación final entre sistemas usar McNemar o bootstrap pareado sobre TEST.

## 3b. Métricas y evidencias añadidas (alineadas con el Plan de tesis)
- **kappa de Cohen** y **F1 por clase** (Tercera y Fábrica, H5) en CV y VAL; predicciones de VAL guardadas en `predicciones_val.csv` de cada corrida.
- **Metas de la hipótesis general** con **IC95 % por bootstrap pareado (B = 1000)** sobre VAL: el límite inferior debe cumplir el umbral (`src/estadistica.py`).
- **H3 / E4:** Fábrica por defecto omitida con 1 vista (0°) frente a 5 vistas, con McNemar (`src/evidencia_hipotesis.py`).
- **H4 / E1:** línea base sin preprocesamiento (F1 de color/madurez y del par Primera–Segunda).
- **Control de atajo por sesión:** F1 de un clasificador que solo conoce lote × fecha.

## 4. Cómo ejecutar

```bash
python src/modelo_baseline.py          # baseline + variantes → logs/metrics_baseline.txt
python src/informe.py                  # informe Word + Markdown en informe/
jupyter notebook notebooks/Baseline_basico.ipynb
```
Configuración: `configs/baseline.yaml` (modelos, semillas, niveles de vistas, modos de remuestreo).
Con datos de demostración: `--procesados data/demo/processed`.

## 5. Qué NO se puede concluir hoy

- Con una sola categoría (solo Fábrica) no hay clasificador que evaluar: la corrida se detiene con un aviso.
- Con clases de pocas decenas de frutos el balanceo es inestable; la prioridad es la **recolección dirigida** de las categorías deficitarias.
- El color aún no está corregido con la carta de color entre sesiones; las variantes de vistas heredan esa limitación.
