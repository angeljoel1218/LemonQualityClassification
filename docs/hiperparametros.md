# Búsqueda de hiperparámetros

Implementa lo que el Plan de tesis fija en la **Tabla 4.2** (*búsqueda acotada de hiperparámetros con presupuesto de cómputo fijo por configuración; se prioriza el orden de la Tabla 3.4*) y en la **sección 4.6.1, paso A.4** (los umbrales θ_def y θ_col se calibran sobre validación).

## Diseño
| Decisión | Valor | Motivo |
|---|---|---|
| Método | muestreo aleatorio sin reemplazo de un espacio acotado (`configs/hiperparametros.yaml`) | presupuesto fijo y reproducible |
| Presupuesto | 24 candidatos **por modelo** (kNN, regresión logística, Random Forest, Naive Bayes) | ningún modelo gana por probar más |
| Balanceo | se busca junto con los hiperparámetros, en el orden de la Tabla 3.4: ninguno · pesos de clase (solo logreg/RF) · sobremuestreo · submuestreo | H5 / E3, solo dentro de `fit` y solo train |
| Selección | F1 macro en CV estratificada 5 × 3 semillas, **solo train** | sin fuga |
| Control | F1 en VAL por candidato; **TEST no se toca** | detectar sobreajuste a la búsqueda |
| Comparación | mejor candidato vs. mismo modelo con valores por defecto, por pliegue: prueba t corregida (Nadeau & Bengio) + Holm | pareada; el mejor de N está sesgado hacia arriba |

## Ejecutar
```bash
python src/busqueda_hiperparametros.py                    # ≈ 6 min en 1 núcleo (RF domina el costo)
python src/busqueda_hiperparametros.py --presupuesto 8 --modelos knn logreg
jupyter notebook notebooks/Hiperparametros.ipynb          # visualización (EJECUTAR_BUSQUEDA = True para re-correr)
```
Salidas: `logs/hiperparametros_runs/<corrida>/` (`trials.csv` con todos los candidatos, `mejores.csv`, `comparacion_vs_baseline.csv`, `cv_pliegues_*.csv`, `matrices_confusion_val.json`), `logs/hiperparametros_historial.csv` y `logs/metrics_hiperparametros.txt`.

## Qué muestra el notebook
Dispersión de candidatos frente al baseline · curva de presupuesto · efecto de cada hiperparámetro · mapa de calor de kNN · balanceo por nivel (F1 macro, Tercera, Fábrica) · mejor candidato CV vs VAL · ganancia pareada con veredicto · matriz de confusión y metas de la hipótesis general con IC95 % bootstrap (B = 1000) · **calibración de θ_def y θ_col** (rejilla sobre VAL, recall/precisión/F1 de Fábrica y estabilidad por bootstrap) · tabla del espacio del modelo profundo.

## Modelo profundo (definido, no ejecutado)
`espacio_profundo` fija, antes de entrenar, AdamW + coseno + early stopping sobre la pérdida de validación (4.6.1 A.3), tasa de aprendizaje y *weight decay* log-uniformes, focal loss (γ, α), pooling de vistas y los λ de la ecuación (3). Se ejecutará con el mismo presupuesto por configuración cuando haya GPU e imágenes.

## Cómo leer los resultados
- Con 359 frutos de train, una diferencia de 1–2 puntos de F1 es ruido: mira la desviación entre pliegues y el veredicto de Holm.
- Si la categoría está casi determinada por la sesión (ver `Baseline_basico` §0), afinar hiperparámetros puede **optimizar el atajo**; el resultado solo es interpretable tras mezclar sesiones por categoría.
