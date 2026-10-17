# Feature Engineering

Convierte las 5 imágenes de cada fruto en variables útiles para clasificar, demostrando con experimentos controlados qué aporta cada decisión y **sin fuga de información**.

## 1. Flujo

```
imágenes ──► features por vista ──► agregación por fruto ──► features_frutos.csv
 (5/fruto)   features_imagen.py       extraer_features.py          │
                                                                   ▼
 particion.csv (congelada, por fruto) ──► train / val / test ──► Pipeline sklearn
                                                                (imputar → B → escalar → [S] → modelo)
                                                                   │
                                  CV solo en train + val ──► fe_experimentos.py ──► logs/
                                                                   │
                                          TEST: una sola vez ◄─────┘
```

## 2. Mapa curso → implementación

| Concepto del curso | Dónde está |
|---|---|
| Divide primero, transforma después | `src/particion.py` (congelada) + todo con estado dentro de `Pipeline` (`src/fe_pipeline.py`) |
| Ajustar SOLO con train | `src/transformadores.py`; `tests/test_fe.py` lo comprueba |
| Creación: ratios, interacciones | `crear_features_dominio` en `src/fe_columnas.py` (croma, amarillez, a/b, distancia a umbrales…) |
| Transformación: escala, log1p, winsorización p5–p95 | `StandardScaler`, `Log1pAsimetricas`, `Winsorizador` |
| Outliers: banderas además de winsorizar | `MarcasOutlier` (IQR, límites aprendidos en train) |
| Faltantes: imputación | `SimpleImputer(mediana)` + `MissingIndicator` |
| Codificación ordinal con orden real | `color_clase_ord` (verde < verde-amarillo < amarillo) |
| Estacionalidad con sen/cos | `CiclicoGrados` aplicado al matiz (`hue_deg`) |
| One-hot / categorías raras / target encoding con CV | Disponibles en el control de fuga (`OneHotEncoder`, `TargetEncoder` con ajuste interno por pliegues); **no se usan como features** (ver §6) |
| Selección: correlación, MI, VIF, permutación | `src/seleccion.py`; `FiltroColinealidad` dentro del pipeline (conjunto S) |
| Pipelines para impedir fugas | `sklearn.pipeline.Pipeline` + `ColumnTransformer` |
| Ablación gradual | `src/fe_experimentos.py`: A, A+B, A+C, A+B+C, A+B+C+S |
| Loguear qué hiciste y con qué parámetros | `configs/fe.yaml` → `logs/fe_runs/<run>/config.json`, `logs/experimentos_fe.csv` |
| Firma de datos (medias, p50–p90, varianza) | `src/firma.py` (hash registrado en cada corrida) |
| Validación estratificada; F1 / PR-AUC en positivo | StratifiedKFold; clase positiva = Fábrica |
| Registrar config + métrica + costo/latencia | columnas `cv_fit_s_media`, `pred_ms_fruto`, `extraccion_ms_fruto`, `latencia_total_ms_fruto` |
| Tabla final con medias ± std (CV) | `logs/fe_runs/<run>/tabla_final.md` |

## 3. Conjuntos de features

**A — Originales (30)**, medidas directas por fruto:

| Grupo | Variables | Cómo se calcula |
|---|---|---|
| Geometría | `diam_ecu_px/mm`, `long_polar_px/mm`, `aspecto`, `area_px`, `circularidad`, `solidez`, `excentricidad`, `asimetria_cv_diam` | Rectángulo de mínima área sobre la máscara: **eje menor = diámetro ecuatorial (calibre)**, eje mayor = longitud polar. Mediana de las 4 vistas ecuatoriales. mm = px × `mm_por_px` de la sesión |
| Pedúnculo | `ped_diam_px`, `ped_circularidad`, `ped_area_ratio` | Vista polar |
| Color (CIELAB real) | `L/a/b_media`, `L/a/b_std`, `L_p10`, `L_p90`, `entropia_L`, `disp_b_vistas` | Superficie **sana** (sin borde ni defectos), media de las vistas ecuatoriales |
| Defecto | `defecto_max`, `defecto_media`, `n_vistas_defecto`, `mancha_max`, `n_manchas_total` | Píxeles cuyo color se aleja del color mediano del fruto (más oscuros, no brillos). **Basta una vista**: máximo sobre las 5 |
| Textura | `lap_var`, `grad_media` | Varianza del laplaciano y gradiente medio sobre superficie sana |

**C — Dominio (10)**, derivadas sin estado de A: `croma`, `hue_deg`, `indice_amarillez` (b\*/L\*), `razon_a_b`, `color_clase_ord`, `cat_calibre_ord`, `dist_umbral_calibre_mm`, `es_fronterizo`, `defecto_alto_flag`, `frac_vistas_defecto`.

**B — Transformaciones (con estado, solo train):** winsorización (p5–p95), `log1p` a columnas no negativas con asimetría > 1, banderas de atípicos (IQR, ≥1 % de atípicos en train), indicadores de faltantes, matiz en sen/cos.

**S — Selección:** elimina, de cada par con |r| > 0,95, la variable con menos información mutua con el objetivo (calculado dentro de cada pliegue).

**Metadatos de calidad** (`nitidez`, `brillo_fondo`, `n_vistas_validas`, `t_extraccion_ms`): se guardan para auditoría y análisis de deriva, pero **no son features** (`usar_calidad_como_feature: false`) porque identifican la sesión de captura.

## 4. Reglas anti-fuga (verificadas por `tests/test_fe.py`)

1. **Columnas prohibidas** como entrada: `categoria_ref`, `calibre_mm_ref`, `color_panel`, `severidad_defecto_panel`, `tipo_defecto` (la categoría se *deriva* de ellas) y `lote`, `fecha`, `id_fruto` (atajos de contexto). `verificar_sin_fuga` lanza un error si alguna aparece.
2. La **extracción** no usa estadísticos del dataset; las **transformaciones con estado** se ajustan dentro del `Pipeline` (en CV, solo con el pliegue de entrenamiento).
3. **Partición por fruto**, congelada y con huella (hash) registrada en cada experimento.
4. **TEST** se evalúa una sola vez (`logs/test_evaluado.json` bloquea repetirlo).
5. El **control de fuga** compara la configuración base contra la misma + `lote` para detectar atajos de contexto.

## 5. Protocolo experimental

- CV estratificada **solo sobre train** (5 pliegues × 3 semillas) + evaluación en validación.
- Métricas: F1 macro, exactitud balanceada, recall y PR-AUC de **Fábrica**.
- Una diferencia menor que la desviación estándar (±) **no es una mejora demostrada**. Los pliegues y semillas no son independientes, así que la desviación es una guía, no un intervalo de confianza. Para conclusiones finales: pruebas pareadas (McNemar / bootstrap pareado) sobre test.
- Modelos de comparación: kNN, regresión logística y Random Forest (más el *dummy* de referencia). Son modelos clásicos sobre features; **no reemplazan** al baseline MVCNN ni al pipeline multitarea de aprendizaje profundo, sino que dan una línea base interpretable y miden cuánto aporta la ingeniería de atributos.

## 6. Lo que NO aplica (y por qué)

- **Texto / RAG / chatbots:** el proyecto trabaja con imágenes.
- **Lags, ventanas y calendario de fechas:** no es una serie de tiempo. La `fecha` se usa para agrupar y detectar deriva, no como feature; incluirla permitiría que el modelo aprenda la sesión en vez del fruto.
- **Target encoding / one-hot de `lote`:** solo en el experimento de control. Como feature sería un atajo de contexto.

## 7. Limitaciones y pendientes con datos reales

- **Escala px→mm:** requiere `data/raw/calibracion_sesiones.csv` (`lote, fecha, mm_por_px`), calculado en cada sesión con el patrón de tamaño conocido (`mm_por_px = tamaño real en mm / tamaño en píxeles del patrón`). Sin él, las features en mm quedan vacías.
- **Corrección de color con la carta de color:** `preprocesamiento.py` aún es un esqueleto. Mientras no se implemente, el color **no está normalizado entre sesiones**; el análisis de deriva por lote/fecha (§5 del notebook) mostrará cuánto afecta.
- **Umbrales de dominio** (`hue_verde_min`, `hue_amarillo_max`, `theta_def`): los valores de `configs/fe.yaml` son de ejemplo; fijarlos con el panel en el piloto.
- **Segmentación clásica de respaldo:** asume fondo uniforme y busca el objeto más grande y centrado. Si la carta de color o el patrón quedan cerca del fruto, o la iluminación varía, usar las máscaras del modelo de segmentación (`features_vista(img, mask=...)` las acepta) o definir una región de interés.
- **Detector de defectos:** heurístico (zonas más oscuras que el fruto). Defectos claros (cicatrices pálidas) pueden no detectarse; se valida contra `severidad_defecto_panel`.
- **Clases:** los experimentos exigen al menos 2 categorías con ≥ 2 frutos en train. Con solo Fábrica la extracción y su validación funcionan, pero no se pueden evaluar clasificadores.

## 8. Cómo ejecutar

```bash
python src/ingesta.py                  # índices
python src/particion.py                # partición congelada (una sola vez)
python src/extraer_features.py         # features por vista y por fruto  (--jobs 4 para paralelizar)
python src/fe_experimentos.py          # ablaciones (--control-fuga para el control de atajos)
python src/fe_experimentos.py --evaluar-test "A+B+C|rf"   # SOLO al final, una vez
python -m pytest -q                    # pruebas anti-fuga y de extracción
jupyter notebook notebooks/Feature_Engineering.ipynb
```
Con datos de demostración añadir `--procesados data/demo/processed` (y `--raw data/demo/raw` en la extracción).
