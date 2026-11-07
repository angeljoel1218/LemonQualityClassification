# 🍋 Clasificación Comercial Post-Cosecha de Limón con Visión por Computadora

Proyecto de investigación (Maestría en IA – UNI) para entrenar un sistema de visión por
computadora basado en Deep Learning que clasifique limones en las categorías comerciales
**Extra, Primera, Segunda, Tercera y Fábrica**, a partir de 5 imágenes por fruto
(4 vistas ecuatoriales + 1 de pedúnculo).

Este repositorio forma parte del curso **Proyecto de Investigación II (MIA 403)**.

---

## 👥 Autores
- Angel Joel Alvarado Crisanto – [@usuario-github](https://github.com/usuario-github)
- Javier Fernandez Purizaca – [@usuario-github](https://github.com/usuario-github)

**Asesor(a):** Mtr. Ing. Maria Fernanda Tejada Begazo

---

## 📊 Dataset
- **Fuente**: captura propia en centros de acopio de post-cosecha de limón sutil,
  distrito de Olmos (Lambayeque).
- **Unidad de análisis**: el fruto individual (5 imágenes: 4 ecuatoriales + pedúnculo, agrupadas por `id_fruto`).
- **Muestra objetivo**: mínimo 1 500 frutos (≈7 500 imágenes), con cuotas mínimas por
  categoría: 250 Extra, 250 Primera, 250 Segunda, 200 Tercera, 200 Fábrica.
- **Variable objetivo**: `categoria_ref` (5 clases), construida con calibre medido +
  panel de agrónomos + adjudicación de desacuerdos.
- **Versión usada**: completar al cerrar la recolección (fecha + hash del dataset).

**Archivos de datos**

| Archivo | Contenido |
|---|---|
| `data/raw/imagenes/` | Imágenes: `<lote>_<fecha>_<id_fruto>_<vista>.jpg` |
| `data/raw/etiquetas.csv` | Una fila por fruto: estándar de referencia |
| `data/raw/panel_evaluaciones.csv` | Evaluaciones individuales (sin adjudicar) de cada agrónomo |

**Variables de `etiquetas.csv`**

| Columna | Descripción breve |
|---|---|
| `id_fruto` | Código único del fruto; coincide con el nombre de sus 5 imágenes |
| `lote`, `fecha` | Origen y fecha de captura; solo para agrupar, **no** entran al modelo |
| `calibre_mm_ref` | Diámetro medido con calibrador (mm) |
| `categoria_ref` | **Variable objetivo**: Extra, Primera, Segunda, Tercera o Fabrica |
| `color_panel` | Color/madurez según el panel (verde, verde_amarillo, amarillo) |
| `severidad_defecto_panel` | Severidad del defecto (0 a 1) según la rúbrica del panel |

📖 Detalle de cada columna, escala de severidad, criterios de `categoria_ref` y protocolo del panel: [`docs/diccionario_datos.md`](docs/diccionario_datos.md).

---

## 🧭 Matriz de consistencia

Problema, objetivos, hipótesis, variables (independientes y dependientes) e indicadores de la tesis, con una fila general y ocho específicas que se corresponden uno a uno (P1–OE1–H1 … P8–OE8–H8, como en el Anexo del Plan de tesis corregido), más la **trazabilidad** de cada hipótesis hacia este repositorio y las **observaciones de consistencia** para revisar con la asesora.

- 📓 **Notebook:** [`notebooks/Matriz_Consistencia.ipynb`](notebooks/Matriz_Consistencia.ipynb): la matriz completa en celdas de texto (visible sin ejecutar) y, al ejecutarlo, la **evidencia actual** de cada indicador con tus datos.
- 📄 Word (hoja horizontal): [`docs/Matriz_de_Consistencia.docx`](docs/Matriz_de_Consistencia.docx)
- 📝 Markdown (se ve directo en GitHub): [`docs/matriz_consistencia.md`](docs/matriz_consistencia.md)
- 📎 También se incluye como **Anexo A** del informe (`informe/Informe_EDA_FE_Baseline.docx`).
- ✏️ Fuente única: [`docs/matriz_consistencia.yaml`](docs/matriz_consistencia.yaml). Tras editarla: `python src/matriz_consistencia.py` (Word y Markdown) y `python scripts/construir_notebook_matriz.py` (notebook). Una prueba falla si el notebook queda desactualizado. Lo marcado con † es redacción propuesta para completar la matriz.

---

## 🗂️ Estructura del repositorio
```
data/
 ├── raw/              # imágenes, etiquetas.csv, panel_evaluaciones.csv, calibracion_sesiones.csv
 ├── processed/        # índices, features, partición congelada
notebooks/
 ├── EDA_basico.ipynb              # EDA: calidad de datos, balance, calibre, imágenes, atajos
 ├── Feature_Engineering.ipynb     # FE: extracción, partición, selección, ablaciones, control de fuga
 ├── Importancia_Variables.ipynb   # importancia por variable y por dimensión, consenso entre métodos
 ├── Matriz_Consistencia.ipynb     # matriz de consistencia + trazabilidad + evidencia actual de cada indicador
 ├── Hiperparametros.ipynb         # búsqueda de hiperparámetros (presupuesto fijo), calibración de θ_def/θ_col y espacio del modelo profundo
 └── Baseline_basico.ipynb         # baseline clásico + variantes de datos + informe
src/
 ├── ingesta.py                # indexa imágenes + etiquetas
 ├── particion.py              # partición train/val/test congelada (por fruto)
 ├── eda.py                    # funciones del EDA
 ├── features_imagen.py        # features de UNA vista (geometría, color CIELAB, defectos, textura)
 ├── extraer_features.py       # extracción masiva + agregación por fruto
 ├── fe_columnas.py            # conjuntos A/C, columnas prohibidas, features de dominio
 ├── transformadores.py        # winsorizador, log1p, banderas, cíclico, filtro de colinealidad
 ├── fe_pipeline.py            # construcción de Pipelines sin fuga
 ├── seleccion.py              # MI, ANOVA, correlación, VIF, permutación
 ├── importancia.py            # importancia de variables (permutación OOF, por dimensión, impureza, MI, control de ruido)
 ├── firma.py                  # firma de datos y deriva
 ├── fe_experimentos.py        # ablaciones, registro, TEST protegido
 ├── preprocesamiento.py       # (esqueleto) corrección de color/escala
 ├── segmentacion.py · atributos.py · regla_comercial.py
 ├── remuestreo.py             # variante de datos: balanceo solo en train
 ├── modelo_baseline.py        # BASELINE clásico (dummy/kNN/NB) + variantes de datos (vistas, balanceo)
 ├── busqueda_hiperparametros.py  # búsqueda acotada con presupuesto fijo por modelo (Tabla 4.2), misma partición y pliegues que el baseline
 ├── estadistica.py            # kappa, bootstrap pareado (B=1000), McNemar, Wilson, Holm, no inferioridad
 ├── calibracion.py            # MAE/RMSE/Bland-Altman del calibre y error en 34,5-42 mm (H2)
 ├── evidencia_hipotesis.py    # E4 (1 vista vs 5, McNemar), línea base de E1, F1 por clase (E3), control de atajo por sesión
 ├── informe.py                # genera el informe Word/Markdown con las cifras reales
 ├── matriz_consistencia.py    # genera docs/Matriz_de_Consistencia.docx y .md desde docs/matriz_consistencia.yaml
 ├── modelo_mvcnn.py           # arquitectura MVCNN (baseline profundo; pendiente de entrenar)
 ├── modelo_multitarea.py      # backbone compartido + cabezales
 └── api_inferencia.py
tests/                         # pruebas anti-fuga y de extracción (pytest)
scripts/                       # generar_datos_demo.py (dataset sintético) · construir_notebook_matriz.py
configs/                       # fe.yaml, baseline.yaml, hiperparametros.yaml y YAML de los experimentos profundos (E2_mvcnn, E5)
docs/                          # diccionario_datos.md, feature_engineering.md, plantillas
logs/                          # métricas, corridas de FE, test_evaluado.json
informe/                       # informe generado (Word + Markdown + figuras)
slides/
README.md · pyproject.toml · requirements.txt · .gitignore
```

---

## ⚙️ Requisitos
Instalar dependencias con `pip`:
```bash
pip install -r requirements.txt
```
O con [Poetry](https://python-poetry.org/):
```bash
poetry install
```

---

## 🔎 EDA (Análisis exploratorio)

**Formato de datos esperado** (`data/raw/`):
```
imagenes/<lote>_<fecha>_<id_fruto>_<vista>.jpg    # vista: 0 | 90 | 180 | 270 | pedunculo
etiquetas.csv   # id_fruto, lote, fecha, calibre_mm_ref, categoria_ref  [+ color_panel, severidad_defecto_panel]
```
**Ejecutar:**
```bash
python src/ingesta.py                      # genera data/processed/indice_*.csv
jupyter notebook notebooks/EDA_basico.ipynb   # con MODO_DEMO = False
```
**Probar sin datos reales** (imágenes sintéticas, solo valida el pipeline):
```bash
python scripts/generar_datos_demo.py
python src/ingesta.py --raw data/demo/raw --salida data/demo/processed
```
El notebook revisa: calidad de etiquetas, capturas incompletas, balance vs. cuotas, cobertura de lotes/fechas,
consistencia calibre–categoría, frutos fronterizos, calidad de imagen (nitidez, exposición, duplicados),
riesgo de atajos (lote/fecha/brillo) y una vista previa de la partición 70/15/15.

---

## 🧪 Feature Engineering

Convierte las 5 imágenes de cada fruto en variables y mide, con ablaciones controladas, qué aporta cada decisión **sin fuga de información**
(partición por fruto congelada, transformaciones ajustadas solo con train dentro de `Pipeline`, TEST evaluado una sola vez).
Detalle completo: [`docs/feature_engineering.md`](docs/feature_engineering.md).

```bash
python src/ingesta.py
python src/particion.py                # una sola vez: queda congelada
python src/extraer_features.py
python src/fe_experimentos.py          # A, A+B, A+C, A+B+C, A+B+C+S  (+ --control-fuga)
python -m pytest -q                    # verifica las garantías anti-fuga
```
Resultados: `logs/fe_runs/<corrida>/tabla_final.md` (media ± desviación), `logs/experimentos_fe.csv` (historial).

---

## 🚀 Cómo ejecutar el pipeline

| # | Paso | Comando | Estado |
|---|---|---|---|
| 1 | Ingesta (índices de imágenes y etiquetas) | `python src/ingesta.py` | ✅ |
| 2 | EDA | `notebooks/EDA_basico.ipynb` | ✅ |
| 3 | Partición congelada (una sola vez) | `python src/particion.py` | ✅ |
| 4 | Features desde las imágenes | `python src/extraer_features.py` | ✅ |
| 5 | Feature Engineering (ablaciones) | `python src/fe_experimentos.py` · `notebooks/Feature_Engineering.ipynb` | ✅ |
| 5b | **Importancia de variables** | `python src/importancia.py` · `notebooks/Importancia_Variables.ipynb` | ✅ |
| 6 | **Baseline + variantes de datos** | `python src/modelo_baseline.py` · `notebooks/Baseline_basico.ipynb` | ✅ |
| 7 | **Informe** (EDA + FE + importancia + baseline + variantes) | `python src/informe.py` | ✅ |
| 6b | **Búsqueda de hiperparámetros** | `python src/busqueda_hiperparametros.py` · `notebooks/Hiperparametros.ipynb` | ✅ |
| 7b | **Matriz de consistencia** | `notebooks/Matriz_Consistencia.ipynb` · `python src/matriz_consistencia.py` → `docs/Matriz_de_Consistencia.docx` | ✅ |
| 8 | Corrección de color y escala por sesión | `src/preprocesamiento.py` | 🚧 esqueleto |
| 9 | MVCNN y pipeline multitarea (GPU) | `src/modelo_mvcnn.py`, `src/modelo_multitarea.py` | 🚧 arquitectura definida, sin entrenamiento |
| 10 | Inferencia sobre un fruto nuevo | `src/api_inferencia.py` | 🚧 esqueleto |

Pruebas automáticas (anti-fuga, extracción, baseline y estadística): `python -m pytest -q`.

**Baseline y variantes de datos.** El baseline (dummy, kNN k=5 y Naive Bayes sobre las variables originales) fija la referencia mínima.
Las variantes cambian el **dato**, no el modelo: **V1** número de imágenes por fruto (1, 2, 4 ecuatoriales y 5) y **V2** balanceo de clases en train
(sobremuestreo y submuestreo). Misma partición y mismos pliegues → comparaciones pareadas con prueba t corregida y corrección de Holm.
Detalle: [`docs/baseline_variantes.md`](docs/baseline_variantes.md). Resultados: `logs/metrics_baseline.txt` y `logs/baseline_runs/<corrida>/`.

---

## 📈 Resultados esperados
- **EDA** → `notebooks/EDA_basico.ipynb`.
- **Feature Engineering** → `logs/fe_runs/<corrida>/tabla_final.md`.
- **Baseline y variantes de datos** → `logs/metrics_baseline.txt`.
- **Búsqueda de hiperparámetros** → `logs/metrics_hiperparametros.txt` y `logs/hiperparametros_runs/<corrida>/` (ver [`docs/hiperparametros.md`](docs/hiperparametros.md)).
- **Importancia de variables** → `logs/importancia_runs/<corrida>/` (ver [`docs/importancia_variables.md`](docs/importancia_variables.md)).
- **Informe** → `informe/Informe_EDA_FE_Baseline.docx` (y `.md`).
- **Matriz de consistencia** → [`docs/Matriz_de_Consistencia.docx`](docs/Matriz_de_Consistencia.docx) y [`docs/matriz_consistencia.md`](docs/matriz_consistencia.md); también es el **Anexo A** del informe.
- **Métrica central**: F1 macro (más F1, recall y PR-AUC de la clase Fábrica).
- **Más adelante**: MVCNN y multitarea → `logs/metrics_multitarea.txt`; comparación final con McNemar / bootstrap pareado sobre TEST.

---

## ⚠️ Lo que los datos actuales NO permiten concluir
Los notebooks lo comprueban automáticamente (ver `Baseline_basico` §0, `EDA_basico` §3 y la sección 4 de `Matriz_Consistencia`):
- **Cada categoría proviene de una sola sesión (lote × fecha).** Un clasificador que solo conoce la sesión ya obtiene F1 macro ≈ 0,73 en validación (el dummy, 0,07), y las variables de imagen identifican la sesión con exactitud ≈ 0,90. La condición de la sección 4.3 de la tesis (ninguna categoría concentrada en un lote o fecha) no se cumple, así que las métricas pueden reflejar el contexto y no el fruto.
- **Falta `data/raw/calibracion_sesiones.csv`**: sin la escala mm/px por sesión `diam_ecu_mm` queda vacío y H2 (MAE ≤ 1,0 mm) no es medible; los píxeles de distintas sesiones no son comparables.
- **Faltan** `panel_evaluaciones.csv` (H1), la fecha de cosecha, el tiempo cosecha–captura, el productor y el tipo de defecto.
- Hay 511 frutos (≈100 por categoría) frente a los ≥ 1 500 de la Tabla 4.1.

---

## 📌 Roadmap
- [x] Semana 2 → Diseño del flujo reproducible (diagrama, partición, riesgos).
- [x] Semana 3 → Ingesta + EDA + Feature Engineering + baseline + variantes de datos + informe.
- [x] Búsqueda de hiperparámetros (clásicos) y calibración de θ_def/θ_col; métricas y estadística de la tesis (kappa, bootstrap, McNemar).
- [ ] Semana 4 → Corrección de color/escala, segmentación, atributos, regla comercial y MVCNN.

---

## 📜 Licencia
Uso académico – Universidad Nacional de Ingeniería (UNI).
