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

## 🗂️ Estructura del repositorio
```
data/
 ├── raw/              # imágenes, etiquetas.csv, panel_evaluaciones.csv, calibracion_sesiones.csv
 ├── processed/        # índices, features, partición congelada
notebooks/
 ├── EDA_basico.ipynb              # EDA: calidad de datos, balance, calibre, imágenes, atajos
 ├── Feature_Engineering.ipynb     # FE: extracción, partición, selección, ablaciones, control de fuga
 └── Baseline_basico.ipynb         # baseline de aprendizaje profundo
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
 ├── firma.py                  # firma de datos y deriva
 ├── fe_experimentos.py        # ablaciones, registro, TEST protegido
 ├── preprocesamiento.py       # (esqueleto) corrección de color/escala
 ├── segmentacion.py · atributos.py · regla_comercial.py
 ├── modelo_baseline.py        # MVCNN (baseline)
 ├── modelo_multitarea.py      # backbone compartido + cabezales
 └── api_inferencia.py
tests/                         # pruebas anti-fuga y de extracción (pytest)
scripts/                       # generar_datos_demo.py (dataset sintético para probar el pipeline)
configs/                       # fe.yaml (Feature Engineering) y YAML por experimento (E2, E5)
docs/                          # diccionario_datos.md, feature_engineering.md, plantillas
logs/                          # métricas, corridas de FE, test_evaluado.json
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

1. **Ingesta de datos** (indexa las 5 vistas por fruto y las etiquetas)
   ```bash
   python src/ingesta.py
   ```

2. **Preprocesamiento** (corrección de color/distorsión con la carta de color y el patrón de escala)
   ```bash
   python src/preprocesamiento.py
   ```

3. **Exploración inicial**
   - Abrir y ejecutar `notebooks/EDA_basico.ipynb`.

4. **Entrenamiento del baseline** (clasificador directo end-to-end)
   ```bash
   python src/modelo_baseline.py --config configs/E2_baseline.yaml
   ```

5. **Entrenamiento del pipeline multitarea propuesto**
   ```bash
   python src/modelo_multitarea.py --config configs/E5_multitarea.yaml
   ```
   Entrena en GPU (Google Colab Pro / AWS). Genera checkpoints por semilla en `logs/`.

6. **Inferencia sobre un fruto nuevo** (una vez entrenado el modelo)
   ```bash
   python src/api_inferencia.py --fruto_id <id>
   ```
   Ejecuta segmentación → atributos → regla comercial, y devuelve la categoría final.

---

## 📈 Resultados esperados
- **EDA inicial** en `notebooks/EDA_basico.ipynb`.
- **Baseline (clasificador directo)** → métricas en `logs/metrics_baseline.txt`.
- **Pipeline multitarea propuesto** → métricas en `logs/metrics_multitarea.txt`.
- **Métrica central**: F1 macro frente al estándar experto, complementada con kappa y
  recall de la clase Fábrica.
- **Comparación baseline vs. propuesto**: McNemar / bootstrap pareado (ablación E2 vs. E5).

---

## 📌 Roadmap
- [x] Semana 2 → Diseño del flujo reproducible (diagrama, partición, riesgos).
- [ ] Semana 3 → Ingesta + preprocesamiento + EDA + baseline.
- [ ] Semana 4 → Segmentación + atributos + regla comercial + comparación con baseline.

---

## 📜 Licencia
Uso académico – Universidad Nacional de Ingeniería (UNI).
