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
- **Unidad de análisis**: el fruto individual (5 imágenes agrupadas por ID de fruto).
- **Muestra objetivo**: mínimo 1 500 frutos (≈7 500 imágenes), con cuotas mínimas por
  categoría: 250 Extra, 250 Primera, 250 Segunda, 200 Tercera, 200 Fábrica.
- **Variable objetivo**: categoría de referencia (5 clases), construida con calibre
  medido + panel de agrónomos + adjudicación de desacuerdos.
- **Versión usada**: completar al cerrar la recolección (fecha + hash del dataset).

---

## 🗂️ Estructura del repositorio
```
data/
 ├── raw/              # imágenes originales por sesión (5 vistas x fruto, sin tocar)
 ├── processed/        # imágenes corregidas (balance de blancos, color, máscaras)
notebooks/
 ├── EDA_basico.ipynb          # EDA: calidad de datos, balance, calibre, imágenes, atajos, partición
 └── Baseline_basico.ipynb     # entrenamiento del clasificador directo (baseline)
src/
 ├── ingesta.py                # indexa imágenes + etiquetas → data/processed/indice_*.csv
 ├── eda.py                    # funciones reutilizables del EDA
 ├── preprocesamiento.py       # corrección de distorsión, balance de blancos, normalización
 ├── segmentacion.py           # segmentación de instancias (Mask R-CNN / YOLOv8-seg)
 ├── atributos.py              # calibre (mm), color/madurez (CIELAB), severidad de defecto
 ├── regla_comercial.py        # regla jerárquica que asigna la categoría final
 ├── modelo_baseline.py        # clasificador directo end-to-end (CNN multi-vista)
 ├── modelo_multitarea.py      # backbone compartido + cabezales (segmentación/calibre/color)
 └── api_inferencia.py         # servicio que ejecuta el pipeline completo sobre un fruto nuevo
scripts/                       # generar_datos_demo.py (dataset sintético para probar el pipeline)
configs/                       # un YAML por experimento de ablación (E1–E6), con seeds y umbrales
logs/                          # métricas por época, resultados de pruebas estadísticas
slides/                        # presentaciones de avance
README.md
pyproject.toml
requirements.txt
.gitignore
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
python src/ingesta.py
jupyter notebook notebooks/EDA_basico.ipynb
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
