# 🍋 Clasificación comercial post-cosecha de limón sutil con visión por computadora

Proyecto de investigación (Maestría en IA – UNI), **alineado con el Plan de tesis v4** (`Plan_de_Tesis_Limon_UNI_v4_corregido`).
Sistema **híbrido**: una red neuronal **solo segmenta** (YOLOv8-seg; Mask R-CNN como único candidato alternativo); el **calibre**, el **color** y la
**categoría** (Extra, Primera, Segunda, Tercera, Fábrica) salen de procedimientos analíticos y de una **regla jerárquica** explicable. Se compara con un
**clasificador directo** (ResNet-50, 5 vistas con pesos compartidos, promedio de logits) y con **operarios**. Curso: Proyecto de Investigación II (MIA 403).

**Autores:** Angel Joel Alvarado Crisanto · Javier Fernandez Purizaca — **Asesor:** Ph.D. Glen Dario Rodriguez Rafael

---

## 🚀 Primeros pasos

```bash
pip install -r requirements.txt          # instala torch con CUDA según tu GPU (https://pytorch.org)
python -m pytest -q                      # verifica el entorno (no necesita GPU ni imágenes)
```

### Archivos de datos (en `data/raw/`)
| Archivo | Contenido |
|---|---|
| `imagenes/<lote>_<fecha>_<id_fruto>_<vista>.jpg` | vista: `0`, `90`, `180`, `270`, `pedunculo` |
| `etiquetas.csv` | `id_fruto, lote, fecha, calibre_mm_ref, categoria_ref` + `color_panel, severidad_defecto_panel` (0–1) o `severidad_nivel_panel` (0–3), `tipo_defecto`, **`piloto`** (1 = fruto del piloto) |
| `calibracion_sesiones.csv` | `lote, fecha, mm_por_px` (o se genera con `escalas_de_sesiones` desde `patrones/patron_<lote>_<fecha>.jpg`) |
| `panel_evaluaciones.csv` | `id_fruto, evaluador, ronda, color, severidad_nivel \| severidad_defecto, [categoria], [rol]` (ronda 2 = repetición de 50 frutos; rol `adjudicador` = 4.º evaluador) |
| `operarios_clasificaciones.csv` | `id_fruto, operario, ronda, categoria, tiempo_s` (≥ 3 operarios; ronda 2 = 50 frutos) |
| `piloto_tiempos.csv` *(opcional)* | `id_fruto, captura_s, vernier_1_mm, vernier_2_mm` |
| `configs/sesiones.yaml` *(opcional)* | parches de la carta de color por sesión (para la variante `brillo_color`) |
| `docs/autorizaciones.md`, `docs/etica.md` *(opcionales)* | licencia/autorizaciones y ética: el informe los incluye si existen; **no se inventan** |

### Orden de ejecución
| # | Paso | Comando |
|---|---|---|
| 1 | Ingesta | `python src/ingesta.py` |
| 2 | Escala por sesión (si tienes los patrones) | `python -c "from src import preprocesamiento as p; p.escalas_de_sesiones('data/raw')"` |
| 3 | Piloto: panel, calibre, tiempos; congelar umbrales | `python src/piloto.py evaluar` · `python src/piloto.py calibrar` · `python src/piloto.py congelar --theta-def 2 --p-am 0.30` |
| 4 | Partición (con la muestra completa; `--forzar` si era provisional) | `python src/particion.py` |
| 5 | Atributos clásicos y baseline previo | `python src/extraer_features.py` · `python src/modelo_baseline.py` |
| 6 | Anotación: preanotar contornos, submuestra de 500, control de calidad | `python src/anotacion.py preanotar` · `submuestra` · `control-calidad` · `iou` |
| 7 | Segmentación (GPU): entrenar y elegir con validación | `python src/segmentacion.py entrenar --tarea fruto` · `... --tarea defectos` · `python src/segmentacion.py elegir --tarea fruto` |
| 8 | Inferencia híbrida en validación (base y ablaciones E3/E5) | `python src/inferir_hibrido.py --subset val --nombre base` · `--preproc brillo --nombre e3_brillo` · `--perturbacion luz_mas --nombre e5_luz_mas` |
| 9 | Clasificador directo (GPU), una corrida por balanceo (E4) | `python src/entrenar_directo.py buscar --nombre directo_ninguno --balanceo ninguno` (`pesos`, `focal`, `aumento`, `combinacion`) |
| 10 | **Congelar** modelos y umbrales | `python src/experimentos.py congelar --directo directo_ninguno` |
| 11 | **Prueba, una sola vez** | `python src/inferir_hibrido.py --subset test --nombre final` · `python src/entrenar_directo.py predecir --nombre directo_ninguno --subset test` |
| 12 | Contrastes (HG, H1–H4) y ablaciones | `python src/experimentos.py analizar` |
| 13 | **Completar el informe Word** | `python src/informe_parcial.py --mes noviembre --docente "Nombre"` |
| — | Simulación de potencia (Anexo 2/F) · API · matriz | `python src/potencia.py` · `uvicorn src.api_inferencia:app` · `python src/matriz_consistencia.py` |

---

## 🗂️ Estructura
```
configs/   proyecto.yaml, fe.yaml, baseline.yaml, hiperparametros.yaml, [umbrales_congelados.json, modelos_congelados.json, sesiones.yaml]
data/      raw/ (imágenes y CSV), processed/ (índices, features, partición + particion_meta.json), anotaciones/, yolo/
src/
 ├─ config.py · regla_comercial.py · atributos.py          parámetros, regla jerárquica (S, P_am, intervalos), calibre d_est y agregación
 ├─ particion.py · eda.py · ingesta.py                     partición por cuotas, EDA, índices
 ├─ features_imagen.py · extraer_features.py               extracción por vista (ancho perpendicular, A_def, P_am) y agregación
 ├─ preprocesamiento.py                                    cámara, escala mm/px, brillo, carta de color, perturbaciones (E3, E5)
 ├─ anotacion.py · segmentacion.py · inferir_hibrido.py    anotación semiautomática, YOLOv8-seg / Mask R-CNN, fase online del híbrido
 ├─ modelo_directo.py · entrenar_directo.py                ResNet-50 5 vistas, balanceo (E4), búsqueda 8 configs + 3 semillas
 ├─ panel.py · operarios.py · piloto.py                    κ de Fleiss/AC1/repetibilidad, adjudicación, operarios, piloto
 ├─ estadistica.py · calibracion.py · contrastes.py        bootstrap estratificado, Holm, H1–H4, HG
 ├─ experimentos.py · protocolo.py · potencia.py           ablaciones E1–E5, prueba única + SHA-256, simulación de potencia
 ├─ informe_parcial.py · docx_util.py · matriz_consistencia.py · api_inferencia.py
 └─ (clásicos previos) fe_*.py, transformadores.py, remuestreo.py, modelo_baseline.py, busqueda_hiperparametros.py, importancia.py, informe.py
tests/     97 pruebas · notebooks/ (EDA, FE, importancia, baseline, hiperparámetros, matriz) · docs/ · informe/ · archivo_v3/
```

## 🧭 Matriz de consistencia
Anexo 1 del plan: HG + P1–OE1–H1 … P4–OE4–H4, con trazabilidad al repositorio y observaciones de consistencia.
📓 [`notebooks/Matriz_Consistencia.ipynb`](notebooks/Matriz_Consistencia.ipynb) · 📄 Word [`docs/Matriz_de_Consistencia.docx`](docs/Matriz_de_Consistencia.docx) ·
📝 Markdown [`docs/matriz_consistencia.md`](docs/matriz_consistencia.md) · ✏️ fuente única `docs/matriz_consistencia.yaml`.

## 📜 Licencia
Uso académico – Universidad Nacional de Ingeniería (UNI).
