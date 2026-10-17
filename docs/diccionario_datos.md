# Diccionario de datos

Describe los archivos de datos del proyecto, sus columnas, los valores permitidos y los criterios para llenarlos.

> ⚠️ Los criterios marcados como **(propuesta)** son una sugerencia de trabajo y deben validarse con la asesora y el panel de agrónomos durante el piloto antes de considerarse definitivos.

---

## 1. Convención de nombres de imágenes

```
data/raw/imagenes/<lote>_<fecha>_<id_fruto>_<vista>.jpg
```
Ejemplo: `L1_20260915_F0001_90.jpg`

| Campo | Significado | Formato |
|---|---|---|
| `lote` | Grupo de fruta que llegó junto y comparte origen/entrega (productor, parcela o entrega; **definir con la asesora**) | Código corto sin guiones bajos (`L1`, `L01`) |
| `fecha` | Fecha de captura | `AAAAMMDD` |
| `id_fruto` | Identificador único del fruto | `F0001`, `F0002`… |
| `vista` | Posición de la captura | `0`, `90`, `180`, `270` (ecuatoriales) o `pedunculo` |

Cada fruto tiene **5 imágenes**. La categoría **no** se codifica en el nombre del archivo (evita fuga de información y porque se conoce después de medir y evaluar).

---

## 2. `data/raw/etiquetas.csv` — una fila por fruto

Estándar de referencia contra el que se entrena y evalúa el sistema. **Nunca** se llena con predicciones del modelo ni con valores estimados a partir de las imágenes.

| Columna | Tipo | Valores / formato | Quién la llena y cuándo |
|---|---|---|---|
| `id_fruto` | texto | Igual al del nombre de las 5 imágenes | Equipo, al capturar |
| `lote` | texto | Igual al del nombre del archivo | Equipo, al capturar |
| `fecha` | texto | `AAAAMMDD` (guardar como texto) | Equipo, al capturar |
| `calibre_mm_ref` | número | mm con punto decimal, rango válido 25–60 | Equipo, medido con calibrador al capturar |
| `categoria_ref` | texto | `Extra`, `Primera`, `Segunda`, `Tercera`, `Fabrica` | Se deriva al final (ver §4) |
| `color_panel` | texto | `verde`, `verde_amarillo`, `amarillo` | Panel, valor final adjudicado |
| `severidad_defecto_panel` | número | 0 a 1 (ver §3) | Panel, valor final adjudicado |
| `tipo_defecto` *(opcional)* | texto | rozadura, cicatriz, mancha, daño_mecanico, plaga, podredumbre_hongo, otro | Panel |

Variables de agrupamiento (`lote`, `fecha`) **no entran como entrada del modelo**; solo sirven para la partición y para detectar atajos.

---

## 3. Severidad del defecto — `severidad_defecto_panel` **(propuesta)**

Nota ordinal de qué tan grave es el defecto de la cáscara. Se califica el **peor defecto** del fruto (coherente con la regla "basta una vista con defecto").

| Valor | Nivel | Criterio referencial |
|---|---|---|
| 0.00 | Sin defecto | Cáscara limpia |
| 0.25 | Leve | Marcas superficiales menores, < ~5 % de la superficie |
| 0.50 | Moderado | Claramente visible, ~5–15 %, sin comprometer la pulpa |
| 0.75 | Severo | ~15–30 % de la superficie, cicatrices o manchas extensas |
| 1.00 | Muy severo | > ~30 %, o daño grave sin importar el área (podredumbre, moho, perforación por plaga, herida abierta, golpe que ablanda) |

Los porcentajes son referenciales. El umbral `θ_def` (severidad desde la cual el fruto es Fábrica) se fija en el piloto con el panel.

Buenas prácticas: atlas de referencia con una foto por nivel y tipo de defecto, y una sesión corta de calibración del panel antes de empezar.

---

## 4. Cómo se decide `categoria_ref`

Mismo orden que la regla comercial, pero aplicado a **mediciones físicas y al juicio del panel**, no a estimaciones del modelo:

1. Defecto con severidad ≥ `θ_def`, **o** color `amarillo` → **Fabrica**.
2. Si no, por calibre medido: ≥ 39,0 mm → **Extra**; 37,0–38,9 → **Primera**; 35,0–36,9 → **Segunda**; < 35,0 → **Tercera**.

Un fruto puede ser Fábrica con severidad 0 si su color es amarillo. Si un fruto marcado Fábrica no cumple ni el criterio de defecto ni el de color, revisar la etiqueta.

---

## 5. `data/raw/panel_evaluaciones.csv` — una fila por fruto × evaluador × ronda

Registro **individual y sin adjudicar** de cada agrónomo. Es la fuente para medir la concordancia del panel y para decidir los valores finales de `etiquetas.csv`.

| Columna | Valores / formato | Descripción |
|---|---|---|
| `id_fruto` | Igual que en `etiquetas.csv` | Fruto evaluado |
| `evaluador` | `E1`, `E2`, `E3`… | Código del agrónomo (no su nombre) |
| `ronda` | `1` o `2` | `1` = evaluación inicial; `2` = re-evaluación de una submuestra, para repetibilidad |
| `color` | `verde`, `verde_amarillo`, `amarillo` | Juicio individual de color/madurez |
| `severidad_defecto` | 0, 0.25, 0.5, 0.75, 1 | Juicio individual de severidad |
| `tipo_defecto` | ver §2 | Tipo del peor defecto (vacío si no hay) |
| `observaciones` | texto libre | Dudas o casos raros |
| `fecha_evaluacion` | `AAAAMMDD` | Cuándo evaluó |

**Evaluación ciega:** el evaluador no ve el calibre, la categoría ni las notas de los otros. Por eso este archivo **no** contiene calibre ni categoría.

Usos: kappa de Fleiss (color, nominal) y kappa ponderado o ICC (severidad, ordinal) entre evaluadores; identificar desacuerdos para adjudicar; repetibilidad intra-evaluador con la ronda 2; trazabilidad de cada valor final.

---

## 6. `data/raw/calibracion_sesiones.csv` — una fila por sesión de captura

Factor de escala para convertir píxeles a milímetros. Se calcula en cada sesión con el **patrón de tamaño conocido** del encuadre: `mm_por_px = tamaño real del patrón (mm) / tamaño del patrón en la imagen (px)`.

| Columna | Descripción |
|---|---|
| `lote`, `fecha` | Igual que en los nombres de imagen (una sesión = un lote en una fecha) |
| `mm_por_px` | Milímetros por píxel (p. ej. `0.1818`) |

Sin este archivo, las features en milímetros (calibre) no se pueden calcular.

---

## 7. Capturas incompletas

- Falta **1 vista ecuatorial** → el calibre y el color se recalculan con las vistas disponibles y el fruto se marca como "captura parcial".
- Falta el **pedúnculo** o **más de una** vista ecuatorial → el fruto se excluye del conjunto definitivo y se registra en `logs/capturas_incompletas.csv`.

---

## 8. Errores comunes al llenar los CSV en Excel

- Guardar como **CSV UTF-8 delimitado por comas** (la configuración regional puede usar `;`).
- Decimal con **punto** (`40.8`, no `40,8`).
- `fecha` como **texto**, porque Excel suele convertir `20260915` en número o fecha.
- Categorías escritas exactamente como en §2 (se normalizan tildes y mayúsculas, no erratas).
- `id_fruto` idéntico al del nombre de las imágenes.
