# Importancia de variables

Responde: **¿en qué variables se apoya el clasificador y tiene sentido físico?** En la regla comercial, el **calibre** decide Extra/Primera/Segunda/Tercera y el **defecto** y el **color** deciden Fábrica. Si el modelo se apoya en otra cosa (brillo de la sesión, ruido), hay que sospechar.

## Métodos (`src/importancia.py`)

| Método | Qué mide | Cuidado |
|---|---|---|
| Permutación **fuera de muestra** | En cada pliegue de la CV (solo train) se ajusta el pipeline y se mide cuánto cae el F1 macro al barajar una variable en el pliegue retenido. Media ± desviación y % de pliegues en que ayuda | Subestima variables redundantes |
| Permutación **agrupada por dimensión** | Barajar juntas Calibre, Forma, Color, Defecto o Textura; también por clase | La lectura más robusta con pocos datos |
| Impureza de Random Forest | Reducción de impureza en los árboles | Sesgada hacia variables continuas y correlacionadas |
| Información mutua | Dependencia con la categoría, sin modelo | No ve interacciones |
| **Control de ruido** | Columna aleatoria: referencia de "importancia por azar" | Una variable cuenta solo si supera el umbral y ayuda en ≥ 60 % de los pliegues |

**Por qué agrupar:** el diámetro en píxeles, en milímetros, el área y la categoría por calibre dicen lo mismo. Al barajar una sola, las otras la sustituyen y su importancia individual parece baja aunque el calibre sea decisivo. El cociente `redundancia` (caída del grupo / suma de caídas individuales) mayor que 1 lo cuantifica.

## Cómo leerlo

- **Dimensión × clase:** ¿Calibre sostiene Extra a Tercera y Defecto/Color sostienen Fábrica? Si sí, el modelo se apoya en lo que dicta la regla comercial.
- **Barras de error:** con pocos frutos la desviación suele ser tan grande como la media; una variable individual solo se considera distinguible del ruido si supera el umbral **y** ayuda en la mayoría de los pliegues.
- **Acuerdo entre métodos** (Spearman de los rangos): si es bajo, no confiar en el puesto de una variable concreta; quedarse con la dimensión.
- **Señales de alarma:** una dimensión inesperada con peso alto (¿atajo de contexto?), ruido con importancia alta (modelo inestable), desacuerdo fuerte entre métodos.

## Límites

- Importancia ≠ causalidad ni necesidad: si hay una variable equivalente, ninguna parece indispensable.
- Es importancia **para estos modelos clásicos sobre features**; no describe al modelo profundo (MVCNN/multitarea), que requerirá su propio análisis (p. ej. mapas de activación).
- Se calcula solo con train; el conjunto de prueba no se usa.

## Ejecutar

```bash
python src/importancia.py              # → logs/importancia_runs/<corrida>/
jupyter notebook notebooks/Importancia_Variables.ipynb
python src/informe.py                  # incluye la sección 4.4 si hay una corrida de importancia
```
Configuración en `configs/fe.yaml` (sección `importancia`): modelos, conjunto de variables, repeticiones, top-k y control de ruido.
