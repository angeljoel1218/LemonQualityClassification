| Modelo | Variante | CV F1 macro | Δ vs baseline | p (Holm) | Veredicto | CV F1 Fábrica | Δ F1 Fábrica | VAL F1 macro | latencia (ms/fruto) |
|---|---|---|---|---|---|---|---|---|---|
| gnb | BASE | 0.745 ± 0.089 | - | - | referencia | 0.758 ± 0.080 | - | 0.811 | 834 |
| gnb | V1_1vista | 0.446 ± 0.071 | -0.299 | 0.000 | empeora | 0.553 ± 0.134 | -0.205 | 0.459 | 167 |
| gnb | V1_2vistas | 0.642 ± 0.141 | -0.102 | 0.266 | sin evidencia | 0.635 ± 0.115 | -0.123 | 0.761 | 333 |
| gnb | V1_4ecuatoriales | 0.703 ± 0.103 | -0.042 | 1.000 | sin evidencia | 0.699 ± 0.095 | -0.059 | 0.766 | 664 |
| gnb | V2_sobremuestreo | 0.739 ± 0.086 | -0.006 | 1.000 | sin evidencia | 0.748 ± 0.088 | -0.010 | 0.811 | 834 |
| gnb | V2_submuestreo | 0.746 ± 0.091 | +0.001 | 1.000 | sin evidencia | 0.761 ± 0.079 | +0.003 | 0.811 | 834 |
| knn | BASE | 0.866 ± 0.048 | - | - | referencia | 0.863 ± 0.077 | - | 0.908 | 835 |
| knn | V1_1vista | 0.815 ± 0.050 | -0.051 | 0.534 | sin evidencia | 0.843 ± 0.075 | -0.020 | 0.855 | 168 |
| knn | V1_2vistas | 0.845 ± 0.054 | -0.021 | 1.000 | sin evidencia | 0.858 ± 0.108 | -0.005 | 0.853 | 334 |
| knn | V1_4ecuatoriales | 0.870 ± 0.047 | +0.004 | 1.000 | sin evidencia | 0.886 ± 0.073 | +0.023 | 0.920 | 666 |
| knn | V2_sobremuestreo | 0.858 ± 0.042 | -0.008 | 1.000 | sin evidencia | 0.864 ± 0.079 | +0.000 | 0.908 | 835 |
| knn | V2_submuestreo | 0.868 ± 0.048 | +0.002 | 1.000 | sin evidencia | 0.869 ± 0.083 | +0.005 | 0.908 | 836 |
