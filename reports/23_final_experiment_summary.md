# Resumen final de experimentos

## Objetivo del resumen

Este informe sintetiza la versión final del sistema de predicción de rendimiento ofensivo. El objetivo principal es evaluar predicciones 2025-2026 generadas a partir de información 2024-2025 y compararlas con valores reales observados.

La métrica principal de selección del modelo es `xG_90`. Sobre esa métrica se comparan varios algoritmos. Una vez seleccionado Ridge, el sistema se amplía a un enfoque multi-métrica con `xG_90`, `goals_90`, `assists_90` y `xA_90`.

## Selección del modelo principal para xG_90

En la evaluación externa global 2025-2026, el mejor modelo para `xG_90` es `Ridge`, con MAE = 0.0516 y R² = 0.8403.

En el subconjunto F/M, el mejor modelo para `xG_90` es `Ridge`, con MAE = 0.0675 y R² = 0.8103.

Por tanto, `Ridge` se mantiene como modelo final, ya que ofrece el mejor rendimiento global y F/M para la métrica principal del sistema.

## Resumen tabular de selección de modelo

| summary_section           | metric_key   | metric_label    | scope                            | model                         | feature_set         |   n_players |    mae |       r2 |   rank_by_mae | input_season   | target_season   | notes                                                                    |   mean_signed_error |
|:--------------------------|:-------------|:----------------|:---------------------------------|:------------------------------|:--------------------|------------:|-------:|---------:|--------------:|:---------------|:----------------|:-------------------------------------------------------------------------|--------------------:|
| internal_temporal_xG_90   | xG_90        | Goles esperados | validación temporal interna      | Ridge                         | without_previous_xg |         954 | 0.0586 |   0.714  |             1 | 2023-2024      | 2024-2025       | Comparación interna de modelos para xG_90.                               |            nan      |
| internal_temporal_xG_90   | xG_90        | Goles esperados | validación temporal interna      | GradientBoostingRegressor     | without_previous_xg |         954 | 0.0595 |   0.7073 |             2 | 2023-2024      | 2024-2025       | Comparación interna de modelos para xG_90.                               |            nan      |
| internal_temporal_xG_90   | xG_90        | Goles esperados | validación temporal interna      | RandomForestRegressor         | without_previous_xg |         954 | 0.0601 |   0.703  |             3 | 2023-2024      | 2024-2025       | Comparación interna de modelos para xG_90.                               |            nan      |
| internal_temporal_xG_90   | xG_90        | Goles esperados | validación temporal interna      | HistGradientBoostingRegressor | without_previous_xg |         954 | 0.0614 |   0.6867 |             4 | 2023-2024      | 2024-2025       | Comparación interna de modelos para xG_90.                               |            nan      |
| internal_temporal_xG_90   | xG_90        | Goles esperados | validación temporal interna      | DummyRegressor                | without_previous_xg |         954 | 0.1286 |  -0.0036 |             5 | 2023-2024      | 2024-2025       | Comparación interna de modelos para xG_90.                               |            nan      |
| external_global_xG_90     | xG_90        | Goles esperados | evaluación externa global        | Ridge                         | without_previous_xg |         205 | 0.0516 |   0.8403 |             1 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 para xG_90.                                 |            nan      |
| external_global_xG_90     | xG_90        | Goles esperados | evaluación externa global        | HistGradientBoostingRegressor | without_previous_xg |         205 | 0.0519 |   0.8118 |             2 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 para xG_90.                                 |            nan      |
| external_global_xG_90     | xG_90        | Goles esperados | evaluación externa global        | RandomForestRegressor         | without_previous_xg |         205 | 0.055  |   0.8051 |             3 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 para xG_90.                                 |            nan      |
| external_global_xG_90     | xG_90        | Goles esperados | evaluación externa global        | GradientBoostingRegressor     | without_previous_xg |         205 | 0.0563 |   0.7857 |             4 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 para xG_90.                                 |            nan      |
| external_global_xG_90     | xG_90        | Goles esperados | evaluación externa global        | DummyRegressor                | without_previous_xg |         205 | 0.1359 |  -0.0004 |             5 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 para xG_90.                                 |            nan      |
| external_f_m_xG_90        | xG_90        | Goles esperados | evaluación externa F/M           | Ridge                         | without_previous_xg |         110 | 0.0675 |   0.8103 |             1 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 restringida a delanteros y centrocampistas. |            nan      |
| external_f_m_xG_90        | xG_90        | Goles esperados | evaluación externa F/M           | HistGradientBoostingRegressor | without_previous_xg |         110 | 0.0689 |   0.7705 |             2 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 restringida a delanteros y centrocampistas. |            nan      |
| external_f_m_xG_90        | xG_90        | Goles esperados | evaluación externa F/M           | RandomForestRegressor         | without_previous_xg |         110 | 0.0737 |   0.7599 |             3 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 restringida a delanteros y centrocampistas. |            nan      |
| external_f_m_xG_90        | xG_90        | Goles esperados | evaluación externa F/M           | GradientBoostingRegressor     | without_previous_xg |         110 | 0.0753 |   0.735  |             4 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 restringida a delanteros y centrocampistas. |            nan      |
| external_f_m_xG_90        | xG_90        | Goles esperados | evaluación externa F/M           | DummyRegressor                | without_previous_xg |         110 | 0.163  |  -0.1913 |             5 | 2024-2025      | 2025-2026       | Evaluación externa 2025-2026 restringida a delanteros y centrocampistas. |            nan      |
| external_high_xG_90_range | xG_90        | Goles esperados | jugadores con xG_90 real >= 0.50 | HistGradientBoostingRegressor | without_previous_xg |          13 | 0.1276 |  -0.5895 |             1 | 2024-2025      | 2025-2026       | Análisis complementario en perfiles ofensivos de alto xG_90.             |             -0.1017 |
| external_high_xG_90_range | xG_90        | Goles esperados | jugadores con xG_90 real >= 0.50 | Ridge                         | without_previous_xg |          13 | 0.1339 |  -0.7533 |             2 | 2024-2025      | 2025-2026       | Análisis complementario en perfiles ofensivos de alto xG_90.             |             -0.0823 |
| external_high_xG_90_range | xG_90        | Goles esperados | jugadores con xG_90 real >= 0.50 | GradientBoostingRegressor     | without_previous_xg |          13 | 0.1347 |  -0.7802 |             3 | 2024-2025      | 2025-2026       | Análisis complementario en perfiles ofensivos de alto xG_90.             |             -0.0993 |
| external_high_xG_90_range | xG_90        | Goles esperados | jugadores con xG_90 real >= 0.50 | RandomForestRegressor         | without_previous_xg |          13 | 0.1517 |  -0.9355 |             4 | 2024-2025      | 2025-2026       | Análisis complementario en perfiles ofensivos de alto xG_90.             |             -0.1277 |
| external_high_xG_90_range | xG_90        | Goles esperados | jugadores con xG_90 real >= 0.50 | DummyRegressor                | without_previous_xg |          13 | 0.4897 | -16.4869 |             5 | 2024-2025      | 2025-2026       | Análisis complementario en perfiles ofensivos de alto xG_90.             |             -0.4897 |

## Evaluación multi-métrica con Ridge

En la evaluación multi-métrica global, la métrica con mayor R² es `xG_90` (Goles esperados), con R² = 0.8403. La métrica más difícil según R² es `assists_90` (Asistencias), con R² = 0.4867.

Para la métrica principal `xG_90`, Ridge obtiene MAE = 0.0516 y R² = 0.8403 en evaluación global, y MAE = 0.0675 y R² = 0.8103 en F/M.

| summary_section              | metric_key   | metric_label          | scope                     | model   |   n_players |    mae |     r2 | input_season   | target_season   |
|:-----------------------------|:-------------|:----------------------|:--------------------------|:--------|------------:|-------:|-------:|:---------------|:----------------|
| external_multi_metric_f_m    | assists_90   | Asistencias           | evaluación externa F/M    | Ridge   |         110 | 0.0657 | 0.3936 | 2024-2025      | 2025-2026       |
| external_multi_metric_f_m    | goals_90     | Goles                 | evaluación externa F/M    | Ridge   |         110 | 0.0883 | 0.687  | 2024-2025      | 2025-2026       |
| external_multi_metric_f_m    | xA_90        | Asistencias esperadas | evaluación externa F/M    | Ridge   |         110 | 0.0449 | 0.6533 | 2024-2025      | 2025-2026       |
| external_multi_metric_f_m    | xG_90        | Goles esperados       | evaluación externa F/M    | Ridge   |         110 | 0.0675 | 0.8103 | 2024-2025      | 2025-2026       |
| external_multi_metric_global | assists_90   | Asistencias           | evaluación externa global | Ridge   |         205 | 0.0546 | 0.4867 | 2024-2025      | 2025-2026       |
| external_multi_metric_global | goals_90     | Goles                 | evaluación externa global | Ridge   |         205 | 0.0658 | 0.7353 | 2024-2025      | 2025-2026       |
| external_multi_metric_global | xA_90        | Asistencias esperadas | evaluación externa global | Ridge   |         205 | 0.0398 | 0.6918 | 2024-2025      | 2025-2026       |
| external_multi_metric_global | xG_90        | Goles esperados       | evaluación externa global | Ridge   |         205 | 0.0516 | 0.8403 | 2024-2025      | 2025-2026       |

## Predicciones prospectivas multi-métrica

Además de la evaluación externa, el sistema genera una aplicación prospectiva sobre jugadores actuales. Estas predicciones no forman parte de la validación principal porque todavía no existe un valor real futuro contra el que compararlas.

| metric_key   | metric_label          | model   |   players_evaluated |   players_f_m_evaluated |   prediction_mean |   prediction_max |   prediction_min | top_player         | top_player_team   |   top_player_prediction | notes                                                             |
|:-------------|:----------------------|:--------|--------------------:|------------------------:|------------------:|-----------------:|-----------------:|:-------------------|:------------------|------------------------:|:------------------------------------------------------------------|
| xG_90        | Goles esperados       | Ridge   |                 309 |                     175 |            0.1391 |           0.8354 |          -0.0511 | Robert Lewandowski | Barcelona         |                0.835401 | Predicción prospectiva no evaluada todavía con valor real futuro. |
| goals_90     | Goles                 | Ridge   |                 309 |                     175 |            0.1265 |           0.864  |          -0.0178 | Robert Lewandowski | Barcelona         |                0.864044 | Predicción prospectiva no evaluada todavía con valor real futuro. |
| assists_90   | Asistencias           | Ridge   |                 309 |                     175 |            0.0986 |           0.369  |          -0.0164 | Raphinha           | Barcelona         |                0.368963 | Predicción prospectiva no evaluada todavía con valor real futuro. |
| xA_90        | Asistencias esperadas | Ridge   |                 309 |                     175 |            0.107  |           0.3929 |          -0.015  | Raphinha           | Barcelona         |                0.392873 | Predicción prospectiva no evaluada todavía con valor real futuro. |

## Análisis complementario de alto xG_90

En el grupo de jugadores con `xG_90` real igual o superior a 0.50, el menor MAE lo obtiene `HistGradientBoostingRegressor`, con MAE = 0.1276. Este resultado se interpreta como complementario, porque el subconjunto de jugadores de alto volumen ofensivo tiene menor tamaño muestral y mayor variabilidad.

## Archivos finales generados

- `data/processed/final_experiment_summary.csv`
- `data/processed/final_multi_metric_summary.csv`
- `data/processed/final_prospective_prediction_summary.csv`
- `data/processed/external_2025_2026_multi_metric_predictions.csv`
- `data/processed/laliga_current_player_predictions_final_ridge.csv`
- `models/ridge_temporal_multi_metric_metadata.json`

## Conclusión

La versión final del sistema queda estructurada como un modelo Ridge multi-métrica. `xG_90` se mantiene como variable principal por su estabilidad y por el rendimiento obtenido en evaluación externa. Las métricas `goals_90`, `assists_90` y `xA_90` amplían la utilidad analítica del sistema al incorporar producción real, asistencias reales y creación esperada de ocasiones.

Los resultados muestran que las métricas esperadas y la producción goleadora son más predecibles que las asistencias reales, lo cual es coherente con la dependencia de las asistencias respecto al acierto finalizador de los compañeros y al contexto colectivo.