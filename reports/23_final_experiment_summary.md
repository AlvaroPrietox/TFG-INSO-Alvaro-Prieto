# Resumen final de experimentos

## Objetivo

Este informe consolida los principales resultados experimentales del sistema de predicción de rendimiento ofensivo de jugadores. La variable objetivo es `xG_90` de la temporada siguiente, estimada a partir de métricas históricas de participación y producción ofensiva.

El resumen integra la validación temporal interna, la evaluación externa sobre LaLiga 2025-2026, el subconjunto de delanteros y centrocampistas, y el análisis específico de jugadores con alto `xG_90` real.

## Tabla sintética final

| evaluation_scope   | subset                             | model                         | feature_set         |   n_train |   n_eval |    mae |       r2 |   mean_signed_error | period                |   rank_mae_in_scope |
|:-------------------|:-----------------------------------|:------------------------------|:--------------------|----------:|---------:|-------:|---------:|--------------------:|:----------------------|--------------------:|
| external_2025_2026 | alto_>=0.50                        | HistGradientBoostingRegressor | without_previous_xg |           |       13 | 0.1276 |  -0.5895 |             -0.1017 | 2024-2025 → 2025-2026 |                   1 |
| external_2025_2026 | alto_>=0.50                        | Ridge                         | without_previous_xg |           |       13 | 0.1339 |  -0.7533 |             -0.0823 | 2024-2025 → 2025-2026 |                   2 |
| external_2025_2026 | alto_>=0.50                        | GradientBoostingRegressor     | without_previous_xg |           |       13 | 0.1347 |  -0.7802 |             -0.0993 | 2024-2025 → 2025-2026 |                   3 |
| external_2025_2026 | alto_>=0.50                        | RandomForestRegressor         | without_previous_xg |           |       13 | 0.151  |  -0.9749 |             -0.1236 | 2024-2025 → 2025-2026 |                   4 |
| external_2025_2026 | alto_>=0.50                        | DummyRegressor                | without_previous_xg |           |       13 | 0.4897 | -16.4869 |             -0.4897 | 2024-2025 → 2025-2026 |                   5 |
| external_2025_2026 | jugadores_emparejados_sin_porteros | Ridge                         | without_previous_xg |      3957 |      205 | 0.0516 |   0.8403 |            nan      | 2024-2025 → 2025-2026 |                   1 |
| external_2025_2026 | jugadores_emparejados_sin_porteros | HistGradientBoostingRegressor | without_previous_xg |      3957 |      205 | 0.0519 |   0.8118 |            nan      | 2024-2025 → 2025-2026 |                   2 |
| external_2025_2026 | jugadores_emparejados_sin_porteros | RandomForestRegressor         | without_previous_xg |      3957 |      205 | 0.055  |   0.8023 |            nan      | 2024-2025 → 2025-2026 |                   3 |
| external_2025_2026 | jugadores_emparejados_sin_porteros | GradientBoostingRegressor     | without_previous_xg |      3957 |      205 | 0.0563 |   0.7857 |            nan      | 2024-2025 → 2025-2026 |                   4 |
| external_2025_2026 | jugadores_emparejados_sin_porteros | DummyRegressor                | without_previous_xg |      3957 |      205 | 0.1359 |  -0.0004 |            nan      | 2024-2025 → 2025-2026 |                   5 |
| external_2025_2026 | solo_F_M                           | Ridge                         | without_previous_xg |      3957 |      110 | 0.0675 |   0.8103 |            nan      | 2024-2025 → 2025-2026 |                   1 |
| external_2025_2026 | solo_F_M                           | HistGradientBoostingRegressor | without_previous_xg |      3957 |      110 | 0.0689 |   0.7705 |            nan      | 2024-2025 → 2025-2026 |                   2 |
| external_2025_2026 | solo_F_M                           | RandomForestRegressor         | without_previous_xg |      3957 |      110 | 0.0738 |   0.7561 |            nan      | 2024-2025 → 2025-2026 |                   3 |
| external_2025_2026 | solo_F_M                           | GradientBoostingRegressor     | without_previous_xg |      3957 |      110 | 0.0753 |   0.735  |            nan      | 2024-2025 → 2025-2026 |                   4 |
| external_2025_2026 | solo_F_M                           | DummyRegressor                | without_previous_xg |      3957 |      110 | 0.163  |  -0.1913 |            nan      | 2024-2025 → 2025-2026 |                   5 |
| temporal_internal  | todos_sin_porteros                 | Ridge                         | without_previous_xg |      3003 |      954 | 0.0586 |   0.714  |            nan      | 2023-2024 → 2024-2025 |                   1 |
| temporal_internal  | todos_sin_porteros                 | GradientBoostingRegressor     | without_previous_xg |      3003 |      954 | 0.0595 |   0.7073 |            nan      | 2023-2024 → 2024-2025 |                   2 |
| temporal_internal  | todos_sin_porteros                 | RandomForestRegressor         | without_previous_xg |      3003 |      954 | 0.0601 |   0.703  |            nan      | 2023-2024 → 2024-2025 |                   3 |
| temporal_internal  | todos_sin_porteros                 | HistGradientBoostingRegressor | without_previous_xg |      3003 |      954 | 0.0614 |   0.6867 |            nan      | 2023-2024 → 2024-2025 |                   4 |
| temporal_internal  | todos_sin_porteros                 | DummyRegressor                | without_previous_xg |      3003 |      954 | 0.1286 |  -0.0036 |            nan      | 2023-2024 → 2024-2025 |                   5 |

## Comparación temporal interna

La validación temporal interna reserva el último par temporal disponible como conjunto de test, manteniendo una separación cronológica entre entrenamiento y evaluación.

| model                         | feature_set         |   train_rows |   test_rows |    mae |      r2 | test_season   | test_next_season   |
|:------------------------------|:--------------------|-------------:|------------:|-------:|--------:|:--------------|:-------------------|
| Ridge                         | without_previous_xg |         3003 |         954 | 0.0586 |  0.714  | 2023-2024     | 2024-2025          |
| GradientBoostingRegressor     | without_previous_xg |         3003 |         954 | 0.0595 |  0.7073 | 2023-2024     | 2024-2025          |
| RandomForestRegressor         | without_previous_xg |         3003 |         954 | 0.0601 |  0.703  | 2023-2024     | 2024-2025          |
| HistGradientBoostingRegressor | without_previous_xg |         3003 |         954 | 0.0614 |  0.6867 | 2023-2024     | 2024-2025          |
| DummyRegressor                | without_previous_xg |         3003 |         954 | 0.1286 | -0.0036 | 2023-2024     | 2024-2025          |

El mejor modelo en validación temporal interna es `Ridge`, con MAE = 0.0586 y R² = 0.7140.

## Evaluación externa global 2025-2026

La evaluación externa utiliza jugadores de 2024-2025 como entrada y los compara con su `xG_90` real observado en la temporada 2025-2026. Esta prueba es especialmente relevante porque actúa como validación fuera del conjunto temporal empleado durante el desarrollo inicial.

| model                         | feature_set         |   training_rows |   matched_players |    mae |      r2 | input_season   | target_season   |
|:------------------------------|:--------------------|----------------:|------------------:|-------:|--------:|:---------------|:----------------|
| Ridge                         | without_previous_xg |            3957 |               205 | 0.0516 |  0.8403 | 2024-2025      | 2025-2026       |
| HistGradientBoostingRegressor | without_previous_xg |            3957 |               205 | 0.0519 |  0.8118 | 2024-2025      | 2025-2026       |
| RandomForestRegressor         | without_previous_xg |            3957 |               205 | 0.055  |  0.8023 | 2024-2025      | 2025-2026       |
| GradientBoostingRegressor     | without_previous_xg |            3957 |               205 | 0.0563 |  0.7857 | 2024-2025      | 2025-2026       |
| DummyRegressor                | without_previous_xg |            3957 |               205 | 0.1359 | -0.0004 | 2024-2025      | 2025-2026       |

El mejor modelo en la evaluación externa global es `Ridge`, con MAE = 0.0516 y R² = 0.8403.

## Evaluación externa en F/M

El subconjunto F/M concentra delanteros y centrocampistas, por lo que está más alineado con el análisis de producción ofensiva que el conjunto global de jugadores no porteros.

| model                         | feature_set         |   training_rows |   matched_players_f_m |    mae |      r2 | input_season   | target_season   |
|:------------------------------|:--------------------|----------------:|----------------------:|-------:|--------:|:---------------|:----------------|
| Ridge                         | without_previous_xg |            3957 |                   110 | 0.0675 |  0.8103 | 2024-2025      | 2025-2026       |
| HistGradientBoostingRegressor | without_previous_xg |            3957 |                   110 | 0.0689 |  0.7705 | 2024-2025      | 2025-2026       |
| RandomForestRegressor         | without_previous_xg |            3957 |                   110 | 0.0738 |  0.7561 | 2024-2025      | 2025-2026       |
| GradientBoostingRegressor     | without_previous_xg |            3957 |                   110 | 0.0753 |  0.735  | 2024-2025      | 2025-2026       |
| DummyRegressor                | without_previous_xg |            3957 |                   110 | 0.163  | -0.1913 | 2024-2025      | 2025-2026       |

El mejor modelo en el subconjunto F/M es `Ridge`, con MAE = 0.0675 y R² = 0.8103.

## Análisis del rango alto de xG_90

Para estudiar el comportamiento del modelo en perfiles de alta producción ofensiva, se analiza de forma separada el grupo de jugadores con `xG_90` real igual o superior a 0.50.

| model                         | xg_range    |   n_players |   actual_xg_90_mean |   predicted_xg_90_mean |    mae |       r2 |   mean_signed_error |   overestimations |   underestimations |
|:------------------------------|:------------|------------:|--------------------:|-----------------------:|-------:|---------:|--------------------:|------------------:|-------------------:|
| HistGradientBoostingRegressor | alto_>=0.50 |          13 |              0.6446 |                 0.5429 | 0.1276 |  -0.5895 |             -0.1017 |                 4 |                  9 |
| Ridge                         | alto_>=0.50 |          13 |              0.6446 |                 0.5623 | 0.1339 |  -0.7533 |             -0.0823 |                 2 |                 11 |
| GradientBoostingRegressor     | alto_>=0.50 |          13 |              0.6446 |                 0.5453 | 0.1347 |  -0.7802 |             -0.0993 |                 3 |                 10 |
| RandomForestRegressor         | alto_>=0.50 |          13 |              0.6446 |                 0.521  | 0.151  |  -0.9749 |             -0.1236 |                 1 |                 12 |
| DummyRegressor                | alto_>=0.50 |          13 |              0.6446 |                 0.1549 | 0.4897 | -16.4869 |             -0.4897 |                 0 |                 13 |

En el rango alto de `xG_90`, el menor MAE corresponde a `HistGradientBoostingRegressor`, con MAE = 0.1276.

No obstante, este subconjunto contiene menos observaciones que la evaluación global, por lo que se usa principalmente como análisis de comportamiento del error y no como único criterio de selección del modelo final.

## Modelo final seleccionado

El modelo final seleccionado es `Ridge` con el conjunto de variables `without_previous_xg`.

En evaluación externa global, `Ridge` obtiene MAE = 0.0516 y R² = 0.8403.

En el subconjunto F/M, `Ridge` obtiene MAE = 0.0675 y R² = 0.8103.

La elección de Ridge se justifica por tres motivos principales: presenta el mejor rendimiento externo global, también lidera el subconjunto F/M y ofrece una estructura más simple e interpretable que los modelos de ensamblado basados en árboles.

## Conclusión experimental

Los resultados muestran que el rendimiento ofensivo futuro de los jugadores puede aproximarse con una precisión razonable a partir de variables de producción ofensiva, participación y posición. Aunque los modelos basados en árboles ofrecen resultados competitivos, Ridge presenta la mejor generalización externa en este experimento.

El análisis por rangos confirma que los jugadores de alto `xG_90` siguen siendo el grupo más difícil de predecir, debido a la mayor variabilidad de los perfiles ofensivos extremos. Por tanto, el modelo final debe interpretarse como una herramienta de apoyo analítico y no como una predicción determinista del rendimiento individual.