# Comparación externa de modelos en LaLiga 2025-2026

## Objetivo

Este experimento compara distintos algoritmos de regresión en una evaluación externa. Cada modelo se entrena con los pares temporales históricos disponibles y se aplica a jugadores de 2024-2025 para predecir su `xG_90` real en LaLiga 2025-2026.

El conjunto de variables utilizado es `without_previous_xg`, por lo que no se emplean `xG_90` ni `npxG_90` previos como predictores directos.

## Resultados globales

| model                         | feature_set         |   training_rows |   matched_players |    mae |      r2 | input_season   | target_season   |
|:------------------------------|:--------------------|----------------:|------------------:|-------:|--------:|:---------------|:----------------|
| Ridge                         | without_previous_xg |            3957 |               205 | 0.0516 |  0.8403 | 2024-2025      | 2025-2026       |
| HistGradientBoostingRegressor | without_previous_xg |            3957 |               205 | 0.0519 |  0.8118 | 2024-2025      | 2025-2026       |
| RandomForestRegressor         | without_previous_xg |            3957 |               205 | 0.055  |  0.8023 | 2024-2025      | 2025-2026       |
| GradientBoostingRegressor     | without_previous_xg |            3957 |               205 | 0.0563 |  0.7857 | 2024-2025      | 2025-2026       |
| DummyRegressor                | without_previous_xg |            3957 |               205 | 0.1359 | -0.0004 | 2024-2025      | 2025-2026       |

## Resultados F/M

| model                         | feature_set         |   training_rows |   matched_players_f_m |    mae |      r2 | input_season   | target_season   |
|:------------------------------|:--------------------|----------------:|----------------------:|-------:|--------:|:---------------|:----------------|
| Ridge                         | without_previous_xg |            3957 |                   110 | 0.0675 |  0.8103 | 2024-2025      | 2025-2026       |
| HistGradientBoostingRegressor | without_previous_xg |            3957 |                   110 | 0.0689 |  0.7705 | 2024-2025      | 2025-2026       |
| RandomForestRegressor         | without_previous_xg |            3957 |                   110 | 0.0738 |  0.7561 | 2024-2025      | 2025-2026       |
| GradientBoostingRegressor     | without_previous_xg |            3957 |                   110 | 0.0753 |  0.735  | 2024-2025      | 2025-2026       |
| DummyRegressor                | without_previous_xg |            3957 |                   110 | 0.163  | -0.1913 | 2024-2025      | 2025-2026       |

## Resultados por rango de xG_90 real

| model                         | xg_range             |   n_players |   actual_xg_90_mean |   predicted_xg_90_mean |    mae |       r2 |   mean_signed_error |   overestimations |   underestimations |
|:------------------------------|:---------------------|------------:|--------------------:|-----------------------:|-------:|---------:|--------------------:|------------------:|-------------------:|
| HistGradientBoostingRegressor | alto_>=0.50          |          13 |              0.6446 |                 0.5429 | 0.1276 |  -0.5895 |             -0.1017 |                 4 |                  9 |
| Ridge                         | alto_>=0.50          |          13 |              0.6446 |                 0.5623 | 0.1339 |  -0.7533 |             -0.0823 |                 2 |                 11 |
| GradientBoostingRegressor     | alto_>=0.50          |          13 |              0.6446 |                 0.5453 | 0.1347 |  -0.7802 |             -0.0993 |                 3 |                 10 |
| RandomForestRegressor         | alto_>=0.50          |          13 |              0.6446 |                 0.521  | 0.151  |  -0.9749 |             -0.1236 |                 1 |                 12 |
| DummyRegressor                | alto_>=0.50          |          13 |              0.6446 |                 0.1549 | 0.4897 | -16.4869 |             -0.4897 |                 0 |                 13 |
| HistGradientBoostingRegressor | bajo_<0.10           |         114 |              0.0439 |                 0.0637 | 0.0289 |  -1.2902 |              0.0198 |                88 |                 26 |
| RandomForestRegressor         | bajo_<0.10           |         114 |              0.0439 |                 0.0661 | 0.0302 |  -1.3627 |              0.0223 |                89 |                 25 |
| Ridge                         | bajo_<0.10           |         114 |              0.0439 |                 0.0627 | 0.0302 |  -1.3414 |              0.0188 |                82 |                 32 |
| GradientBoostingRegressor     | bajo_<0.10           |         114 |              0.0439 |                 0.066  | 0.0305 |  -1.4301 |              0.0221 |                89 |                 25 |
| DummyRegressor                | bajo_<0.10           |         114 |              0.0439 |                 0.1549 | 0.111  | -17.9416 |              0.111  |               114 |                  0 |
| Ridge                         | medio_alto_0.25_0.50 |          34 |              0.3653 |                 0.3296 | 0.0744 |  -0.8005 |             -0.0357 |                 8 |                 26 |
| HistGradientBoostingRegressor | medio_alto_0.25_0.50 |          34 |              0.3653 |                 0.3399 | 0.0874 |  -2.0519 |             -0.0254 |                11 |                 23 |
| RandomForestRegressor         | medio_alto_0.25_0.50 |          34 |              0.3653 |                 0.3316 | 0.0876 |  -1.9787 |             -0.0337 |                 8 |                 26 |
| GradientBoostingRegressor     | medio_alto_0.25_0.50 |          34 |              0.3653 |                 0.3321 | 0.0949 |  -2.3914 |             -0.0332 |                10 |                 24 |
| DummyRegressor                | medio_alto_0.25_0.50 |          34 |              0.3653 |                 0.1549 | 0.2104 | -10.3791 |             -0.2104 |                 0 |                 34 |
| DummyRegressor                | medio_bajo_0.10_0.25 |          44 |              0.1525 |                 0.1549 | 0.0384 |  -0.0029 |              0.0024 |                24 |                 20 |
| HistGradientBoostingRegressor | medio_bajo_0.10_0.25 |          44 |              0.1525 |                 0.1569 | 0.0615 |  -2.5581 |              0.0044 |                18 |                 26 |
| Ridge                         | medio_bajo_0.10_0.25 |          44 |              0.1525 |                 0.1591 | 0.065  |  -2.0934 |              0.0066 |                22 |                 22 |
| RandomForestRegressor         | medio_bajo_0.10_0.25 |          44 |              0.1525 |                 0.1613 | 0.0658 |  -2.4854 |              0.0088 |                18 |                 26 |
| GradientBoostingRegressor     | medio_bajo_0.10_0.25 |          44 |              0.1525 |                 0.1606 | 0.0704 |  -3.4167 |              0.0081 |                19 |                 25 |

## Figuras generadas

![Comparación externa MAE](figures/external_2025_2026_model_comparison_mae.png)

![Comparación externa R²](figures/external_2025_2026_model_comparison_r2.png)

![Comparación externa F/M MAE](figures/external_2025_2026_model_comparison_f_m_mae.png)

## Interpretación

El menor MAE global lo obtiene `Ridge`, con MAE = 0.0516 y R² = 0.8403.

En el subconjunto F/M, el menor MAE lo obtiene `Ridge`, con MAE = 0.0675 y R² = 0.8103.

En jugadores con `xG_90 >= 0.50`, el menor MAE corresponde a `HistGradientBoostingRegressor`, con MAE = 0.1276.

Esta comparación permite decidir si el modelo final debe mantenerse como Random Forest o si una alternativa más simple o más robusta, como Ridge o Gradient Boosting, ofrece mejor generalización externa.