# Modelo temporal de rendimiento de jugadores

## Métricas de evaluación

| feature_set         | model                 |       mae |         r2 |   train_rows |   test_rows | test_season   | test_next_season   |
|:--------------------|:----------------------|----------:|-----------:|-------------:|------------:|:--------------|:-------------------|
| full                | DummyRegressor        | 0.128569  | -0.0035582 |         3003 |         954 | 2023-2024     | 2024-2025          |
| full                | RandomForestRegressor | 0.0576742 |  0.724179  |         3003 |         954 | 2023-2024     | 2024-2025          |
| without_previous_xg | DummyRegressor        | 0.128569  | -0.0035582 |         3003 |         954 | 2023-2024     | 2024-2025          |
| without_previous_xg | RandomForestRegressor | 0.0600737 |  0.703034  |         3003 |         954 | 2023-2024     | 2024-2025          |

## Variables más importantes por experimento

### Experimento: full

| feature_set   | feature                      |   importance |   importance_percent |
|:--------------|:-----------------------------|-------------:|---------------------:|
| full          | numeric__xG_90               |   0.806489   |            80.6489   |
| full          | numeric__shots_90            |   0.0486353  |             4.86353  |
| full          | numeric__npxG_90             |   0.0287774  |             2.87774  |
| full          | numeric__goals_90            |   0.0156723  |             1.56723  |
| full          | numeric__npg_90              |   0.0109551  |             1.09551  |
| full          | numeric__yellow_cards_90     |   0.00980156 |             0.980156 |
| full          | numeric__xGBuildup_90        |   0.00963665 |             0.963665 |
| full          | numeric__xA_90               |   0.00950537 |             0.950537 |
| full          | numeric__assists_90          |   0.00906044 |             0.906044 |
| full          | numeric__minutes_per_game    |   0.00897518 |             0.897518 |
| full          | numeric__key_passes_90       |   0.00823553 |             0.823553 |
| full          | categorical__position_main_F |   0.00725238 |             0.725238 |
| full          | numeric__time                |   0.00710545 |             0.710545 |
| full          | numeric__xGChain_90          |   0.00657658 |             0.657658 |
| full          | numeric__games               |   0.00615191 |             0.615191 |

### Experimento: without_previous_xg

| feature_set         | feature                            |   importance |   importance_percent |
|:--------------------|:-----------------------------------|-------------:|---------------------:|
| without_previous_xg | numeric__shots_90                  |   0.539942   |            53.9942   |
| without_previous_xg | numeric__goals_90                  |   0.27073    |            27.073    |
| without_previous_xg | categorical__position_main_F       |   0.0533108  |             5.33108  |
| without_previous_xg | numeric__xGChain_90                |   0.020604   |             2.0604   |
| without_previous_xg | numeric__key_passes_90             |   0.0176043  |             1.76043  |
| without_previous_xg | numeric__xGBuildup_90              |   0.0162579  |             1.62579  |
| without_previous_xg | numeric__npg_90                    |   0.0155007  |             1.55007  |
| without_previous_xg | numeric__yellow_cards_90           |   0.0117038  |             1.17038  |
| without_previous_xg | numeric__minutes_per_game          |   0.0116812  |             1.16812  |
| without_previous_xg | numeric__assists_90                |   0.0102607  |             1.02607  |
| without_previous_xg | numeric__xA_90                     |   0.0101709  |             1.01709  |
| without_previous_xg | numeric__time                      |   0.0083462  |             0.83462  |
| without_previous_xg | numeric__games                     |   0.00570522 |             0.570522 |
| without_previous_xg | categorical__league_Premier-League |   0.0021061  |             0.21061  |
| without_previous_xg | categorical__league_Serie-A        |   0.00148347 |             0.148347 |

## Interpretación metodológica

Este experimento utiliza una partición temporal: el modelo se entrena con temporadas anteriores y se evalúa sobre el último par de temporadas disponible. Por tanto, la evaluación se aproxima más a un escenario real de predicción futura que una división aleatoria.

Se comparan dos configuraciones. La primera utiliza todas las variables históricas disponibles. La segunda elimina xG_90 y npxG_90 de la temporada anterior para comprobar si el modelo mantiene capacidad predictiva sin utilizar directamente métricas de goles esperados previos.

La variable objetivo es target_next_xG_90, que representa el xG por 90 minutos del jugador en la temporada siguiente.