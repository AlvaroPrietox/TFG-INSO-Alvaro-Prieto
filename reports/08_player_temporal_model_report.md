# Modelo temporal de rendimiento de jugadores

## Métricas de evaluación

| model                 |       mae |          r2 |   train_rows |   test_rows | test_season   | test_next_season   |
|:----------------------|----------:|------------:|-------------:|------------:|:--------------|:-------------------|
| DummyRegressor        | 0.12659   | -0.00259795 |         3236 |        1034 | 2023-2024     | 2024-2025          |
| RandomForestRegressor | 0.0533837 |  0.740378   |         3236 |        1034 | 2023-2024     | 2024-2025          |

## Variables más importantes

| feature                      |   importance |   importance_percent |
|:-----------------------------|-------------:|---------------------:|
| numeric__xG_90               |   0.817296   |            81.7296   |
| numeric__shots_90            |   0.0546365  |             5.46365  |
| numeric__npxG_90             |   0.0205394  |             2.05394  |
| numeric__goals_90            |   0.0162049  |             1.62049  |
| numeric__npg_90              |   0.00932128 |             0.932128 |
| numeric__yellow_cards_90     |   0.00925859 |             0.925859 |
| numeric__xGBuildup_90        |   0.0085567  |             0.85567  |
| numeric__xA_90               |   0.00832768 |             0.832768 |
| numeric__minutes_per_game    |   0.00824152 |             0.824152 |
| numeric__key_passes_90       |   0.00796006 |             0.796006 |
| numeric__assists_90          |   0.00775899 |             0.775899 |
| categorical__position_main_F |   0.00691463 |             0.691463 |
| numeric__time                |   0.00651269 |             0.651269 |
| numeric__xGChain_90          |   0.00586126 |             0.586126 |
| numeric__games               |   0.00519851 |             0.519851 |

## Interpretación metodológica

Este experimento utiliza una partición temporal: el modelo se entrena con temporadas anteriores y se evalúa sobre el último par de temporadas disponible. Por tanto, la evaluación se aproxima más a un escenario real de predicción futura que una división aleatoria.

La variable objetivo es target_next_xG_90, que representa el xG por 90 minutos del jugador en la temporada siguiente.