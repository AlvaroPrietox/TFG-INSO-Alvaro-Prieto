# Análisis del baseline de rendimiento de jugadores

## Métricas de evaluación

| model                 |       mae |          r2 |
|:----------------------|----------:|------------:|
| DummyRegressor        | 0.116936  | -0.00148959 |
| RandomForestRegressor | 0.0327023 |  0.900833   |

## Variables más importantes

| feature                            |   importance |   importance_percent |
|:-----------------------------------|-------------:|---------------------:|
| numeric__shots_90                  |  0.84386     |           84.386     |
| numeric__xGChain_90                |  0.0670766   |            6.70766   |
| numeric__xGBuildup_90              |  0.0367917   |            3.67917   |
| numeric__key_passes_90             |  0.0195788   |            1.95788   |
| categorical__position_main_F       |  0.00978392  |            0.978392  |
| numeric__xA_90                     |  0.00899867  |            0.899867  |
| numeric__minutes_per_game          |  0.00396208  |            0.396208  |
| numeric__time                      |  0.00266677  |            0.266677  |
| numeric__assists_90                |  0.00226248  |            0.226248  |
| numeric__yellow_cards_90           |  0.00225863  |            0.225863  |
| numeric__games                     |  0.00159091  |            0.159091  |
| categorical__league_Premier-League |  0.00018117  |            0.018117  |
| categorical__league_Serie-A        |  0.000180313 |            0.0180313 |
| numeric__red_cards_90              |  0.000147443 |            0.0147443 |
| categorical__position_main_M       |  0.000146199 |            0.0146199 |

## Interpretación preliminar

El análisis de importancia de variables permite comprobar qué atributos utiliza principalmente el modelo para estimar el rendimiento ofensivo esperado del jugador.

Si las primeras variables están muy próximas conceptualmente a la variable objetivo, el resultado debe interpretarse como un modelo descriptivo del rendimiento agregado, no todavía como una predicción futura estricta.