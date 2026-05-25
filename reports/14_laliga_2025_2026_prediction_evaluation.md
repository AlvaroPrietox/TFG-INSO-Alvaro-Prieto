# Evaluación de predicciones 2025-2026 en LaLiga

## Planteamiento

Este experimento utiliza datos de jugadores de la temporada 2024-2025 para predecir su `xG_90` en la temporada 2025-2026. Posteriormente, las predicciones se comparan con los valores reales de LaLiga 2025-2026 filtrados por un mínimo de 900 minutos.

## Cobertura de emparejamiento

- Jugadores disponibles en 2024-2025: 1437
- Jugadores disponibles en objetivo 2025-2026: 345
- Jugadores emparejados por nombre normalizado: 205

## Métricas

- MAE: 0.0550
- R²: 0.8023

## Muestra representativa

| player_name           | team_title      | target_team_title   | position_main   |   predicted_2025_2026_xG_90 |   actual_2025_2026_xG_90 |   signed_error |   absolute_error |
|:----------------------|:----------------|:--------------------|:----------------|----------------------------:|-------------------------:|---------------:|-----------------:|
| Jon Moncayola         | Osasuna         | Osasuna             | M               |                      0.0699 |                     0.07 |        -0.0001 |           0.0001 |
| Sergio Carreira       | Celta Vigo      | Celta Vigo          | D               |                      0.0602 |                     0.06 |         0.0002 |           0.0002 |
| Ante Budimir          | Osasuna         | Osasuna             | F               |                      0.5792 |                     0.58 |        -0.0008 |           0.0008 |
| Pol Lozano            | Espanyol        | Espanyol            | M               |                      0.0314 |                     0.03 |         0.0014 |           0.0014 |
| Eduardo Camavinga     | Real Madrid     | Real Madrid         | D               |                      0.068  |                     0.07 |        -0.002  |           0.002  |
| Manuel Sánchez        | Alaves          | Levante             | D               |                      0.037  |                     0.04 |        -0.003  |           0.003  |
| Luis Rioja            | Alaves,Valencia | Valencia            | D               |                      0.0932 |                     0.09 |         0.0032 |           0.0032 |
| Leo Petrot            | Saint-Etienne   | Elche               | D               |                      0.0334 |                     0.03 |         0.0034 |           0.0034 |
| Alex Berenguer        | Athletic Club   | Athletic Club       | F               |                      0.2526 |                     0.22 |         0.0326 |           0.0326 |
| Djené Dakonam         | Getafe          | Getafe              | D               |                      0.0328 |                     0    |         0.0328 |           0.0328 |
| Fran García           | Real Madrid     | Real Madrid         | D               |                      0.0571 |                     0.09 |        -0.0329 |           0.0329 |
| César Tárrega         | Valencia        | Valencia            | D               |                      0.054  |                     0.02 |         0.034  |           0.034  |
| Pau Cubarsí           | Barcelona       | Barcelona           | D               |                      0.0541 |                     0.02 |         0.0341 |           0.0341 |
| Antonio Blanco        | Alaves          | Alaves              | M               |                      0.0643 |                     0.03 |         0.0343 |           0.0343 |
| Rubén García          | Osasuna         | Osasuna             | F               |                      0.1346 |                     0.1  |         0.0346 |           0.0346 |
| Kike Salas            | Sevilla         | Sevilla             | D               |                      0.0746 |                     0.04 |         0.0346 |           0.0346 |
| Robert Lewandowski    | Barcelona       | Barcelona           | F               |                      0.7016 |                     0.89 |        -0.1884 |           0.1884 |
| Raphinha              | Barcelona       | Barcelona           | F               |                      0.4884 |                     0.7  |        -0.2116 |           0.2116 |
| Álex Baena            | Villarreal      | Atletico Madrid     | F               |                      0.3186 |                     0.1  |         0.2186 |           0.2186 |
| Abdessamad Ezzalzouli | Real Betis      | Real Betis          | M               |                      0.1881 |                     0.41 |        -0.2219 |           0.2219 |
| Mikel Oyarzabal       | Real Sociedad   | Real Sociedad       | F               |                      0.3003 |                     0.53 |        -0.2297 |           0.2297 |
| Antoine Griezmann     | Atletico Madrid | Atletico Madrid     | F               |                      0.2736 |                     0.52 |        -0.2464 |           0.2464 |
| Oihan Sancet          | Athletic Club   | Athletic Club       | F               |                      0.5982 |                     0.35 |         0.2482 |           0.2482 |
| Vedat Muriqi          | Mallorca        | Mallorca            | F               |                      0.3537 |                     0.62 |        -0.2663 |           0.2663 |
| Ayoze Pérez           | Villarreal      | Villarreal          | F               |                      0.793  |                     0.46 |         0.333  |           0.333  |

## Interpretación

Esta evaluación permite comprobar el comportamiento del modelo sobre una temporada no utilizada como objetivo durante la fase inicial del pipeline. Debe interpretarse con cautela porque el emparejamiento se realiza por nombre normalizado, no por identificador único de jugador.

Los errores altos pueden deberse a cambios de equipo, cambios de rol, lesiones, variaciones tácticas o incrementos abruptos de rendimiento que el modelo no observa directamente en las variables de entrada.