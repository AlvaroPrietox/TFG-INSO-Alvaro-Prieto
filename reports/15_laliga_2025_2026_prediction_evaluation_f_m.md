# Evaluación 2025-2026 en jugadores F/M

## Objetivo

Este informe analiza el rendimiento del modelo únicamente en jugadores con posición principal `F` o `M`. Este subconjunto es especialmente relevante porque la variable objetivo, `xG_90`, representa una métrica ofensiva cuya interpretación es más significativa en delanteros, atacantes y centrocampistas.

## Cobertura

- Jugadores emparejados totales: 205
- Jugadores F/M analizados: 110

## Métricas globales F/M

- MAE F/M: 0.0738
- R² F/M: 0.7561

## Métricas por posición

| position_main   |   n_players |    mae |     r2 |
|:----------------|------------:|-------:|-------:|
| F               |          61 | 0.0995 | 0.6258 |
| M               |          49 | 0.0418 | 0.6919 |

## Muestra representativa F/M

| player_name           | position_main   | team_title      | target_team_title   |   games |   time |   target_games |   target_time |   predicted_2025_2026_xG_90 |   actual_2025_2026_xG_90 |   signed_error |   absolute_error |
|:----------------------|:----------------|:----------------|:--------------------|--------:|-------:|---------------:|--------------:|----------------------------:|-------------------------:|---------------:|-----------------:|
| Jon Moncayola         | M               | Osasuna         | Osasuna             |      32 |   2144 |             36 |          3061 |                      0.0699 |                     0.07 |        -0.0001 |           0.0001 |
| Ante Budimir          | F               | Osasuna         | Osasuna             |      38 |   3008 |             37 |          2995 |                      0.5792 |                     0.58 |        -0.0008 |           0.0008 |
| Pol Lozano            | M               | Espanyol        | Espanyol            |      30 |   2015 |             33 |          2211 |                      0.0314 |                     0.03 |         0.0014 |           0.0014 |
| Frenkie de Jong       | M               | Barcelona       | Barcelona           |      26 |   1105 |             25 |          1597 |                      0.0651 |                     0.06 |         0.0051 |           0.0051 |
| Iván Martín           | M               | Girona          | Girona              |      32 |   2054 |             35 |          2558 |                      0.0454 |                     0.04 |         0.0054 |           0.0054 |
| Jon Guridi            | F               | Alaves          | Alaves              |      33 |   2149 |             28 |          1251 |                      0.1261 |                     0.12 |         0.0061 |           0.0061 |
| Koke                  | M               | Atletico Madrid | Atletico Madrid     |      32 |   1915 |             34 |          2237 |                      0.0468 |                     0.04 |         0.0068 |           0.0068 |
| Beñat Turrientes      | M               | Real Sociedad   | Real Sociedad       |      21 |    933 |             27 |          1355 |                      0.0519 |                     0.06 |        -0.0081 |           0.0081 |
| Javier Guerra         | F               | Valencia        | Valencia            |      36 |   2642 |             36 |          2086 |                      0.1757 |                     0.13 |         0.0457 |           0.0457 |
| Fran Beltrán          | M               | Celta Vigo      | Celta Vigo,Girona   |      34 |   2443 |             27 |          1544 |                      0.0859 |                     0.04 |         0.0459 |           0.0459 |
| Santiago Comesaña     | M               | Villarreal      | Villarreal          |      35 |   2597 |             34 |          2428 |                      0.1338 |                     0.18 |        -0.0462 |           0.0462 |
| Mikel Jauregizar      | M               | Athletic Club   | Athletic Club       |      34 |   2244 |             36 |          2872 |                      0.0875 |                     0.04 |         0.0475 |           0.0475 |
| Bryan Zaragoza        | F               | Osasuna         | Celta Vigo          |      27 |   1887 |             19 |          1138 |                      0.1979 |                     0.15 |         0.0479 |           0.0479 |
| Borja Mayoral         | F               | Getafe          | Getafe              |      24 |   1064 |             16 |          1072 |                      0.3398 |                     0.29 |         0.0498 |           0.0498 |
| Isaac Romero          | F               | Sevilla         | Sevilla             |      31 |   2200 |             29 |          1506 |                      0.2801 |                     0.33 |        -0.0499 |           0.0499 |
| Pablo Ibáñez          | M               | Osasuna         | Alaves              |      28 |   1286 |             37 |          2278 |                      0.106  |                     0.16 |        -0.054  |           0.054  |
| Robert Lewandowski    | F               | Barcelona       | Barcelona           |      34 |   2716 |             31 |          1630 |                      0.7016 |                     0.89 |        -0.1884 |           0.1884 |
| Raphinha              | F               | Barcelona       | Barcelona           |      36 |   2862 |             22 |          1424 |                      0.4884 |                     0.7  |        -0.2116 |           0.2116 |
| Álex Baena            | F               | Villarreal      | Atletico Madrid     |      32 |   2639 |             27 |          1615 |                      0.3186 |                     0.1  |         0.2186 |           0.2186 |
| Abdessamad Ezzalzouli | M               | Real Betis      | Real Betis          |      32 |   2038 |             29 |          2275 |                      0.1881 |                     0.41 |        -0.2219 |           0.2219 |
| Mikel Oyarzabal       | F               | Real Sociedad   | Real Sociedad       |      35 |   2291 |             34 |          2727 |                      0.3003 |                     0.53 |        -0.2297 |           0.2297 |
| Antoine Griezmann     | F               | Atletico Madrid | Atletico Madrid     |      38 |   2514 |             34 |          1428 |                      0.2736 |                     0.52 |        -0.2464 |           0.2464 |
| Oihan Sancet          | F               | Athletic Club   | Athletic Club       |      29 |   1645 |             28 |          1832 |                      0.5982 |                     0.35 |         0.2482 |           0.2482 |
| Vedat Muriqi          | F               | Mallorca        | Mallorca            |      29 |   2081 |             37 |          3150 |                      0.3537 |                     0.62 |        -0.2663 |           0.2663 |
| Ayoze Pérez           | F               | Villarreal      | Villarreal          |      30 |   1987 |             26 |          1154 |                      0.793  |                     0.46 |         0.333  |           0.333  |

## Diez mejores aciertos F/M

| player_name           | position_main   | team_title      | target_team_title   |   games |   time |   target_games |   target_time |   predicted_2025_2026_xG_90 |   actual_2025_2026_xG_90 |   signed_error |   absolute_error |
|:----------------------|:----------------|:----------------|:--------------------|--------:|-------:|---------------:|--------------:|----------------------------:|-------------------------:|---------------:|-----------------:|
| Jon Moncayola         | M               | Osasuna         | Osasuna             |      32 |   2144 |             36 |          3061 |                      0.0699 |                     0.07 |        -0.0001 |           0.0001 |
| Ante Budimir          | F               | Osasuna         | Osasuna             |      38 |   3008 |             37 |          2995 |                      0.5792 |                     0.58 |        -0.0008 |           0.0008 |
| Pol Lozano            | M               | Espanyol        | Espanyol            |      30 |   2015 |             33 |          2211 |                      0.0314 |                     0.03 |         0.0014 |           0.0014 |
| Frenkie de Jong       | M               | Barcelona       | Barcelona           |      26 |   1105 |             25 |          1597 |                      0.0651 |                     0.06 |         0.0051 |           0.0051 |
| Iván Martín           | M               | Girona          | Girona              |      32 |   2054 |             35 |          2558 |                      0.0454 |                     0.04 |         0.0054 |           0.0054 |
| Jon Guridi            | F               | Alaves          | Alaves              |      33 |   2149 |             28 |          1251 |                      0.1261 |                     0.12 |         0.0061 |           0.0061 |
| Koke                  | M               | Atletico Madrid | Atletico Madrid     |      32 |   1915 |             34 |          2237 |                      0.0468 |                     0.04 |         0.0068 |           0.0068 |
| Beñat Turrientes      | M               | Real Sociedad   | Real Sociedad       |      21 |    933 |             27 |          1355 |                      0.0519 |                     0.06 |        -0.0081 |           0.0081 |
| Marc Casadó           | M               | Barcelona       | Barcelona           |      23 |   1640 |             24 |           957 |                      0.0919 |                     0.1  |        -0.0081 |           0.0081 |
| Juan Camilo Hernández | F               | Real Betis      | Real Betis          |      15 |   1172 |             32 |          2623 |                      0.4217 |                     0.43 |        -0.0083 |           0.0083 |

## Diez mayores errores F/M

| player_name           | position_main   | team_title      | target_team_title   |   games |   time |   target_games |   target_time |   predicted_2025_2026_xG_90 |   actual_2025_2026_xG_90 |   signed_error |   absolute_error |
|:----------------------|:----------------|:----------------|:--------------------|--------:|-------:|---------------:|--------------:|----------------------------:|-------------------------:|---------------:|-----------------:|
| Ayoze Pérez           | F               | Villarreal      | Villarreal          |      30 |   1987 |             26 |          1154 |                      0.793  |                     0.46 |         0.333  |           0.333  |
| Vedat Muriqi          | F               | Mallorca        | Mallorca            |      29 |   2081 |             37 |          3150 |                      0.3537 |                     0.62 |        -0.2663 |           0.2663 |
| Oihan Sancet          | F               | Athletic Club   | Athletic Club       |      29 |   1645 |             28 |          1832 |                      0.5982 |                     0.35 |         0.2482 |           0.2482 |
| Antoine Griezmann     | F               | Atletico Madrid | Atletico Madrid     |      38 |   2514 |             34 |          1428 |                      0.2736 |                     0.52 |        -0.2464 |           0.2464 |
| Mikel Oyarzabal       | F               | Real Sociedad   | Real Sociedad       |      35 |   2291 |             34 |          2727 |                      0.3003 |                     0.53 |        -0.2297 |           0.2297 |
| Abdessamad Ezzalzouli | M               | Real Betis      | Real Betis          |      32 |   2038 |             29 |          2275 |                      0.1881 |                     0.41 |        -0.2219 |           0.2219 |
| Álex Baena            | F               | Villarreal      | Atletico Madrid     |      32 |   2639 |             27 |          1615 |                      0.3186 |                     0.1  |         0.2186 |           0.2186 |
| Raphinha              | F               | Barcelona       | Barcelona           |      36 |   2862 |             22 |          1424 |                      0.4884 |                     0.7  |        -0.2116 |           0.2116 |
| Robert Lewandowski    | F               | Barcelona       | Barcelona           |      34 |   2716 |             31 |          1630 |                      0.7016 |                     0.89 |        -0.1884 |           0.1884 |
| Alexander Sørloth     | F               | Atletico Madrid | Atletico Madrid     |      35 |   1525 |             35 |          1955 |                      0.8178 |                     0.64 |         0.1778 |           0.1778 |

## Mayores sobreestimaciones

| player_name       | position_main   | team_title              | target_team_title   |   games |   time |   target_games |   target_time |   predicted_2025_2026_xG_90 |   actual_2025_2026_xG_90 |   signed_error |   absolute_error |
|:------------------|:----------------|:------------------------|:--------------------|--------:|-------:|---------------:|--------------:|----------------------------:|-------------------------:|---------------:|-----------------:|
| Ayoze Pérez       | F               | Villarreal              | Villarreal          |      30 |   1987 |             26 |          1154 |                      0.793  |                     0.46 |         0.333  |           0.333  |
| Oihan Sancet      | F               | Athletic Club           | Athletic Club       |      29 |   1645 |             28 |          1832 |                      0.5982 |                     0.35 |         0.2482 |           0.2482 |
| Álex Baena        | F               | Villarreal              | Atletico Madrid     |      32 |   2639 |             27 |          1615 |                      0.3186 |                     0.1  |         0.2186 |           0.2186 |
| Alexander Sørloth | F               | Atletico Madrid         | Atletico Madrid     |      35 |   1525 |             35 |          1955 |                      0.8178 |                     0.64 |         0.1778 |           0.1778 |
| Gonçalo Guedes    | F               | Wolverhampton Wanderers | Real Sociedad       |      29 |    968 |             32 |          1833 |                      0.3683 |                     0.2  |         0.1683 |           0.1683 |
| Tete Morente      | F               | Lecce                   | Elche               |      31 |   2036 |             13 |           940 |                      0.2377 |                     0.07 |         0.1677 |           0.1677 |
| Ramón Terrats     | F               | Getafe,Villarreal       | Espanyol            |      26 |   1356 |             29 |          1020 |                      0.2597 |                     0.1  |         0.1597 |           0.1597 |
| Giovani Lo Celso  | M               | Real Betis              | Real Betis          |      25 |   1437 |             24 |          1396 |                      0.3431 |                     0.19 |         0.1531 |           0.1531 |
| Brahim Diaz       | F               | Real Madrid             | Real Madrid         |      31 |   1368 |             30 |          1243 |                      0.2709 |                     0.12 |         0.1509 |           0.1509 |
| Arda Güler        | F               | Real Madrid             | Real Madrid         |      28 |   1252 |             33 |          2076 |                      0.3394 |                     0.21 |         0.1294 |           0.1294 |

## Mayores infraestimaciones

| player_name           | position_main   | team_title      | target_team_title   |   games |   time |   target_games |   target_time |   predicted_2025_2026_xG_90 |   actual_2025_2026_xG_90 |   signed_error |   absolute_error |
|:----------------------|:----------------|:----------------|:--------------------|--------:|-------:|---------------:|--------------:|----------------------------:|-------------------------:|---------------:|-----------------:|
| Vedat Muriqi          | F               | Mallorca        | Mallorca            |      29 |   2081 |             37 |          3150 |                      0.3537 |                     0.62 |        -0.2663 |           0.2663 |
| Antoine Griezmann     | F               | Atletico Madrid | Atletico Madrid     |      38 |   2514 |             34 |          1428 |                      0.2736 |                     0.52 |        -0.2464 |           0.2464 |
| Mikel Oyarzabal       | F               | Real Sociedad   | Real Sociedad       |      35 |   2291 |             34 |          2727 |                      0.3003 |                     0.53 |        -0.2297 |           0.2297 |
| Abdessamad Ezzalzouli | M               | Real Betis      | Real Betis          |      32 |   2038 |             29 |          2275 |                      0.1881 |                     0.41 |        -0.2219 |           0.2219 |
| Raphinha              | F               | Barcelona       | Barcelona           |      36 |   2862 |             22 |          1424 |                      0.4884 |                     0.7  |        -0.2116 |           0.2116 |
| Robert Lewandowski    | F               | Barcelona       | Barcelona           |      34 |   2716 |             31 |          1630 |                      0.7016 |                     0.89 |        -0.1884 |           0.1884 |
| Lamine Yamal          | F               | Barcelona       | Barcelona           |      35 |   2879 |             28 |          2291 |                      0.4154 |                     0.58 |        -0.1646 |           0.1646 |
| Kylian Mbappe-Lottin  | F               | Real Madrid     | Real Madrid         |      34 |   2938 |             31 |          2623 |                      0.7405 |                     0.89 |        -0.1495 |           0.1495 |
| Giuliano Simeone      | F               | Atletico Madrid | Atletico Madrid     |      33 |   1988 |             31 |          2153 |                      0.1215 |                     0.26 |        -0.1385 |           0.1385 |
| Roberto Navarro       | M               | Mallorca        | Athletic Club       |      23 |   1302 |             32 |          1246 |                      0.1074 |                     0.24 |        -0.1326 |           0.1326 |

## Interpretación

El análisis F/M permite evaluar el modelo en el subconjunto de jugadores donde el `xG_90` tiene mayor significado deportivo. Si el rendimiento se mantiene próximo al obtenido en la evaluación global, se refuerza la validez del modelo para perfiles ofensivos y creativos.

Los mayores errores deben analizarse como casos de cambio contextual. Una sobreestimación puede indicar que el jugador no mantuvo el volumen ofensivo esperado, mientras que una infraestimación puede reflejar una mejora de rol, mayor protagonismo ofensivo o cambio táctico favorable durante la temporada 2025-2026.