# Análisis de predicciones individuales de jugadores

## Objetivo del análisis

Este informe analiza una muestra de predicciones individuales generadas por el modelo temporal sin utilizar `xG_90` ni `npxG_90` de la temporada anterior como variables predictoras directas.

La variable objetivo es `actual_next_xG_90`, que representa el valor real de goles esperados por 90 minutos del jugador en la temporada siguiente. La columna `predicted_next_xG_90` representa la estimación del modelo.

## Resumen cuantitativo de la muestra

- Número de jugadores analizados: 15
- Error absoluto medio de la muestra: 0.2293
- Mediana del error absoluto: 0.0524
- Casos con error bajo: 5
- Casos con error moderado: 5
- Casos con error alto: 5
- Sobreestimaciones: 8
- Infraestimaciones: 7

## Tabla completa de predicciones

| player_name          | position_main   | team_title          | next_team_title               | season    | next_season   |   actual_next_xG_90 |   predicted_next_xG_90 |   signed_error |   absolute_error | error_group    | error_direction   |
|:---------------------|:----------------|:--------------------|:------------------------------|:----------|:--------------|--------------------:|-----------------------:|---------------:|-----------------:|:---------------|:------------------|
| Mikel Vesga          | M               | Athletic Club       | Athletic Club                 | 2023-2024 | 2024-2025     |              0.1144 |                 0.1142 |        -0.0002 |           0.0002 | error_bajo     | infraestimación   |
| Jack Harrison        | M               | Everton             | Everton                       | 2023-2024 | 2024-2025     |              0.1445 |                 0.1447 |         0.0002 |           0.0002 | error_bajo     | sobreestimación   |
| Albert Sambi Lokonga | M               | Luton               | Sevilla                       | 2023-2024 | 2024-2025     |              0.0467 |                 0.0466 |        -0.0002 |           0.0002 | error_bajo     | infraestimación   |
| Milan Badelj         | M               | Genoa               | Genoa                         | 2023-2024 | 2024-2025     |              0.0565 |                 0.057  |         0.0005 |           0.0005 | error_bajo     | sobreestimación   |
| Bernardo Silva       | M               | Manchester City     | Manchester City               | 2023-2024 | 2024-2025     |              0.1622 |                 0.1628 |         0.0006 |           0.0006 | error_bajo     | sobreestimación   |
| Carles Aleñá         | M               | Getafe              | Alaves,Getafe                 | 2023-2024 | 2024-2025     |              0.0715 |                 0.1207 |         0.0492 |           0.0492 | error_moderado | sobreestimación   |
| Marcus Rashford      | F               | Manchester United   | Aston Villa,Manchester United | 2023-2024 | 2024-2025     |              0.3127 |                 0.3638 |         0.051  |           0.051  | error_moderado | sobreestimación   |
| Dejan Kulusevski     | F               | Tottenham           | Tottenham                     | 2023-2024 | 2024-2025     |              0.2079 |                 0.2603 |         0.0524 |           0.0524 | error_moderado | sobreestimación   |
| Marcel Sabitzer      | M               | Borussia Dortmund   | Borussia Dortmund             | 2023-2024 | 2024-2025     |              0.1242 |                 0.1771 |         0.0529 |           0.0529 | error_moderado | sobreestimación   |
| Oliver McBurnie      | F               | Sheffield United    | Las Palmas                    | 2023-2024 | 2024-2025     |              0.2961 |                 0.349  |         0.0529 |           0.0529 | error_moderado | sobreestimación   |
| Mateo Retegui        | F               | Genoa               | Atalanta                      | 2023-2024 | 2024-2025     |              0.8523 |                 0.3274 |        -0.5249 |           0.5249 | error_alto     | infraestimación   |
| Nick Woltemade       | F               | Werder Bremen       | VfB Stuttgart                 | 2023-2024 | 2024-2025     |              0.7249 |                 0.1975 |        -0.5274 |           0.5274 | error_alto     | infraestimación   |
| Alexander Sørloth    | F               | Villarreal          | Atletico Madrid               | 2023-2024 | 2024-2025     |              1.2072 |                 0.6332 |        -0.5739 |           0.5739 | error_alto     | infraestimación   |
| Ousmane Dembélé      | F               | Paris Saint Germain | Paris Saint Germain           | 2023-2024 | 2024-2025     |              0.9549 |                 0.2832 |        -0.6717 |           0.6717 | error_alto     | infraestimación   |
| Gonçalo Ramos        | F               | Paris Saint Germain | Paris Saint Germain           | 2023-2024 | 2024-2025     |              1.3982 |                 0.5174 |        -0.8808 |           0.8808 | error_alto     | infraestimación   |

## Cinco predicciones más precisas

| player_name          | position_main   | team_title      | next_team_title   | season    | next_season   |   actual_next_xG_90 |   predicted_next_xG_90 |   signed_error |   absolute_error | error_group   | error_direction   |
|:---------------------|:----------------|:----------------|:------------------|:----------|:--------------|--------------------:|-----------------------:|---------------:|-----------------:|:--------------|:------------------|
| Mikel Vesga          | M               | Athletic Club   | Athletic Club     | 2023-2024 | 2024-2025     |              0.1144 |                 0.1142 |        -0.0002 |           0.0002 | error_bajo    | infraestimación   |
| Jack Harrison        | M               | Everton         | Everton           | 2023-2024 | 2024-2025     |              0.1445 |                 0.1447 |         0.0002 |           0.0002 | error_bajo    | sobreestimación   |
| Albert Sambi Lokonga | M               | Luton           | Sevilla           | 2023-2024 | 2024-2025     |              0.0467 |                 0.0466 |        -0.0002 |           0.0002 | error_bajo    | infraestimación   |
| Milan Badelj         | M               | Genoa           | Genoa             | 2023-2024 | 2024-2025     |              0.0565 |                 0.057  |         0.0005 |           0.0005 | error_bajo    | sobreestimación   |
| Bernardo Silva       | M               | Manchester City | Manchester City   | 2023-2024 | 2024-2025     |              0.1622 |                 0.1628 |         0.0006 |           0.0006 | error_bajo    | sobreestimación   |

## Cinco mayores errores

| player_name       | position_main   | team_title          | next_team_title     | season    | next_season   |   actual_next_xG_90 |   predicted_next_xG_90 |   signed_error |   absolute_error | error_group   | error_direction   |
|:------------------|:----------------|:--------------------|:--------------------|:----------|:--------------|--------------------:|-----------------------:|---------------:|-----------------:|:--------------|:------------------|
| Gonçalo Ramos     | F               | Paris Saint Germain | Paris Saint Germain | 2023-2024 | 2024-2025     |              1.3982 |                 0.5174 |        -0.8808 |           0.8808 | error_alto    | infraestimación   |
| Ousmane Dembélé   | F               | Paris Saint Germain | Paris Saint Germain | 2023-2024 | 2024-2025     |              0.9549 |                 0.2832 |        -0.6717 |           0.6717 | error_alto    | infraestimación   |
| Alexander Sørloth | F               | Villarreal          | Atletico Madrid     | 2023-2024 | 2024-2025     |              1.2072 |                 0.6332 |        -0.5739 |           0.5739 | error_alto    | infraestimación   |
| Nick Woltemade    | F               | Werder Bremen       | VfB Stuttgart       | 2023-2024 | 2024-2025     |              0.7249 |                 0.1975 |        -0.5274 |           0.5274 | error_alto    | infraestimación   |
| Mateo Retegui     | F               | Genoa               | Atalanta            | 2023-2024 | 2024-2025     |              0.8523 |                 0.3274 |        -0.5249 |           0.5249 | error_alto    | infraestimación   |

## Interpretación

Los casos con menor error corresponden principalmente a jugadores con rendimiento ofensivo estable o moderado, donde el modelo consigue aproximar con precisión el `xG_90` de la temporada siguiente.

En cambio, los mayores errores se concentran en jugadores ofensivos que incrementan de forma notable su producción de ocasiones en la temporada objetivo. En estos casos, el error suele ser negativo, lo que indica una infraestimación del rendimiento real.

Este comportamiento sugiere que el modelo presenta una tendencia conservadora: reproduce adecuadamente patrones históricos estables, pero tiene más dificultad para anticipar saltos bruscos de rendimiento asociados a cambios de rol, equipo, contexto táctico o eficiencia ofensiva.