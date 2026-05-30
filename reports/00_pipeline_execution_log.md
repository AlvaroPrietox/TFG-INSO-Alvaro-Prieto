# Log de ejecución del pipeline

- Fecha de ejecución: 2026-05-30 13:43:00
- Directorio del proyecto: `C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO`

## Configuración de ejecución

- Fase solicitada: `all`
- Omitir tests: `False`
- Omitir comprobación de inputs: `False`
- Omitir comprobación de outputs: `False`


# Fase: diagnostics


## Diagnóstico inicial de los datasets raw.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\01_data_diagnostics.py
```


### Salida estándar

```text
Cargando dataset: matches
Cargando dataset: historical_players
Cargando dataset: laliga_players
Informe generado correctamente en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\01_data_diagnostics.md
```


**Estado:** ejecutado correctamente.


# Fase: preprocessing


## Limpieza del dataset histórico de partidos.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\02_preprocess_matches.py
```


### Salida estándar

```text
Preprocesamiento de partidos completado.
Filas finales: 8972
Columnas finales: 25
Archivo generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\matches_clean.csv
```


**Estado:** ejecutado correctamente.


## Limpieza de datasets históricos y actuales de jugadores.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\03_preprocess_players.py
```


### Salida estándar

```text
Jornada latest seleccionada: Jornada_14
Filas en latest antes del filtro de minutos: 490
Filtro latest aplicado: jugadores con 300 minutos o más.
Jugadores eliminados en latest por bajo volumen de minutos: 156
Preprocesamiento de jugadores completado.

Jugadores históricos:
- Filas: 8063
- Columnas: 23
- Archivo generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\historical_players_clean.csv

Jugadores actuales de LaLiga:
- Filas: 2422
- Columnas: 24
- Archivo generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_players_current_clean.csv
```


**Estado:** ejecutado correctamente.


## Preparación del target real 2025-2026 de LaLiga.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\13_prepare_2025_2026_laliga_target.py
```


### Salida estándar

```text
Filtro de minutos aplicado en 2025-2026: jugadores con 900 minutos o más.
Jugadores eliminados por bajo volumen de minutos: 255
Dataset objetivo 2025-2026 limpiado correctamente.
Filas finales: 345
Columnas finales: 14
Archivo generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_2025_2026_target_clean.csv

Top 10 por xG_90 real:
            player_name          team_title   league  ... xG_90  assists_90  xA_90
0    Robert Lewandowski           Barcelona  La-Liga  ...  0.89    0.110429   0.13
1  Kylian Mbappe-Lottin         Real Madrid  La-Liga  ...  0.89    0.171559   0.25
2         Ferrán Torres           Barcelona  La-Liga  ...  0.72    0.088889   0.14
3              Raphinha           Barcelona  La-Liga  ...  0.70    0.189607   0.37
4           Raul García             Osasuna  La-Liga  ...  0.67    0.097192   0.11
5           Carlos Espí             Levante  La-Liga  ...  0.67    0.000000   0.05
6     Alexander Sørloth     Atletico Madrid  La-Liga  ...  0.64    0.000000   0.06
7    Georges Mikautadze          Villarreal  La-Liga  ...  0.63    0.250580   0.17
8          Vedat Muriqi            Mallorca  La-Liga  ...  0.62    0.028571   0.07
9            Etta Eyong  Levante,Villarreal  La-Liga  ...  0.60    0.217260   0.15

[10 rows x 14 columns]
```


**Estado:** ejecutado correctamente.


# Fase: baseline


## Construcción del dataset descriptivo inicial de modelado.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\04_build_player_modeling_dataset.py
```


### Salida estándar

```text
Dataset de modelado de jugadores generado correctamente.
Filas finales: 8063
Columnas finales: 21
Archivo generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\player_modeling_dataset.csv
```


**Estado:** ejecutado correctamente.


## Entrenamiento del baseline descriptivo inicial.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\05_train_player_baseline.py
```


### Salida estándar

```text
Entrenamiento baseline completado.

                   model       mae        r2
0         DummyRegressor  0.116936 -0.001490
1  RandomForestRegressor  0.032702  0.900833

Resultados guardados en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\player_baseline_evaluation.csv
```


**Estado:** ejecutado correctamente.


## Análisis de importancia de variables del baseline.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\06_analyze_player_baseline.py
```


### Salida estándar

```text
Análisis del baseline completado.

Métricas:
                   model       mae        r2
0         DummyRegressor  0.116936 -0.001490
1  RandomForestRegressor  0.032702  0.900833

Top 15 variables más importantes:
                               feature  importance  importance_percent
0                    numeric__shots_90    0.843860           84.385964
1                  numeric__xGChain_90    0.067077            6.707663
2                numeric__xGBuildup_90    0.036792            3.679174
3               numeric__key_passes_90    0.019579            1.957882
4         categorical__position_main_F    0.009784            0.978392
5                       numeric__xA_90    0.008999            0.899867
6            numeric__minutes_per_game    0.003962            0.396208
7                        numeric__time    0.002667            0.266677
8                  numeric__assists_90    0.002262            0.226248
9             numeric__yellow_cards_90    0.002259            0.225863
10                      numeric__games    0.001591            0.159091
11  categorical__league_Premier-League    0.000181            0.018117
12         categorical__league_Serie-A    0.000180            0.018031
13               numeric__red_cards_90    0.000147            0.014744
14        categorical__position_main_M    0.000146            0.014620

Importancia de variables guardada en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\player_feature_importance.csv
Informe guardado en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\06_player_baseline_analysis.md
```


**Estado:** ejecutado correctamente.


# Fase: temporal_dataset


## Construcción del dataset temporal jugador-temporada.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\07_build_player_temporal_dataset.py
```


### Salida estándar

```text
Porteros eliminados del dataset temporal: 599
Dataset temporal de jugadores generado correctamente.
Filas finales: 3957
Columnas finales: 30
Archivo generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\player_temporal_modeling_dataset.csv

Pares temporada actual -> temporada siguiente:
   season next_season
2020-2021   2021-2022
2021-2022   2022-2023
2022-2023   2023-2024
2023-2024   2024-2025
```


**Estado:** ejecutado correctamente.


# Fase: temporal_model


## Entrenamiento y evaluación temporal inicial.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\08_train_player_temporal_model.py
```


### Salida estándar

```text
Ejecutando experimento: full
Ejecutando experimento: without_previous_xg

Entrenamiento temporal completado.

Métricas:
           feature_set                  model  ...  test_season  test_next_season
0                 full         DummyRegressor  ...    2023-2024         2024-2025
1                 full  RandomForestRegressor  ...    2023-2024         2024-2025
2  without_previous_xg         DummyRegressor  ...    2023-2024         2024-2025
3  without_previous_xg  RandomForestRegressor  ...    2023-2024         2024-2025

[4 rows x 8 columns]

Top 15 variables más importantes por experimento:

Experimento: full
   feature_set                       feature  importance  importance_percent
0         full                numeric__xG_90    0.806489           80.648901
1         full             numeric__shots_90    0.048635            4.863527
2         full              numeric__npxG_90    0.028777            2.877737
3         full             numeric__goals_90    0.015672            1.567229
4         full               numeric__npg_90    0.010955            1.095508
5         full      numeric__yellow_cards_90    0.009802            0.980156
6         full         numeric__xGBuildup_90    0.009637            0.963665
7         full                numeric__xA_90    0.009505            0.950537
8         full           numeric__assists_90    0.009060            0.906044
9         full     numeric__minutes_per_game    0.008975            0.897518
10        full        numeric__key_passes_90    0.008236            0.823553
11        full  categorical__position_main_F    0.007252            0.725238
12        full                 numeric__time    0.007105            0.710545
13        full           numeric__xGChain_90    0.006577            0.657658
14        full                numeric__games    0.006152            0.615191

Experimento: without_previous_xg
            feature_set  ... importance_percent
24  without_previous_xg  ...          53.994202
25  without_previous_xg  ...          27.072963
26  without_previous_xg  ...           5.331080
27  without_previous_xg  ...           2.060396
28  without_previous_xg  ...           1.760430
29  without_previous_xg  ...           1.625788
30  without_previous_xg  ...           1.550071
31  without_previous_xg  ...           1.170377
32  without_previous_xg  ...           1.168122
33  without_previous_xg  ...           1.026072
34  without_previous_xg  ...           1.017095
35  without_previous_xg  ...           0.834620
36  without_previous_xg  ...           0.570522
37  without_previous_xg  ...           0.210610
38  without_previous_xg  ...           0.148347

[15 rows x 4 columns]

Evaluación guardada en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\player_temporal_evaluation.csv
Importancia de variables guardada en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\player_temporal_feature_importance.csv
Informe guardado en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\08_player_temporal_model_report.md
```


**Estado:** ejecutado correctamente.


## Generación de predicciones temporales de ejemplo.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\09_generate_player_predictions.py
```


### Salida estándar

```text
Predicciones individuales generadas correctamente.

Predicciones calculadas en test: 954

Muestra general:
- Filas guardadas: 15
- Archivo generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\player_temporal_predictions_without_previous_xg.csv

Muestra F/M:
- Filas guardadas: 15
- Archivo generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\player_temporal_predictions_f_m_sample_15.csv

Vista previa muestra F/M:
       id           player_name  ... signed_error absolute_error
0    5077           Mikel Vesga  ...    -0.000150       0.000150
1    8720         Jack Harrison  ...     0.000161       0.000161
2    9689  Albert Sambi Lokonga  ...    -0.000186       0.000186
3    1430          Milan Badelj  ...     0.000467       0.000467
4    3635        Bernardo Silva  ...     0.000607       0.000607
5    5058          Carles Aleñá  ...     0.049169       0.049169
6     556       Marcus Rashford  ...     0.051041       0.051041
7    6691      Dejan Kulusevski  ...     0.052414       0.052414
8    5248       Marcel Sabitzer  ...     0.052892       0.052892
9    1736       Oliver McBurnie  ...     0.052929       0.052929
10  11898         Mateo Retegui  ...    -0.524852       0.524852
11   8117        Nick Woltemade  ...    -0.527408       0.527408
12   6531     Alexander Sørloth  ...    -0.573934       0.573934
13   3226       Ousmane Dembélé  ...    -0.671686       0.671686
14  11767         Gonçalo Ramos  ...    -0.880847       0.880847

[15 rows x 18 columns]
```


**Estado:** ejecutado correctamente.


## Análisis cualitativo de predicciones individuales.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\10_analyze_player_predictions.py
```


### Salida estándar

```text
Informe de predicciones individuales generado correctamente.
Archivo leído: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\player_temporal_predictions_f_m_sample_15.csv
Informe generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\10_player_prediction_examples.md
```


**Estado:** ejecutado correctamente.


# Fase: random_forest_legacy


## Entrenamiento y guardado del modelo Random Forest legacy.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\11_train_and_save_temporal_model.py
```


### Salida estándar

```text
Modelo temporal final entrenado y guardado correctamente.

Filas usadas para entrenamiento final: 3957
Modelo guardado en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\models\random_forest_temporal_without_previous_xg.joblib
Metadatos guardados en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\models\random_forest_temporal_without_previous_xg_metadata.json
```


**Estado:** ejecutado correctamente.


## Predicciones prospectivas legacy con Random Forest.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\12_predict_laliga_current_players.py
```


### Salida estándar

```text
Porteros eliminados antes de predecir: 25
Predicciones actuales de LaLiga generadas correctamente.

Jugadores predichos: 309
Archivo completo: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_current_player_predictions.csv

Top 25 F/M por predicted_next_xG_90:
       id            player_name  ... xGBuildup_90 predicted_next_xG_90
0    3423   Kylian Mbappe-Lottin  ...     0.353753             0.856466
1     227     Robert Lewandowski  ...     0.355635             0.836339
2    6441          Ferrán Torres  ...     0.223929             0.756117
3    9002           Vedat Muriqi  ...     0.162577             0.576016
4    2120          Gerard Moreno  ...     0.429585             0.545118
5    6531      Alexander Sørloth  ...     0.023924             0.528387
6   11822           Fermín López  ...     0.555313             0.502671
7    2270      Antoine Griezmann  ...     0.469586             0.487447
8    8026               Raphinha  ...     0.655859             0.486751
9    4175             Pere Milla  ...     0.176643             0.484841
10  11527           Lamine Yamal  ...     0.530921             0.482337
11  10930             Etta Eyong  ...     0.048965             0.479777
12  12160        Alberto Moleiro  ...     0.181577             0.462999
13   5074            Kike García  ...     0.132605             0.436978
14  11094                 Antony  ...     0.390181             0.436238
15   2589               Rafa Mir  ...     0.084964             0.433325
16   6170            André Silva  ...     0.120736             0.432986
17   6954  Juan Camilo Hernández  ...     0.179951             0.424163
18   9275            Iván Romero  ...     0.063706             0.423987
19  11783             Akor Adams  ...     0.083521             0.416046
20   1235           Ante Budimir  ...     0.061101             0.412010
21   2543         Borja Iglesias  ...     0.194245             0.411575
22  13358          Thiago Almada  ...     0.473861             0.406521
23  13996         Tani Oluwaseyi  ...     0.048276             0.405053
24  10846         Julián Álvarez  ...     0.354258             0.399588

[25 rows x 20 columns]

Archivo Top 25 F/M: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_current_top_25_f_m_predictions.csv
```


**Estado:** ejecutado correctamente.


## Evaluación externa legacy 2025-2026 con Random Forest.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\14_evaluate_laliga_2025_2026_predictions.py
```


### Salida estándar

```text
Porteros eliminados de 2024-2025: 119
Evaluación 2025-2026 completada.

Jugadores 2024-2025 preparados: 1437
Jugadores objetivo 2025-2026: 345
Jugadores emparejados: 205

Métricas:
{'mae': 0.05503688725877283, 'r2': 0.8023194863547602}

Comparación completa: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_2025_2026_prediction_evaluation.csv
Muestra representativa: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_2025_2026_prediction_sample_25.csv
Informe generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\14_laliga_2025_2026_prediction_evaluation.md
```


**Estado:** ejecutado correctamente.


## Evaluación externa legacy restringida a F/M.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\15_analyze_laliga_2025_2026_f_m.py
```


### Salida estándar

```text
Análisis F/M 2025-2026 completado.

Jugadores emparejados totales: 205
Jugadores F/M analizados: 110

Métricas F/M:
{'mae': 0.07380105967398197, 'r2': 0.756051435409534}

Métricas por posición:
  position_main  n_players       mae        r2
0             F         61  0.099489  0.625805
1             M         49  0.041823  0.691852

Evaluación F/M guardada en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_2025_2026_prediction_evaluation_f_m.csv
Muestra F/M guardada en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_2025_2026_prediction_sample_f_m_25.csv
Informe F/M generado en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\15_laliga_2025_2026_prediction_evaluation_f_m.md
```


**Estado:** ejecutado correctamente.


## Resumen experimental legacy previo a la selección final.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\16_build_experiment_summary.py
```


### Salida estándar

```text
Resumen final de experimentos generado correctamente.

                         experiment  ...            test_period
0     external_laliga_2025_2026_f_m  ...  2024-2025 → 2025-2026
1  external_laliga_2025_2026_global  ...  2024-2025 → 2025-2026
2       external_laliga_2025_2026_F  ...  2024-2025 → 2025-2026
3       external_laliga_2025_2026_M  ...  2024-2025 → 2025-2026
4                     temporal_full  ...  2023-2024 → 2024-2025
5      temporal_without_previous_xg  ...  2023-2024 → 2024-2025

[6 rows x 6 columns]

CSV generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\experiment_summary.csv
Informe generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\16_experiment_summary.md
```


**Estado:** ejecutado correctamente.


## Generación de figuras legacy de resultados.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\17_generate_result_figures.py
```


### Salida estándar

```text
Figuras de resultados generadas correctamente.

Carpeta de figuras: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\figures

Archivos generados:
- experiment_mae.png
- experiment_r2.png
- laliga_2025_2026_actual_vs_predicted.png
- laliga_2025_2026_residuals.png
- laliga_2025_2026_error_by_position.png
- temporal_feature_importance_without_previous_xg.png
- reports/17_result_figures_index.md
```


**Estado:** ejecutado correctamente.


# Fase: error_analysis


## Análisis del error por rangos de xG_90 real.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\18_analyze_error_by_xg_range.py
```


### Salida estándar

```text
Análisis de error por rangos de xG_90 completado.

               xg_range  n_players  ...  overestimations  underestimations
0            bajo_<0.10        114  ...               89                25
1  medio_bajo_0.10_0.25         44  ...               18                26
2  medio_alto_0.25_0.50         34  ...                8                26
3           alto_>=0.50         13  ...                1                12

[4 rows x 9 columns]

CSV generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\error_by_xg_range.csv
Informe generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\18_error_by_xg_range.md
Figura generada: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\figures\error_by_xg_range.png
```


**Estado:** ejecutado correctamente.


# Fase: model_comparison


## Comparación temporal interna de modelos candidatos.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\19_compare_temporal_models.py
```


### Salida estándar

```text
Entrenando modelo: DummyRegressor
Entrenando modelo: Ridge
Entrenando modelo: RandomForestRegressor
Entrenando modelo: GradientBoostingRegressor
Entrenando modelo: HistGradientBoostingRegressor
Comparación de modelos temporales completada.

Resultados globales:
                           model  train_rows  ...  test_season  test_next_season
0                          Ridge        3003  ...    2023-2024         2024-2025
1      GradientBoostingRegressor        3003  ...    2023-2024         2024-2025
2          RandomForestRegressor        3003  ...    2023-2024         2024-2025
3  HistGradientBoostingRegressor        3003  ...    2023-2024         2024-2025
4                 DummyRegressor        3003  ...    2023-2024         2024-2025

[5 rows x 7 columns]

Resultados por rango de xG_90:
                            model  ... underestimations
0                           Ridge  ...               51
1       GradientBoostingRegressor  ...               50
2   HistGradientBoostingRegressor  ...               50
3           RandomForestRegressor  ...               53
4                  DummyRegressor  ...               58
5       GradientBoostingRegressor  ...              126
6   HistGradientBoostingRegressor  ...              140
7                           Ridge  ...              161
8           RandomForestRegressor  ...              127
9                  DummyRegressor  ...                0
10                          Ridge  ...               95
11          RandomForestRegressor  ...               90
12      GradientBoostingRegressor  ...               86
13  HistGradientBoostingRegressor  ...               85
14                 DummyRegressor  ...              153
15                 DummyRegressor  ...              121
16                          Ridge  ...              136
17          RandomForestRegressor  ...              142
18      GradientBoostingRegressor  ...              145
19  HistGradientBoostingRegressor  ...              147

[20 rows x 10 columns]

CSV global generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\temporal_model_comparison.csv
CSV por rango generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\temporal_model_comparison_by_xg_range.csv
Informe generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\19_temporal_model_comparison.md
Figura MAE generada: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\figures\temporal_model_comparison_mae.png
Figura R² generada: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\figures\temporal_model_comparison_r2.png
```


**Estado:** ejecutado correctamente.


## Comparación externa de modelos para xG_90 y evaluación multi-métrica con Ridge.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\20_compare_external_2025_2026_models.py
```


### Salida estándar

```text
Porteros eliminados del dataset temporal multi-métrica: 599
Porteros eliminados de 2024-2025: 119
Filas temporales de entrenamiento: 3957
Jugadores 2024-2025 preparados: 1437
Jugadores objetivo 2025-2026: 345
Jugadores emparejados: 205
Entrenando y evaluando modelo externo xG_90: DummyRegressor
Entrenando y evaluando modelo externo xG_90: Ridge
Entrenando y evaluando modelo externo xG_90: RandomForestRegressor
Entrenando y evaluando modelo externo xG_90: GradientBoostingRegressor
Entrenando y evaluando modelo externo xG_90: HistGradientBoostingRegressor
Entrenando y evaluando Ridge multi-métrica: xG_90
Entrenando y evaluando Ridge multi-métrica: goals_90
Entrenando y evaluando Ridge multi-métrica: assists_90
Entrenando y evaluando Ridge multi-métrica: xA_90
Comparación externa 2025-2026 y evaluación multi-métrica completadas.

Resultados globales xG_90:
                           model  ... target_season
0                          Ridge  ...     2025-2026
1  HistGradientBoostingRegressor  ...     2025-2026
2          RandomForestRegressor  ...     2025-2026
3      GradientBoostingRegressor  ...     2025-2026
4                 DummyRegressor  ...     2025-2026

[5 rows x 8 columns]

Resultados F/M xG_90:
                           model  ... target_season
0                          Ridge  ...     2025-2026
1  HistGradientBoostingRegressor  ...     2025-2026
2          RandomForestRegressor  ...     2025-2026
3      GradientBoostingRegressor  ...     2025-2026
4                 DummyRegressor  ...     2025-2026

[5 rows x 8 columns]

Resultados multi-métrica globales Ridge:
   metric_key           metric_label  ... input_season  target_season
0  assists_90            Asistencias  ...    2024-2025      2025-2026
1    goals_90                  Goles  ...    2024-2025      2025-2026
2       xA_90  Asistencias esperadas  ...    2024-2025      2025-2026
3       xG_90        Goles esperados  ...    2024-2025      2025-2026

[4 rows x 9 columns]

Resultados multi-métrica F/M Ridge:
   metric_key           metric_label  ... input_season  target_season
0  assists_90            Asistencias  ...    2024-2025      2025-2026
1    goals_90                  Goles  ...    2024-2025      2025-2026
2       xA_90  Asistencias esperadas  ...    2024-2025      2025-2026
3       xG_90        Goles esperados  ...    2024-2025      2025-2026

[4 rows x 9 columns]

CSV comparación xG_90 generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\external_2025_2026_model_comparison.csv
CSV comparación F/M xG_90 generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\external_2025_2026_model_comparison_f_m.csv
Predicciones xG_90 por modelo generadas: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\external_2025_2026_model_predictions.csv
CSV rango xG_90 generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\external_2025_2026_model_comparison_by_xg_range.csv
CSV multi-métrica wide generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\external_2025_2026_multi_metric_predictions.csv
CSV multi-métrica long generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\external_2025_2026_multi_metric_predictions_long.csv
CSV multi-métrica global generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\external_2025_2026_multi_metric_comparison.csv
CSV multi-métrica F/M generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\external_2025_2026_multi_metric_comparison_f_m.csv
Informe actualizado generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\20_external_2025_2026_model_comparison.md
```


**Estado:** ejecutado correctamente.


# Fase: final_model


## Entrenamiento y guardado de modelos Ridge finales multi-métrica.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\21_train_and_save_final_ridge_model.py
```


### Salida estándar

```text
Porteros eliminados del dataset temporal multi-métrica: 599
Entrenando modelo final Ridge para: xG_90
Modelo guardado en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\models\ridge_temporal_xG_90.joblib
Entrenando modelo final Ridge para: goals_90
Modelo guardado en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\models\ridge_temporal_goals_90.joblib
Entrenando modelo final Ridge para: assists_90
Modelo guardado en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\models\ridge_temporal_assists_90.joblib
Entrenando modelo final Ridge para: xA_90
Modelo guardado en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\models\ridge_temporal_xA_90.joblib

Modelos Ridge finales multi-métrica entrenados y guardados correctamente.
Filas temporales usadas: 3957
Metadatos multi-métrica guardados en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\models\ridge_temporal_multi_metric_metadata.json
Modelo legacy xG_90 guardado en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\models\ridge_temporal_without_previous_xg.joblib
Metadatos legacy xG_90 guardados en: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\models\ridge_temporal_without_previous_xg_metadata.json
```


**Estado:** ejecutado correctamente.


## Predicciones prospectivas multi-métrica con Ridge.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\22_predict_laliga_current_players_final_model.py
```


### Salida estándar

```text
Porteros eliminados: 25
Generando predicciones prospectivas para: xG_90
Generando predicciones prospectivas para: goals_90
Generando predicciones prospectivas para: assists_90
Generando predicciones prospectivas para: xA_90
Predicciones prospectivas multi-métrica con Ridge generadas correctamente.

Jugadores evaluados: 309
Jugadores F/M evaluados: 175

Resumen por métrica:
   metric_key           metric_label  ...  prediction_max  prediction_min
0       xG_90        Goles esperados  ...        0.835401       -0.051106
1    goals_90                  Goles  ...        0.864044       -0.017819
2  assists_90            Asistencias  ...        0.368963       -0.016410
3       xA_90  Asistencias esperadas  ...        0.392873       -0.014951

[4 rows x 6 columns]

Top 25 F/M por xG_90 predicho:
    model     id  ... predicted_next_assists_90 predicted_next_xA_90
0   Ridge    227  ...                  0.201459             0.167457
1   Ridge   3423  ...                  0.327938             0.349572
2   Ridge   6441  ...                  0.237592             0.235415
3   Ridge  13996  ...                  0.227967             0.193469
4   Ridge   5074  ...                  0.208201             0.206516
5   Ridge   8026  ...                  0.368963             0.392873
6   Ridge  11527  ...                  0.291531             0.322679
7   Ridge   2270  ...                  0.268645             0.270974
8   Ridge   8187  ...                  0.175412             0.165411
9   Ridge   9002  ...                  0.083772             0.091686
10  Ridge   7008  ...                  0.277055             0.307715
11  Ridge   6531  ...                  0.120352             0.114436
12  Ridge   2234  ...                  0.201924             0.219640
13  Ridge   2543  ...                  0.164364             0.174836
14  Ridge  10095  ...                  0.214186             0.207129
15  Ridge  10930  ...                  0.114969             0.106116
16  Ridge  10846  ...                  0.239587             0.250150
17  Ridge   9275  ...                  0.122931             0.124779
18  Ridge   4175  ...                  0.185282             0.197354
19  Ridge   2120  ...                  0.256787             0.259566
20  Ridge   6170  ...                  0.190311             0.198752
21  Ridge  11822  ...                  0.284008             0.280262
22  Ridge  12160  ...                  0.184010             0.193753
23  Ridge   5656  ...                  0.281506             0.302021
24  Ridge   6954  ...                  0.175396             0.192271

[25 rows x 23 columns]

Predicciones wide generadas: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_current_player_predictions_final_ridge.csv
Predicciones long generadas: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_current_player_predictions_final_ridge_long.csv
Top 25 F/M legacy generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_current_top_25_f_m_predictions_final_ridge.csv
Top 25 F/M por métrica generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\laliga_current_top_25_f_m_predictions_by_metric_final_ridge.csv
Informe generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\22_laliga_current_predictions_final_model.md
```


**Estado:** ejecutado correctamente.


## Resumen final de experimentos multi-métrica.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\scripts\23_build_final_experiment_summary.py
```


### Salida estándar

```text
Resumen final de experimentos actualizado correctamente.

Resumen de selección de modelo:
              summary_section  ... mean_signed_error
0     internal_temporal_xG_90  ...               NaN
1     internal_temporal_xG_90  ...               NaN
2     internal_temporal_xG_90  ...               NaN
3     internal_temporal_xG_90  ...               NaN
4     internal_temporal_xG_90  ...               NaN
5       external_global_xG_90  ...               NaN
6       external_global_xG_90  ...               NaN
7       external_global_xG_90  ...               NaN
8       external_global_xG_90  ...               NaN
9       external_global_xG_90  ...               NaN
10         external_f_m_xG_90  ...               NaN
11         external_f_m_xG_90  ...               NaN
12         external_f_m_xG_90  ...               NaN
13         external_f_m_xG_90  ...               NaN
14         external_f_m_xG_90  ...               NaN
15  external_high_xG_90_range  ...           -0.1017
16  external_high_xG_90_range  ...           -0.0823
17  external_high_xG_90_range  ...           -0.0993
18  external_high_xG_90_range  ...           -0.1277
19  external_high_xG_90_range  ...           -0.4897

[20 rows x 14 columns]

Resumen multi-métrica:
                summary_section  metric_key  ... input_season target_season
0     external_multi_metric_f_m  assists_90  ...    2024-2025     2025-2026
1     external_multi_metric_f_m    goals_90  ...    2024-2025     2025-2026
2     external_multi_metric_f_m       xA_90  ...    2024-2025     2025-2026
3     external_multi_metric_f_m       xG_90  ...    2024-2025     2025-2026
4  external_multi_metric_global  assists_90  ...    2024-2025     2025-2026
5  external_multi_metric_global    goals_90  ...    2024-2025     2025-2026
6  external_multi_metric_global       xA_90  ...    2024-2025     2025-2026
7  external_multi_metric_global       xG_90  ...    2024-2025     2025-2026

[8 rows x 10 columns]

Resumen prospectivo:
   metric_key  ...                                              notes
0       xG_90  ...  Predicción prospectiva no evaluada todavía con...
1    goals_90  ...  Predicción prospectiva no evaluada todavía con...
2  assists_90  ...  Predicción prospectiva no evaluada todavía con...
3       xA_90  ...  Predicción prospectiva no evaluada todavía con...

[4 rows x 12 columns]

CSV final generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\final_experiment_summary.csv
CSV multi-métrica generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\final_multi_metric_summary.csv
CSV prospectivo generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\data\processed\final_prospective_prediction_summary.csv
Informe final generado: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\reports\23_final_experiment_summary.md
```


**Estado:** ejecutado correctamente.


## Aviso de salidas faltantes en `preprocessing`

- `data\processed\laliga_players_all_clean.csv`


# Fase: tests


## Ejecución de tests automatizados con pytest.

```powershell
C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\.venv\Scripts\python.exe -m pytest C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO\tests
```


### Salida estándar

```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\apc01\OneDrive - Luca Soluciones en TIC, S.L\Documentos\U-tad\5º\TFGs\INSO
plugins: anyio-4.13.0
collected 10 items

tests\test_player_temporal_features.py .....                             [ 50%]
tests\test_temporal_dataset_structure.py .                               [ 60%]
tests\test_text_normalization.py ....                                    [100%]

============================= 10 passed in 0.72s ==============================
```


**Estado:** ejecutado correctamente.


# Ejecución completada

Finalización: 2026-05-30 13:44:25

