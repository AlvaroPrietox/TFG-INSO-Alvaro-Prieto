# Resumen final de experimentos

## Objetivo

Este informe consolida los principales experimentos realizados para la predicción del rendimiento ofensivo futuro de jugadores de fútbol, utilizando `xG_90` de la temporada siguiente como variable objetivo.

## Tabla resumen

| experiment                       | evaluation_type                 | subset                             |   n_train |   n_test |    mae |     r2 | test_period           |
|:---------------------------------|:--------------------------------|:-----------------------------------|----------:|---------:|-------:|-------:|:----------------------|
| external_laliga_2025_2026_f_m    | evaluación externa              | F y M                              |       nan |      110 | 0.0738 | 0.7561 | 2024-2025 → 2025-2026 |
| external_laliga_2025_2026_global | evaluación externa              | jugadores emparejados sin porteros |       nan |      205 | 0.055  | 0.8023 | 2024-2025 → 2025-2026 |
| external_laliga_2025_2026_F      | evaluación externa por posición | solo F                             |       nan |       61 | 0.0995 | 0.6258 | 2024-2025 → 2025-2026 |
| external_laliga_2025_2026_M      | evaluación externa por posición | solo M                             |       nan |       49 | 0.0418 | 0.6919 | 2024-2025 → 2025-2026 |
| temporal_full                    | validación temporal interna     | todos sin porteros                 |      3003 |      954 | 0.0577 | 0.7242 | 2023-2024 → 2024-2025 |
| temporal_without_previous_xg     | validación temporal interna     | todos sin porteros                 |      3003 |      954 | 0.0601 | 0.703  | 2023-2024 → 2024-2025 |

## Interpretación por experimento

### external_laliga_2025_2026_f_m

- Tipo de evaluación: evaluación externa
- Subconjunto: F y M
- MAE: 0.0738
- R²: 0.7561
- Periodo de prueba: 2024-2025 → 2025-2026

Evaluación externa restringida a delanteros, atacantes y centrocampistas. Es el subconjunto más relevante para analizar una métrica ofensiva como xG_90.

### external_laliga_2025_2026_global

- Tipo de evaluación: evaluación externa
- Subconjunto: jugadores emparejados sin porteros
- MAE: 0.055
- R²: 0.8023
- Periodo de prueba: 2024-2025 → 2025-2026

Evaluación externa usando datos de 2024-2025 para predecir xG_90 real en LaLiga 2025-2026. La unión se realiza mediante nombre normalizado, por lo que debe interpretarse con cautela.

### external_laliga_2025_2026_F

- Tipo de evaluación: evaluación externa por posición
- Subconjunto: solo F
- MAE: 0.0995
- R²: 0.6258
- Periodo de prueba: 2024-2025 → 2025-2026

Evaluación externa restringida a delanteros y atacantes. Es el grupo con mayor variabilidad ofensiva, por lo que se espera un error superior.

### external_laliga_2025_2026_M

- Tipo de evaluación: evaluación externa por posición
- Subconjunto: solo M
- MAE: 0.0418
- R²: 0.6919
- Periodo de prueba: 2024-2025 → 2025-2026

Evaluación externa restringida a centrocampistas. Presenta un comportamiento más estable que los delanteros en términos de xG_90.

### temporal_full

- Tipo de evaluación: validación temporal interna
- Subconjunto: todos sin porteros
- MAE: 0.0577
- R²: 0.7242
- Periodo de prueba: 2023-2024 → 2024-2025

Modelo temporal con todas las variables históricas, incluyendo xG_90 y npxG_90 de la temporada anterior. Representa el escenario con máxima información ofensiva previa.

### temporal_without_previous_xg

- Tipo de evaluación: validación temporal interna
- Subconjunto: todos sin porteros
- MAE: 0.0601
- R²: 0.703
- Periodo de prueba: 2023-2024 → 2024-2025

Modelo temporal sin xG_90 ni npxG_90 previos. Evalúa si variables como tiros, goles, asistencias, participación y posición permiten anticipar el rendimiento futuro sin usar directamente xG previo.

## Conclusión general

Los resultados muestran que el modelo temporal mantiene una capacidad predictiva sólida tanto en validación interna como en una evaluación externa sobre LaLiga 2025-2026. La eliminación de variables directas de xG previo reduce ligeramente el rendimiento, pero el modelo sigue capturando señal mediante variables como volumen de tiro, goles por 90, participación y posición principal.

La evaluación específica sobre jugadores F/M es más exigente que la evaluación global, ya que excluye perfiles defensivos de bajo xG_90 y se centra en jugadores con mayor variabilidad ofensiva. Aun así, los resultados siguen siendo suficientemente consistentes para defender la utilidad del enfoque propuesto.

La principal limitación detectada es que el modelo tiende a comportarse de forma conservadora: predice bien perfiles estables, pero tiene más dificultad para anticipar saltos abruptos de rendimiento asociados a cambios de contexto, rol táctico, equipo, lesiones o variaciones en el modelo de juego.