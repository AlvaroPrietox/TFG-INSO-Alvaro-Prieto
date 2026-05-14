# Dataset: matches

- Filas: 8972
- Columnas: 24
- Filas duplicadas: 0

## Columnas

- Date (str)
- HomeTeam (str)
- AwayTeam (str)
- FTHG (int64)
- FTAG (int64)
- FTR (str)
- HTHG (float64)
- HTAG (float64)
- HTR (str)
- Referee (str)
- HS (float64)
- AS (float64)
- HST (float64)
- AST (float64)
- HF (float64)
- AF (float64)
- HC (float64)
- AC (float64)
- HY (float64)
- AY (float64)
- HR (float64)
- AR (float64)
- liga (str)
- temporada (str)

## Valores nulos por columna

- Referee: 7072 nulos (78.82%)
- HTHG: 1 nulos (0.01%)
- HTAG: 1 nulos (0.01%)
- HTR: 1 nulos (0.01%)
- HS: 1 nulos (0.01%)
- AS: 1 nulos (0.01%)
- HST: 1 nulos (0.01%)
- AST: 1 nulos (0.01%)
- HF: 1 nulos (0.01%)
- AF: 1 nulos (0.01%)
- HC: 1 nulos (0.01%)
- AC: 1 nulos (0.01%)
- HY: 1 nulos (0.01%)
- AY: 1 nulos (0.01%)
- HR: 1 nulos (0.01%)
- AR: 1 nulos (0.01%)

## Primeras filas

| Date     | HomeTeam      | AwayTeam   |   FTHG |   FTAG | FTR   |   HTHG |   HTAG | HTR   |   Referee |   HS |   AS |   HST |   AST |   HF |   AF |   HC |   AC |   HY |   AY |   HR |   AR | liga       | temporada   |
|:---------|:--------------|:-----------|-------:|-------:|:------|-------:|-------:|:------|----------:|-----:|-----:|------:|------:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|:-----------|:------------|
| 18/09/20 | Bayern Munich | Schalke 04 |      8 |      0 | H     |      3 |      0 | H     |       nan |   22 |    5 |    12 |     1 |   11 |   12 |    9 |    2 |    1 |    2 |    0 |    0 | Bundesliga | 2020-2021   |
| 19/09/20 | Ein Frankfurt | Bielefeld  |      1 |      1 | D     |      0 |      0 | D     |       nan |   18 |   14 |     6 |     4 |   14 |   13 |   14 |    3 |    2 |    2 |    0 |    0 | Bundesliga | 2020-2021   |
| 19/09/20 | FC Koln       | Hoffenheim |      2 |      3 | A     |      1 |      2 | A     |       nan |   13 |   13 |     6 |     7 |   12 |   13 |    1 |    6 |    0 |    0 |    0 |    0 | Bundesliga | 2020-2021   |
| 19/09/20 | Stuttgart     | Freiburg   |      2 |      3 | A     |      0 |      2 | A     |       nan |   22 |    7 |     7 |     6 |   12 |   16 |    7 |    2 |    2 |    2 |    0 |    0 | Bundesliga | 2020-2021   |
| 19/09/20 | Union Berlin  | Augsburg   |      1 |      3 | A     |      0 |      1 | A     |       nan |   13 |    9 |     3 |     5 |    9 |    9 |    8 |    1 |    2 |    0 |    0 |    0 | Bundesliga | 2020-2021   |

---

# Dataset: historical_players

- Filas: 8063
- Columnas: 20
- Filas duplicadas: 0

## Columnas

- id (int64)
- player_name (str)
- games (int64)
- time (int64)
- position (str)
- team_title (str)
- league (str)
- season (str)
- goals_90 (float64)
- xG_90 (float64)
- assists_90 (float64)
- xA_90 (float64)
- shots_90 (float64)
- key_passes_90 (float64)
- yellow_cards_90 (float64)
- red_cards_90 (float64)
- npg_90 (float64)
- npxG_90 (float64)
- xGChain_90 (float64)
- xGBuildup_90 (float64)

## Valores nulos por columna

No se han detectado valores nulos.

## Primeras filas

|   id | player_name        |   games |   time | position   | team_title          | league     | season    |   goals_90 |    xG_90 |   assists_90 |    xA_90 |   shots_90 |   key_passes_90 |   yellow_cards_90 |   red_cards_90 |   npg_90 |   npxG_90 |   xGChain_90 |   xGBuildup_90 |
|-----:|:-------------------|--------:|-------:|:-----------|:--------------------|:-----------|:----------|-----------:|---------:|-------------:|---------:|-----------:|----------------:|------------------:|---------------:|---------:|----------:|-------------:|---------------:|
|  227 | Robert Lewandowski |      29 |   2467 | F S        | Bayern Munich       | Bundesliga | 2020-2021 |   1.49574  | 1.17023  |     0.255371 | 0.175679 |    4.92501 |         1.16741 |         0.145926  |              0 | 1.20389  |  0.921427 |     1.15793  |       0.207556 |
| 6170 | André Silva        |      32 |   2787 | F          | Eintracht Frankfurt | Bundesliga | 2020-2021 |   0.904198 | 0.826667 |     0.161464 | 0.176547 |    3.68138 |         1.00108 |         0.0322928 |              0 | 0.678149 |  0.655375 |     0.863722 |       0.132285 |
| 8260 | Erling Haaland     |      28 |   2416 | F S        | Borussia Dortmund   | Bundesliga | 2020-2021 |   1.00579  | 0.879083 |     0.22351  | 0.150327 |    3.42715 |         1.00579 |         0.0745033 |              0 | 0.931291 |  0.766169 |     1.01314  |       0.219643 |
|  956 | Andrej Kramaric    |      28 |   2386 | F M S      | Hoffenheim          | Bundesliga | 2020-2021 |   0.754401 | 0.585626 |     0.1886   | 0.153612 |    3.5834  |         1.43336 |         0.0754401 |              0 | 0.565801 |  0.44271  |     0.681137 |       0.202643 |
| 7052 | Wout Weghorst      |      34 |   2954 | F S        | Wolfsburg           | Bundesliga | 2020-2021 |   0.609343 | 0.557866 |     0.243737 | 0.165355 |    2.83345 |         1.37102 |         0.0914015 |              0 | 0.548409 |  0.488604 |     0.740179 |       0.181427 |

---

# Dataset: laliga_players

- Filas: 2422
- Columnas: 22
- Filas duplicadas: 0

## Columnas

- id (int64)
- player_name (str)
- games (int64)
- time (int64)
- goals (int64)
- xG (float64)
- assists (int64)
- xA (float64)
- shots (int64)
- key_passes (int64)
- yellow_cards (int64)
- red_cards (int64)
- position (str)
- team_title (str)
- npg (int64)
- npxG (float64)
- xGChain (float64)
- xGBuildup (float64)
- league (str)
- season (str)
- jornada (str)
- fecha_descarga (str)

## Valores nulos por columna

No se han detectado valores nulos.

## Primeras filas

|    id | player_name          |   games |   time |   goals |      xG |   assists |       xA |   shots |   key_passes |   yellow_cards |   red_cards | position   | team_title         |   npg |    npxG |   xGChain |   xGBuildup | league   | season    | jornada    | fecha_descarga      |
|------:|:---------------------|--------:|-------:|--------:|--------:|----------:|---------:|--------:|-------------:|---------------:|------------:|:-----------|:-------------------|------:|--------:|----------:|------------:|:---------|:----------|:-----------|:--------------------|
|  3423 | Kylian Mbappe-Lottin |      10 |    886 |      11 | 9.70221 |         2 | 2.51507  |      52 |           27 |              2 |           0 | F          | Real Madrid        |     9 | 7.47238 |  10.8098  |    2.79997  | La-Liga  | 2025-2026 | Jornada_10 | 2025-10-30 16:00:57 |
| 10846 | Julián Álvarez       |      10 |    794 |       6 | 5.15938 |         2 | 3.06654  |      22 |           17 |              1 |           0 | F          | Atletico Madrid    |     5 | 3.67282 |   7.63787 |    3.05471  | La-Liga  | 2025-2026 | Jornada_10 | 2025-10-30 16:00:57 |
| 10930 | Etta Eyong           |      10 |    826 |       6 | 6.50574 |         3 | 1.67035  |      24 |            8 |              2 |           0 | F          | Levante,Villarreal |     6 | 6.50574 |   6.47082 |    0.305498 | La-Liga  | 2025-2026 | Jornada_10 | 2025-10-30 16:00:57 |
|  7008 | Vinícius Júnior      |      10 |    716 |       5 | 3.75609 |         4 | 2.90668  |      27 |           22 |              1 |           0 | F M S      | Real Madrid        |     4 | 3.01281 |   7.83878 |    2.20236  | La-Liga  | 2025-2026 | Jornada_10 | 2025-10-30 16:00:57 |
|  9002 | Vedat Muriqi         |       9 |    735 |       5 | 3.56027 |         0 | 0.125388 |      25 |            4 |              1 |           1 | F          | Mallorca           |     4 | 2.81699 |   4.05233 |    1.75771  | La-Liga  | 2025-2026 | Jornada_10 | 2025-10-30 16:00:57 |

---
