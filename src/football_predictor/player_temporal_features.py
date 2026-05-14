import pandas as pd


PLAYER_IDENTIFIER_COLUMNS = [
    "id",
    "player_name",
]


CONTEXT_COLUMNS = [
    "position",
    "team_title",
    "league",
    "season",
    "games",
    "time",
]


PERFORMANCE_COLUMNS = [
    "goals_90",
    "xG_90",
    "assists_90",
    "xA_90",
    "shots_90",
    "key_passes_90",
    "yellow_cards_90",
    "red_cards_90",
    "npg_90",
    "npxG_90",
    "xGChain_90",
    "xGBuildup_90",
]


DEFAULT_TEMPORAL_TARGET = "xG_90"


def validate_temporal_input_columns(df: pd.DataFrame) -> None:
    """
    Comprueba que el dataset histórico contiene las columnas necesarias
    para construir pares temporales jugador-temporada.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio de jugadores históricos.
    """
    required_columns = (
        PLAYER_IDENTIFIER_COLUMNS
        + CONTEXT_COLUMNS
        + PERFORMANCE_COLUMNS
    )

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para construir el dataset temporal: "
            f"{missing_columns}"
        )


def simplify_position(position: str) -> str:
    """
    Simplifica la posición del jugador a su rol principal.

    Examples
    --------
    'F M S' -> 'F'
    'M S'   -> 'M'
    """
    if pd.isna(position):
        return "Unknown"

    position = str(position).strip()

    if not position:
        return "Unknown"

    return position.split()[0]


def parse_season_start_year(season: str) -> int:
    """
    Extrae el año inicial de una temporada.

    Example
    -------
    '2020-2021' -> 2020
    """
    try:
        return int(str(season).split("-")[0])
    except (ValueError, IndexError):
        raise ValueError(f"Formato de temporada no válido: {season}")


def build_next_season_label(season: str) -> str:
    """
    Construye la etiqueta de la temporada siguiente.

    Example
    -------
    '2020-2021' -> '2021-2022'
    """
    start_year = parse_season_start_year(season)
    next_start_year = start_year + 1
    next_end_year = start_year + 2

    return f"{next_start_year}-{next_end_year}"


def add_temporal_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade variables básicas necesarias para el modelado temporal.
    """
    temporal_df = df.copy()

    temporal_df["position_main"] = temporal_df["position"].apply(
        simplify_position
    )

    temporal_df["season_start_year"] = temporal_df["season"].apply(
        parse_season_start_year
    )

    temporal_df["next_season"] = temporal_df["season"].apply(
        build_next_season_label
    )

    temporal_df["minutes_per_game"] = (
        temporal_df["time"] / temporal_df["games"]
    )

    temporal_df["high_participation"] = (
        temporal_df["minutes_per_game"] >= 60
    ).astype(int)

    return temporal_df


def build_player_temporal_modeling_dataset(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TEMPORAL_TARGET,
) -> pd.DataFrame:
    """
    Construye un dataset temporal para predecir rendimiento futuro.

    Cada fila utiliza las métricas de un jugador en una temporada t como
    variables explicativas, y el rendimiento del mismo jugador en la
    temporada t+1 como variable objetivo.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio de jugadores históricos.
    target_column : str
        Métrica de rendimiento futuro que se quiere predecir.

    Returns
    -------
    pd.DataFrame
        Dataset temporal de modelado.
    """
    validate_temporal_input_columns(df)

    if target_column not in df.columns:
        raise ValueError(f"La variable objetivo no existe: {target_column}")

    current_df = add_temporal_base_features(df)

    future_columns = [
        "id",
        "season",
        "team_title",
        "league",
        "games",
        "time",
        target_column,
    ]

    future_df = df[future_columns].copy()

    future_df = future_df.rename(
        columns={
            "season": "next_season",
            "team_title": "next_team_title",
            "league": "next_league",
            "games": "next_games",
            "time": "next_time",
            target_column: f"target_next_{target_column}",
        }
    )

    temporal_df = current_df.merge(
        future_df,
        how="inner",
        on=["id", "next_season"],
    )

    selected_columns = [
        "id",
        "player_name",
        "position",
        "position_main",
        "team_title",
        "league",
        "season",
        "season_start_year",
        "next_season",
        "next_team_title",
        "next_league",
        "games",
        "time",
        "minutes_per_game",
        "high_participation",
    ]

    selected_columns += PERFORMANCE_COLUMNS

    selected_columns += [
        "next_games",
        "next_time",
        f"target_next_{target_column}",
    ]

    temporal_df = temporal_df[selected_columns].copy()

    temporal_df = temporal_df.sort_values(
        by=["season_start_year", "league", "team_title", "player_name"]
    ).reset_index(drop=True)

    return temporal_df