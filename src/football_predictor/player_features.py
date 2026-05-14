import pandas as pd


PLAYER_ID_COLUMNS = [
    "id",
    "player_name",
    "position",
    "team_title",
    "league",
    "season",
]


PLAYER_CONTEXT_COLUMNS = [
    "games",
    "time",
]


PLAYER_PERFORMANCE_COLUMNS = [
    "goals_90",
    "assists_90",
    "xA_90",
    "shots_90",
    "key_passes_90",
    "yellow_cards_90",
    "red_cards_90",
    "xGChain_90",
    "xGBuildup_90",
]


DEFAULT_TARGET_COLUMN = "xG_90"


def validate_player_modeling_columns(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> None:
    """
    Comprueba que el dataset contiene las columnas necesarias para construir
    la primera tabla de modelado de rendimiento individual.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio de jugadores.
    target_column : str
        Variable objetivo que se quiere predecir.
    """
    required_columns = (
        PLAYER_ID_COLUMNS
        + PLAYER_CONTEXT_COLUMNS
        + PLAYER_PERFORMANCE_COLUMNS
        + [target_column]
    )

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para modelado de jugadores: {missing_columns}"
        )


def simplify_position(position: str) -> str:
    """
    Simplifica la posición del jugador a una categoría principal.

    El dataset puede contener valores compuestos como 'F M S'. Para una
    primera versión del modelo, se toma la primera letra como posición
    principal.

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


def add_basic_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade variables básicas derivadas del jugador.

    De momento incorporamos:
    - position_main: posición principal simplificada.
    - minutes_per_game: minutos medios disputados por partido.
    - high_participation: indicador binario de participación alta.
    """
    featured_df = df.copy()

    featured_df["position_main"] = featured_df["position"].apply(simplify_position)

    featured_df["minutes_per_game"] = (
        featured_df["time"] / featured_df["games"]
    )

    featured_df["high_participation"] = (
        featured_df["minutes_per_game"] >= 60
    ).astype(int)

    return featured_df


def build_player_modeling_dataset(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Construye el dataset base para predecir rendimiento individual.

    Para esta primera versión se plantea un problema de regresión en el que
    la variable objetivo es `target_column`. Por defecto se utiliza xG_90.

    No se aplica aquí un filtro adicional de minutos, porque el control de
    volumen mínimo de participación ya se realiza previamente en la fase de
    preprocesamiento.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio de jugadores históricos.
    target_column : str
        Métrica de rendimiento que se quiere predecir.

    Returns
    -------
    pd.DataFrame
        Dataset preparado para modelado supervisado.
    """
    validate_player_modeling_columns(df, target_column)

    modeling_df = df.copy()

    modeling_df = add_basic_player_features(modeling_df)

    modeling_df[f"target_{target_column}"] = modeling_df[target_column]

    selected_columns = (
        PLAYER_ID_COLUMNS
        + [
            "position_main",
            "games",
            "time",
            "minutes_per_game",
            "high_participation",
        ]
        + PLAYER_PERFORMANCE_COLUMNS
        + [f"target_{target_column}"]
    )

    modeling_df = modeling_df[selected_columns].copy()

    modeling_df = modeling_df.sort_values(
        by=["season", "league", "team_title", "player_name"]
    ).reset_index(drop=True)

    return modeling_df