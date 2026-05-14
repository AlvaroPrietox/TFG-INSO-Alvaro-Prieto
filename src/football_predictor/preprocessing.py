from pathlib import Path

import pandas as pd


# Preprocesamiento de datos históricos de partidos

REQUIRED_MATCH_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "liga",
    "temporada",
]


RESULT_MAPPING = {
    "H": 2,  # Victoria local
    "D": 1,  # Empate
    "A": 0,  # Victoria visitante
}


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Comprueba que el DataFrame contiene todas las columnas necesarias.

    Esta función evita que el pipeline continúe si falta alguna variable
    esencial para construir el dataset de entrenamiento.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")


def normalize_team_name(team_name: str) -> str:
    """
    Normaliza el nombre de un equipo eliminando espacios innecesarios.

    Más adelante ampliaremos esta función con equivalencias entre nombres,
    por ejemplo: 'Ath Madrid' -> 'Atletico Madrid'.
    """
    if pd.isna(team_name):
        return team_name

    return str(team_name).strip()


def clean_matches_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset histórico de partidos.

    Operaciones realizadas:
    1. Valida que existen las columnas necesarias.
    2. Convierte la fecha a formato datetime.
    3. Normaliza nombres de equipos.
    4. Crea una variable objetivo numérica para modelado.
    5. Ordena los partidos cronológicamente.
    """
    validate_required_columns(df, REQUIRED_MATCH_COLUMNS)

    clean_df = df.copy()

    clean_df["Date"] = pd.to_datetime(
        clean_df["Date"],
        format="%d/%m/%y",
        errors="coerce",
    )

    clean_df["HomeTeam"] = clean_df["HomeTeam"].apply(normalize_team_name)
    clean_df["AwayTeam"] = clean_df["AwayTeam"].apply(normalize_team_name)

    clean_df["target_result"] = clean_df["FTR"].map(RESULT_MAPPING)

    rows_before = len(clean_df)

    clean_df = clean_df.dropna(
        subset=[
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
            "target_result",
        ]
    )

    rows_after = len(clean_df)
    removed_rows = rows_before - rows_after

    if removed_rows > 0:
        print(f"Se han eliminado {removed_rows} filas con valores críticos nulos.")

    clean_df = clean_df.sort_values(
        by=["Date", "liga", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    return clean_df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Guarda un DataFrame en CSV creando antes la carpeta de destino si no existe.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    

# Preprocesamiento de datos históricos y actuales de jugadores

HISTORICAL_PLAYER_REQUIRED_COLUMNS = [
    "id",
    "player_name",
    "games",
    "time",
    "position",
    "team_title",
    "league",
    "season",
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


LALIGA_PLAYER_REQUIRED_COLUMNS = [
    "id",
    "player_name",
    "games",
    "time",
    "goals",
    "xG",
    "assists",
    "xA",
    "shots",
    "key_passes",
    "yellow_cards",
    "red_cards",
    "position",
    "team_title",
    "npg",
    "npxG",
    "xGChain",
    "xGBuildup",
    "league",
    "season",
    "jornada",
    "fecha_descarga",
]


PLAYER_BASE_COLUMNS = [
    "id",
    "player_name",
    "games",
    "time",
    "position",
    "team_title",
    "league",
    "season",
]


PLAYER_PER90_COLUMNS = [
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


TOTAL_TO_PER90_MAPPING = {
    "goals": "goals_90",
    "xG": "xG_90",
    "assists": "assists_90",
    "xA": "xA_90",
    "shots": "shots_90",
    "key_passes": "key_passes_90",
    "yellow_cards": "yellow_cards_90",
    "red_cards": "red_cards_90",
    "npg": "npg_90",
    "npxG": "npxG_90",
    "xGChain": "xGChain_90",
    "xGBuildup": "xGBuildup_90",
}


def clean_historical_players_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset histórico de jugadores.

    Este dataset ya contiene métricas normalizadas por 90 minutos,
    por lo que la limpieza se centra en validar columnas, normalizar
    nombres de equipos y seleccionar una estructura común.
    """
    validate_required_columns(df, HISTORICAL_PLAYER_REQUIRED_COLUMNS)

    clean_df = df.copy()

    clean_df["team_title"] = clean_df["team_title"].apply(normalize_team_name)
    clean_df["player_name"] = clean_df["player_name"].astype(str).str.strip()
    clean_df["league"] = clean_df["league"].astype(str).str.strip()
    clean_df["season"] = clean_df["season"].astype(str).str.strip()

    output_columns = PLAYER_BASE_COLUMNS + PLAYER_PER90_COLUMNS

    clean_df = clean_df[output_columns].copy()
    clean_df["data_source"] = "historical"
    clean_df["jornada"] = None
    clean_df["fecha_descarga"] = None

    return clean_df.reset_index(drop=True)


def clean_laliga_players_dataset(df: pd.DataFrame, latest_jornada: int = 14, latest_min_minutes: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Limpia el dataset de jugadores de LaLiga y devuelve dos versiones:

    1. Dataset completo con todas las jornadas/capturas disponibles.
    2. Dataset latest con la última jornada seleccionada y filtrada por minutos.

    El dataset original contiene métricas acumuladas. Para hacerlo comparable
    con el dataset histórico, se transforman las variables principales a
    métricas por 90 minutos.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original de jugadores de LaLiga.
    latest_jornada : int
        Jornada que se considera como última fotografía disponible.
    latest_min_minutes : int
        Umbral mínimo de minutos para incluir jugadores en el dataset latest.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Primer elemento: dataset limpio con todas las jornadas.
        Segundo elemento: dataset limpio de la última jornada filtrado por minutos.
    """
    validate_required_columns(df, LALIGA_PLAYER_REQUIRED_COLUMNS)

    clean_df = df.copy()

    clean_df["fecha_descarga"] = pd.to_datetime(
        clean_df["fecha_descarga"],
        errors="coerce",
    )

    rows_before_cleaning = len(clean_df)

    clean_df = clean_df.dropna(
        subset=[
            "id",
            "player_name",
            "time",
            "team_title",
            "league",
            "season",
            "jornada",
            "fecha_descarga",
        ]
    )

    clean_df = clean_df[clean_df["time"] > 0].copy()

    rows_after_basic_cleaning = len(clean_df)
    removed_invalid_rows = rows_before_cleaning - rows_after_basic_cleaning

    if removed_invalid_rows > 0:
        print(f"Se han eliminado {removed_invalid_rows} jugadores sin datos válidos.")

    clean_df["team_title"] = clean_df["team_title"].apply(normalize_team_name)
    clean_df["player_name"] = clean_df["player_name"].astype(str).str.strip()
    clean_df["league"] = clean_df["league"].astype(str).str.strip()
    clean_df["season"] = clean_df["season"].astype(str).str.strip()
    clean_df["jornada"] = clean_df["jornada"].astype(str).str.strip()

    clean_df["jornada_num"] = (
        clean_df["jornada"]
        .str.extract(r"(\d+)")
        .astype("Int64")
    )

    if clean_df["jornada_num"].isna().any():
        raise ValueError(
            "No se ha podido extraer el número de jornada en algunas filas."
        )

    for total_column, per90_column in TOTAL_TO_PER90_MAPPING.items():
        clean_df[per90_column] = clean_df[total_column] / clean_df["time"] * 90

    output_columns = (
        PLAYER_BASE_COLUMNS
        + PLAYER_PER90_COLUMNS
        + ["jornada", "jornada_num", "fecha_descarga"]
    )

    laliga_players_all_clean = clean_df[output_columns].copy()
    laliga_players_all_clean["data_source"] = "laliga_all"

    laliga_players_all_clean = laliga_players_all_clean.sort_values(
        by=["jornada_num", "fecha_descarga", "team_title", "player_name"]
    ).reset_index(drop=True)

    available_jornadas = sorted(
        laliga_players_all_clean["jornada_num"].dropna().unique().tolist()
    )

    if latest_jornada not in available_jornadas:
        raise ValueError(
            f"La jornada {latest_jornada} no está disponible. "
            f"Jornadas encontradas: {available_jornadas}"
        )

    laliga_players_latest_clean = laliga_players_all_clean[
        laliga_players_all_clean["jornada_num"] == latest_jornada
    ].copy()

    print(f"Jornada latest seleccionada: Jornada_{latest_jornada}")
    print(
        "Filas en latest antes del filtro de minutos: "
        f"{len(laliga_players_latest_clean)}"
    )

    rows_before_minutes_filter = len(laliga_players_latest_clean)

    laliga_players_latest_clean = laliga_players_latest_clean[
        laliga_players_latest_clean["time"] >= latest_min_minutes
    ].copy()

    rows_after_minutes_filter = len(laliga_players_latest_clean)
    removed_low_minutes_players = (
        rows_before_minutes_filter - rows_after_minutes_filter
    )

    print(
        f"Filtro latest aplicado: jugadores con "
        f"{latest_min_minutes} minutos o más."
    )
    print(
        "Jugadores eliminados en latest por bajo volumen de minutos: "
        f"{removed_low_minutes_players}"
    )

    laliga_players_latest_clean["data_source"] = "laliga_latest"

    laliga_players_latest_clean = laliga_players_latest_clean.sort_values(
        by=["team_title", "player_name"]
    ).reset_index(drop=True)

    return laliga_players_all_clean, laliga_players_latest_clean