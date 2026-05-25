from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "laliga_players_2526.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "laliga_2025_2026_target_clean.csv"


REQUIRED_COLUMNS = [
    "player",
    "team",
    "apps",
    "min",
    "goals",
    "a",
    "xG",
    "xA",
    "xG90",
    "xA90",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias en el CSV 2025-2026: {missing_columns}"
        )


def clean_laliga_2025_2026_target(
    df: pd.DataFrame,
    min_minutes: int = 900,
) -> pd.DataFrame:
    """
    Limpia el archivo de jugadores de LaLiga 2025-2026 para utilizarlo
    como temporada objetivo real.

    Se aplica un filtro mínimo de minutos para evitar que jugadores con
    muestras muy pequeñas generen valores extremos de xG_90.
    """
    validate_columns(df)

    clean_df = df.copy()

    clean_df = clean_df.rename(
        columns={
            "player": "player_name",
            "team": "team_title",
            "apps": "games",
            "min": "time",
            "a": "assists",
            "xG90": "xG_90",
            "xA90": "xA_90",
        }
    )

    clean_df["player_name"] = clean_df["player_name"].astype(str).str.strip()
    clean_df["team_title"] = clean_df["team_title"].astype(str).str.strip()

    clean_df = clean_df.dropna(
        subset=[
            "player_name",
            "team_title",
            "games",
            "time",
            "goals",
            "assists",
            "xG",
            "xA",
            "xG_90",
            "xA_90",
        ]
    )

    clean_df = clean_df[clean_df["time"] > 0].copy()

    rows_before_minutes_filter = len(clean_df)

    clean_df = clean_df[clean_df["time"] >= min_minutes].copy()

    rows_after_minutes_filter = len(clean_df)
    removed_low_minutes_players = (
        rows_before_minutes_filter - rows_after_minutes_filter
    )

    print(
        f"Filtro de minutos aplicado en 2025-2026: "
        f"jugadores con {min_minutes} minutos o más."
    )
    print(
        f"Jugadores eliminados por bajo volumen de minutos: "
        f"{removed_low_minutes_players}"
    )

    clean_df["goals_90"] = clean_df["goals"] / clean_df["time"] * 90
    clean_df["assists_90"] = clean_df["assists"] / clean_df["time"] * 90

    clean_df["league"] = "La-Liga"
    clean_df["season"] = "2025-2026"

    output_columns = [
        "player_name",
        "team_title",
        "league",
        "season",
        "games",
        "time",
        "goals",
        "assists",
        "xG",
        "xA",
        "goals_90",
        "xG_90",
        "assists_90",
        "xA_90",
    ]

    clean_df = clean_df[output_columns].copy()

    clean_df = clean_df.sort_values(
        by="xG_90",
        ascending=False,
    ).reset_index(drop=True)

    return clean_df

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(
        RAW_INPUT_PATH,
        sep=";",
        encoding="utf-8-sig",
    )

    clean_df = clean_laliga_2025_2026_target(
        raw_df,
        min_minutes=900,
    )

    clean_df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset objetivo 2025-2026 limpiado correctamente.")
    print(f"Filas finales: {clean_df.shape[0]}")
    print(f"Columnas finales: {clean_df.shape[1]}")
    print(f"Archivo generado: {OUTPUT_PATH}")
    print("")
    print("Top 10 por xG_90 real:")
    print(clean_df.head(10))


if __name__ == "__main__":
    main()