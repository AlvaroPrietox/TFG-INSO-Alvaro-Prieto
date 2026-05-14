import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.preprocessing import (
    clean_historical_players_dataset,
    clean_laliga_players_dataset,
    save_dataframe,
)


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

HISTORICAL_PLAYERS_PATH = RAW_DATA_DIR / "all_players_normalized.csv"
LALIGA_PLAYERS_PATH = RAW_DATA_DIR / "players_laliga.csv"

HISTORICAL_OUTPUT_PATH = PROCESSED_DATA_DIR / "historical_players_clean.csv"
LALIGA_OUTPUT_PATH = PROCESSED_DATA_DIR / "laliga_players_current_clean.csv"
LALIGA_LATEST_OUTPUT_PATH = PROCESSED_DATA_DIR / "laliga_players_latest_clean.csv"


def main() -> None:
    historical_players_df = pd.read_csv(HISTORICAL_PLAYERS_PATH)
    laliga_players_df = pd.read_csv(LALIGA_PLAYERS_PATH)

    historical_players_clean = clean_historical_players_dataset(historical_players_df)
    laliga_players_clean, laliga_players_latest_clean = clean_laliga_players_dataset(laliga_players_df, latest_jornada=14, latest_min_minutes=300)

    save_dataframe(historical_players_clean, HISTORICAL_OUTPUT_PATH)
    save_dataframe(laliga_players_clean, LALIGA_OUTPUT_PATH)
    save_dataframe(laliga_players_latest_clean, LALIGA_LATEST_OUTPUT_PATH)

    print("Preprocesamiento de jugadores completado.")
    print("")
    print("Jugadores históricos:")
    print(f"- Filas: {historical_players_clean.shape[0]}")
    print(f"- Columnas: {historical_players_clean.shape[1]}")
    print(f"- Archivo generado: {HISTORICAL_OUTPUT_PATH}")
    print("")
    print("Jugadores actuales de LaLiga:")
    print(f"- Filas: {laliga_players_clean.shape[0]}")
    print(f"- Columnas: {laliga_players_clean.shape[1]}")
    print(f"- Archivo generado: {LALIGA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()