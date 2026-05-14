import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_features import (
    build_player_temporal_modeling_dataset,
)
from football_predictor.preprocessing import save_dataframe


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

HISTORICAL_PLAYERS_PATH = PROCESSED_DATA_DIR / "historical_players_clean.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "player_temporal_modeling_dataset.csv"


def main() -> None:
    historical_players_df = pd.read_csv(HISTORICAL_PLAYERS_PATH)

    temporal_modeling_df = build_player_temporal_modeling_dataset(
        historical_players_df,
        target_column="xG_90",
    )

    save_dataframe(temporal_modeling_df, OUTPUT_PATH)

    print("Dataset temporal de jugadores generado correctamente.")
    print(f"Filas finales: {temporal_modeling_df.shape[0]}")
    print(f"Columnas finales: {temporal_modeling_df.shape[1]}")
    print(f"Archivo generado: {OUTPUT_PATH}")

    print("")
    print("Pares temporada actual -> temporada siguiente:")
    print(
        temporal_modeling_df[["season", "next_season"]]
        .drop_duplicates()
        .sort_values(by=["season", "next_season"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()