import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.preprocessing import clean_matches_dataset, save_dataframe


RAW_MATCHES_PATH = PROJECT_ROOT / "data" / "raw" / "matches_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"


def main() -> None:
    matches_df = pd.read_csv(RAW_MATCHES_PATH)

    clean_matches_df = clean_matches_dataset(matches_df)

    save_dataframe(clean_matches_df, OUTPUT_PATH)

    print("Preprocesamiento de partidos completado.")
    print(f"Filas finales: {clean_matches_df.shape[0]}")
    print(f"Columnas finales: {clean_matches_df.shape[1]}")
    print(f"Archivo generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()