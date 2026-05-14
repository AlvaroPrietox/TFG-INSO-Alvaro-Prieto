import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_modeling import train_and_evaluate_baseline_models


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PLAYER_MODELING_DATASET_PATH = PROCESSED_DATA_DIR / "player_modeling_dataset.csv"
EVALUATION_OUTPUT_PATH = PROCESSED_DATA_DIR / "player_baseline_evaluation.csv"


def main() -> None:
    player_modeling_df = pd.read_csv(PLAYER_MODELING_DATASET_PATH)

    results, _ = train_and_evaluate_baseline_models(player_modeling_df)

    evaluation_df = pd.DataFrame(
        [
            {
                "model": result.model_name,
                "mae": result.mae,
                "r2": result.r2,
            }
            for result in results
        ]
    )

    evaluation_df.to_csv(EVALUATION_OUTPUT_PATH, index=False)

    print("Entrenamiento baseline completado.")
    print("")
    print(evaluation_df)
    print("")
    print(f"Resultados guardados en: {EVALUATION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()