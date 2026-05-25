import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_modeling import (
    build_temporal_test_predictions,
)


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TEMPORAL_DATASET_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_modeling_dataset.csv"
)

GENERAL_PREDICTIONS_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_predictions_without_previous_xg.csv"
)

ATTACKING_PREDICTIONS_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_predictions_f_m_sample_15.csv"
)


def select_representative_prediction_sample(
    predictions_df: pd.DataFrame,
    sample_size: int = 15,
) -> pd.DataFrame:
    """
    Selecciona una muestra representativa de predicciones individuales.

    La muestra incluye:
    - jugadores con bajo error,
    - jugadores con error intermedio,
    - jugadores con alto error.
    """
    if len(predictions_df) <= sample_size:
        return predictions_df.copy()

    sorted_df = predictions_df.sort_values(
        by="absolute_error",
        ascending=True,
    ).reset_index(drop=True)

    n_best = sample_size // 3
    n_middle = sample_size // 3
    n_worst = sample_size - n_best - n_middle

    best_cases = sorted_df.head(n_best)

    middle_start = max((len(sorted_df) // 2) - (n_middle // 2), 0)
    middle_cases = sorted_df.iloc[
        middle_start: middle_start + n_middle
    ]

    worst_cases = sorted_df.tail(n_worst)

    sample_df = pd.concat(
        [
            best_cases,
            middle_cases,
            worst_cases,
        ],
        ignore_index=True,
    )

    sample_df = sample_df.drop_duplicates(
        subset=["id", "player_name", "season", "next_season"]
    )

    sample_df = sample_df.sort_values(
        by="absolute_error",
        ascending=True,
    ).reset_index(drop=True)

    return sample_df


def select_attacking_prediction_sample(
    predictions_df: pd.DataFrame,
    sample_size: int = 15,
) -> pd.DataFrame:
    """
    Selecciona una muestra representativa únicamente con jugadores de perfil
    ofensivo o creativo.

    Se conservan:
    - F: delanteros / atacantes.
    - M: centrocampistas.

    Esto permite analizar predicciones más relevantes para un objetivo basado
    en xG_90, ya que defensas y porteros tienden a tener valores ofensivos
    más bajos y menos interpretables.
    """
    attacking_df = predictions_df[
        predictions_df["position_main"].isin(["F", "M"])
    ].copy()

    return select_representative_prediction_sample(
        attacking_df,
        sample_size=sample_size,
    )


def main() -> None:
    temporal_df = pd.read_csv(TEMPORAL_DATASET_PATH)

    predictions_df = build_temporal_test_predictions(
        temporal_df,
        feature_set_name="without_previous_xg",
        random_state=42,
    )

    general_sample_df = select_representative_prediction_sample(
        predictions_df,
        sample_size=15,
    )

    attacking_sample_df = select_attacking_prediction_sample(
        predictions_df,
        sample_size=15,
    )

    general_sample_df.to_csv(
        GENERAL_PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    attacking_sample_df.to_csv(
        ATTACKING_PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    print("Predicciones individuales generadas correctamente.")
    print("")
    print(f"Predicciones calculadas en test: {predictions_df.shape[0]}")
    print("")
    print("Muestra general:")
    print(f"- Filas guardadas: {general_sample_df.shape[0]}")
    print(f"- Archivo generado: {GENERAL_PREDICTIONS_OUTPUT_PATH}")
    print("")
    print("Muestra F/M:")
    print(f"- Filas guardadas: {attacking_sample_df.shape[0]}")
    print(f"- Archivo generado: {ATTACKING_PREDICTIONS_OUTPUT_PATH}")
    print("")
    print("Vista previa muestra F/M:")
    print(attacking_sample_df)


if __name__ == "__main__":
    main()