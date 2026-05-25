import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_features import simplify_position
from football_predictor.player_temporal_modeling import get_temporal_feature_set


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

LALIGA_LATEST_PATH = PROCESSED_DATA_DIR / "laliga_players_latest_clean.csv"

MODEL_PATH = (
    MODELS_DIR / "random_forest_temporal_without_previous_xg.joblib"
)

ALL_PREDICTIONS_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_player_predictions.csv"
)

TOP_ATTACKING_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_top_25_f_m_predictions.csv"
)


FEATURE_SET_NAME = "without_previous_xg"


def prepare_laliga_latest_for_prediction(
    df: pd.DataFrame,
    feature_set_name: str = FEATURE_SET_NAME,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepara el dataset latest de LaLiga para aplicar el modelo temporal.

    El modelo entrenado espera las mismas variables utilizadas durante el
    entrenamiento temporal. Por tanto, aquí se generan las variables derivadas
    necesarias y se eliminan los porteros.
    """
    prediction_df = df.copy()

    prediction_df["position_main"] = prediction_df["position"].apply(
        simplify_position
    )

    rows_before_goalkeeper_filter = len(prediction_df)

    prediction_df = prediction_df[
        prediction_df["position_main"] != "GK"
    ].copy()

    removed_goalkeepers = rows_before_goalkeeper_filter - len(prediction_df)

    print(f"Porteros eliminados antes de predecir: {removed_goalkeepers}")

    prediction_df["minutes_per_game"] = (
        prediction_df["time"] / prediction_df["games"]
    )

    prediction_df["high_participation"] = (
        prediction_df["minutes_per_game"] >= 60
    ).astype(int)

    numeric_features, categorical_features = get_temporal_feature_set(
        feature_set_name
    )

    feature_columns = numeric_features + categorical_features

    missing_columns = [
        column for column in feature_columns if column not in prediction_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para generar predicciones: "
            f"{missing_columns}"
        )

    return prediction_df, feature_columns


def build_laliga_current_predictions(
    laliga_latest_df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Genera predicciones de rendimiento futuro para jugadores actuales de LaLiga.
    """
    prediction_df, feature_columns = prepare_laliga_latest_for_prediction(
        laliga_latest_df,
        feature_set_name=FEATURE_SET_NAME,
    )

    X = prediction_df[feature_columns]

    predictions = model.predict(X)

    prediction_df["predicted_next_xG_90"] = predictions

    output_columns = [
        "id",
        "player_name",
        "position",
        "position_main",
        "team_title",
        "league",
        "season",
        "jornada",
        "fecha_descarga",
        "games",
        "time",
        "minutes_per_game",
        "goals_90",
        "assists_90",
        "xA_90",
        "shots_90",
        "key_passes_90",
        "xGChain_90",
        "xGBuildup_90",
        "predicted_next_xG_90",
    ]

    prediction_df = prediction_df[output_columns].copy()

    prediction_df = prediction_df.sort_values(
        by="predicted_next_xG_90",
        ascending=False,
    ).reset_index(drop=True)

    return prediction_df


def build_top_attacking_predictions(
    predictions_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Selecciona los jugadores F/M con mayor predicción de xG_90 futuro.
    """
    attacking_df = predictions_df[
        predictions_df["position_main"].isin(["F", "M"])
    ].copy()

    attacking_df = attacking_df.sort_values(
        by="predicted_next_xG_90",
        ascending=False,
    ).head(top_n).reset_index(drop=True)

    return attacking_df


def main() -> None:
    laliga_latest_df = pd.read_csv(LALIGA_LATEST_PATH)

    model = joblib.load(MODEL_PATH)

    predictions_df = build_laliga_current_predictions(
        laliga_latest_df=laliga_latest_df,
        model=model,
    )

    top_attacking_df = build_top_attacking_predictions(
        predictions_df,
        top_n=25,
    )

    predictions_df.to_csv(
        ALL_PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    top_attacking_df.to_csv(
        TOP_ATTACKING_OUTPUT_PATH,
        index=False,
    )

    print("Predicciones actuales de LaLiga generadas correctamente.")
    print("")
    print(f"Jugadores predichos: {predictions_df.shape[0]}")
    print(f"Archivo completo: {ALL_PREDICTIONS_OUTPUT_PATH}")
    print("")
    print("Top 25 F/M por predicted_next_xG_90:")
    print(top_attacking_df)
    print("")
    print(f"Archivo Top 25 F/M: {TOP_ATTACKING_OUTPUT_PATH}")


if __name__ == "__main__":
    main()