import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_modeling import (
    TARGET_COLUMN,
    build_random_forest_temporal_regressor,
    get_temporal_feature_set,
)


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TEMPORAL_DATASET_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_modeling_dataset.csv"
)

MODEL_OUTPUT_PATH = (
    MODELS_DIR / "random_forest_temporal_without_previous_xg.joblib"
)

METADATA_OUTPUT_PATH = (
    MODELS_DIR / "random_forest_temporal_without_previous_xg_metadata.json"
)


FEATURE_SET_NAME = "without_previous_xg"


def train_final_temporal_model(
    temporal_df: pd.DataFrame,
    feature_set_name: str = FEATURE_SET_NAME,
):
    """
    Entrena el modelo temporal final usando todos los pares temporales
    disponibles.

    Este modelo se guarda posteriormente para reutilizarlo en predicciones
    sobre nuevos jugadores o nuevas temporadas.
    """
    numeric_features, categorical_features = get_temporal_feature_set(
        feature_set_name
    )

    feature_columns = numeric_features + categorical_features

    X = temporal_df[feature_columns]
    y = temporal_df[TARGET_COLUMN]

    model = build_random_forest_temporal_regressor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=42,
    )

    model.fit(X, y)

    return model, numeric_features, categorical_features


def build_model_metadata(
    temporal_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    feature_set_name: str = FEATURE_SET_NAME,
) -> dict:
    """
    Construye metadatos del modelo entrenado.

    Estos metadatos documentan qué variable se predice, qué variables se han
    usado y con qué datos se entrenó el modelo.
    """
    metadata = {
        "model_name": "RandomForestRegressor temporal",
        "feature_set": feature_set_name,
        "target_column": TARGET_COLUMN,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "training_rows": int(len(temporal_df)),
        "seasons_used": sorted(temporal_df["season"].unique().tolist()),
        "next_seasons_used": sorted(temporal_df["next_season"].unique().tolist()),
        "sklearn_version": sklearn.__version__,
        "model_file": MODEL_OUTPUT_PATH.name,
        "notes": (
            "Modelo temporal entrenado para predecir xG_90 de la temporada "
            "siguiente sin utilizar xG_90 ni npxG_90 de la temporada anterior "
            "como predictores directos."
        ),
    }

    return metadata


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    temporal_df = pd.read_csv(TEMPORAL_DATASET_PATH)

    model, numeric_features, categorical_features = train_final_temporal_model(
        temporal_df=temporal_df,
        feature_set_name=FEATURE_SET_NAME,
    )

    metadata = build_model_metadata(
        temporal_df=temporal_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_set_name=FEATURE_SET_NAME,
    )

    joblib.dump(model, MODEL_OUTPUT_PATH)

    METADATA_OUTPUT_PATH.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Modelo temporal final entrenado y guardado correctamente.")
    print("")
    print(f"Filas usadas para entrenamiento final: {len(temporal_df)}")
    print(f"Modelo guardado en: {MODEL_OUTPUT_PATH}")
    print(f"Metadatos guardados en: {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()