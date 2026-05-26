import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_modeling import (  # noqa: E402
    TARGET_COLUMN,
    get_temporal_feature_set,
)


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TEMPORAL_DATASET_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_modeling_dataset.csv"
)

MODEL_OUTPUT_PATH = (
    MODELS_DIR / "ridge_temporal_without_previous_xg.joblib"
)

METADATA_OUTPUT_PATH = (
    MODELS_DIR / "ridge_temporal_without_previous_xg_metadata.json"
)


FEATURE_SET_NAME = "without_previous_xg"
MODEL_NAME = "Ridge"


def build_one_hot_encoder() -> OneHotEncoder:
    """
    Construye un OneHotEncoder compatible con distintas versiones de
    scikit-learn.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def build_ridge_temporal_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Construye el pipeline final con preprocesamiento y Ridge.

    Las variables categóricas se codifican mediante one-hot encoding y las
    variables numéricas se estandarizan, ya que Ridge es un modelo lineal
    regularizado sensible a la escala de las variables.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                build_one_hot_encoder(),
                categorical_features,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    model = Ridge(alpha=1.0)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def validate_training_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Comprueba que el dataset temporal contiene todas las columnas necesarias.
    """
    required_columns = feature_columns + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para entrenar el modelo final: "
            f"{missing_columns}"
        )


def train_final_ridge_model(
    temporal_df: pd.DataFrame,
    feature_set_name: str = FEATURE_SET_NAME,
) -> tuple[Pipeline, list[str], list[str]]:
    """
    Entrena el modelo Ridge final usando todos los pares temporales disponibles.

    La evaluación externa previa mostró que Ridge ofrece mejor rendimiento
    global que Random Forest en la predicción de xG_90 2025-2026.
    """
    numeric_features, categorical_features = get_temporal_feature_set(
        feature_set_name
    )

    feature_columns = numeric_features + categorical_features

    validate_training_columns(
        df=temporal_df,
        feature_columns=feature_columns,
    )

    X = temporal_df[feature_columns]
    y = temporal_df[TARGET_COLUMN]

    model = build_ridge_temporal_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
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
    Construye metadatos del modelo final entrenado.
    """
    metadata = {
        "model_name": MODEL_NAME,
        "model_type": "linear_regularized_regression",
        "feature_set": feature_set_name,
        "target_column": TARGET_COLUMN,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "training_rows": int(len(temporal_df)),
        "seasons_used": sorted(temporal_df["season"].unique().tolist()),
        "next_seasons_used": sorted(temporal_df["next_season"].unique().tolist()),
        "sklearn_version": sklearn.__version__,
        "model_file": MODEL_OUTPUT_PATH.name,
        "selection_reason": (
            "Ridge se selecciona como modelo final porque obtuvo el mejor "
            "rendimiento en la comparación externa 2025-2026, con menor MAE "
            "y mayor R² que RandomForestRegressor en el conjunto global y en "
            "el subconjunto F/M."
        ),
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

    model, numeric_features, categorical_features = train_final_ridge_model(
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

    print("Modelo Ridge temporal final entrenado y guardado correctamente.")
    print("")
    print(f"Filas usadas para entrenamiento final: {len(temporal_df)}")
    print(f"Modelo guardado en: {MODEL_OUTPUT_PATH}")
    print(f"Metadatos guardados en: {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()