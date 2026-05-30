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

from football_predictor.player_temporal_features import add_temporal_base_features  # noqa: E402


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

HISTORICAL_PLAYERS_PATH = (
    PROCESSED_DATA_DIR / "historical_players_clean.csv"
)

METADATA_OUTPUT_PATH = (
    MODELS_DIR / "ridge_temporal_multi_metric_metadata.json"
)

# Archivo mantenido por compatibilidad con versiones anteriores del proyecto.
LEGACY_XG_MODEL_OUTPUT_PATH = (
    MODELS_DIR / "ridge_temporal_without_previous_xg.joblib"
)

LEGACY_XG_METADATA_OUTPUT_PATH = (
    MODELS_DIR / "ridge_temporal_without_previous_xg_metadata.json"
)


MODEL_NAME = "Ridge"
MODEL_TYPE = "linear_regularized_regression"

CATEGORICAL_FEATURES = [
    "position_main",
    "league",
]

BASE_NUMERIC_FEATURES = [
    "games",
    "time",
    "minutes_per_game",
    "high_participation",
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

METRIC_CONFIG = {
    "xG_90": {
        "label": "Goles esperados",
        "target_column": "target_next_xG_90",
        "model_filename": "ridge_temporal_xG_90.joblib",
        "excluded_input_features": ["xG_90", "npxG_90"],
        "notes": (
            "Modelo Ridge para estimar goles esperados por 90 minutos en la "
            "temporada siguiente. Se excluyen xG_90 y npxG_90 como predictores "
            "directos para evitar dependencia inmediata de la misma familia de "
            "métricas."
        ),
    },
    "goals_90": {
        "label": "Goles",
        "target_column": "target_next_goals_90",
        "model_filename": "ridge_temporal_goals_90.joblib",
        "excluded_input_features": ["goals_90", "npg_90"],
        "notes": (
            "Modelo Ridge para estimar goles reales por 90 minutos en la "
            "temporada siguiente. Se excluyen goals_90 y npg_90 como señales "
            "directamente equivalentes de producción goleadora previa."
        ),
    },
    "assists_90": {
        "label": "Asistencias",
        "target_column": "target_next_assists_90",
        "model_filename": "ridge_temporal_assists_90.joblib",
        "excluded_input_features": ["assists_90"],
        "notes": (
            "Modelo Ridge para estimar asistencias reales por 90 minutos en la "
            "temporada siguiente. Las asistencias son una métrica más ruidosa, "
            "ya que dependen también de la finalización de los compañeros."
        ),
    },
    "xA_90": {
        "label": "Asistencias esperadas",
        "target_column": "target_next_xA_90",
        "model_filename": "ridge_temporal_xA_90.joblib",
        "excluded_input_features": ["xA_90"],
        "notes": (
            "Modelo Ridge para estimar asistencias esperadas por 90 minutos en "
            "la temporada siguiente. Se excluye xA_90 previo como predictor "
            "directo de la misma métrica objetivo."
        ),
    },
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """
    Comprueba que un DataFrame contiene las columnas necesarias.
    """
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en {dataset_name}: {missing_columns}"
        )


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


def get_numeric_features_for_metric(metric_key: str) -> list[str]:
    """
    Devuelve las variables numéricas de entrada para una métrica concreta.
    """
    excluded_features = METRIC_CONFIG[metric_key]["excluded_input_features"]

    return [
        feature for feature in BASE_NUMERIC_FEATURES
        if feature not in excluded_features
    ]


def build_ridge_temporal_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Construye el pipeline Ridge final.

    Ridge es un modelo lineal regularizado sensible a la escala de las
    variables, por lo que las variables numéricas se estandarizan. Las variables
    categóricas se codifican mediante one-hot encoding.
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset temporal multi-métrica
# ---------------------------------------------------------------------------


def build_multi_metric_temporal_dataset(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye un dataset temporal con varios targets futuros.

    Cada fila representa una observación jugador-temporada en t y contiene como
    objetivos las métricas observadas en t+1:
    - target_next_xG_90
    - target_next_goals_90
    - target_next_assists_90
    - target_next_xA_90
    """
    required_columns = [
        "id",
        "player_name",
        "position",
        "team_title",
        "league",
        "season",
        "games",
        "time",
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

    validate_columns(
        df=historical_df,
        required_columns=required_columns,
        dataset_name="historical_players_clean.csv",
    )

    base_df = add_temporal_base_features(historical_df)

    rows_before_gk_filter = len(base_df)
    base_df = base_df[base_df["position_main"] != "GK"].copy()
    removed_goalkeepers = rows_before_gk_filter - len(base_df)

    print(
        "Porteros eliminados del dataset temporal multi-métrica: "
        f"{removed_goalkeepers}"
    )

    next_target_df = base_df[
        [
            "id",
            "season",
            "team_title",
            "league",
            "games",
            "time",
            "xG_90",
            "goals_90",
            "assists_90",
            "xA_90",
        ]
    ].copy()

    next_target_df = next_target_df.rename(
        columns={
            "season": "next_season",
            "team_title": "next_team_title",
            "league": "next_league",
            "games": "next_games",
            "time": "next_time",
            "xG_90": "target_next_xG_90",
            "goals_90": "target_next_goals_90",
            "assists_90": "target_next_assists_90",
            "xA_90": "target_next_xA_90",
        }
    )

    temporal_df = base_df.merge(
        next_target_df,
        how="inner",
        on=["id", "next_season"],
    )

    target_columns = [
        config["target_column"]
        for config in METRIC_CONFIG.values()
    ]

    temporal_df = temporal_df.dropna(
        subset=target_columns,
    ).reset_index(drop=True)

    return temporal_df


# ---------------------------------------------------------------------------
# Entrenamiento y guardado
# ---------------------------------------------------------------------------


def train_metric_model(
    temporal_df: pd.DataFrame,
    metric_key: str,
) -> tuple[Pipeline, dict]:
    """
    Entrena el modelo Ridge final para una métrica concreta.
    """
    config = METRIC_CONFIG[metric_key]

    numeric_features = get_numeric_features_for_metric(metric_key)
    categorical_features = CATEGORICAL_FEATURES
    feature_columns = numeric_features + categorical_features

    validate_columns(
        df=temporal_df,
        required_columns=feature_columns + [config["target_column"]],
        dataset_name=f"dataset temporal para {metric_key}",
    )

    train_df = temporal_df.dropna(
        subset=feature_columns + [config["target_column"]],
    ).copy()

    X = train_df[feature_columns]
    y = train_df[config["target_column"]]

    model = build_ridge_temporal_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model.fit(X, y)

    model_metadata = {
        "metric_key": metric_key,
        "metric_label": config["label"],
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "target_column": config["target_column"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_input_features": config["excluded_input_features"],
        "training_rows": int(len(train_df)),
        "model_file": config["model_filename"],
        "notes": config["notes"],
    }

    return model, model_metadata


def build_global_metadata(
    temporal_df: pd.DataFrame,
    metric_metadata: list[dict],
) -> dict:
    """
    Construye metadatos globales de los modelos Ridge finales multi-métrica.
    """
    metadata = {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "approach": "one_independent_ridge_model_per_metric",
        "metrics": metric_metadata,
        "training_rows_total": int(len(temporal_df)),
        "seasons_used": sorted(temporal_df["season"].unique().tolist()),
        "next_seasons_used": sorted(temporal_df["next_season"].unique().tolist()),
        "sklearn_version": sklearn.__version__,
        "selection_reason": (
            "Ridge se mantiene como modelo final porque obtuvo el mejor "
            "rendimiento externo para xG_90 en el conjunto global y F/M. "
            "La versión multi-métrica conserva el mismo algoritmo seleccionado "
            "y entrena un modelo independiente por cada métrica ofensiva."
        ),
        "notes": (
            "Todos los objetivos se expresan por 90 minutos. Para cada métrica "
            "se excluye de las variables de entrada la versión previa directa "
            "de la métrica objetivo, con el fin de evitar una dependencia "
            "excesiva del mismo valor histórico."
        ),
    }

    return metadata


def build_legacy_xg_metadata(
    temporal_df: pd.DataFrame,
    xg_metadata: dict,
) -> dict:
    """
    Construye el archivo de metadatos histórico para xG_90, manteniendo
    compatibilidad con la versión anterior del proyecto.
    """
    metadata = {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "feature_set": "without_previous_xg",
        "target_column": xg_metadata["target_column"],
        "numeric_features": xg_metadata["numeric_features"],
        "categorical_features": xg_metadata["categorical_features"],
        "training_rows": xg_metadata["training_rows"],
        "seasons_used": sorted(temporal_df["season"].unique().tolist()),
        "next_seasons_used": sorted(temporal_df["next_season"].unique().tolist()),
        "sklearn_version": sklearn.__version__,
        "model_file": LEGACY_XG_MODEL_OUTPUT_PATH.name,
        "selection_reason": (
            "Ridge se selecciona como modelo final para xG_90 porque obtuvo "
            "el mejor rendimiento en la comparación externa 2025-2026, con "
            "menor MAE y mayor R² que RandomForestRegressor en el conjunto "
            "global y en el subconjunto F/M."
        ),
        "notes": (
            "Archivo mantenido por compatibilidad con la versión previa del "
            "proyecto. El modelo equivalente también se guarda como "
            "ridge_temporal_xG_90.joblib dentro de la versión multi-métrica."
        ),
    }

    return metadata


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    historical_df = pd.read_csv(HISTORICAL_PLAYERS_PATH)

    temporal_df = build_multi_metric_temporal_dataset(historical_df)

    metric_metadata = []
    trained_models = {}

    for metric_key in METRIC_CONFIG:
        print(f"Entrenando modelo final Ridge para: {metric_key}")

        model, metadata = train_metric_model(
            temporal_df=temporal_df,
            metric_key=metric_key,
        )

        model_output_path = MODELS_DIR / METRIC_CONFIG[metric_key]["model_filename"]

        joblib.dump(model, model_output_path)

        print(f"Modelo guardado en: {model_output_path}")

        metric_metadata.append(metadata)
        trained_models[metric_key] = model

    # Compatibilidad con el nombre anterior del modelo final de xG_90.
    joblib.dump(
        trained_models["xG_90"],
        LEGACY_XG_MODEL_OUTPUT_PATH,
    )

    xg_metadata = next(
        metadata for metadata in metric_metadata
        if metadata["metric_key"] == "xG_90"
    )

    legacy_metadata = build_legacy_xg_metadata(
        temporal_df=temporal_df,
        xg_metadata=xg_metadata,
    )

    LEGACY_XG_METADATA_OUTPUT_PATH.write_text(
        json.dumps(legacy_metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    global_metadata = build_global_metadata(
        temporal_df=temporal_df,
        metric_metadata=metric_metadata,
    )

    METADATA_OUTPUT_PATH.write_text(
        json.dumps(global_metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print("Modelos Ridge finales multi-métrica entrenados y guardados correctamente.")
    print(f"Filas temporales usadas: {len(temporal_df)}")
    print(f"Metadatos multi-métrica guardados en: {METADATA_OUTPUT_PATH}")
    print(f"Modelo legacy xG_90 guardado en: {LEGACY_XG_MODEL_OUTPUT_PATH}")
    print(f"Metadatos legacy xG_90 guardados en: {LEGACY_XG_METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()