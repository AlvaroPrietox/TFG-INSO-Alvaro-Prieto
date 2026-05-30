import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_features import add_temporal_base_features  # noqa: E402
from football_predictor.text_normalization import normalize_player_key  # noqa: E402


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

HISTORICAL_PLAYERS_PATH = (
    PROCESSED_DATA_DIR / "historical_players_clean.csv"
)

TARGET_2025_2026_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_target_clean.csv"
)

# Salidas ya existentes para comparación de modelos sobre xG_90.
COMPARISON_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison.csv"
)

COMPARISON_FM_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_f_m.csv"
)

PREDICTIONS_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_predictions.csv"
)

RANGE_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_by_xg_range.csv"
)

# Nuevas salidas multi-métrica para el modelo final Ridge.
MULTI_METRIC_WIDE_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_predictions.csv"
)

MULTI_METRIC_LONG_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_predictions_long.csv"
)

MULTI_METRIC_COMPARISON_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_comparison.csv"
)

MULTI_METRIC_COMPARISON_FM_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_comparison_f_m.csv"
)

REPORT_OUTPUT_PATH = (
    REPORTS_DIR / "20_external_2025_2026_model_comparison.md"
)

MAE_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "external_2025_2026_model_comparison_mae.png"
)

R2_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "external_2025_2026_model_comparison_r2.png"
)

FM_MAE_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "external_2025_2026_model_comparison_f_m_mae.png"
)

MULTI_METRIC_MAE_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "external_2025_2026_multi_metric_mae.png"
)

MULTI_METRIC_R2_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "external_2025_2026_multi_metric_r2.png"
)


MODEL_NAME = "Ridge"
INPUT_SEASON = "2024-2025"
TARGET_SEASON = "2025-2026"

XG_METRIC_KEY = "xG_90"
XG_RANGE_LABELS = [
    "bajo_<0.10",
    "medio_bajo_0.10_0.25",
    "medio_alto_0.25_0.50",
    "alto_>=0.50",
]

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
        "predicted_column": "predicted_2025_2026_xG_90",
        "actual_column": "actual_2025_2026_xG_90",
        "signed_error_column": "signed_error_xG_90",
        "absolute_error_column": "absolute_error_xG_90",
        "excluded_input_features": ["xG_90", "npxG_90"],
    },
    "goals_90": {
        "label": "Goles",
        "target_column": "target_next_goals_90",
        "predicted_column": "predicted_2025_2026_goals_90",
        "actual_column": "actual_2025_2026_goals_90",
        "signed_error_column": "signed_error_goals_90",
        "absolute_error_column": "absolute_error_goals_90",
        "excluded_input_features": ["goals_90", "npg_90"],
    },
    "assists_90": {
        "label": "Asistencias",
        "target_column": "target_next_assists_90",
        "predicted_column": "predicted_2025_2026_assists_90",
        "actual_column": "actual_2025_2026_assists_90",
        "signed_error_column": "signed_error_assists_90",
        "absolute_error_column": "absolute_error_assists_90",
        "excluded_input_features": ["assists_90"],
    },
    "xA_90": {
        "label": "Asistencias esperadas",
        "target_column": "target_next_xA_90",
        "predicted_column": "predicted_2025_2026_xA_90",
        "actual_column": "actual_2025_2026_xA_90",
        "signed_error_column": "signed_error_xA_90",
        "absolute_error_column": "absolute_error_xA_90",
        "excluded_input_features": ["xA_90"],
    },
}


# ---------------------------------------------------------------------------
# Utilidades generales
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
    Construye un OneHotEncoder compatible con versiones recientes y antiguas de
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
    Devuelve las variables numéricas para una métrica concreta.

    Para evitar que el modelo dependa de forma directa de la misma métrica en
    la temporada anterior, se excluye el predictor más cercano al objetivo.
    """
    excluded_features = METRIC_CONFIG[metric_key]["excluded_input_features"]

    return [
        feature for feature in BASE_NUMERIC_FEATURES
        if feature not in excluded_features
    ]


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """
    Construye el preprocesador común.

    Ridge utiliza escalado numérico. Los modelos basados en árboles no lo
    necesitan.
    """
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                build_one_hot_encoder(),
                categorical_features,
            ),
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
        ],
        remainder="drop",
    )


def build_ridge_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Construye el modelo Ridge final para una métrica concreta.
    """
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    scale_numeric=True,
                ),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def build_candidate_models_for_xg(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    """
    Construye los modelos candidatos de la comparación externa sobre xG_90.
    """
    tree_preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=False,
    )

    linear_preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=True,
    )

    return {
        "DummyRegressor": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoostingRegressor": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "HistGradientBoostingRegressor": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def compute_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """
    Calcula MAE y R².
    """
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


# ---------------------------------------------------------------------------
# Preparación de datasets
# ---------------------------------------------------------------------------


def build_multi_metric_temporal_dataset(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye un dataset temporal con varios targets futuros.

    Cada fila representa la temporada t de un jugador y contiene como variables
    objetivo sus métricas observadas en la temporada t+1.
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


def prepare_2024_2025_players_for_prediction(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara jugadores de 2024-2025 como entrada para predecir 2025-2026.
    """
    input_df = historical_df[
        historical_df["season"] == INPUT_SEASON
    ].copy()

    input_df = add_temporal_base_features(input_df)

    rows_before_gk_filter = len(input_df)
    input_df = input_df[input_df["position_main"] != "GK"].copy()
    removed_goalkeepers = rows_before_gk_filter - len(input_df)

    print(f"Porteros eliminados de {INPUT_SEASON}: {removed_goalkeepers}")

    input_df["player_key"] = input_df["player_name"].apply(
        normalize_player_key
    )

    input_df = input_df.sort_values(
        by="time",
        ascending=False,
    ).drop_duplicates(
        subset=["player_key"],
        keep="first",
    )

    return input_df.reset_index(drop=True)


def prepare_2025_2026_target(
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara los valores reales 2025-2026 para las métricas evaluadas.
    """
    required_columns = [
        "player_name",
        "team_title",
        "games",
        "time",
        "xG_90",
        "goals_90",
        "assists_90",
        "xA_90",
    ]

    validate_columns(
        df=target_df,
        required_columns=required_columns,
        dataset_name="laliga_2025_2026_target_clean.csv",
    )

    target_clean = target_df.copy()

    target_clean["player_key"] = target_clean["player_name"].apply(
        normalize_player_key
    )

    target_clean = target_clean.sort_values(
        by="time",
        ascending=False,
    ).drop_duplicates(
        subset=["player_key"],
        keep="first",
    )

    target_clean = target_clean.rename(
        columns={
            "player_name": "target_player_name",
            "team_title": "target_team_title",
            "games": "target_games",
            "time": "target_time",
            "xG_90": "actual_2025_2026_xG_90",
            "goals_90": "actual_2025_2026_goals_90",
            "assists_90": "actual_2025_2026_assists_90",
            "xA_90": "actual_2025_2026_xA_90",
        }
    )

    selected_columns = [
        "player_key",
        "target_player_name",
        "target_team_title",
        "target_games",
        "target_time",
        "actual_2025_2026_xG_90",
        "actual_2025_2026_goals_90",
        "actual_2025_2026_assists_90",
        "actual_2025_2026_xA_90",
    ]

    return target_clean[selected_columns].copy()


def build_external_base_dataset(
    input_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une jugadores 2024-2025 con valores reales 2025-2026.
    """
    merged_df = input_df.merge(
        target_df,
        how="inner",
        on="player_key",
    )

    return merged_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Comparación externa de modelos para xG_90
# ---------------------------------------------------------------------------


def assign_xg_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna rangos según el xG_90 real 2025-2026.
    """
    ranged_df = df.copy()

    ranged_df["xg_range"] = pd.cut(
        ranged_df["actual_2025_2026_xG_90"],
        bins=[-float("inf"), 0.10, 0.25, 0.50, float("inf")],
        labels=XG_RANGE_LABELS,
        right=False,
    )

    return ranged_df


def compare_models_for_xg(
    temporal_df: pd.DataFrame,
    external_base_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compara varios modelos únicamente para xG_90.

    Esta comparación mantiene la justificación de selección de Ridge como
    modelo final principal.
    """
    config = METRIC_CONFIG[XG_METRIC_KEY]

    numeric_features = get_numeric_features_for_metric(XG_METRIC_KEY)
    categorical_features = CATEGORICAL_FEATURES
    feature_columns = numeric_features + categorical_features

    validate_columns(
        df=temporal_df,
        required_columns=feature_columns + [config["target_column"]],
        dataset_name="dataset temporal xG_90",
    )

    validate_columns(
        df=external_base_df,
        required_columns=feature_columns + [config["actual_column"]],
        dataset_name="dataset externo xG_90",
    )

    train_df = temporal_df.dropna(
        subset=feature_columns + [config["target_column"]],
    ).copy()

    X_train = train_df[feature_columns]
    y_train = train_df[config["target_column"]]

    models = build_candidate_models_for_xg(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    comparison_rows = []
    fm_rows = []
    prediction_frames = []
    range_rows = []

    for model_name, model in models.items():
        print(f"Entrenando y evaluando modelo externo xG_90: {model_name}")

        model.fit(X_train, y_train)

        model_df = external_base_df.copy()

        model_df[config["predicted_column"]] = model.predict(
            model_df[feature_columns]
        )

        model_df["signed_error"] = (
            model_df[config["predicted_column"]]
            - model_df[config["actual_column"]]
        )

        model_df["absolute_error"] = model_df["signed_error"].abs()
        model_df["model"] = model_name
        model_df["feature_set"] = "without_previous_xg"
        model_df["input_season"] = INPUT_SEASON
        model_df["target_season"] = TARGET_SEASON

        metrics = compute_metrics(
            actual=model_df[config["actual_column"]],
            predicted=model_df[config["predicted_column"]],
        )

        comparison_rows.append(
            {
                "model": model_name,
                "feature_set": "without_previous_xg",
                "training_rows": int(len(X_train)),
                "matched_players": int(len(model_df)),
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "input_season": INPUT_SEASON,
                "target_season": TARGET_SEASON,
            }
        )

        fm_df = model_df[model_df["position_main"].isin(["F", "M"])].copy()

        fm_metrics = compute_metrics(
            actual=fm_df[config["actual_column"]],
            predicted=fm_df[config["predicted_column"]],
        )

        fm_rows.append(
            {
                "model": model_name,
                "feature_set": "without_previous_xg",
                "training_rows": int(len(X_train)),
                "matched_players_f_m": int(len(fm_df)),
                "mae": fm_metrics["mae"],
                "r2": fm_metrics["r2"],
                "input_season": INPUT_SEASON,
                "target_season": TARGET_SEASON,
            }
        )

        ranged_df = assign_xg_ranges(model_df)

        for xg_range in XG_RANGE_LABELS:
            group_df = ranged_df[ranged_df["xg_range"] == xg_range].copy()

            if group_df.empty:
                continue

            group_metrics = compute_metrics(
                actual=group_df[config["actual_column"]],
                predicted=group_df[config["predicted_column"]],
            )

            range_rows.append(
                {
                    "model": model_name,
                    "xg_range": xg_range,
                    "n_players": int(len(group_df)),
                    "actual_xg_90_mean": float(
                        group_df[config["actual_column"]].mean()
                    ),
                    "predicted_xg_90_mean": float(
                        group_df[config["predicted_column"]].mean()
                    ),
                    "mae": group_metrics["mae"],
                    "r2": group_metrics["r2"],
                    "mean_signed_error": float(group_df["signed_error"].mean()),
                    "overestimations": int((group_df["signed_error"] > 0).sum()),
                    "underestimations": int((group_df["signed_error"] < 0).sum()),
                }
            )

        output_columns = [
            "model",
            "feature_set",
            "player_key",
            "id",
            "player_name",
            "position",
            "position_main",
            "team_title",
            "league",
            "season",
            "games",
            "time",
            "minutes_per_game",
            "target_player_name",
            "target_team_title",
            "target_games",
            "target_time",
            config["actual_column"],
            config["predicted_column"],
            "signed_error",
            "absolute_error",
        ]

        prediction_frames.append(model_df[output_columns].copy())

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by="mae",
        ascending=True,
    ).reset_index(drop=True)

    fm_comparison_df = pd.DataFrame(fm_rows).sort_values(
        by="mae",
        ascending=True,
    ).reset_index(drop=True)

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    range_comparison_df = pd.DataFrame(range_rows).sort_values(
        by=["xg_range", "mae"],
        ascending=True,
    ).reset_index(drop=True)

    return comparison_df, fm_comparison_df, predictions_df, range_comparison_df


# ---------------------------------------------------------------------------
# Evaluación multi-métrica con Ridge
# ---------------------------------------------------------------------------


def evaluate_metric_with_ridge(
    metric_key: str,
    temporal_df: pd.DataFrame,
    external_base_df: pd.DataFrame,
) -> tuple[pd.Series, dict]:
    """
    Entrena Ridge para una métrica y devuelve sus predicciones externas.
    """
    config = METRIC_CONFIG[metric_key]

    numeric_features = get_numeric_features_for_metric(metric_key)
    categorical_features = CATEGORICAL_FEATURES
    feature_columns = numeric_features + categorical_features

    validate_columns(
        df=temporal_df,
        required_columns=feature_columns + [config["target_column"]],
        dataset_name=f"dataset temporal {metric_key}",
    )

    validate_columns(
        df=external_base_df,
        required_columns=feature_columns + [config["actual_column"]],
        dataset_name=f"dataset externo {metric_key}",
    )

    train_df = temporal_df.dropna(
        subset=feature_columns + [config["target_column"]],
    ).copy()

    model = build_ridge_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    X_train = train_df[feature_columns]
    y_train = train_df[config["target_column"]]

    model.fit(X_train, y_train)

    predictions = pd.Series(
        model.predict(external_base_df[feature_columns]),
        index=external_base_df.index,
        name=config["predicted_column"],
    )

    metadata = {
        "metric_key": metric_key,
        "metric_label": config["label"],
        "training_rows": int(len(train_df)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    return predictions, metadata


def evaluate_multi_metric_ridge(
    temporal_df: pd.DataFrame,
    external_base_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evalúa Ridge de forma independiente para cada métrica ofensiva.
    """
    wide_predictions_df = external_base_df.copy()
    comparison_rows = []
    comparison_fm_rows = []
    long_prediction_frames = []

    for metric_key, config in METRIC_CONFIG.items():
        print(f"Entrenando y evaluando Ridge multi-métrica: {metric_key}")

        predictions, metadata = evaluate_metric_with_ridge(
            metric_key=metric_key,
            temporal_df=temporal_df,
            external_base_df=external_base_df,
        )

        predicted_column = config["predicted_column"]
        actual_column = config["actual_column"]
        signed_error_column = config["signed_error_column"]
        absolute_error_column = config["absolute_error_column"]

        wide_predictions_df[predicted_column] = predictions
        wide_predictions_df[signed_error_column] = (
            wide_predictions_df[predicted_column]
            - wide_predictions_df[actual_column]
        )
        wide_predictions_df[absolute_error_column] = (
            wide_predictions_df[signed_error_column].abs()
        )

        metrics = compute_metrics(
            actual=wide_predictions_df[actual_column],
            predicted=wide_predictions_df[predicted_column],
        )

        comparison_rows.append(
            {
                "metric_key": metric_key,
                "metric_label": config["label"],
                "model": MODEL_NAME,
                "training_rows": metadata["training_rows"],
                "matched_players": int(len(wide_predictions_df)),
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "input_season": INPUT_SEASON,
                "target_season": TARGET_SEASON,
            }
        )

        fm_df = wide_predictions_df[
            wide_predictions_df["position_main"].isin(["F", "M"])
        ].copy()

        fm_metrics = compute_metrics(
            actual=fm_df[actual_column],
            predicted=fm_df[predicted_column],
        )

        comparison_fm_rows.append(
            {
                "metric_key": metric_key,
                "metric_label": config["label"],
                "model": MODEL_NAME,
                "training_rows": metadata["training_rows"],
                "matched_players_f_m": int(len(fm_df)),
                "mae": fm_metrics["mae"],
                "r2": fm_metrics["r2"],
                "input_season": INPUT_SEASON,
                "target_season": TARGET_SEASON,
            }
        )

        long_df = wide_predictions_df[
            [
                "player_key",
                "id",
                "player_name",
                "position",
                "position_main",
                "team_title",
                "league",
                "season",
                "games",
                "time",
                "minutes_per_game",
                "target_player_name",
                "target_team_title",
                "target_games",
                "target_time",
                actual_column,
                predicted_column,
                signed_error_column,
                absolute_error_column,
            ]
        ].copy()

        long_df = long_df.rename(
            columns={
                actual_column: "actual_value",
                predicted_column: "predicted_value",
                signed_error_column: "signed_error",
                absolute_error_column: "absolute_error",
            }
        )

        long_df["metric_key"] = metric_key
        long_df["metric_label"] = config["label"]
        long_df["model"] = MODEL_NAME
        long_df["input_season"] = INPUT_SEASON
        long_df["target_season"] = TARGET_SEASON

        long_prediction_frames.append(long_df)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by="metric_key",
        ascending=True,
    ).reset_index(drop=True)

    comparison_fm_df = pd.DataFrame(comparison_fm_rows).sort_values(
        by="metric_key",
        ascending=True,
    ).reset_index(drop=True)

    long_predictions_df = pd.concat(
        long_prediction_frames,
        ignore_index=True,
    )

    return (
        wide_predictions_df,
        long_predictions_df,
        comparison_df,
        comparison_fm_df,
    )


# ---------------------------------------------------------------------------
# Figuras e informe
# ---------------------------------------------------------------------------


def plot_model_metric(
    comparison_df: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    title: str,
    x_label: str,
    ascending: bool,
) -> None:
    """
    Genera gráfico horizontal para comparar modelos.
    """
    plot_df = comparison_df.sort_values(
        by=metric_column,
        ascending=ascending,
    ).copy()

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["model"], plot_df[metric_column])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Modelo")

    for index, value in enumerate(plot_df[metric_column]):
        plt.text(
            value,
            index,
            f" {value:.4f}",
            va="center",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multi_metric_bar(
    comparison_df: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    """
    Genera gráfico de barras para comparar métricas ofensivas con Ridge.
    """
    plot_df = comparison_df.copy()

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["metric_label"], plot_df[metric_column])

    plt.title(title)
    plt.xlabel("Métrica ofensiva")
    plt.ylabel(y_label)
    plt.xticks(rotation=20, ha="right")

    for index, value in enumerate(plot_df[metric_column]):
        plt.text(
            index,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def round_report_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redondea métricas para informes Markdown.
    """
    report_df = df.copy()

    for column in [
        "mae",
        "r2",
        "mean_signed_error",
        "actual_xg_90_mean",
        "predicted_xg_90_mean",
    ]:
        if column in report_df.columns:
            report_df[column] = pd.to_numeric(
                report_df[column],
                errors="coerce",
            ).round(4)

    return report_df


def build_markdown_report(
    comparison_df: pd.DataFrame,
    fm_comparison_df: pd.DataFrame,
    range_comparison_df: pd.DataFrame,
    multi_metric_comparison_df: pd.DataFrame,
    multi_metric_fm_comparison_df: pd.DataFrame,
) -> str:
    """
    Construye el informe Markdown actualizado del script 20.
    """
    report_global_df = round_report_df(comparison_df)
    report_fm_df = round_report_df(fm_comparison_df)
    report_range_df = round_report_df(range_comparison_df)
    report_multi_metric_df = round_report_df(multi_metric_comparison_df)
    report_multi_metric_fm_df = round_report_df(multi_metric_fm_comparison_df)

    best_global_row = comparison_df.sort_values(
        by="mae",
        ascending=True,
    ).iloc[0]

    best_fm_row = fm_comparison_df.sort_values(
        by="mae",
        ascending=True,
    ).iloc[0]

    high_xg_df = range_comparison_df[
        range_comparison_df["xg_range"] == "alto_>=0.50"
    ].copy()

    best_high_xg_row = high_xg_df.sort_values(
        by="mae",
        ascending=True,
    ).iloc[0]

    lines = []

    lines.append("# Comparación externa de modelos y evaluación multi-métrica")
    lines.append("")

    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este experimento evalúa el rendimiento del sistema sobre una prueba "
        "externa 2025-2026. En primer lugar se comparan distintos modelos para "
        "predecir `xG_90`. En segundo lugar, una vez seleccionado Ridge como "
        "modelo final, se amplía la evaluación a varias métricas ofensivas "
        "normalizadas por 90 minutos."
    )
    lines.append("")

    lines.append("## Comparación externa de modelos para xG_90")
    lines.append("")
    lines.append(report_global_df.to_markdown(index=False))
    lines.append("")
    lines.append(
        f"El menor MAE global para `xG_90` lo obtiene "
        f"`{best_global_row['model']}`, con MAE = "
        f"{best_global_row['mae']:.4f} y R² = "
        f"{best_global_row['r2']:.4f}."
    )
    lines.append("")

    lines.append("## Comparación externa F/M para xG_90")
    lines.append("")
    lines.append(report_fm_df.to_markdown(index=False))
    lines.append("")
    lines.append(
        f"En el subconjunto F/M, el menor MAE para `xG_90` lo obtiene "
        f"`{best_fm_row['model']}`, con MAE = "
        f"{best_fm_row['mae']:.4f} y R² = "
        f"{best_fm_row['r2']:.4f}."
    )
    lines.append("")

    lines.append("## Análisis del rango alto de xG_90")
    lines.append("")
    lines.append(report_range_df.to_markdown(index=False))
    lines.append("")
    lines.append(
        f"En jugadores con `xG_90 >= 0.50`, el menor MAE corresponde a "
        f"`{best_high_xg_row['model']}`, con MAE = "
        f"{best_high_xg_row['mae']:.4f}."
    )
    lines.append("")

    lines.append("## Evaluación multi-métrica con Ridge")
    lines.append("")
    lines.append(
        "Tras seleccionar Ridge como modelo final principal, se entrena un "
        "modelo Ridge independiente para cada una de las siguientes métricas: "
        "`xG_90`, `goals_90`, `assists_90` y `xA_90`."
    )
    lines.append("")
    lines.append(report_multi_metric_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Evaluación multi-métrica F/M con Ridge")
    lines.append("")
    lines.append(report_multi_metric_fm_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Figuras generadas")
    lines.append("")
    lines.append("![Comparación externa MAE](figures/external_2025_2026_model_comparison_mae.png)")
    lines.append("")
    lines.append("![Comparación externa R²](figures/external_2025_2026_model_comparison_r2.png)")
    lines.append("")
    lines.append("![Comparación externa F/M MAE](figures/external_2025_2026_model_comparison_f_m_mae.png)")
    lines.append("")
    lines.append("![MAE multi-métrica](figures/external_2025_2026_multi_metric_mae.png)")
    lines.append("")
    lines.append("![R² multi-métrica](figures/external_2025_2026_multi_metric_r2.png)")
    lines.append("")

    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        "La comparación externa mantiene `xG_90` como métrica principal de "
        "selección del modelo, ya que fue el objetivo central del sistema. "
        "La extensión multi-métrica permite analizar dimensiones adicionales "
        "del rendimiento ofensivo, como producción goleadora real, asistencia "
        "real y generación esperada de asistencias."
    )
    lines.append("")
    lines.append(
        "Las métricas esperadas, como `xG_90` y `xA_90`, suelen ser más "
        "estables conceptualmente que goles y asistencias reales, que dependen "
        "en mayor medida de factores contextuales y de la varianza propia de "
        "la finalización."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    historical_df = pd.read_csv(HISTORICAL_PLAYERS_PATH)
    target_2025_2026_df = pd.read_csv(TARGET_2025_2026_PATH)

    temporal_df = build_multi_metric_temporal_dataset(historical_df)

    input_2024_2025_df = prepare_2024_2025_players_for_prediction(
        historical_df
    )

    target_df = prepare_2025_2026_target(target_2025_2026_df)

    external_base_df = build_external_base_dataset(
        input_df=input_2024_2025_df,
        target_df=target_df,
    )

    print(f"Filas temporales de entrenamiento: {len(temporal_df)}")
    print(f"Jugadores 2024-2025 preparados: {len(input_2024_2025_df)}")
    print(f"Jugadores objetivo 2025-2026: {len(target_df)}")
    print(f"Jugadores emparejados: {len(external_base_df)}")

    (
        comparison_df,
        fm_comparison_df,
        predictions_df,
        range_comparison_df,
    ) = compare_models_for_xg(
        temporal_df=temporal_df,
        external_base_df=external_base_df,
    )

    (
        multi_metric_predictions_df,
        multi_metric_long_predictions_df,
        multi_metric_comparison_df,
        multi_metric_fm_comparison_df,
    ) = evaluate_multi_metric_ridge(
        temporal_df=temporal_df,
        external_base_df=external_base_df,
    )

    comparison_df.to_csv(COMPARISON_OUTPUT_PATH, index=False)
    fm_comparison_df.to_csv(COMPARISON_FM_OUTPUT_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    range_comparison_df.to_csv(RANGE_OUTPUT_PATH, index=False)

    multi_metric_predictions_df.to_csv(
        MULTI_METRIC_WIDE_OUTPUT_PATH,
        index=False,
    )
    multi_metric_long_predictions_df.to_csv(
        MULTI_METRIC_LONG_OUTPUT_PATH,
        index=False,
    )
    multi_metric_comparison_df.to_csv(
        MULTI_METRIC_COMPARISON_OUTPUT_PATH,
        index=False,
    )
    multi_metric_fm_comparison_df.to_csv(
        MULTI_METRIC_COMPARISON_FM_OUTPUT_PATH,
        index=False,
    )

    plot_model_metric(
        comparison_df=comparison_df,
        metric_column="mae",
        output_path=MAE_FIGURE_OUTPUT_PATH,
        title="Evaluación externa 2025-2026: MAE por modelo",
        x_label="MAE",
        ascending=False,
    )

    plot_model_metric(
        comparison_df=comparison_df,
        metric_column="r2",
        output_path=R2_FIGURE_OUTPUT_PATH,
        title="Evaluación externa 2025-2026: R² por modelo",
        x_label="R²",
        ascending=True,
    )

    plot_model_metric(
        comparison_df=fm_comparison_df,
        metric_column="mae",
        output_path=FM_MAE_FIGURE_OUTPUT_PATH,
        title="Evaluación externa F/M: MAE por modelo",
        x_label="MAE",
        ascending=False,
    )

    plot_multi_metric_bar(
        comparison_df=multi_metric_comparison_df,
        metric_column="mae",
        output_path=MULTI_METRIC_MAE_FIGURE_OUTPUT_PATH,
        title="Evaluación externa multi-métrica: MAE con Ridge",
        y_label="MAE",
    )

    plot_multi_metric_bar(
        comparison_df=multi_metric_comparison_df,
        metric_column="r2",
        output_path=MULTI_METRIC_R2_FIGURE_OUTPUT_PATH,
        title="Evaluación externa multi-métrica: R² con Ridge",
        y_label="R²",
    )

    report = build_markdown_report(
        comparison_df=comparison_df,
        fm_comparison_df=fm_comparison_df,
        range_comparison_df=range_comparison_df,
        multi_metric_comparison_df=multi_metric_comparison_df,
        multi_metric_fm_comparison_df=multi_metric_fm_comparison_df,
    )

    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print("Comparación externa 2025-2026 y evaluación multi-métrica completadas.")
    print("")
    print("Resultados globales xG_90:")
    print(comparison_df)
    print("")
    print("Resultados F/M xG_90:")
    print(fm_comparison_df)
    print("")
    print("Resultados multi-métrica globales Ridge:")
    print(multi_metric_comparison_df)
    print("")
    print("Resultados multi-métrica F/M Ridge:")
    print(multi_metric_fm_comparison_df)
    print("")
    print(f"CSV comparación xG_90 generado: {COMPARISON_OUTPUT_PATH}")
    print(f"CSV comparación F/M xG_90 generado: {COMPARISON_FM_OUTPUT_PATH}")
    print(f"Predicciones xG_90 por modelo generadas: {PREDICTIONS_OUTPUT_PATH}")
    print(f"CSV rango xG_90 generado: {RANGE_OUTPUT_PATH}")
    print(f"CSV multi-métrica wide generado: {MULTI_METRIC_WIDE_OUTPUT_PATH}")
    print(f"CSV multi-métrica long generado: {MULTI_METRIC_LONG_OUTPUT_PATH}")
    print(f"CSV multi-métrica global generado: {MULTI_METRIC_COMPARISON_OUTPUT_PATH}")
    print(f"CSV multi-métrica F/M generado: {MULTI_METRIC_COMPARISON_FM_OUTPUT_PATH}")
    print(f"Informe actualizado generado: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
