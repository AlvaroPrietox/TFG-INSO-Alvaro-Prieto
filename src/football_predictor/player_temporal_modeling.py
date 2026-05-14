from dataclasses import dataclass

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "target_next_xG_90"


NUMERIC_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "position_main",
    "league",
]


@dataclass
class TemporalModelEvaluationResult:
    """
    Representa el resultado de evaluación de un modelo temporal de regresión.
    """
    model_name: str
    mae: float
    r2: float
    train_rows: int
    test_rows: int
    test_season: str
    test_next_season: str


def validate_temporal_modeling_dataset(df: pd.DataFrame) -> None:
    """
    Comprueba que el dataset temporal contiene las columnas necesarias
    para entrenar y evaluar el modelo.
    """
    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "season",
            "next_season",
            "season_start_year",
            TARGET_COLUMN,
        ]
    )

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para entrenar el modelo temporal: "
            f"{missing_columns}"
        )


def build_temporal_preprocessor() -> ColumnTransformer:
    """
    Construye el preprocesador del modelo temporal.

    Las variables categóricas se codifican mediante one-hot encoding.
    Las variables numéricas se pasan directamente al modelo.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )


def split_temporal_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str, str]:
    """
    Divide el dataset usando una partición temporal.

    Se reserva como prueba el último par temporal disponible. El resto de
    temporadas se utilizan para entrenamiento.
    """
    validate_temporal_modeling_dataset(df)

    latest_season_start_year = df["season_start_year"].max()

    train_df = df[df["season_start_year"] < latest_season_start_year].copy()
    test_df = df[df["season_start_year"] == latest_season_start_year].copy()

    if train_df.empty:
        raise ValueError(
            "No hay suficientes temporadas para entrenar con partición temporal."
        )

    if test_df.empty:
        raise ValueError(
            "No hay registros en la última temporada para evaluar el modelo."
        )

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    test_season = str(test_df["season"].iloc[0])
    test_next_season = str(test_df["next_season"].iloc[0])

    return X_train, X_test, y_train, y_test, test_season, test_next_season


def build_dummy_temporal_regressor() -> DummyRegressor:
    """
    Construye un modelo ingenuo que predice siempre la media del target.
    """
    return DummyRegressor(strategy="mean")


def build_random_forest_temporal_regressor(
    random_state: int = 42,
) -> Pipeline:
    """
    Construye un pipeline temporal con preprocesamiento y Random Forest.
    """
    preprocessor = build_temporal_preprocessor()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_temporal_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    train_rows: int,
    test_rows: int,
    test_season: str,
    test_next_season: str,
) -> TemporalModelEvaluationResult:
    """
    Evalúa un modelo temporal mediante MAE y R².
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return TemporalModelEvaluationResult(
        model_name=model_name,
        mae=mae,
        r2=r2,
        train_rows=train_rows,
        test_rows=test_rows,
        test_season=test_season,
        test_next_season=test_next_season,
    )


def train_and_evaluate_temporal_models(
    df: pd.DataFrame,
) -> tuple[list[TemporalModelEvaluationResult], Pipeline]:
    """
    Entrena y evalúa modelos temporales de rendimiento individual.

    Returns
    -------
    tuple[list[TemporalModelEvaluationResult], Pipeline]
        Resultados de evaluación y modelo Random Forest entrenado.
    """
    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_season,
        test_next_season,
    ) = split_temporal_dataset(df)

    dummy_model = build_dummy_temporal_regressor()
    dummy_model.fit(X_train, y_train)

    random_forest_model = build_random_forest_temporal_regressor()
    random_forest_model.fit(X_train, y_train)

    train_rows = len(X_train)
    test_rows = len(X_test)

    results = [
        evaluate_temporal_model(
            dummy_model,
            X_test,
            y_test,
            model_name="DummyRegressor",
            train_rows=train_rows,
            test_rows=test_rows,
            test_season=test_season,
            test_next_season=test_next_season,
        ),
        evaluate_temporal_model(
            random_forest_model,
            X_test,
            y_test,
            model_name="RandomForestRegressor",
            train_rows=train_rows,
            test_rows=test_rows,
            test_season=test_season,
            test_next_season=test_next_season,
        ),
    ]

    return results, random_forest_model


def extract_temporal_feature_importance(
    trained_model: Pipeline,
) -> pd.DataFrame:
    """
    Extrae la importancia de variables del Random Forest temporal.
    """
    preprocessor = trained_model.named_steps["preprocessor"]
    model = trained_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "El número de variables no coincide con el número de importancias."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df["importance_percent"] = (
        importance_df["importance"] / importance_df["importance"].sum() * 100
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df