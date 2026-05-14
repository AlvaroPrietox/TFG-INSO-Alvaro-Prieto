from dataclasses import dataclass

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "target_xG_90"


NUMERIC_FEATURES = [
    "games",
    "time",
    "minutes_per_game",
    "high_participation",
    "assists_90",
    "xA_90",
    "shots_90",
    "key_passes_90",
    "yellow_cards_90",
    "red_cards_90",
    "xGChain_90",
    "xGBuildup_90",
]


CATEGORICAL_FEATURES = [
    "position_main",
    "league",
]


@dataclass
class ModelEvaluationResult:
    """
    Representa el resultado de evaluación de un modelo de regresión.
    """
    model_name: str
    mae: float
    r2: float


def validate_modeling_dataset(df: pd.DataFrame) -> None:
    """
    Comprueba que el dataset contiene las columnas necesarias para entrenar
    el modelo de rendimiento individual.
    """
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para entrenar el modelo: {missing_columns}"
        )


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el preprocesador de variables.

    Las variables numéricas pasan directamente al modelo.
    Las variables categóricas se transforman mediante one-hot encoding.
    """
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def split_player_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en subconjuntos de entrenamiento y prueba.

    Esta división permite evaluar el modelo en datos que no se han usado
    durante el aprendizaje.
    """
    validate_modeling_dataset(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def build_dummy_regressor() -> DummyRegressor:
    """
    Construye un modelo baseline ingenuo.

    Este modelo predice siempre el valor medio de la variable objetivo.
    """
    return DummyRegressor(strategy="mean")


def build_random_forest_regressor(
    random_state: int = 42,
) -> Pipeline:
    """
    Construye un pipeline con preprocesamiento y RandomForestRegressor.
    """
    preprocessor = build_preprocessor()

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


def evaluate_regression_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> ModelEvaluationResult:
    """
    Evalúa un modelo de regresión mediante MAE y R².
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return ModelEvaluationResult(
        model_name=model_name,
        mae=mae,
        r2=r2,
    )


def train_and_evaluate_baseline_models(
    df: pd.DataFrame,
) -> tuple[list[ModelEvaluationResult], Pipeline]:
    """
    Entrena y evalúa los primeros modelos baseline.

    Returns
    -------
    tuple[list[ModelEvaluationResult], Pipeline]
        Lista de métricas de evaluación y modelo Random Forest entrenado.
    """
    X_train, X_test, y_train, y_test = split_player_dataset(df)

    dummy_model = build_dummy_regressor()
    dummy_model.fit(X_train, y_train)

    random_forest_model = build_random_forest_regressor()
    random_forest_model.fit(X_train, y_train)

    results = [
        evaluate_regression_model(
            dummy_model,
            X_test,
            y_test,
            model_name="DummyRegressor",
        ),
        evaluate_regression_model(
            random_forest_model,
            X_test,
            y_test,
            model_name="RandomForestRegressor",
        ),
    ]

    return results, random_forest_model


def extract_feature_importance(
    trained_model: Pipeline,
) -> pd.DataFrame:
    """
    Extrae la importancia de variables de un RandomForestRegressor entrenado.

    El modelo está dentro de un Pipeline, por lo que primero se recuperan
    los nombres de variables generados por el preprocesador y después se
    asocian con la importancia calculada por el modelo.

    Parameters
    ----------
    trained_model : Pipeline
        Pipeline entrenado que contiene el preprocesador y el modelo.

    Returns
    -------
    pd.DataFrame
        Tabla ordenada con variables e importancia relativa.
    """
    preprocessor = trained_model.named_steps["preprocessor"]
    model = trained_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "El número de nombres de variables no coincide con el número "
            "de importancias del modelo."
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