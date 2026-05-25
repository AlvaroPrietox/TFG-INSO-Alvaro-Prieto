from dataclasses import dataclass

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "target_next_xG_90"


FULL_NUMERIC_FEATURES = [
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


WITHOUT_PREVIOUS_XG_NUMERIC_FEATURES = [
    feature
    for feature in FULL_NUMERIC_FEATURES
    if feature not in ["xG_90", "npxG_90"]
]


CATEGORICAL_FEATURES = [
    "position_main",
    "league",
]


TEMPORAL_FEATURE_SETS = {
    "full": {
        "numeric": FULL_NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
    "without_previous_xg": {
        "numeric": WITHOUT_PREVIOUS_XG_NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
}


@dataclass
class TemporalModelEvaluationResult:
    """
    Representa el resultado de evaluación de un modelo temporal de regresión.
    """
    feature_set: str
    model_name: str
    mae: float
    r2: float
    train_rows: int
    test_rows: int
    test_season: str
    test_next_season: str


def get_temporal_feature_set(
    feature_set_name: str,
) -> tuple[list[str], list[str]]:
    """
    Devuelve las variables numéricas y categóricas de un experimento temporal.

    Parameters
    ----------
    feature_set_name : str
        Nombre del conjunto de variables que se desea utilizar.

    Returns
    -------
    tuple[list[str], list[str]]
        Lista de variables numéricas y lista de variables categóricas.
    """
    if feature_set_name not in TEMPORAL_FEATURE_SETS:
        available_sets = list(TEMPORAL_FEATURE_SETS.keys())
        raise ValueError(
            f"Conjunto de variables no válido: {feature_set_name}. "
            f"Disponibles: {available_sets}"
        )

    feature_set = TEMPORAL_FEATURE_SETS[feature_set_name]

    return feature_set["numeric"], feature_set["categorical"]


def validate_temporal_modeling_dataset(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    """
    Comprueba que el dataset temporal contiene las columnas necesarias
    para entrenar y evaluar el modelo.
    """
    required_columns = (
        numeric_features
        + categorical_features
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


def build_temporal_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
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
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )


def split_temporal_dataset(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str, str]:
    """
    Divide el dataset usando una partición temporal.

    Se reserva como prueba el último par temporal disponible. El resto de
    temporadas se utilizan para entrenamiento.
    """
    validate_temporal_modeling_dataset(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

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

    feature_columns = numeric_features + categorical_features

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    test_season = str(test_df["season"].iloc[0])
    test_next_season = str(test_df["next_season"].iloc[0])

    return X_train, X_test, y_train, y_test, test_season, test_next_season


def build_dummy_temporal_regressor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Construye un modelo ingenuo que predice siempre la media del target.

    Se incluye el mismo preprocesador que en el Random Forest para mantener
    una estructura homogénea de pipeline.
    """
    preprocessor = build_temporal_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model = DummyRegressor(strategy="mean")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_random_forest_temporal_regressor(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    """
    Construye un pipeline temporal con preprocesamiento y Random Forest.
    """
    preprocessor = build_temporal_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

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
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_set_name: str,
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
        feature_set=feature_set_name,
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
    feature_set_name: str = "full",
) -> tuple[list[TemporalModelEvaluationResult], Pipeline]:
    """
    Entrena y evalúa modelos temporales de rendimiento individual.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset temporal de jugadores.
    feature_set_name : str
        Conjunto de variables a utilizar. Puede ser:
        - 'full'
        - 'without_previous_xg'

    Returns
    -------
    tuple[list[TemporalModelEvaluationResult], Pipeline]
        Resultados de evaluación y modelo Random Forest entrenado.
    """
    numeric_features, categorical_features = get_temporal_feature_set(
        feature_set_name
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_season,
        test_next_season,
    ) = split_temporal_dataset(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    dummy_model = build_dummy_temporal_regressor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    dummy_model.fit(X_train, y_train)

    random_forest_model = build_random_forest_temporal_regressor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    random_forest_model.fit(X_train, y_train)

    train_rows = len(X_train)
    test_rows = len(X_test)

    results = [
        evaluate_temporal_model(
            dummy_model,
            X_test,
            y_test,
            feature_set_name=feature_set_name,
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
            feature_set_name=feature_set_name,
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
    feature_set_name: str,
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
            "feature_set": feature_set_name,
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

def build_temporal_test_predictions(
    df: pd.DataFrame,
    feature_set_name: str = "without_previous_xg",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Entrena un Random Forest temporal y genera predicciones sobre el último
    par de temporadas disponible.

    El objetivo es obtener una tabla interpretable con el valor real, el valor
    predicho y el error de predicción para cada jugador del conjunto de test.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset temporal de jugadores.
    feature_set_name : str
        Conjunto de variables que se quiere utilizar.
    random_state : int
        Semilla de reproducibilidad del modelo.

    Returns
    -------
    pd.DataFrame
        Tabla con predicciones individuales sobre el conjunto de test.
    """
    numeric_features, categorical_features = get_temporal_feature_set(
        feature_set_name
    )

    validate_temporal_modeling_dataset(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    latest_season_start_year = df["season_start_year"].max()

    train_df = df[df["season_start_year"] < latest_season_start_year].copy()
    test_df = df[df["season_start_year"] == latest_season_start_year].copy()

    if train_df.empty:
        raise ValueError(
            "No hay suficientes temporadas para entrenar el modelo temporal."
        )

    if test_df.empty:
        raise ValueError(
            "No hay registros en la última temporada para generar predicciones."
        )

    feature_columns = numeric_features + categorical_features

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]

    model = build_random_forest_temporal_regressor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    output_columns = [
        "id",
        "player_name",
        "position_main",
        "team_title",
        "league",
        "season",
        "next_team_title",
        "next_league",
        "next_season",
        "games",
        "time",
        "next_games",
        "next_time",
    ]

    predictions_df = test_df[output_columns].copy()

    predictions_df["feature_set"] = feature_set_name
    predictions_df["actual_next_xG_90"] = test_df[TARGET_COLUMN].values
    predictions_df["predicted_next_xG_90"] = predictions
    predictions_df["signed_error"] = (
        predictions_df["predicted_next_xG_90"]
        - predictions_df["actual_next_xG_90"]
    )
    predictions_df["absolute_error"] = predictions_df["signed_error"].abs()

    predictions_df = predictions_df.sort_values(
        by="absolute_error",
        ascending=True,
    ).reset_index(drop=True)

    return predictions_df