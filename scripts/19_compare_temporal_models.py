from pathlib import Path
import sys

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

from football_predictor.player_temporal_modeling import (  # noqa: E402
    TARGET_COLUMN,
    get_temporal_feature_set,
)


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TEMPORAL_DATASET_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_modeling_dataset.csv"
)

COMPARISON_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "temporal_model_comparison.csv"
)

COMPARISON_BY_XG_RANGE_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "temporal_model_comparison_by_xg_range.csv"
)

REPORT_OUTPUT_PATH = (
    REPORTS_DIR / "19_temporal_model_comparison.md"
)

MAE_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "temporal_model_comparison_mae.png"
)

R2_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "temporal_model_comparison_r2.png"
)


FEATURE_SET_NAME = "without_previous_xg"

XG_RANGE_LABELS = [
    "bajo_<0.10",
    "medio_bajo_0.10_0.25",
    "medio_alto_0.25_0.50",
    "alto_>=0.50",
]


def build_one_hot_encoder() -> OneHotEncoder:
    """
    Construye un OneHotEncoder compatible con versiones recientes de
    scikit-learn.

    Se fuerza salida densa para que también pueda ser utilizada por modelos
    como HistGradientBoostingRegressor.
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


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """
    Construye el preprocesador común.

    Para modelos lineales se escalan las variables numéricas.
    Para modelos basados en árboles se mantienen sin escalar.
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


def split_temporal_train_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str, str]:
    """
    Divide el dataset temporal reservando como test el último par temporal.

    Ejemplo:
    train -> temporadas anteriores
    test  -> 2023-2024 -> 2024-2025
    """
    latest_season_start_year = df["season_start_year"].max()

    train_df = df[
        df["season_start_year"] < latest_season_start_year
    ].copy()

    test_df = df[
        df["season_start_year"] == latest_season_start_year
    ].copy()

    if train_df.empty:
        raise ValueError(
            "No hay suficientes temporadas para construir el conjunto de entrenamiento."
        )

    if test_df.empty:
        raise ValueError(
            "No hay registros disponibles para el conjunto de test temporal."
        )

    numeric_features, categorical_features = get_temporal_feature_set(
        FEATURE_SET_NAME
    )

    feature_columns = numeric_features + categorical_features

    missing_columns = [
        column for column in feature_columns + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para comparar modelos: {missing_columns}"
        )

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    test_season = str(test_df["season"].iloc[0])
    test_next_season = str(test_df["next_season"].iloc[0])

    return X_train, X_test, y_train, y_test, test_season, test_next_season


def build_candidate_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    """
    Construye los modelos candidatos a comparar.
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

    models = {
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

    return models


def assign_xg_ranges(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna cada jugador a un rango según su xG_90 real futuro.
    """
    ranged_df = prediction_df.copy()

    ranged_df["xg_range"] = pd.cut(
        ranged_df["actual_next_xG_90"],
        bins=[-float("inf"), 0.10, 0.25, 0.50, float("inf")],
        labels=XG_RANGE_LABELS,
        right=False,
    )

    return ranged_df


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


def evaluate_models() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Entrena y evalúa todos los modelos candidatos.
    """
    temporal_df = pd.read_csv(TEMPORAL_DATASET_PATH)

    numeric_features, categorical_features = get_temporal_feature_set(
        FEATURE_SET_NAME
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_season,
        test_next_season,
    ) = split_temporal_train_test(temporal_df)

    models = build_candidate_models(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    comparison_rows = []
    range_rows = []

    for model_name, model in models.items():
        print(f"Entrenando modelo: {model_name}")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = compute_metrics(
            actual=y_test,
            predicted=predictions,
        )

        comparison_rows.append(
            {
                "feature_set": FEATURE_SET_NAME,
                "model": model_name,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "test_season": test_season,
                "test_next_season": test_next_season,
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

        prediction_df = pd.DataFrame(
            {
                "actual_next_xG_90": y_test.values,
                "predicted_next_xG_90": predictions,
            }
        )

        prediction_df["signed_error"] = (
            prediction_df["predicted_next_xG_90"]
            - prediction_df["actual_next_xG_90"]
        )

        prediction_df["absolute_error"] = (
            prediction_df["signed_error"].abs()
        )

        ranged_prediction_df = assign_xg_ranges(prediction_df)

        for xg_range in XG_RANGE_LABELS:
            group_df = ranged_prediction_df[
                ranged_prediction_df["xg_range"] == xg_range
            ].copy()

            if group_df.empty:
                continue

            range_metrics = compute_metrics(
                actual=group_df["actual_next_xG_90"],
                predicted=group_df["predicted_next_xG_90"],
            )

            range_rows.append(
                {
                    "model": model_name,
                    "xg_range": xg_range,
                    "n_players": int(len(group_df)),
                    "actual_xg_90_mean": float(
                        group_df["actual_next_xG_90"].mean()
                    ),
                    "predicted_xg_90_mean": float(
                        group_df["predicted_next_xG_90"].mean()
                    ),
                    "mae": range_metrics["mae"],
                    "r2": range_metrics["r2"],
                    "mean_signed_error": float(
                        group_df["signed_error"].mean()
                    ),
                    "overestimations": int(
                        (group_df["signed_error"] > 0).sum()
                    ),
                    "underestimations": int(
                        (group_df["signed_error"] < 0).sum()
                    ),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    range_comparison_df = pd.DataFrame(range_rows)

    comparison_df = comparison_df.sort_values(
        by="mae",
        ascending=True,
    ).reset_index(drop=True)

    range_comparison_df = range_comparison_df.sort_values(
        by=["xg_range", "mae"],
        ascending=True,
    ).reset_index(drop=True)

    return comparison_df, range_comparison_df, test_season, test_next_season


def plot_model_metric(
    comparison_df: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    title: str,
    x_label: str,
    ascending: bool,
) -> None:
    """
    Genera gráfico de barras horizontal para comparar modelos.
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


def build_markdown_report(
    comparison_df: pd.DataFrame,
    range_comparison_df: pd.DataFrame,
    test_season: str,
    test_next_season: str,
) -> str:
    """
    Construye el informe Markdown de comparación de modelos.
    """
    report_comparison_df = comparison_df.copy()
    report_comparison_df["mae"] = report_comparison_df["mae"].round(4)
    report_comparison_df["r2"] = report_comparison_df["r2"].round(4)

    report_range_df = range_comparison_df.copy()

    numeric_columns = [
        "actual_xg_90_mean",
        "predicted_xg_90_mean",
        "mae",
        "r2",
        "mean_signed_error",
    ]

    for column in numeric_columns:
        report_range_df[column] = report_range_df[column].round(4)

    best_model_row = comparison_df.sort_values(
        by="mae",
        ascending=True,
    ).iloc[0]

    best_r2_row = comparison_df.sort_values(
        by="r2",
        ascending=False,
    ).iloc[0]

    high_xg_df = range_comparison_df[
        range_comparison_df["xg_range"] == "alto_>=0.50"
    ].copy()

    best_high_xg_row = high_xg_df.sort_values(
        by="mae",
        ascending=True,
    ).iloc[0]

    lines = []

    lines.append("# Comparación de modelos temporales")
    lines.append("")

    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este experimento compara distintos algoritmos de regresión para "
        "predecir el `xG_90` de la temporada siguiente a partir de métricas "
        "de la temporada actual."
    )
    lines.append("")
    lines.append(
        "La comparación utiliza el conjunto de variables `without_previous_xg`, "
        "por lo que no se emplean `xG_90` ni `npxG_90` de la temporada anterior "
        "como predictores directos."
    )
    lines.append("")

    lines.append("## Configuración experimental")
    lines.append("")
    lines.append(f"- Periodo de test: {test_season} → {test_next_season}")
    lines.append(f"- Conjunto de variables: `{FEATURE_SET_NAME}`")
    lines.append("- Variable objetivo: `target_next_xG_90`")
    lines.append("")

    lines.append("## Resultados globales")
    lines.append("")
    lines.append(
        report_comparison_df[
            [
                "model",
                "train_rows",
                "test_rows",
                "mae",
                "r2",
                "test_season",
                "test_next_season",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Resultados por rango de xG_90")
    lines.append("")
    lines.append(
        report_range_df.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Figuras generadas")
    lines.append("")
    lines.append("![Comparación MAE](figures/temporal_model_comparison_mae.png)")
    lines.append("")
    lines.append("![Comparación R²](figures/temporal_model_comparison_r2.png)")
    lines.append("")

    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        f"El modelo con menor MAE global es `{best_model_row['model']}`, "
        f"con MAE = {best_model_row['mae']:.4f}."
    )
    lines.append("")
    lines.append(
        f"El modelo con mayor R² global es `{best_r2_row['model']}`, "
        f"con R² = {best_r2_row['r2']:.4f}."
    )
    lines.append("")
    lines.append(
        f"En el rango de alto rendimiento ofensivo (`xG_90 >= 0.50`), "
        f"el menor MAE lo obtiene `{best_high_xg_row['model']}`, "
        f"con MAE = {best_high_xg_row['mae']:.4f}."
    )
    lines.append("")
    lines.append(
        "Esta comparación permite valorar si el Random Forest utilizado como "
        "modelo principal es competitivo frente a modelos lineales y métodos "
        "de boosting. Si los modelos de boosting reducen el error en perfiles "
        "de alto `xG_90`, podrían considerarse como alternativa o extensión "
        "del sistema."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df, range_comparison_df, test_season, test_next_season = (
        evaluate_models()
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    range_comparison_df.to_csv(
        COMPARISON_BY_XG_RANGE_OUTPUT_PATH,
        index=False,
    )

    plot_model_metric(
        comparison_df=comparison_df,
        metric_column="mae",
        output_path=MAE_FIGURE_OUTPUT_PATH,
        title="Comparación de MAE por modelo temporal",
        x_label="MAE",
        ascending=False,
    )

    plot_model_metric(
        comparison_df=comparison_df,
        metric_column="r2",
        output_path=R2_FIGURE_OUTPUT_PATH,
        title="Comparación de R² por modelo temporal",
        x_label="R²",
        ascending=True,
    )

    report = build_markdown_report(
        comparison_df=comparison_df,
        range_comparison_df=range_comparison_df,
        test_season=test_season,
        test_next_season=test_next_season,
    )

    REPORT_OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Comparación de modelos temporales completada.")
    print("")
    print("Resultados globales:")
    print(
        comparison_df[
            [
                "model",
                "train_rows",
                "test_rows",
                "mae",
                "r2",
                "test_season",
                "test_next_season",
            ]
        ]
    )
    print("")
    print("Resultados por rango de xG_90:")
    print(range_comparison_df)
    print("")
    print(f"CSV global generado: {COMPARISON_OUTPUT_PATH}")
    print(f"CSV por rango generado: {COMPARISON_BY_XG_RANGE_OUTPUT_PATH}")
    print(f"Informe generado: {REPORT_OUTPUT_PATH}")
    print(f"Figura MAE generada: {MAE_FIGURE_OUTPUT_PATH}")
    print(f"Figura R² generada: {R2_FIGURE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()