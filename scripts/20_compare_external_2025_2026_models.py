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

from football_predictor.player_temporal_features import simplify_position  # noqa: E402
from football_predictor.player_temporal_modeling import (  # noqa: E402
    TARGET_COLUMN,
    get_temporal_feature_set,
)
from football_predictor.text_normalization import normalize_player_key  # noqa: E402


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TEMPORAL_DATASET_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_modeling_dataset.csv"
)

HISTORICAL_PLAYERS_PATH = (
    PROCESSED_DATA_DIR / "historical_players_clean.csv"
)

TARGET_2025_2026_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_target_clean.csv"
)

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


FEATURE_SET_NAME = "without_previous_xg"
INPUT_SEASON = "2024-2025"
TARGET_SEASON = "2025-2026"

ACTUAL_COLUMN = "actual_2025_2026_xG_90"
PREDICTED_COLUMN = "predicted_2025_2026_xG_90"

XG_RANGE_LABELS = [
    "bajo_<0.10",
    "medio_bajo_0.10_0.25",
    "medio_alto_0.25_0.50",
    "alto_>=0.50",
]


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


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """
    Construye el preprocesador de variables.

    Los modelos lineales usan escalado numérico. Los modelos basados en
    árboles no lo necesitan.
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


def build_candidate_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    """
    Construye los modelos candidatos para la evaluación externa.
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


def prepare_training_data(
    temporal_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Prepara los datos de entrenamiento temporal.

    Se usan todos los pares temporales disponibles, porque la evaluación
    externa 2025-2026 actúa como nuevo horizonte de validación.
    """
    numeric_features, categorical_features = get_temporal_feature_set(
        FEATURE_SET_NAME
    )

    feature_columns = numeric_features + categorical_features

    missing_columns = [
        column
        for column in feature_columns + [TARGET_COLUMN]
        if column not in temporal_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias en el dataset temporal: {missing_columns}"
        )

    X_train = temporal_df[feature_columns]
    y_train = temporal_df[TARGET_COLUMN]

    return X_train, y_train, numeric_features, categorical_features


def prepare_2024_2025_players_for_prediction(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara jugadores 2024-2025 como entrada para predecir 2025-2026.
    """
    input_df = historical_df[
        historical_df["season"] == INPUT_SEASON
    ].copy()

    input_df["position_main"] = input_df["position"].apply(
        simplify_position
    )

    rows_before_gk_filter = len(input_df)

    input_df = input_df[
        input_df["position_main"] != "GK"
    ].copy()

    removed_goalkeepers = rows_before_gk_filter - len(input_df)

    print(f"Porteros eliminados de {INPUT_SEASON}: {removed_goalkeepers}")

    input_df["minutes_per_game"] = input_df["time"] / input_df["games"]
    input_df["high_participation"] = (
        input_df["minutes_per_game"] >= 60
    ).astype(int)

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
    Prepara el target real 2025-2026 para la evaluación externa.
    """
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
            "xG_90": ACTUAL_COLUMN,
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
        ACTUAL_COLUMN,
        "actual_2025_2026_goals_90",
        "actual_2025_2026_assists_90",
        "actual_2025_2026_xA_90",
    ]

    return target_clean[selected_columns].copy()


def build_external_prediction_input(
    input_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une jugadores 2024-2025 con el target real 2025-2026.

    El emparejamiento se realiza por nombre normalizado.
    """
    merged_df = input_df.merge(
        target_df,
        how="inner",
        on="player_key",
    )

    return merged_df.reset_index(drop=True)


def compute_metrics(
    df: pd.DataFrame,
) -> dict[str, float]:
    """
    Calcula MAE y R².
    """
    return {
        "mae": float(
            mean_absolute_error(
                df[ACTUAL_COLUMN],
                df[PREDICTED_COLUMN],
            )
        ),
        "r2": float(
            r2_score(
                df[ACTUAL_COLUMN],
                df[PREDICTED_COLUMN],
            )
        ),
    }


def assign_xg_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna rangos según el xG_90 real 2025-2026.
    """
    ranged_df = df.copy()

    ranged_df["xg_range"] = pd.cut(
        ranged_df[ACTUAL_COLUMN],
        bins=[-float("inf"), 0.10, 0.25, 0.50, float("inf")],
        labels=XG_RANGE_LABELS,
        right=False,
    )

    return ranged_df


def evaluate_models_external() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Entrena y evalúa todos los modelos candidatos sobre la evaluación externa
    2025-2026.
    """
    temporal_df = pd.read_csv(TEMPORAL_DATASET_PATH)
    historical_df = pd.read_csv(HISTORICAL_PLAYERS_PATH)
    target_df = pd.read_csv(TARGET_2025_2026_PATH)

    X_train, y_train, numeric_features, categorical_features = prepare_training_data(
        temporal_df
    )

    input_2024_2025_df = prepare_2024_2025_players_for_prediction(
        historical_df
    )

    target_2025_2026_df = prepare_2025_2026_target(
        target_df
    )

    external_base_df = build_external_prediction_input(
        input_df=input_2024_2025_df,
        target_df=target_2025_2026_df,
    )

    feature_columns = numeric_features + categorical_features

    missing_columns = [
        column for column in feature_columns
        if column not in external_base_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas para predecir en evaluación externa: {missing_columns}"
        )

    models = build_candidate_models(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    comparison_rows = []
    fm_rows = []
    prediction_frames = []
    range_rows = []

    for model_name, model in models.items():
        print(f"Entrenando y evaluando modelo externo: {model_name}")

        model.fit(X_train, y_train)

        model_df = external_base_df.copy()

        model_df[PREDICTED_COLUMN] = model.predict(
            model_df[feature_columns]
        )

        model_df["signed_error"] = (
            model_df[PREDICTED_COLUMN] - model_df[ACTUAL_COLUMN]
        )

        model_df["absolute_error"] = model_df["signed_error"].abs()
        model_df["model"] = model_name
        model_df["feature_set"] = FEATURE_SET_NAME
        model_df["input_season"] = INPUT_SEASON
        model_df["target_season"] = TARGET_SEASON

        metrics = compute_metrics(model_df)

        comparison_rows.append(
            {
                "model": model_name,
                "feature_set": FEATURE_SET_NAME,
                "training_rows": int(len(X_train)),
                "matched_players": int(len(model_df)),
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "input_season": INPUT_SEASON,
                "target_season": TARGET_SEASON,
            }
        )

        fm_df = model_df[
            model_df["position_main"].isin(["F", "M"])
        ].copy()

        fm_metrics = compute_metrics(fm_df)

        fm_rows.append(
            {
                "model": model_name,
                "feature_set": FEATURE_SET_NAME,
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
            group_df = ranged_df[
                ranged_df["xg_range"] == xg_range
            ].copy()

            if group_df.empty:
                continue

            group_metrics = compute_metrics(group_df)

            range_rows.append(
                {
                    "model": model_name,
                    "xg_range": xg_range,
                    "n_players": int(len(group_df)),
                    "actual_xg_90_mean": float(group_df[ACTUAL_COLUMN].mean()),
                    "predicted_xg_90_mean": float(group_df[PREDICTED_COLUMN].mean()),
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
            ACTUAL_COLUMN,
            PREDICTED_COLUMN,
            "signed_error",
            "absolute_error",
        ]

        prediction_frames.append(
            model_df[output_columns].copy()
        )

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
    fm_comparison_df: pd.DataFrame,
    range_comparison_df: pd.DataFrame,
) -> str:
    """
    Construye el informe Markdown de comparación externa.
    """
    report_global_df = comparison_df.copy()
    report_fm_df = fm_comparison_df.copy()
    report_range_df = range_comparison_df.copy()

    for df in [report_global_df, report_fm_df, report_range_df]:
        for column in ["mae", "r2", "mean_signed_error"]:
            if column in df.columns:
                df[column] = df[column].round(4)

    for df in [report_range_df]:
        for column in ["actual_xg_90_mean", "predicted_xg_90_mean"]:
            if column in df.columns:
                df[column] = df[column].round(4)

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

    lines.append("# Comparación externa de modelos en LaLiga 2025-2026")
    lines.append("")

    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este experimento compara distintos algoritmos de regresión en una "
        "evaluación externa. Cada modelo se entrena con los pares temporales "
        "históricos disponibles y se aplica a jugadores de 2024-2025 para "
        "predecir su `xG_90` real en LaLiga 2025-2026."
    )
    lines.append("")
    lines.append(
        "El conjunto de variables utilizado es `without_previous_xg`, por lo "
        "que no se emplean `xG_90` ni `npxG_90` previos como predictores "
        "directos."
    )
    lines.append("")

    lines.append("## Resultados globales")
    lines.append("")
    lines.append(
        report_global_df.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Resultados F/M")
    lines.append("")
    lines.append(
        report_fm_df.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Resultados por rango de xG_90 real")
    lines.append("")
    lines.append(
        report_range_df.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Figuras generadas")
    lines.append("")
    lines.append(
        "![Comparación externa MAE](figures/external_2025_2026_model_comparison_mae.png)"
    )
    lines.append("")
    lines.append(
        "![Comparación externa R²](figures/external_2025_2026_model_comparison_r2.png)"
    )
    lines.append("")
    lines.append(
        "![Comparación externa F/M MAE](figures/external_2025_2026_model_comparison_f_m_mae.png)"
    )
    lines.append("")

    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        f"El menor MAE global lo obtiene `{best_global_row['model']}`, "
        f"con MAE = {best_global_row['mae']:.4f} y "
        f"R² = {best_global_row['r2']:.4f}."
    )
    lines.append("")
    lines.append(
        f"En el subconjunto F/M, el menor MAE lo obtiene "
        f"`{best_fm_row['model']}`, con MAE = {best_fm_row['mae']:.4f} "
        f"y R² = {best_fm_row['r2']:.4f}."
    )
    lines.append("")
    lines.append(
        f"En jugadores con `xG_90 >= 0.50`, el menor MAE corresponde a "
        f"`{best_high_xg_row['model']}`, con MAE = "
        f"{best_high_xg_row['mae']:.4f}."
    )
    lines.append("")
    lines.append(
        "Esta comparación permite decidir si el modelo final debe mantenerse "
        "como Random Forest o si una alternativa más simple o más robusta, como "
        "Ridge o Gradient Boosting, ofrece mejor generalización externa."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    (
        comparison_df,
        fm_comparison_df,
        predictions_df,
        range_comparison_df,
    ) = evaluate_models_external()

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    fm_comparison_df.to_csv(
        COMPARISON_FM_OUTPUT_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    range_comparison_df.to_csv(
        RANGE_OUTPUT_PATH,
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

    report = build_markdown_report(
        comparison_df=comparison_df,
        fm_comparison_df=fm_comparison_df,
        range_comparison_df=range_comparison_df,
    )

    REPORT_OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Comparación externa 2025-2026 completada.")
    print("")
    print("Resultados globales:")
    print(comparison_df)
    print("")
    print("Resultados F/M:")
    print(fm_comparison_df)
    print("")
    print("Resultados por rango de xG_90:")
    print(range_comparison_df)
    print("")
    print(f"CSV global generado: {COMPARISON_OUTPUT_PATH}")
    print(f"CSV F/M generado: {COMPARISON_FM_OUTPUT_PATH}")
    print(f"Predicciones por modelo generadas: {PREDICTIONS_OUTPUT_PATH}")
    print(f"CSV por rango generado: {RANGE_OUTPUT_PATH}")
    print(f"Informe generado: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()