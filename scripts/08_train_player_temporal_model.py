import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_modeling import (
    extract_temporal_feature_importance,
    train_and_evaluate_temporal_models,
)


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TEMPORAL_DATASET_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_modeling_dataset.csv"
)

EVALUATION_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_evaluation.csv"
)

FEATURE_IMPORTANCE_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_feature_importance.csv"
)

REPORT_OUTPUT_PATH = REPORTS_DIR / "08_player_temporal_model_report.md"


FEATURE_SETS_TO_RUN = [
    "full",
    "without_previous_xg",
]


def build_temporal_report(
    evaluation_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    top_n: int = 15,
) -> str:
    """
    Construye un informe Markdown del modelo temporal.
    """
    lines = []

    lines.append("# Modelo temporal de rendimiento de jugadores")
    lines.append("")

    lines.append("## Métricas de evaluación")
    lines.append("")
    lines.append(evaluation_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Variables más importantes por experimento")
    lines.append("")

    for feature_set in evaluation_df["feature_set"].unique():
        lines.append(f"### Experimento: {feature_set}")
        lines.append("")

        subset_importance = feature_importance_df[
            feature_importance_df["feature_set"] == feature_set
        ]

        lines.append(subset_importance.head(top_n).to_markdown(index=False))
        lines.append("")

    lines.append("## Interpretación metodológica")
    lines.append("")
    lines.append(
        "Este experimento utiliza una partición temporal: el modelo se entrena "
        "con temporadas anteriores y se evalúa sobre el último par de temporadas "
        "disponible. Por tanto, la evaluación se aproxima más a un escenario real "
        "de predicción futura que una división aleatoria."
    )
    lines.append("")
    lines.append(
        "Se comparan dos configuraciones. La primera utiliza todas las variables "
        "históricas disponibles. La segunda elimina xG_90 y npxG_90 de la "
        "temporada anterior para comprobar si el modelo mantiene capacidad "
        "predictiva sin utilizar directamente métricas de goles esperados previos."
    )
    lines.append("")
    lines.append(
        "La variable objetivo es target_next_xG_90, que representa el xG por 90 "
        "minutos del jugador en la temporada siguiente."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    temporal_df = pd.read_csv(TEMPORAL_DATASET_PATH)

    all_results = []
    all_feature_importances = []

    for feature_set_name in FEATURE_SETS_TO_RUN:
        print(f"Ejecutando experimento: {feature_set_name}")

        results, random_forest_model = train_and_evaluate_temporal_models(
            temporal_df,
            feature_set_name=feature_set_name,
        )

        all_results.extend(results)

        feature_importance_df = extract_temporal_feature_importance(
            trained_model=random_forest_model,
            feature_set_name=feature_set_name,
        )

        all_feature_importances.append(feature_importance_df)

    evaluation_df = pd.DataFrame(
        [
            {
                "feature_set": result.feature_set,
                "model": result.model_name,
                "mae": result.mae,
                "r2": result.r2,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
                "test_season": result.test_season,
                "test_next_season": result.test_next_season,
            }
            for result in all_results
        ]
    )

    feature_importance_df = pd.concat(
        all_feature_importances,
        ignore_index=True,
    )

    evaluation_df.to_csv(EVALUATION_OUTPUT_PATH, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_OUTPUT_PATH, index=False)

    report = build_temporal_report(
        evaluation_df=evaluation_df,
        feature_importance_df=feature_importance_df,
        top_n=15,
    )

    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print("")
    print("Entrenamiento temporal completado.")
    print("")
    print("Métricas:")
    print(evaluation_df)
    print("")
    print("Top 15 variables más importantes por experimento:")

    for feature_set_name in FEATURE_SETS_TO_RUN:
        print("")
        print(f"Experimento: {feature_set_name}")
        print(
            feature_importance_df[
                feature_importance_df["feature_set"] == feature_set_name
            ].head(15)
        )

    print("")
    print(f"Evaluación guardada en: {EVALUATION_OUTPUT_PATH}")
    print(f"Importancia de variables guardada en: {FEATURE_IMPORTANCE_OUTPUT_PATH}")
    print(f"Informe guardado en: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()