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

    lines.append("## Variables más importantes")
    lines.append("")
    lines.append(feature_importance_df.head(top_n).to_markdown(index=False))
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
        "La variable objetivo es target_next_xG_90, que representa el xG por 90 "
        "minutos del jugador en la temporada siguiente."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    temporal_df = pd.read_csv(TEMPORAL_DATASET_PATH)

    results, random_forest_model = train_and_evaluate_temporal_models(
        temporal_df
    )

    evaluation_df = pd.DataFrame(
        [
            {
                "model": result.model_name,
                "mae": result.mae,
                "r2": result.r2,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
                "test_season": result.test_season,
                "test_next_season": result.test_next_season,
            }
            for result in results
        ]
    )

    feature_importance_df = extract_temporal_feature_importance(
        random_forest_model
    )

    evaluation_df.to_csv(EVALUATION_OUTPUT_PATH, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_OUTPUT_PATH, index=False)

    report = build_temporal_report(
        evaluation_df=evaluation_df,
        feature_importance_df=feature_importance_df,
        top_n=15,
    )

    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print("Entrenamiento temporal completado.")
    print("")
    print("Métricas:")
    print(evaluation_df)
    print("")
    print("Top 15 variables más importantes:")
    print(feature_importance_df.head(15))
    print("")
    print(f"Evaluación guardada en: {EVALUATION_OUTPUT_PATH}")
    print(f"Importancia de variables guardada en: {FEATURE_IMPORTANCE_OUTPUT_PATH}")
    print(f"Informe guardado en: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()