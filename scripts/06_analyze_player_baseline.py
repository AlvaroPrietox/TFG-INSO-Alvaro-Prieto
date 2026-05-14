import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_modeling import (
    extract_feature_importance,
    train_and_evaluate_baseline_models,
)


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PLAYER_MODELING_DATASET_PATH = PROCESSED_DATA_DIR / "player_modeling_dataset.csv"

FEATURE_IMPORTANCE_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "player_feature_importance.csv"
)

REPORT_OUTPUT_PATH = REPORTS_DIR / "06_player_baseline_analysis.md"


def build_markdown_report(
    evaluation_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    top_n: int = 15,
) -> str:
    """
    Construye un informe Markdown con los resultados del baseline.

    Parameters
    ----------
    evaluation_df : pd.DataFrame
        Tabla con las métricas de evaluación de los modelos.
    feature_importance_df : pd.DataFrame
        Tabla con la importancia de variables del Random Forest.
    top_n : int
        Número de variables más importantes que se incluyen en el informe.

    Returns
    -------
    str
        Informe en formato Markdown.
    """
    lines = []

    lines.append("# Análisis del baseline de rendimiento de jugadores")
    lines.append("")
    lines.append("## Métricas de evaluación")
    lines.append("")
    lines.append(evaluation_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Variables más importantes")
    lines.append("")
    lines.append(
        feature_importance_df.head(top_n).to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Interpretación preliminar")
    lines.append("")
    lines.append(
        "El análisis de importancia de variables permite comprobar qué "
        "atributos utiliza principalmente el modelo para estimar el rendimiento "
        "ofensivo esperado del jugador."
    )
    lines.append("")
    lines.append(
        "Si las primeras variables están muy próximas conceptualmente a la "
        "variable objetivo, el resultado debe interpretarse como un modelo "
        "descriptivo del rendimiento agregado, no todavía como una predicción "
        "futura estricta."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    player_modeling_df = pd.read_csv(PLAYER_MODELING_DATASET_PATH)

    results, random_forest_model = train_and_evaluate_baseline_models(
        player_modeling_df
    )

    evaluation_df = pd.DataFrame(
        [
            {
                "model": result.model_name,
                "mae": result.mae,
                "r2": result.r2,
            }
            for result in results
        ]
    )

    feature_importance_df = extract_feature_importance(random_forest_model)

    feature_importance_df.to_csv(
        FEATURE_IMPORTANCE_OUTPUT_PATH,
        index=False,
    )

    report = build_markdown_report(
        evaluation_df=evaluation_df,
        feature_importance_df=feature_importance_df,
        top_n=15,
    )

    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print("Análisis del baseline completado.")
    print("")
    print("Métricas:")
    print(evaluation_df)
    print("")
    print("Top 15 variables más importantes:")
    print(feature_importance_df.head(15))
    print("")
    print(f"Importancia de variables guardada en: {FEATURE_IMPORTANCE_OUTPUT_PATH}")
    print(f"Informe guardado en: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()