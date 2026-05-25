import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_predictions_f_m_sample_15.csv"
)

REPORT_OUTPUT_PATH = REPORTS_DIR / "10_player_prediction_examples.md"


DISPLAY_COLUMNS = [
    "player_name",
    "position_main",
    "team_title",
    "next_team_title",
    "season",
    "next_season",
    "actual_next_xG_90",
    "predicted_next_xG_90",
    "signed_error",
    "absolute_error",
]


def classify_prediction_error(row: pd.Series) -> str:
    """
    Clasifica cualitativamente el error de predicción.

    La clasificación se basa en el error absoluto entre el valor real y
    el valor predicho de xG_90 en la temporada siguiente.
    """
    absolute_error = row["absolute_error"]

    if absolute_error < 0.02:
        return "error_bajo"

    if absolute_error < 0.10:
        return "error_moderado"

    return "error_alto"


def classify_prediction_direction(row: pd.Series) -> str:
    """
    Clasifica si el modelo sobreestima o infraestima al jugador.

    signed_error = predicción - valor real
    """
    signed_error = row["signed_error"]

    if signed_error > 0:
        return "sobreestimación"

    if signed_error < 0:
        return "infraestimación"

    return "predicción_exacta"


def prepare_predictions_for_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara la tabla de predicciones para el informe.

    Añade columnas interpretativas y redondea los valores numéricos para
    facilitar la lectura en Markdown.
    """
    report_df = df.copy()

    report_df["error_group"] = report_df.apply(
        classify_prediction_error,
        axis=1,
    )

    report_df["error_direction"] = report_df.apply(
        classify_prediction_direction,
        axis=1,
    )

    numeric_columns = [
        "actual_next_xG_90",
        "predicted_next_xG_90",
        "signed_error",
        "absolute_error",
    ]

    for column in numeric_columns:
        report_df[column] = report_df[column].round(4)

    return report_df


def build_prediction_examples_report(df: pd.DataFrame) -> str:
    """
    Construye un informe Markdown a partir de una muestra de predicciones
    individuales de jugadores.
    """
    report_df = prepare_predictions_for_report(df)

    best_predictions = report_df.sort_values(
        by="absolute_error",
        ascending=True,
    ).head(5)

    worst_predictions = report_df.sort_values(
        by="absolute_error",
        ascending=False,
    ).head(5)

    mean_absolute_error = report_df["absolute_error"].mean()
    median_absolute_error = report_df["absolute_error"].median()

    low_error_count = (report_df["error_group"] == "error_bajo").sum()
    moderate_error_count = (report_df["error_group"] == "error_moderado").sum()
    high_error_count = (report_df["error_group"] == "error_alto").sum()

    overestimated_count = (
        report_df["error_direction"] == "sobreestimación"
    ).sum()
    underestimated_count = (
        report_df["error_direction"] == "infraestimación"
    ).sum()

    lines = []

    lines.append("# Análisis de predicciones individuales de jugadores")
    lines.append("")

    lines.append("## Objetivo del análisis")
    lines.append("")
    lines.append(
        "Este informe analiza una muestra de predicciones individuales "
        "generadas por el modelo temporal sin utilizar `xG_90` ni `npxG_90` "
        "de la temporada anterior como variables predictoras directas."
    )
    lines.append("")
    lines.append(
        "La variable objetivo es `actual_next_xG_90`, que representa el valor "
        "real de goles esperados por 90 minutos del jugador en la temporada "
        "siguiente. La columna `predicted_next_xG_90` representa la estimación "
        "del modelo."
    )
    lines.append("")

    lines.append("## Resumen cuantitativo de la muestra")
    lines.append("")
    lines.append(f"- Número de jugadores analizados: {len(report_df)}")
    lines.append(f"- Error absoluto medio de la muestra: {mean_absolute_error:.4f}")
    lines.append(f"- Mediana del error absoluto: {median_absolute_error:.4f}")
    lines.append(f"- Casos con error bajo: {low_error_count}")
    lines.append(f"- Casos con error moderado: {moderate_error_count}")
    lines.append(f"- Casos con error alto: {high_error_count}")
    lines.append(f"- Sobreestimaciones: {overestimated_count}")
    lines.append(f"- Infraestimaciones: {underestimated_count}")
    lines.append("")

    lines.append("## Tabla completa de predicciones")
    lines.append("")
    lines.append(
        report_df[
            DISPLAY_COLUMNS + ["error_group", "error_direction"]
        ].to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Cinco predicciones más precisas")
    lines.append("")
    lines.append(
        best_predictions[
            DISPLAY_COLUMNS + ["error_group", "error_direction"]
        ].to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Cinco mayores errores")
    lines.append("")
    lines.append(
        worst_predictions[
            DISPLAY_COLUMNS + ["error_group", "error_direction"]
        ].to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        "Los casos con menor error corresponden principalmente a jugadores "
        "con rendimiento ofensivo estable o moderado, donde el modelo consigue "
        "aproximar con precisión el `xG_90` de la temporada siguiente."
    )
    lines.append("")
    lines.append(
        "En cambio, los mayores errores se concentran en jugadores ofensivos "
        "que incrementan de forma notable su producción de ocasiones en la "
        "temporada objetivo. En estos casos, el error suele ser negativo, lo "
        "que indica una infraestimación del rendimiento real."
    )
    lines.append("")
    lines.append(
        "Este comportamiento sugiere que el modelo presenta una tendencia "
        "conservadora: reproduce adecuadamente patrones históricos estables, "
        "pero tiene más dificultad para anticipar saltos bruscos de rendimiento "
        "asociados a cambios de rol, equipo, contexto táctico o eficiencia "
        "ofensiva."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    report = build_prediction_examples_report(predictions_df)

    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print("Informe de predicciones individuales generado correctamente.")
    print(f"Archivo leído: {PREDICTIONS_PATH}")
    print(f"Informe generado: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()