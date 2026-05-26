from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

INPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation.csv"
)

OUTPUT_CSV_PATH = (
    PROCESSED_DATA_DIR / "error_by_xg_range.csv"
)

OUTPUT_REPORT_PATH = (
    REPORTS_DIR / "18_error_by_xg_range.md"
)

OUTPUT_FIGURE_PATH = (
    FIGURES_DIR / "error_by_xg_range.png"
)


ACTUAL_COLUMN = "actual_2025_2026_xG_90"
PREDICTED_COLUMN = "predicted_2025_2026_xG_90"


XG_RANGE_LABELS = [
    "bajo_<0.10",
    "medio_bajo_0.10_0.25",
    "medio_alto_0.25_0.50",
    "alto_>=0.50",
]


def validate_input_columns(df: pd.DataFrame) -> None:
    """
    Comprueba que el dataset contiene las columnas necesarias para analizar
    el error por rangos de xG_90.
    """
    required_columns = [
        "player_name",
        "position_main",
        ACTUAL_COLUMN,
        PREDICTED_COLUMN,
        "signed_error",
        "absolute_error",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para el análisis por rangos: "
            f"{missing_columns}"
        )


def assign_xg_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna cada jugador a un rango de xG_90 real.

    Los rangos se definen sobre el valor real de xG_90 en 2025-2026.
    """
    ranged_df = df.copy()

    ranged_df["xg_range"] = pd.cut(
        ranged_df[ACTUAL_COLUMN],
        bins=[-float("inf"), 0.10, 0.25, 0.50, float("inf")],
        labels=XG_RANGE_LABELS,
        right=False,
    )

    return ranged_df


def compute_r2(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calcula R² manualmente para permitir análisis por grupos pequeños.
    """
    if len(actual) < 2:
        return float("nan")

    total_sum_of_squares = ((actual - actual.mean()) ** 2).sum()

    if total_sum_of_squares == 0:
        return float("nan")

    residual_sum_of_squares = ((actual - predicted) ** 2).sum()

    return 1 - (residual_sum_of_squares / total_sum_of_squares)


def summarize_error_by_xg_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de error para cada rango de xG_90 real.
    """
    validate_input_columns(df)

    ranged_df = assign_xg_ranges(df)

    rows = []

    for xg_range in XG_RANGE_LABELS:
        group_df = ranged_df[ranged_df["xg_range"] == xg_range].copy()

        if group_df.empty:
            rows.append(
                {
                    "xg_range": xg_range,
                    "n_players": 0,
                    "actual_xg_90_mean": None,
                    "predicted_xg_90_mean": None,
                    "mae": None,
                    "mean_signed_error": None,
                    "r2": None,
                    "overestimations": 0,
                    "underestimations": 0,
                }
            )
            continue

        actual = group_df[ACTUAL_COLUMN]
        predicted = group_df[PREDICTED_COLUMN]

        rows.append(
            {
                "xg_range": xg_range,
                "n_players": int(len(group_df)),
                "actual_xg_90_mean": float(actual.mean()),
                "predicted_xg_90_mean": float(predicted.mean()),
                "mae": float(group_df["absolute_error"].mean()),
                "mean_signed_error": float(group_df["signed_error"].mean()),
                "r2": float(compute_r2(actual, predicted)),
                "overestimations": int((group_df["signed_error"] > 0).sum()),
                "underestimations": int((group_df["signed_error"] < 0).sum()),
            }
        )

    summary_df = pd.DataFrame(rows)

    return summary_df


def plot_error_by_xg_range(summary_df: pd.DataFrame) -> None:
    """
    Genera un gráfico de barras con el MAE por rango de xG_90.
    """
    plot_df = summary_df.copy()

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["xg_range"], plot_df["mae"])

    plt.title("Error absoluto medio por rango de xG_90 real")
    plt.xlabel("Rango de xG_90 real en 2025-2026")
    plt.ylabel("MAE")

    plt.xticks(rotation=25, ha="right")

    for index, value in enumerate(plot_df["mae"]):
        if pd.notna(value):
            plt.text(
                index,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def build_markdown_report(summary_df: pd.DataFrame) -> str:
    """
    Construye un informe Markdown con el análisis de error por rango de xG_90.
    """
    report_df = summary_df.copy()

    numeric_columns = [
        "actual_xg_90_mean",
        "predicted_xg_90_mean",
        "mae",
        "mean_signed_error",
        "r2",
    ]

    for column in numeric_columns:
        report_df[column] = report_df[column].round(4)

    highest_error_row = summary_df.sort_values(
        by="mae",
        ascending=False,
    ).iloc[0]

    lowest_error_row = summary_df.sort_values(
        by="mae",
        ascending=True,
    ).iloc[0]

    lines = []

    lines.append("# Análisis del error por rangos de xG_90")
    lines.append("")

    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este análisis evalúa si el error del modelo cambia según el nivel "
        "real de `xG_90` del jugador en la temporada 2025-2026."
    )
    lines.append("")
    lines.append(
        "La hipótesis de partida es que el modelo debería cometer errores "
        "menores en jugadores con baja producción ofensiva y errores mayores "
        "en jugadores de alto `xG_90`, ya que estos perfiles suelen presentar "
        "mayor variabilidad contextual y deportiva."
    )
    lines.append("")

    lines.append("## Rangos utilizados")
    lines.append("")
    lines.append("- `bajo_<0.10`: jugadores con xG_90 real inferior a 0.10")
    lines.append("- `medio_bajo_0.10_0.25`: jugadores entre 0.10 y 0.25")
    lines.append("- `medio_alto_0.25_0.50`: jugadores entre 0.25 y 0.50")
    lines.append("- `alto_>=0.50`: jugadores con xG_90 real igual o superior a 0.50")
    lines.append("")

    lines.append("## Tabla resumen")
    lines.append("")
    lines.append(report_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Figura generada")
    lines.append("")
    lines.append("![Error por rango de xG_90](figures/error_by_xg_range.png)")
    lines.append("")

    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        f"El rango con mayor MAE es `{highest_error_row['xg_range']}`, "
        f"con un error absoluto medio de {highest_error_row['mae']:.4f}."
    )
    lines.append("")
    lines.append(
        f"El rango con menor MAE es `{lowest_error_row['xg_range']}`, "
        f"con un error absoluto medio de {lowest_error_row['mae']:.4f}."
    )
    lines.append("")
    lines.append(
        "Si el error aumenta en los rangos de mayor `xG_90`, esto refuerza "
        "la interpretación de que el modelo presenta un comportamiento "
        "conservador: predice con mayor estabilidad perfiles de bajo o medio "
        "rendimiento ofensivo, pero tiene más dificultad para anticipar "
        "jugadores que alcanzan valores ofensivos altos en la temporada objetivo."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_df = pd.read_csv(INPUT_PATH)

    summary_df = summarize_error_by_xg_range(evaluation_df)

    summary_df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
    )

    plot_error_by_xg_range(summary_df)

    report = build_markdown_report(summary_df)

    OUTPUT_REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Análisis de error por rangos de xG_90 completado.")
    print("")
    print(summary_df)
    print("")
    print(f"CSV generado: {OUTPUT_CSV_PATH}")
    print(f"Informe generado: {OUTPUT_REPORT_PATH}")
    print(f"Figura generada: {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()