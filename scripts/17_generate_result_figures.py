from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

EXPERIMENT_SUMMARY_PATH = PROCESSED_DATA_DIR / "experiment_summary.csv"

LALIGA_EVALUATION_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation.csv"
)

LALIGA_FM_EVALUATION_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation_f_m.csv"
)

TEMPORAL_FEATURE_IMPORTANCE_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_feature_importance.csv"
)


ACTUAL_COLUMN = "actual_2025_2026_xG_90"
PREDICTED_COLUMN = "predicted_2025_2026_xG_90"


def save_current_figure(output_path: Path) -> None:
    """
    Guarda la figura actual y libera memoria.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_experiment_metric(
    summary_df: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    """
    Genera un gráfico de barras horizontal para comparar experimentos.
    """
    plot_df = summary_df.sort_values(
        by=metric_column,
        ascending=True,
    ).copy()

    labels = plot_df["experiment"].str.replace("_", " ", regex=False)

    plt.figure(figsize=(11, 6))
    plt.barh(labels, plot_df[metric_column])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Experimento")

    for index, value in enumerate(plot_df[metric_column]):
        plt.text(
            value,
            index,
            f" {value:.4f}",
            va="center",
        )

    save_current_figure(output_path)


def plot_actual_vs_predicted(
    evaluation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Genera un gráfico de dispersión entre xG_90 real y predicho.

    Una predicción perfecta se situaría sobre la diagonal.
    """
    actual = evaluation_df[ACTUAL_COLUMN]
    predicted = evaluation_df[PREDICTED_COLUMN]

    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, alpha=0.7)

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    plt.title("LaLiga 2025-2026: xG_90 real vs predicho")
    plt.xlabel("xG_90 real 2025-2026")
    plt.ylabel("xG_90 predicho 2025-2026")

    save_current_figure(output_path)


def plot_residuals(
    evaluation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Genera un histograma de residuos.

    El residuo se define como:
    predicción - valor real.
    """
    residuals = evaluation_df["signed_error"]

    plt.figure(figsize=(9, 6))
    plt.hist(residuals, bins=30)

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.title("Distribución de residuos en LaLiga 2025-2026")
    plt.xlabel("Residuo: predicción - valor real")
    plt.ylabel("Número de jugadores")

    save_current_figure(output_path)


def plot_error_by_position(
    fm_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compara el error absoluto medio entre F y M.
    """
    plot_df = (
        fm_df.groupby("position_main")["absolute_error"]
        .mean()
        .reset_index()
        .sort_values(by="absolute_error", ascending=False)
    )

    plt.figure(figsize=(7, 5))
    plt.bar(plot_df["position_main"], plot_df["absolute_error"])

    plt.title("Error absoluto medio por posición principal")
    plt.xlabel("Posición principal")
    plt.ylabel("MAE")

    for index, value in enumerate(plot_df["absolute_error"]):
        plt.text(
            index,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    save_current_figure(output_path)


def plot_feature_importance_without_previous_xg(
    feature_importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """
    Genera un gráfico con las variables más importantes del experimento
    without_previous_xg.
    """
    plot_df = feature_importance_df[
        feature_importance_df["feature_set"] == "without_previous_xg"
    ].copy()

    plot_df = plot_df.sort_values(
        by="importance_percent",
        ascending=False,
    ).head(top_n)

    plot_df = plot_df.sort_values(
        by="importance_percent",
        ascending=True,
    )

    labels = (
        plot_df["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(labels, plot_df["importance_percent"])

    plt.title("Importancia de variables sin xG previo")
    plt.xlabel("Importancia relativa (%)")
    plt.ylabel("Variable")

    for index, value in enumerate(plot_df["importance_percent"]):
        plt.text(
            value,
            index,
            f" {value:.2f}%",
            va="center",
        )

    save_current_figure(output_path)


def build_figure_index() -> None:
    """
    Crea un índice Markdown con las figuras generadas.
    """
    lines = []

    lines.append("# Figuras de resultados")
    lines.append("")
    lines.append("Este archivo resume las figuras generadas automáticamente.")
    lines.append("")

    figures = [
        (
            "Comparación de MAE por experimento",
            "figures/experiment_mae.png",
        ),
        (
            "Comparación de R² por experimento",
            "figures/experiment_r2.png",
        ),
        (
            "xG_90 real vs predicho en LaLiga 2025-2026",
            "figures/laliga_2025_2026_actual_vs_predicted.png",
        ),
        (
            "Distribución de residuos en LaLiga 2025-2026",
            "figures/laliga_2025_2026_residuals.png",
        ),
        (
            "Error absoluto medio por posición F/M",
            "figures/laliga_2025_2026_error_by_position.png",
        ),
        (
            "Importancia de variables sin xG previo",
            "figures/temporal_feature_importance_without_previous_xg.png",
        ),
    ]

    for title, path in figures:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{title}]({path})")
        lines.append("")

    output_path = PROJECT_ROOT / "reports" / "17_result_figures_index.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    experiment_summary_df = pd.read_csv(EXPERIMENT_SUMMARY_PATH)
    laliga_evaluation_df = pd.read_csv(LALIGA_EVALUATION_PATH)
    laliga_fm_evaluation_df = pd.read_csv(LALIGA_FM_EVALUATION_PATH)
    feature_importance_df = pd.read_csv(TEMPORAL_FEATURE_IMPORTANCE_PATH)

    plot_experiment_metric(
        summary_df=experiment_summary_df,
        metric_column="mae",
        output_path=FIGURES_DIR / "experiment_mae.png",
        title="MAE por experimento",
        x_label="MAE",
    )

    plot_experiment_metric(
        summary_df=experiment_summary_df,
        metric_column="r2",
        output_path=FIGURES_DIR / "experiment_r2.png",
        title="R² por experimento",
        x_label="R²",
    )

    plot_actual_vs_predicted(
        evaluation_df=laliga_evaluation_df,
        output_path=FIGURES_DIR / "laliga_2025_2026_actual_vs_predicted.png",
    )

    plot_residuals(
        evaluation_df=laliga_evaluation_df,
        output_path=FIGURES_DIR / "laliga_2025_2026_residuals.png",
    )

    plot_error_by_position(
        fm_df=laliga_fm_evaluation_df,
        output_path=FIGURES_DIR / "laliga_2025_2026_error_by_position.png",
    )

    plot_feature_importance_without_previous_xg(
        feature_importance_df=feature_importance_df,
        output_path=FIGURES_DIR / "temporal_feature_importance_without_previous_xg.png",
        top_n=15,
    )

    build_figure_index()

    print("Figuras de resultados generadas correctamente.")
    print("")
    print(f"Carpeta de figuras: {FIGURES_DIR}")
    print("")
    print("Archivos generados:")
    print("- experiment_mae.png")
    print("- experiment_r2.png")
    print("- laliga_2025_2026_actual_vs_predicted.png")
    print("- laliga_2025_2026_residuals.png")
    print("- laliga_2025_2026_error_by_position.png")
    print("- temporal_feature_importance_without_previous_xg.png")
    print("- reports/17_result_figures_index.md")


if __name__ == "__main__":
    main()