from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TEMPORAL_EVALUATION_PATH = (
    PROCESSED_DATA_DIR / "player_temporal_evaluation.csv"
)

LALIGA_2025_2026_EVALUATION_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation.csv"
)

LALIGA_2025_2026_FM_EVALUATION_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation_f_m.csv"
)

OUTPUT_CSV_PATH = (
    PROCESSED_DATA_DIR / "experiment_summary.csv"
)

OUTPUT_REPORT_PATH = (
    REPORTS_DIR / "16_experiment_summary.md"
)


ACTUAL_2025_COLUMN = "actual_2025_2026_xG_90"
PREDICTED_2025_COLUMN = "predicted_2025_2026_xG_90"


def compute_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calcula el error absoluto medio.
    """
    return (actual - predicted).abs().mean()


def compute_r2(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calcula el coeficiente de determinación R².

    Se implementa manualmente para evitar depender de una importación adicional
    dentro del script de resumen.
    """
    residual_sum_of_squares = ((actual - predicted) ** 2).sum()
    total_sum_of_squares = ((actual - actual.mean()) ** 2).sum()

    if total_sum_of_squares == 0:
        return float("nan")

    return 1 - (residual_sum_of_squares / total_sum_of_squares)


def load_temporal_experiments() -> list[dict]:
    """
    Carga los resultados del modelo temporal interno.

    Este archivo contiene los experimentos:
    - full
    - without_previous_xg
    """
    temporal_df = pd.read_csv(TEMPORAL_EVALUATION_PATH)

    rows = []

    for _, row in temporal_df.iterrows():
        if row["model"] != "RandomForestRegressor":
            continue

        feature_set = row["feature_set"]

        if feature_set == "full":
            interpretation = (
                "Modelo temporal con todas las variables históricas, incluyendo "
                "xG_90 y npxG_90 de la temporada anterior. Representa el escenario "
                "con máxima información ofensiva previa."
            )
        elif feature_set == "without_previous_xg":
            interpretation = (
                "Modelo temporal sin xG_90 ni npxG_90 previos. Evalúa si variables "
                "como tiros, goles, asistencias, participación y posición permiten "
                "anticipar el rendimiento futuro sin usar directamente xG previo."
            )
        else:
            interpretation = "Experimento temporal de rendimiento individual."

        rows.append(
            {
                "experiment": f"temporal_{feature_set}",
                "evaluation_type": "validación temporal interna",
                "subset": "todos sin porteros",
                "model": row["model"],
                "n_train": int(row["train_rows"]),
                "n_test": int(row["test_rows"]),
                "mae": float(row["mae"]),
                "r2": float(row["r2"]),
                "test_period": f"{row['test_season']} → {row['test_next_season']}",
                "interpretation": interpretation,
            }
        )

    return rows


def load_laliga_2025_2026_global_experiment() -> dict:
    """
    Calcula métricas de la evaluación externa contra LaLiga 2025-2026.
    """
    evaluation_df = pd.read_csv(LALIGA_2025_2026_EVALUATION_PATH)

    mae = compute_mae(
        evaluation_df[ACTUAL_2025_COLUMN],
        evaluation_df[PREDICTED_2025_COLUMN],
    )

    r2 = compute_r2(
        evaluation_df[ACTUAL_2025_COLUMN],
        evaluation_df[PREDICTED_2025_COLUMN],
    )

    return {
        "experiment": "external_laliga_2025_2026_global",
        "evaluation_type": "evaluación externa",
        "subset": "jugadores emparejados sin porteros",
        "model": "RandomForestRegressor",
        "n_train": None,
        "n_test": int(len(evaluation_df)),
        "mae": float(mae),
        "r2": float(r2),
        "test_period": "2024-2025 → 2025-2026",
        "interpretation": (
            "Evaluación externa usando datos de 2024-2025 para predecir xG_90 "
            "real en LaLiga 2025-2026. La unión se realiza mediante nombre "
            "normalizado, por lo que debe interpretarse con cautela."
        ),
    }


def load_laliga_2025_2026_fm_experiment() -> list[dict]:
    """
    Calcula métricas de la evaluación externa F/M y por posición.
    """
    fm_df = pd.read_csv(LALIGA_2025_2026_FM_EVALUATION_PATH)

    rows = []

    global_mae = compute_mae(
        fm_df[ACTUAL_2025_COLUMN],
        fm_df[PREDICTED_2025_COLUMN],
    )

    global_r2 = compute_r2(
        fm_df[ACTUAL_2025_COLUMN],
        fm_df[PREDICTED_2025_COLUMN],
    )

    rows.append(
        {
            "experiment": "external_laliga_2025_2026_f_m",
            "evaluation_type": "evaluación externa",
            "subset": "F y M",
            "model": "RandomForestRegressor",
            "n_train": None,
            "n_test": int(len(fm_df)),
            "mae": float(global_mae),
            "r2": float(global_r2),
            "test_period": "2024-2025 → 2025-2026",
            "interpretation": (
                "Evaluación externa restringida a delanteros, atacantes y "
                "centrocampistas. Es el subconjunto más relevante para analizar "
                "una métrica ofensiva como xG_90."
            ),
        }
    )

    for position_main in ["F", "M"]:
        position_df = fm_df[fm_df["position_main"] == position_main].copy()

        if position_df.empty:
            continue

        mae = compute_mae(
            position_df[ACTUAL_2025_COLUMN],
            position_df[PREDICTED_2025_COLUMN],
        )

        r2 = compute_r2(
            position_df[ACTUAL_2025_COLUMN],
            position_df[PREDICTED_2025_COLUMN],
        )

        if position_main == "F":
            subset_label = "solo F"
            interpretation = (
                "Evaluación externa restringida a delanteros y atacantes. "
                "Es el grupo con mayor variabilidad ofensiva, por lo que se "
                "espera un error superior."
            )
        else:
            subset_label = "solo M"
            interpretation = (
                "Evaluación externa restringida a centrocampistas. Presenta "
                "un comportamiento más estable que los delanteros en términos "
                "de xG_90."
            )

        rows.append(
            {
                "experiment": f"external_laliga_2025_2026_{position_main}",
                "evaluation_type": "evaluación externa por posición",
                "subset": subset_label,
                "model": "RandomForestRegressor",
                "n_train": None,
                "n_test": int(len(position_df)),
                "mae": float(mae),
                "r2": float(r2),
                "test_period": "2024-2025 → 2025-2026",
                "interpretation": interpretation,
            }
        )

    return rows


def build_experiment_summary() -> pd.DataFrame:
    """
    Construye la tabla resumen de experimentos del proyecto.
    """
    rows = []

    rows.extend(load_temporal_experiments())
    rows.append(load_laliga_2025_2026_global_experiment())
    rows.extend(load_laliga_2025_2026_fm_experiment())

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df[
        [
            "experiment",
            "evaluation_type",
            "subset",
            "model",
            "n_train",
            "n_test",
            "mae",
            "r2",
            "test_period",
            "interpretation",
        ]
    ].copy()

    summary_df = summary_df.sort_values(
        by=["evaluation_type", "experiment"],
        ascending=True,
    ).reset_index(drop=True)

    return summary_df


def build_markdown_report(summary_df: pd.DataFrame) -> str:
    """
    Construye el informe Markdown final de experimentos.
    """
    display_df = summary_df.copy()

    display_df["mae"] = display_df["mae"].round(4)
    display_df["r2"] = display_df["r2"].round(4)

    compact_columns = [
        "experiment",
        "evaluation_type",
        "subset",
        "n_train",
        "n_test",
        "mae",
        "r2",
        "test_period",
    ]

    lines = []

    lines.append("# Resumen final de experimentos")
    lines.append("")

    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este informe consolida los principales experimentos realizados para "
        "la predicción del rendimiento ofensivo futuro de jugadores de fútbol, "
        "utilizando `xG_90` de la temporada siguiente como variable objetivo."
    )
    lines.append("")

    lines.append("## Tabla resumen")
    lines.append("")
    lines.append(display_df[compact_columns].to_markdown(index=False))
    lines.append("")

    lines.append("## Interpretación por experimento")
    lines.append("")

    for _, row in display_df.iterrows():
        lines.append(f"### {row['experiment']}")
        lines.append("")
        lines.append(f"- Tipo de evaluación: {row['evaluation_type']}")
        lines.append(f"- Subconjunto: {row['subset']}")
        lines.append(f"- MAE: {row['mae']}")
        lines.append(f"- R²: {row['r2']}")
        lines.append(f"- Periodo de prueba: {row['test_period']}")
        lines.append("")
        lines.append(row["interpretation"])
        lines.append("")

    lines.append("## Conclusión general")
    lines.append("")
    lines.append(
        "Los resultados muestran que el modelo temporal mantiene una capacidad "
        "predictiva sólida tanto en validación interna como en una evaluación "
        "externa sobre LaLiga 2025-2026. La eliminación de variables directas "
        "de xG previo reduce ligeramente el rendimiento, pero el modelo sigue "
        "capturando señal mediante variables como volumen de tiro, goles por 90, "
        "participación y posición principal."
    )
    lines.append("")
    lines.append(
        "La evaluación específica sobre jugadores F/M es más exigente que la "
        "evaluación global, ya que excluye perfiles defensivos de bajo xG_90 y "
        "se centra en jugadores con mayor variabilidad ofensiva. Aun así, los "
        "resultados siguen siendo suficientemente consistentes para defender la "
        "utilidad del enfoque propuesto."
    )
    lines.append("")
    lines.append(
        "La principal limitación detectada es que el modelo tiende a comportarse "
        "de forma conservadora: predice bien perfiles estables, pero tiene más "
        "dificultad para anticipar saltos abruptos de rendimiento asociados a "
        "cambios de contexto, rol táctico, equipo, lesiones o variaciones en el "
        "modelo de juego."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = build_experiment_summary()

    summary_df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
    )

    report = build_markdown_report(summary_df)

    OUTPUT_REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Resumen final de experimentos generado correctamente.")
    print("")
    print(summary_df[[
        "experiment",
        "subset",
        "n_test",
        "mae",
        "r2",
        "test_period",
    ]])
    print("")
    print(f"CSV generado: {OUTPUT_CSV_PATH}")
    print(f"Informe generado: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()