from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation.csv"
)

FM_EVALUATION_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation_f_m.csv"
)

FM_SAMPLE_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_sample_f_m_25.csv"
)

REPORT_OUTPUT_PATH = (
    REPORTS_DIR / "15_laliga_2025_2026_prediction_evaluation_f_m.md"
)


ACTUAL_COLUMN = "actual_2025_2026_xG_90"
PREDICTED_COLUMN = "predicted_2025_2026_xG_90"


REPORT_COLUMNS = [
    "player_name",
    "position_main",
    "team_title",
    "target_team_title",
    "games",
    "time",
    "target_games",
    "target_time",
    PREDICTED_COLUMN,
    ACTUAL_COLUMN,
    "signed_error",
    "absolute_error",
]


def validate_input_columns(df: pd.DataFrame) -> None:
    """
    Comprueba que el dataset de evaluación contiene las columnas necesarias.
    """
    required_columns = [
        "player_name",
        "position_main",
        "team_title",
        "target_team_title",
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
            f"Faltan columnas necesarias para el análisis F/M: {missing_columns}"
        )


def filter_attacking_and_midfield_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra jugadores ofensivos y centrocampistas.

    Se conservan:
    - F: delanteros / atacantes.
    - M: centrocampistas.

    Este subconjunto es más apropiado para evaluar predicciones de xG_90,
    ya que la métrica tiene mayor relevancia futbolística en perfiles con
    participación ofensiva.
    """
    validate_input_columns(df)

    filtered_df = df[
        df["position_main"].isin(["F", "M"])
    ].copy()

    filtered_df = filtered_df.sort_values(
        by="absolute_error",
        ascending=True,
    ).reset_index(drop=True)

    return filtered_df


def compute_regression_metrics(df: pd.DataFrame) -> dict:
    """
    Calcula MAE y R² sobre un subconjunto de jugadores.
    """
    if df.empty:
        return {
            "n_players": 0,
            "mae": None,
            "r2": None,
        }

    mae = mean_absolute_error(
        df[ACTUAL_COLUMN],
        df[PREDICTED_COLUMN],
    )

    r2 = r2_score(
        df[ACTUAL_COLUMN],
        df[PREDICTED_COLUMN],
    )

    return {
        "n_players": int(len(df)),
        "mae": float(mae),
        "r2": float(r2),
    }


def build_metrics_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas separadas para F y M.
    """
    rows = []

    for position in ["F", "M"]:
        position_df = df[df["position_main"] == position].copy()
        metrics = compute_regression_metrics(position_df)

        rows.append(
            {
                "position_main": position,
                "n_players": metrics["n_players"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

    return pd.DataFrame(rows)


def select_representative_sample(
    df: pd.DataFrame,
    sample_size: int = 25,
) -> pd.DataFrame:
    """
    Selecciona una muestra representativa con:
    - mejores aciertos,
    - errores intermedios,
    - mayores errores.
    """
    if len(df) <= sample_size:
        return df.copy()

    sorted_df = df.sort_values(
        by="absolute_error",
        ascending=True,
    ).reset_index(drop=True)

    n_best = sample_size // 3
    n_middle = sample_size // 3
    n_worst = sample_size - n_best - n_middle

    best_cases = sorted_df.head(n_best)

    middle_start = max((len(sorted_df) // 2) - (n_middle // 2), 0)
    middle_cases = sorted_df.iloc[
        middle_start: middle_start + n_middle
    ]

    worst_cases = sorted_df.tail(n_worst)

    sample_df = pd.concat(
        [
            best_cases,
            middle_cases,
            worst_cases,
        ],
        ignore_index=True,
    )

    sample_df = sample_df.sort_values(
        by="absolute_error",
        ascending=True,
    ).reset_index(drop=True)

    return sample_df


def round_report_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redondea columnas numéricas para mejorar la lectura del informe.
    """
    report_df = df.copy()

    numeric_columns = [
        PREDICTED_COLUMN,
        ACTUAL_COLUMN,
        "signed_error",
        "absolute_error",
    ]

    for column in numeric_columns:
        if column in report_df.columns:
            report_df[column] = report_df[column].round(4)

    return report_df


def build_markdown_report(
    fm_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    total_players: int,
    metrics: dict,
    metrics_by_position_df: pd.DataFrame,
) -> str:
    """
    Construye un informe Markdown del análisis específico F/M.
    """
    best_cases = fm_df.sort_values(
        by="absolute_error",
        ascending=True,
    ).head(10)

    worst_cases = fm_df.sort_values(
        by="absolute_error",
        ascending=False,
    ).head(10)

    overestimations = fm_df[fm_df["signed_error"] > 0]
    underestimations = fm_df[fm_df["signed_error"] < 0]

    biggest_overestimations = overestimations.sort_values(
        by="signed_error",
        ascending=False,
    ).head(10)

    biggest_underestimations = underestimations.sort_values(
        by="signed_error",
        ascending=True,
    ).head(10)

    metrics_by_position_report = metrics_by_position_df.copy()
    metrics_by_position_report["mae"] = metrics_by_position_report["mae"].round(4)
    metrics_by_position_report["r2"] = metrics_by_position_report["r2"].round(4)

    lines = []

    lines.append("# Evaluación 2025-2026 en jugadores F/M")
    lines.append("")

    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este informe analiza el rendimiento del modelo únicamente en jugadores "
        "con posición principal `F` o `M`. Este subconjunto es especialmente "
        "relevante porque la variable objetivo, `xG_90`, representa una métrica "
        "ofensiva cuya interpretación es más significativa en delanteros, "
        "atacantes y centrocampistas."
    )
    lines.append("")

    lines.append("## Cobertura")
    lines.append("")
    lines.append(f"- Jugadores emparejados totales: {total_players}")
    lines.append(f"- Jugadores F/M analizados: {metrics['n_players']}")
    lines.append("")

    lines.append("## Métricas globales F/M")
    lines.append("")
    lines.append(f"- MAE F/M: {metrics['mae']:.4f}")
    lines.append(f"- R² F/M: {metrics['r2']:.4f}")
    lines.append("")

    lines.append("## Métricas por posición")
    lines.append("")
    lines.append(metrics_by_position_report.to_markdown(index=False))
    lines.append("")

    lines.append("## Muestra representativa F/M")
    lines.append("")
    lines.append(
        round_report_dataframe(sample_df[REPORT_COLUMNS]).to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Diez mejores aciertos F/M")
    lines.append("")
    lines.append(
        round_report_dataframe(best_cases[REPORT_COLUMNS]).to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Diez mayores errores F/M")
    lines.append("")
    lines.append(
        round_report_dataframe(worst_cases[REPORT_COLUMNS]).to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Mayores sobreestimaciones")
    lines.append("")
    lines.append(
        round_report_dataframe(
            biggest_overestimations[REPORT_COLUMNS]
        ).to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Mayores infraestimaciones")
    lines.append("")
    lines.append(
        round_report_dataframe(
            biggest_underestimations[REPORT_COLUMNS]
        ).to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        "El análisis F/M permite evaluar el modelo en el subconjunto de jugadores "
        "donde el `xG_90` tiene mayor significado deportivo. Si el rendimiento "
        "se mantiene próximo al obtenido en la evaluación global, se refuerza "
        "la validez del modelo para perfiles ofensivos y creativos."
    )
    lines.append("")
    lines.append(
        "Los mayores errores deben analizarse como casos de cambio contextual. "
        "Una sobreestimación puede indicar que el jugador no mantuvo el volumen "
        "ofensivo esperado, mientras que una infraestimación puede reflejar una "
        "mejora de rol, mayor protagonismo ofensivo o cambio táctico favorable "
        "durante la temporada 2025-2026."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_df = pd.read_csv(INPUT_PATH)

    fm_df = filter_attacking_and_midfield_players(evaluation_df)

    metrics = compute_regression_metrics(fm_df)
    metrics_by_position_df = build_metrics_by_position(fm_df)

    sample_df = select_representative_sample(
        fm_df,
        sample_size=25,
    )

    fm_df.to_csv(
        FM_EVALUATION_OUTPUT_PATH,
        index=False,
    )

    sample_df.to_csv(
        FM_SAMPLE_OUTPUT_PATH,
        index=False,
    )

    report = build_markdown_report(
        fm_df=fm_df,
        sample_df=sample_df,
        total_players=len(evaluation_df),
        metrics=metrics,
        metrics_by_position_df=metrics_by_position_df,
    )

    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print("Análisis F/M 2025-2026 completado.")
    print("")
    print(f"Jugadores emparejados totales: {len(evaluation_df)}")
    print(f"Jugadores F/M analizados: {metrics['n_players']}")
    print("")
    print("Métricas F/M:")
    print(
        {
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        }
    )
    print("")
    print("Métricas por posición:")
    print(metrics_by_position_df)
    print("")
    print(f"Evaluación F/M guardada en: {FM_EVALUATION_OUTPUT_PATH}")
    print(f"Muestra F/M guardada en: {FM_SAMPLE_OUTPUT_PATH}")
    print(f"Informe F/M generado en: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()