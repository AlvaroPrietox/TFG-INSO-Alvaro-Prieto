from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


TEMPORAL_MODEL_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "temporal_model_comparison.csv"
)

EXTERNAL_MODEL_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison.csv"
)

EXTERNAL_MODEL_COMPARISON_FM_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_f_m.csv"
)

EXTERNAL_MODEL_COMPARISON_BY_RANGE_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_by_xg_range.csv"
)

MULTI_METRIC_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_comparison.csv"
)

MULTI_METRIC_COMPARISON_FM_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_comparison_f_m.csv"
)

CURRENT_PROSPECTIVE_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_player_predictions_final_ridge.csv"
)

CURRENT_PROSPECTIVE_TOP_BY_METRIC_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_top_25_f_m_predictions_by_metric_final_ridge.csv"
)

FINAL_SUMMARY_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "final_experiment_summary.csv"
)

FINAL_MULTI_METRIC_SUMMARY_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "final_multi_metric_summary.csv"
)

FINAL_PROSPECTIVE_SUMMARY_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "final_prospective_prediction_summary.csv"
)

REPORT_OUTPUT_PATH = (
    REPORTS_DIR / "23_final_experiment_summary.md"
)


FINAL_MODEL_NAME = "Ridge"
FINAL_FEATURE_SET = "without_previous_xg"
INPUT_SEASON = "2024-2025"
TARGET_SEASON = "2025-2026"

METRIC_CONFIG = {
    "xG_90": {
        "label": "Goles esperados",
        "predicted_next_column": "predicted_next_xG_90",
    },
    "goals_90": {
        "label": "Goles",
        "predicted_next_column": "predicted_next_goals_90",
    },
    "assists_90": {
        "label": "Asistencias",
        "predicted_next_column": "predicted_next_assists_90",
    },
    "xA_90": {
        "label": "Asistencias esperadas",
        "predicted_next_column": "predicted_next_xA_90",
    },
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> pd.DataFrame:
    """
    Lee un CSV requerido por el resumen final.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el archivo necesario: {path}"
        )

    return pd.read_csv(path)


def round_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redondea columnas métricas para hacer los CSV e informes más legibles.
    """
    rounded_df = df.copy()

    metric_columns = [
        "mae",
        "r2",
        "mean_signed_error",
        "actual_xg_90_mean",
        "predicted_xg_90_mean",
        "prediction_mean",
        "prediction_max",
        "prediction_min",
    ]

    for column in metric_columns:
        if column in rounded_df.columns:
            rounded_df[column] = pd.to_numeric(
                rounded_df[column],
                errors="coerce",
            ).round(4)

    return rounded_df


def get_count_value(row: pd.Series) -> int | None:
    """
    Recupera el tamaño muestral de una fila, independientemente del nombre de
    columna usado por cada experimento.
    """
    candidate_columns = [
        "test_rows",
        "matched_players",
        "matched_players_f_m",
        "n_players",
        "players_evaluated",
    ]

    for column in candidate_columns:
        if column in row.index and pd.notna(row[column]):
            return int(row[column])

    return None


def add_rank_by_mae(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade ranking por MAE dentro de cada bloque de comparación.
    """
    ranked_df = df.copy()

    if "mae" not in ranked_df.columns:
        ranked_df["rank_by_mae"] = None
        return ranked_df

    ranked_df["rank_by_mae"] = (
        ranked_df["mae"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return ranked_df


def to_markdown_table(df: pd.DataFrame) -> str:
    """
    Convierte un DataFrame a tabla Markdown tras redondear métricas.
    """
    if df.empty:
        return "No hay datos disponibles."

    return round_metric_columns(df).to_markdown(index=False)


# ---------------------------------------------------------------------------
# Construcción de resúmenes tabulares
# ---------------------------------------------------------------------------


def build_model_selection_summary(
    temporal_df: pd.DataFrame,
    external_df: pd.DataFrame,
    external_fm_df: pd.DataFrame,
    range_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una tabla final de selección de modelo centrada en xG_90.
    """
    rows = []

    temporal_ranked = add_rank_by_mae(temporal_df)

    for _, row in temporal_ranked.iterrows():
        rows.append(
            {
                "summary_section": "internal_temporal_xG_90",
                "metric_key": "xG_90",
                "metric_label": "Goles esperados",
                "scope": "validación temporal interna",
                "model": row.get("model"),
                "feature_set": row.get("feature_set", FINAL_FEATURE_SET),
                "n_players": get_count_value(row),
                "mae": row.get("mae"),
                "r2": row.get("r2"),
                "rank_by_mae": row.get("rank_by_mae"),
                "input_season": row.get("input_season", "2023-2024"),
                "target_season": row.get("target_season", "2024-2025"),
                "notes": "Comparación interna de modelos para xG_90.",
            }
        )

    external_ranked = add_rank_by_mae(external_df)

    for _, row in external_ranked.iterrows():
        rows.append(
            {
                "summary_section": "external_global_xG_90",
                "metric_key": "xG_90",
                "metric_label": "Goles esperados",
                "scope": "evaluación externa global",
                "model": row.get("model"),
                "feature_set": row.get("feature_set", FINAL_FEATURE_SET),
                "n_players": get_count_value(row),
                "mae": row.get("mae"),
                "r2": row.get("r2"),
                "rank_by_mae": row.get("rank_by_mae"),
                "input_season": row.get("input_season", INPUT_SEASON),
                "target_season": row.get("target_season", TARGET_SEASON),
                "notes": "Evaluación externa 2025-2026 para xG_90.",
            }
        )

    external_fm_ranked = add_rank_by_mae(external_fm_df)

    for _, row in external_fm_ranked.iterrows():
        rows.append(
            {
                "summary_section": "external_f_m_xG_90",
                "metric_key": "xG_90",
                "metric_label": "Goles esperados",
                "scope": "evaluación externa F/M",
                "model": row.get("model"),
                "feature_set": row.get("feature_set", FINAL_FEATURE_SET),
                "n_players": get_count_value(row),
                "mae": row.get("mae"),
                "r2": row.get("r2"),
                "rank_by_mae": row.get("rank_by_mae"),
                "input_season": row.get("input_season", INPUT_SEASON),
                "target_season": row.get("target_season", TARGET_SEASON),
                "notes": "Evaluación externa 2025-2026 restringida a delanteros y centrocampistas.",
            }
        )

    high_xg_df = range_df[range_df["xg_range"] == "alto_>=0.50"].copy()
    high_xg_ranked = add_rank_by_mae(high_xg_df)

    for _, row in high_xg_ranked.iterrows():
        rows.append(
            {
                "summary_section": "external_high_xG_90_range",
                "metric_key": "xG_90",
                "metric_label": "Goles esperados",
                "scope": "jugadores con xG_90 real >= 0.50",
                "model": row.get("model"),
                "feature_set": FINAL_FEATURE_SET,
                "n_players": get_count_value(row),
                "mae": row.get("mae"),
                "r2": row.get("r2"),
                "mean_signed_error": row.get("mean_signed_error"),
                "rank_by_mae": row.get("rank_by_mae"),
                "input_season": INPUT_SEASON,
                "target_season": TARGET_SEASON,
                "notes": "Análisis complementario en perfiles ofensivos de alto xG_90.",
            }
        )

    summary_df = pd.DataFrame(rows)

    return round_metric_columns(summary_df)


def build_multi_metric_summary(
    multi_metric_df: pd.DataFrame,
    multi_metric_fm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una tabla resumen por métrica para el modelo final Ridge.
    """
    rows = []

    for _, row in multi_metric_df.iterrows():
        rows.append(
            {
                "summary_section": "external_multi_metric_global",
                "metric_key": row.get("metric_key"),
                "metric_label": row.get("metric_label"),
                "scope": "evaluación externa global",
                "model": row.get("model", FINAL_MODEL_NAME),
                "n_players": row.get("matched_players"),
                "mae": row.get("mae"),
                "r2": row.get("r2"),
                "input_season": row.get("input_season", INPUT_SEASON),
                "target_season": row.get("target_season", TARGET_SEASON),
            }
        )

    for _, row in multi_metric_fm_df.iterrows():
        rows.append(
            {
                "summary_section": "external_multi_metric_f_m",
                "metric_key": row.get("metric_key"),
                "metric_label": row.get("metric_label"),
                "scope": "evaluación externa F/M",
                "model": row.get("model", FINAL_MODEL_NAME),
                "n_players": row.get("matched_players_f_m"),
                "mae": row.get("mae"),
                "r2": row.get("r2"),
                "input_season": row.get("input_season", INPUT_SEASON),
                "target_season": row.get("target_season", TARGET_SEASON),
            }
        )

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df.sort_values(
        by=["summary_section", "metric_key"],
        ascending=True,
    ).reset_index(drop=True)

    return round_metric_columns(summary_df)


def build_prospective_summary(
    current_predictions_df: pd.DataFrame,
    top_by_metric_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resume las predicciones prospectivas actuales generadas por el script 22.
    """
    rows = []

    f_m_df = current_predictions_df[
        current_predictions_df["position_main"].isin(["F", "M"])
    ].copy()

    for metric_key, config in METRIC_CONFIG.items():
        predicted_column = config["predicted_next_column"]

        metric_top_df = top_by_metric_df[
            top_by_metric_df["metric_key"] == metric_key
        ].copy()

        if not metric_top_df.empty:
            top_player_row = metric_top_df.sort_values(
                by="predicted_next_value",
                ascending=False,
            ).iloc[0]

            top_player_name = top_player_row.get("player_name")
            top_player_team = top_player_row.get("team_title")
            top_player_prediction = top_player_row.get("predicted_next_value")
        else:
            top_player_name = None
            top_player_team = None
            top_player_prediction = None

        rows.append(
            {
                "metric_key": metric_key,
                "metric_label": config["label"],
                "model": FINAL_MODEL_NAME,
                "players_evaluated": int(len(current_predictions_df)),
                "players_f_m_evaluated": int(len(f_m_df)),
                "prediction_mean": float(current_predictions_df[predicted_column].mean()),
                "prediction_max": float(current_predictions_df[predicted_column].max()),
                "prediction_min": float(current_predictions_df[predicted_column].min()),
                "top_player": top_player_name,
                "top_player_team": top_player_team,
                "top_player_prediction": top_player_prediction,
                "notes": "Predicción prospectiva no evaluada todavía con valor real futuro.",
            }
        )

    summary_df = pd.DataFrame(rows)

    return round_metric_columns(summary_df)


# ---------------------------------------------------------------------------
# Informe Markdown
# ---------------------------------------------------------------------------


def build_markdown_report(
    final_summary_df: pd.DataFrame,
    multi_metric_summary_df: pd.DataFrame,
    prospective_summary_df: pd.DataFrame,
    external_df: pd.DataFrame,
    external_fm_df: pd.DataFrame,
    multi_metric_df: pd.DataFrame,
    multi_metric_fm_df: pd.DataFrame,
    range_df: pd.DataFrame,
) -> str:
    """
    Construye el informe final del experimento.
    """
    best_external = external_df.sort_values(by="mae", ascending=True).iloc[0]
    best_external_fm = external_fm_df.sort_values(by="mae", ascending=True).iloc[0]

    high_xg_df = range_df[range_df["xg_range"] == "alto_>=0.50"].copy()
    best_high_xg = high_xg_df.sort_values(by="mae", ascending=True).iloc[0]

    ridge_xg_global = multi_metric_df[
        multi_metric_df["metric_key"] == "xG_90"
    ].iloc[0]

    ridge_xg_fm = multi_metric_fm_df[
        multi_metric_fm_df["metric_key"] == "xG_90"
    ].iloc[0]

    weakest_metric_global = multi_metric_df.sort_values(
        by="r2",
        ascending=True,
    ).iloc[0]

    strongest_metric_global = multi_metric_df.sort_values(
        by="r2",
        ascending=False,
    ).iloc[0]

    lines = []

    lines.append("# Resumen final de experimentos")
    lines.append("")

    lines.append("## Objetivo del resumen")
    lines.append("")
    lines.append(
        "Este informe sintetiza la versión final del sistema de predicción de "
        "rendimiento ofensivo. El objetivo principal es evaluar predicciones "
        "2025-2026 generadas a partir de información 2024-2025 y compararlas "
        "con valores reales observados."
    )
    lines.append("")
    lines.append(
        "La métrica principal de selección del modelo es `xG_90`. Sobre esa "
        "métrica se comparan varios algoritmos. Una vez seleccionado Ridge, "
        "el sistema se amplía a un enfoque multi-métrica con `xG_90`, "
        "`goals_90`, `assists_90` y `xA_90`."
    )
    lines.append("")

    lines.append("## Selección del modelo principal para xG_90")
    lines.append("")
    lines.append(
        f"En la evaluación externa global 2025-2026, el mejor modelo para "
        f"`xG_90` es `{best_external['model']}`, con MAE = "
        f"{best_external['mae']:.4f} y R² = {best_external['r2']:.4f}."
    )
    lines.append("")
    lines.append(
        f"En el subconjunto F/M, el mejor modelo para `xG_90` es "
        f"`{best_external_fm['model']}`, con MAE = "
        f"{best_external_fm['mae']:.4f} y R² = "
        f"{best_external_fm['r2']:.4f}."
    )
    lines.append("")
    lines.append(
        f"Por tanto, `{FINAL_MODEL_NAME}` se mantiene como modelo final, ya "
        f"que ofrece el mejor rendimiento global y F/M para la métrica "
        f"principal del sistema."
    )
    lines.append("")

    lines.append("## Resumen tabular de selección de modelo")
    lines.append("")
    lines.append(to_markdown_table(final_summary_df))
    lines.append("")

    lines.append("## Evaluación multi-métrica con Ridge")
    lines.append("")
    lines.append(
        f"En la evaluación multi-métrica global, la métrica con mayor R² es "
        f"`{strongest_metric_global['metric_key']}` "
        f"({strongest_metric_global['metric_label']}), con R² = "
        f"{strongest_metric_global['r2']:.4f}. La métrica más difícil según "
        f"R² es `{weakest_metric_global['metric_key']}` "
        f"({weakest_metric_global['metric_label']}), con R² = "
        f"{weakest_metric_global['r2']:.4f}."
    )
    lines.append("")
    lines.append(
        f"Para la métrica principal `xG_90`, Ridge obtiene MAE = "
        f"{ridge_xg_global['mae']:.4f} y R² = {ridge_xg_global['r2']:.4f} "
        f"en evaluación global, y MAE = {ridge_xg_fm['mae']:.4f} y R² = "
        f"{ridge_xg_fm['r2']:.4f} en F/M."
    )
    lines.append("")
    lines.append(to_markdown_table(multi_metric_summary_df))
    lines.append("")

    lines.append("## Predicciones prospectivas multi-métrica")
    lines.append("")
    lines.append(
        "Además de la evaluación externa, el sistema genera una aplicación "
        "prospectiva sobre jugadores actuales. Estas predicciones no forman "
        "parte de la validación principal porque todavía no existe un valor "
        "real futuro contra el que compararlas."
    )
    lines.append("")
    lines.append(to_markdown_table(prospective_summary_df))
    lines.append("")

    lines.append("## Análisis complementario de alto xG_90")
    lines.append("")
    lines.append(
        f"En el grupo de jugadores con `xG_90` real igual o superior a 0.50, "
        f"el menor MAE lo obtiene `{best_high_xg['model']}`, con MAE = "
        f"{best_high_xg['mae']:.4f}. Este resultado se interpreta como "
        f"complementario, porque el subconjunto de jugadores de alto volumen "
        f"ofensivo tiene menor tamaño muestral y mayor variabilidad."
    )
    lines.append("")

    lines.append("## Archivos finales generados")
    lines.append("")
    lines.append("- `data/processed/final_experiment_summary.csv`")
    lines.append("- `data/processed/final_multi_metric_summary.csv`")
    lines.append("- `data/processed/final_prospective_prediction_summary.csv`")
    lines.append("- `data/processed/external_2025_2026_multi_metric_predictions.csv`")
    lines.append("- `data/processed/laliga_current_player_predictions_final_ridge.csv`")
    lines.append("- `models/ridge_temporal_multi_metric_metadata.json`")
    lines.append("")

    lines.append("## Conclusión")
    lines.append("")
    lines.append(
        "La versión final del sistema queda estructurada como un modelo "
        "Ridge multi-métrica. `xG_90` se mantiene como variable principal "
        "por su estabilidad y por el rendimiento obtenido en evaluación "
        "externa. Las métricas `goals_90`, `assists_90` y `xA_90` amplían "
        "la utilidad analítica del sistema al incorporar producción real, "
        "asistencias reales y creación esperada de ocasiones."
    )
    lines.append("")
    lines.append(
        "Los resultados muestran que las métricas esperadas y la producción "
        "goleadora son más predecibles que las asistencias reales, lo cual es "
        "coherente con la dependencia de las asistencias respecto al acierto "
        "finalizador de los compañeros y al contexto colectivo."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    temporal_df = read_csv(TEMPORAL_MODEL_COMPARISON_PATH)
    external_df = read_csv(EXTERNAL_MODEL_COMPARISON_PATH)
    external_fm_df = read_csv(EXTERNAL_MODEL_COMPARISON_FM_PATH)
    range_df = read_csv(EXTERNAL_MODEL_COMPARISON_BY_RANGE_PATH)
    multi_metric_df = read_csv(MULTI_METRIC_COMPARISON_PATH)
    multi_metric_fm_df = read_csv(MULTI_METRIC_COMPARISON_FM_PATH)
    current_predictions_df = read_csv(CURRENT_PROSPECTIVE_PREDICTIONS_PATH)
    top_by_metric_df = read_csv(CURRENT_PROSPECTIVE_TOP_BY_METRIC_PATH)

    final_summary_df = build_model_selection_summary(
        temporal_df=temporal_df,
        external_df=external_df,
        external_fm_df=external_fm_df,
        range_df=range_df,
    )

    multi_metric_summary_df = build_multi_metric_summary(
        multi_metric_df=multi_metric_df,
        multi_metric_fm_df=multi_metric_fm_df,
    )

    prospective_summary_df = build_prospective_summary(
        current_predictions_df=current_predictions_df,
        top_by_metric_df=top_by_metric_df,
    )

    final_summary_df.to_csv(
        FINAL_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    multi_metric_summary_df.to_csv(
        FINAL_MULTI_METRIC_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    prospective_summary_df.to_csv(
        FINAL_PROSPECTIVE_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    report = build_markdown_report(
        final_summary_df=final_summary_df,
        multi_metric_summary_df=multi_metric_summary_df,
        prospective_summary_df=prospective_summary_df,
        external_df=external_df,
        external_fm_df=external_fm_df,
        multi_metric_df=multi_metric_df,
        multi_metric_fm_df=multi_metric_fm_df,
        range_df=range_df,
    )

    REPORT_OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Resumen final de experimentos actualizado correctamente.")
    print("")
    print("Resumen de selección de modelo:")
    print(final_summary_df)
    print("")
    print("Resumen multi-métrica:")
    print(multi_metric_summary_df)
    print("")
    print("Resumen prospectivo:")
    print(prospective_summary_df)
    print("")
    print(f"CSV final generado: {FINAL_SUMMARY_OUTPUT_PATH}")
    print(f"CSV multi-métrica generado: {FINAL_MULTI_METRIC_SUMMARY_OUTPUT_PATH}")
    print(f"CSV prospectivo generado: {FINAL_PROSPECTIVE_SUMMARY_OUTPUT_PATH}")
    print(f"Informe final generado: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
