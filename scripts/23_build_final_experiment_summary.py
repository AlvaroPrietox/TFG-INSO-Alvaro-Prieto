from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TEMPORAL_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "temporal_model_comparison.csv"
)

EXTERNAL_GLOBAL_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison.csv"
)

EXTERNAL_FM_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_f_m.csv"
)

EXTERNAL_RANGE_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_by_xg_range.csv"
)

FINAL_SUMMARY_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "final_experiment_summary.csv"
)

REPORT_OUTPUT_PATH = (
    REPORTS_DIR / "23_final_experiment_summary.md"
)


FINAL_MODEL_NAME = "Ridge"
FEATURE_SET_NAME = "without_previous_xg"
HIGH_XG_RANGE = "alto_>=0.50"


def validate_input_files() -> None:
    """
    Comprueba que existen todos los ficheros de resultados necesarios para
    construir el resumen experimental final.
    """
    required_paths = [
        TEMPORAL_COMPARISON_PATH,
        EXTERNAL_GLOBAL_COMPARISON_PATH,
        EXTERNAL_FM_COMPARISON_PATH,
        EXTERNAL_RANGE_COMPARISON_PATH,
    ]

    missing_paths = [
        path for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Faltan ficheros necesarios para construir el resumen final: "
            f"{missing_paths}"
        )


def round_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redondea columnas métricas para mejorar la legibilidad de tablas e informes.

    Convierte previamente los valores a numéricos para evitar errores cuando
    existen valores None en columnas como mean_signed_error.
    """
    rounded_df = df.copy()

    metric_columns = [
        "mae",
        "r2",
        "mean_signed_error",
        "actual_xg_90_mean",
        "predicted_xg_90_mean",
    ]

    for column in metric_columns:
        if column in rounded_df.columns:
            rounded_df[column] = pd.to_numeric(
                rounded_df[column],
                errors="coerce",
            ).round(4)

    return rounded_df


def build_temporal_summary(
    temporal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte la comparación temporal interna al formato común del resumen.
    """
    summary_df = temporal_df.copy()

    summary_df = summary_df.sort_values(
        by="mae",
        ascending=True,
    ).reset_index(drop=True)

    summary_df["rank_mae_in_scope"] = summary_df.index + 1

    output_df = pd.DataFrame(
        {
            "evaluation_scope": "temporal_internal",
            "subset": "todos_sin_porteros",
            "model": summary_df["model"],
            "feature_set": summary_df["feature_set"],
            "n_train": summary_df["train_rows"],
            "n_eval": summary_df["test_rows"],
            "mae": summary_df["mae"],
            "r2": summary_df["r2"],
            "mean_signed_error": None,
            "period": (
                summary_df["test_season"].astype(str)
                + " → "
                + summary_df["test_next_season"].astype(str)
            ),
            "rank_mae_in_scope": summary_df["rank_mae_in_scope"],
        }
    )

    return output_df


def build_external_global_summary(
    external_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte la comparación externa global al formato común del resumen.
    """
    summary_df = external_df.copy()

    summary_df = summary_df.sort_values(
        by="mae",
        ascending=True,
    ).reset_index(drop=True)

    summary_df["rank_mae_in_scope"] = summary_df.index + 1

    output_df = pd.DataFrame(
        {
            "evaluation_scope": "external_2025_2026",
            "subset": "jugadores_emparejados_sin_porteros",
            "model": summary_df["model"],
            "feature_set": summary_df["feature_set"],
            "n_train": summary_df["training_rows"],
            "n_eval": summary_df["matched_players"],
            "mae": summary_df["mae"],
            "r2": summary_df["r2"],
            "mean_signed_error": None,
            "period": (
                summary_df["input_season"].astype(str)
                + " → "
                + summary_df["target_season"].astype(str)
            ),
            "rank_mae_in_scope": summary_df["rank_mae_in_scope"],
        }
    )

    return output_df


def build_external_fm_summary(
    external_fm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte la comparación externa F/M al formato común del resumen.
    """
    summary_df = external_fm_df.copy()

    summary_df = summary_df.sort_values(
        by="mae",
        ascending=True,
    ).reset_index(drop=True)

    summary_df["rank_mae_in_scope"] = summary_df.index + 1

    output_df = pd.DataFrame(
        {
            "evaluation_scope": "external_2025_2026",
            "subset": "solo_F_M",
            "model": summary_df["model"],
            "feature_set": summary_df["feature_set"],
            "n_train": summary_df["training_rows"],
            "n_eval": summary_df["matched_players_f_m"],
            "mae": summary_df["mae"],
            "r2": summary_df["r2"],
            "mean_signed_error": None,
            "period": (
                summary_df["input_season"].astype(str)
                + " → "
                + summary_df["target_season"].astype(str)
            ),
            "rank_mae_in_scope": summary_df["rank_mae_in_scope"],
        }
    )

    return output_df


def build_external_high_xg_summary(
    external_range_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extrae el resumen específico para jugadores de alto xG_90 real.
    """
    high_xg_df = external_range_df[
        external_range_df["xg_range"] == HIGH_XG_RANGE
    ].copy()

    high_xg_df = high_xg_df.sort_values(
        by="mae",
        ascending=True,
    ).reset_index(drop=True)

    high_xg_df["rank_mae_in_scope"] = high_xg_df.index + 1

    output_df = pd.DataFrame(
        {
            "evaluation_scope": "external_2025_2026",
            "subset": HIGH_XG_RANGE,
            "model": high_xg_df["model"],
            "feature_set": FEATURE_SET_NAME,
            "n_train": None,
            "n_eval": high_xg_df["n_players"],
            "mae": high_xg_df["mae"],
            "r2": high_xg_df["r2"],
            "mean_signed_error": high_xg_df["mean_signed_error"],
            "period": "2024-2025 → 2025-2026",
            "rank_mae_in_scope": high_xg_df["rank_mae_in_scope"],
        }
    )

    return output_df


def build_final_summary_table(
    temporal_df: pd.DataFrame,
    external_global_df: pd.DataFrame,
    external_fm_df: pd.DataFrame,
    external_range_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una tabla común con los resultados principales del proyecto.
    """
    temporal_summary_df = build_temporal_summary(temporal_df)
    external_global_summary_df = build_external_global_summary(
        external_global_df
    )
    external_fm_summary_df = build_external_fm_summary(
        external_fm_df
    )
    external_high_xg_summary_df = build_external_high_xg_summary(
        external_range_df
    )

    final_summary_df = pd.concat(
        [
            temporal_summary_df,
            external_global_summary_df,
            external_fm_summary_df,
            external_high_xg_summary_df,
        ],
        ignore_index=True,
    )

    final_summary_df = final_summary_df.sort_values(
        by=[
            "evaluation_scope",
            "subset",
            "rank_mae_in_scope",
        ],
        ascending=True,
    ).reset_index(drop=True)

    return final_summary_df


def get_row(
    df: pd.DataFrame,
    evaluation_scope: str,
    subset: str,
    model: str,
) -> pd.Series:
    """
    Recupera una fila concreta del resumen final.
    """
    filtered_df = df[
        (df["evaluation_scope"] == evaluation_scope)
        & (df["subset"] == subset)
        & (df["model"] == model)
    ]

    if filtered_df.empty:
        raise ValueError(
            f"No se encontró la fila solicitada: "
            f"{evaluation_scope}, {subset}, {model}"
        )

    return filtered_df.iloc[0]


def get_best_row(
    df: pd.DataFrame,
    evaluation_scope: str,
    subset: str,
) -> pd.Series:
    """
    Recupera el mejor modelo por MAE dentro de un ámbito y subconjunto.
    """
    filtered_df = df[
        (df["evaluation_scope"] == evaluation_scope)
        & (df["subset"] == subset)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No se encontraron filas para: {evaluation_scope}, {subset}"
        )

    return filtered_df.sort_values(
        by="mae",
        ascending=True,
    ).iloc[0]


def build_markdown_report(
    final_summary_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    external_global_df: pd.DataFrame,
    external_fm_df: pd.DataFrame,
    external_range_df: pd.DataFrame,
) -> str:
    """
    Construye el informe Markdown final de resultados experimentales.
    """
    report_summary_df = round_metric_columns(final_summary_df)

    report_temporal_df = round_metric_columns(
        temporal_df[
            [
                "model",
                "feature_set",
                "train_rows",
                "test_rows",
                "mae",
                "r2",
                "test_season",
                "test_next_season",
            ]
        ].copy()
    )

    report_external_global_df = round_metric_columns(
        external_global_df[
            [
                "model",
                "feature_set",
                "training_rows",
                "matched_players",
                "mae",
                "r2",
                "input_season",
                "target_season",
            ]
        ].copy()
    )

    report_external_fm_df = round_metric_columns(
        external_fm_df[
            [
                "model",
                "feature_set",
                "training_rows",
                "matched_players_f_m",
                "mae",
                "r2",
                "input_season",
                "target_season",
            ]
        ].copy()
    )

    high_xg_df = external_range_df[
        external_range_df["xg_range"] == HIGH_XG_RANGE
    ].copy()

    report_high_xg_df = round_metric_columns(
        high_xg_df[
            [
                "model",
                "xg_range",
                "n_players",
                "actual_xg_90_mean",
                "predicted_xg_90_mean",
                "mae",
                "r2",
                "mean_signed_error",
                "overestimations",
                "underestimations",
            ]
        ].copy()
    )

    best_temporal_row = get_best_row(
        final_summary_df,
        evaluation_scope="temporal_internal",
        subset="todos_sin_porteros",
    )

    best_external_global_row = get_best_row(
        final_summary_df,
        evaluation_scope="external_2025_2026",
        subset="jugadores_emparejados_sin_porteros",
    )

    best_external_fm_row = get_best_row(
        final_summary_df,
        evaluation_scope="external_2025_2026",
        subset="solo_F_M",
    )

    best_high_xg_row = get_best_row(
        final_summary_df,
        evaluation_scope="external_2025_2026",
        subset=HIGH_XG_RANGE,
    )

    ridge_external_global_row = get_row(
        final_summary_df,
        evaluation_scope="external_2025_2026",
        subset="jugadores_emparejados_sin_porteros",
        model=FINAL_MODEL_NAME,
    )

    ridge_external_fm_row = get_row(
        final_summary_df,
        evaluation_scope="external_2025_2026",
        subset="solo_F_M",
        model=FINAL_MODEL_NAME,
    )

    lines = []

    lines.append("# Resumen final de experimentos")
    lines.append("")

    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este informe consolida los principales resultados experimentales del "
        "sistema de predicción de rendimiento ofensivo de jugadores. La "
        "variable objetivo es `xG_90` de la temporada siguiente, estimada a "
        "partir de métricas históricas de participación y producción ofensiva."
    )
    lines.append("")
    lines.append(
        "El resumen integra la validación temporal interna, la evaluación "
        "externa sobre LaLiga 2025-2026, el subconjunto de delanteros y "
        "centrocampistas, y el análisis específico de jugadores con alto "
        "`xG_90` real."
    )
    lines.append("")

    lines.append("## Tabla sintética final")
    lines.append("")
    lines.append(
        report_summary_df.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Comparación temporal interna")
    lines.append("")
    lines.append(
        "La validación temporal interna reserva el último par temporal "
        "disponible como conjunto de test, manteniendo una separación "
        "cronológica entre entrenamiento y evaluación."
    )
    lines.append("")
    lines.append(
        report_temporal_df.to_markdown(index=False)
    )
    lines.append("")
    lines.append(
        f"El mejor modelo en validación temporal interna es "
        f"`{best_temporal_row['model']}`, con MAE = "
        f"{best_temporal_row['mae']:.4f} y R² = "
        f"{best_temporal_row['r2']:.4f}."
    )
    lines.append("")

    lines.append("## Evaluación externa global 2025-2026")
    lines.append("")
    lines.append(
        "La evaluación externa utiliza jugadores de 2024-2025 como entrada y "
        "los compara con su `xG_90` real observado en la temporada 2025-2026. "
        "Esta prueba es especialmente relevante porque actúa como validación "
        "fuera del conjunto temporal empleado durante el desarrollo inicial."
    )
    lines.append("")
    lines.append(
        report_external_global_df.to_markdown(index=False)
    )
    lines.append("")
    lines.append(
        f"El mejor modelo en la evaluación externa global es "
        f"`{best_external_global_row['model']}`, con MAE = "
        f"{best_external_global_row['mae']:.4f} y R² = "
        f"{best_external_global_row['r2']:.4f}."
    )
    lines.append("")

    lines.append("## Evaluación externa en F/M")
    lines.append("")
    lines.append(
        "El subconjunto F/M concentra delanteros y centrocampistas, por lo que "
        "está más alineado con el análisis de producción ofensiva que el "
        "conjunto global de jugadores no porteros."
    )
    lines.append("")
    lines.append(
        report_external_fm_df.to_markdown(index=False)
    )
    lines.append("")
    lines.append(
        f"El mejor modelo en el subconjunto F/M es "
        f"`{best_external_fm_row['model']}`, con MAE = "
        f"{best_external_fm_row['mae']:.4f} y R² = "
        f"{best_external_fm_row['r2']:.4f}."
    )
    lines.append("")

    lines.append("## Análisis del rango alto de xG_90")
    lines.append("")
    lines.append(
        "Para estudiar el comportamiento del modelo en perfiles de alta "
        "producción ofensiva, se analiza de forma separada el grupo de "
        "jugadores con `xG_90` real igual o superior a 0.50."
    )
    lines.append("")
    lines.append(
        report_high_xg_df.to_markdown(index=False)
    )
    lines.append("")
    lines.append(
        f"En el rango alto de `xG_90`, el menor MAE corresponde a "
        f"`{best_high_xg_row['model']}`, con MAE = "
        f"{best_high_xg_row['mae']:.4f}."
    )
    lines.append("")
    lines.append(
        "No obstante, este subconjunto contiene menos observaciones que la "
        "evaluación global, por lo que se usa principalmente como análisis de "
        "comportamiento del error y no como único criterio de selección del "
        "modelo final."
    )
    lines.append("")

    lines.append("## Modelo final seleccionado")
    lines.append("")
    lines.append(
        f"El modelo final seleccionado es `{FINAL_MODEL_NAME}` con el conjunto "
        f"de variables `{FEATURE_SET_NAME}`."
    )
    lines.append("")
    lines.append(
        f"En evaluación externa global, `{FINAL_MODEL_NAME}` obtiene MAE = "
        f"{ridge_external_global_row['mae']:.4f} y R² = "
        f"{ridge_external_global_row['r2']:.4f}."
    )
    lines.append("")
    lines.append(
        f"En el subconjunto F/M, `{FINAL_MODEL_NAME}` obtiene MAE = "
        f"{ridge_external_fm_row['mae']:.4f} y R² = "
        f"{ridge_external_fm_row['r2']:.4f}."
    )
    lines.append("")
    lines.append(
        "La elección de Ridge se justifica por tres motivos principales: "
        "presenta el mejor rendimiento externo global, también lidera el "
        "subconjunto F/M y ofrece una estructura más simple e interpretable "
        "que los modelos de ensamblado basados en árboles."
    )
    lines.append("")

    lines.append("## Conclusión experimental")
    lines.append("")
    lines.append(
        "Los resultados muestran que el rendimiento ofensivo futuro de los "
        "jugadores puede aproximarse con una precisión razonable a partir de "
        "variables de producción ofensiva, participación y posición. Aunque "
        "los modelos basados en árboles ofrecen resultados competitivos, Ridge "
        "presenta la mejor generalización externa en este experimento."
    )
    lines.append("")
    lines.append(
        "El análisis por rangos confirma que los jugadores de alto `xG_90` "
        "siguen siendo el grupo más difícil de predecir, debido a la mayor "
        "variabilidad de los perfiles ofensivos extremos. Por tanto, el modelo "
        "final debe interpretarse como una herramienta de apoyo analítico y no "
        "como una predicción determinista del rendimiento individual."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    validate_input_files()

    temporal_df = pd.read_csv(TEMPORAL_COMPARISON_PATH)
    external_global_df = pd.read_csv(EXTERNAL_GLOBAL_COMPARISON_PATH)
    external_fm_df = pd.read_csv(EXTERNAL_FM_COMPARISON_PATH)
    external_range_df = pd.read_csv(EXTERNAL_RANGE_COMPARISON_PATH)

    final_summary_df = build_final_summary_table(
        temporal_df=temporal_df,
        external_global_df=external_global_df,
        external_fm_df=external_fm_df,
        external_range_df=external_range_df,
    )

    final_summary_df.to_csv(
        FINAL_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    report = build_markdown_report(
        final_summary_df=final_summary_df,
        temporal_df=temporal_df,
        external_global_df=external_global_df,
        external_fm_df=external_fm_df,
        external_range_df=external_range_df,
    )

    REPORT_OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Resumen final de experimentos generado correctamente.")
    print("")
    print("Tabla sintética final:")
    print(round_metric_columns(final_summary_df))
    print("")
    print(f"CSV generado: {FINAL_SUMMARY_OUTPUT_PATH}")
    print(f"Informe generado: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()