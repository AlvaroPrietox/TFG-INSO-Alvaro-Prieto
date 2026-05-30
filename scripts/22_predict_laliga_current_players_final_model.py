import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_features import simplify_position  # noqa: E402


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CURRENT_PLAYERS_PATH = (
    PROCESSED_DATA_DIR / "laliga_players_latest_clean.csv"
)

PREDICTIONS_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_player_predictions_final_ridge.csv"
)

PREDICTIONS_LONG_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_player_predictions_final_ridge_long.csv"
)

TOP_FM_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_top_25_f_m_predictions_final_ridge.csv"
)

TOP_FM_BY_METRIC_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_top_25_f_m_predictions_by_metric_final_ridge.csv"
)

REPORT_OUTPUT_PATH = (
    REPORTS_DIR / "22_laliga_current_predictions_final_model.md"
)

# Figura legacy para mantener compatibilidad con la versión previa centrada en xG_90.
TOP_FM_XG_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "laliga_current_top_25_final_ridge.png"
)


MODEL_NAME = "Ridge"
MODEL_TYPE = "linear_regularized_regression"

CATEGORICAL_FEATURES = [
    "position_main",
    "league",
]

BASE_NUMERIC_FEATURES = [
    "games",
    "time",
    "minutes_per_game",
    "high_participation",
    "goals_90",
    "xG_90",
    "assists_90",
    "xA_90",
    "shots_90",
    "key_passes_90",
    "yellow_cards_90",
    "red_cards_90",
    "npg_90",
    "npxG_90",
    "xGChain_90",
    "xGBuildup_90",
]

METRIC_CONFIG = {
    "xG_90": {
        "label": "Goles esperados",
        "model_filename": "ridge_temporal_xG_90.joblib",
        "predicted_column": "predicted_next_xG_90",
        "excluded_input_features": ["xG_90", "npxG_90"],
    },
    "goals_90": {
        "label": "Goles",
        "model_filename": "ridge_temporal_goals_90.joblib",
        "predicted_column": "predicted_next_goals_90",
        "excluded_input_features": ["goals_90", "npg_90"],
    },
    "assists_90": {
        "label": "Asistencias",
        "model_filename": "ridge_temporal_assists_90.joblib",
        "predicted_column": "predicted_next_assists_90",
        "excluded_input_features": ["assists_90"],
    },
    "xA_90": {
        "label": "Asistencias esperadas",
        "model_filename": "ridge_temporal_xA_90.joblib",
        "predicted_column": "predicted_next_xA_90",
        "excluded_input_features": ["xA_90"],
    },
}


# ---------------------------------------------------------------------------
# Preparación de datos
# ---------------------------------------------------------------------------


def get_numeric_features_for_metric(metric_key: str) -> list[str]:
    """
    Devuelve las variables numéricas de entrada para una métrica concreta.
    """
    excluded_features = METRIC_CONFIG[metric_key]["excluded_input_features"]

    return [
        feature for feature in BASE_NUMERIC_FEATURES
        if feature not in excluded_features
    ]


def validate_prediction_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    metric_key: str,
) -> None:
    """
    Comprueba que el dataset actual contiene todas las columnas necesarias
    para aplicar el modelo de una métrica concreta.
    """
    missing_columns = [
        column for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para generar predicciones de "
            f"{metric_key}: {missing_columns}"
        )


def prepare_current_players_for_prediction(
    current_players_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara los jugadores actuales de LaLiga para aplicar los modelos finales.

    Se generan las variables auxiliares usadas durante el entrenamiento:
    - position_main
    - minutes_per_game
    - high_participation

    Además, se eliminan porteros para mantener coherencia con los datasets
    temporales utilizados durante el entrenamiento.
    """
    prediction_df = current_players_df.copy()

    prediction_df["position_main"] = prediction_df["position"].apply(
        simplify_position
    )

    rows_before_gk_filter = len(prediction_df)

    prediction_df = prediction_df[
        prediction_df["position_main"] != "GK"
    ].copy()

    removed_goalkeepers = rows_before_gk_filter - len(prediction_df)

    print(f"Porteros eliminados: {removed_goalkeepers}")

    prediction_df["minutes_per_game"] = (
        prediction_df["time"] / prediction_df["games"]
    )

    prediction_df["high_participation"] = (
        prediction_df["minutes_per_game"] >= 60
    ).astype(int)

    return prediction_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Predicción multi-métrica
# ---------------------------------------------------------------------------


def load_metric_model(metric_key: str):
    """
    Carga el modelo Ridge correspondiente a una métrica concreta.
    """
    model_path = MODELS_DIR / METRIC_CONFIG[metric_key]["model_filename"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el modelo para {metric_key}: {model_path}. "
            f"Ejecuta antes scripts/21_train_and_save_final_ridge_model.py"
        )

    return joblib.load(model_path)


def build_multi_metric_predictions(
    prediction_input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera predicciones prospectivas para todas las métricas configuradas.
    """
    output_df = prediction_input_df.copy()

    output_df["model"] = MODEL_NAME
    output_df["model_type"] = MODEL_TYPE

    for metric_key, config in METRIC_CONFIG.items():
        print(f"Generando predicciones prospectivas para: {metric_key}")

        numeric_features = get_numeric_features_for_metric(metric_key)
        feature_columns = numeric_features + CATEGORICAL_FEATURES

        validate_prediction_columns(
            df=output_df,
            feature_columns=feature_columns,
            metric_key=metric_key,
        )

        model = load_metric_model(metric_key)

        output_df[config["predicted_column"]] = model.predict(
            output_df[feature_columns]
        )

    return output_df


def build_long_predictions(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una versión long de las predicciones prospectivas, útil para la
    interfaz y para análisis por métrica.
    """
    base_columns = [
        "id",
        "player_name",
        "position",
        "position_main",
        "team_title",
        "league",
        "season",
        "games",
        "time",
        "minutes_per_game",
        "model",
    ]

    available_base_columns = [
        column for column in base_columns
        if column in predictions_df.columns
    ]

    long_frames = []

    for metric_key, config in METRIC_CONFIG.items():
        metric_df = predictions_df[available_base_columns].copy()

        metric_df["metric_key"] = metric_key
        metric_df["metric_label"] = config["label"]
        metric_df["predicted_next_value"] = predictions_df[
            config["predicted_column"]
        ]

        if metric_key in predictions_df.columns:
            metric_df["current_value"] = predictions_df[metric_key]
        else:
            metric_df["current_value"] = None

        long_frames.append(metric_df)

    return pd.concat(long_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Selección de rankings y figuras
# ---------------------------------------------------------------------------


def select_top_f_m_predictions_by_metric(
    predictions_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Selecciona el top F/M para cada métrica predicha.
    """
    rows = []

    f_m_df = predictions_df[
        predictions_df["position_main"].isin(["F", "M"])
    ].copy()

    for metric_key, config in METRIC_CONFIG.items():
        predicted_column = config["predicted_column"]

        metric_top_df = f_m_df.sort_values(
            by=predicted_column,
            ascending=False,
        ).head(top_n).copy()

        metric_top_df["metric_key"] = metric_key
        metric_top_df["metric_label"] = config["label"]
        metric_top_df["predicted_next_value"] = metric_top_df[predicted_column]

        rows.append(metric_top_df)

    return pd.concat(rows, ignore_index=True)


def select_top_f_m_xg_legacy(
    predictions_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Mantiene la salida legacy del top 25 F/M ordenado por xG_90 predicho.
    """
    xg_column = METRIC_CONFIG["xG_90"]["predicted_column"]

    top_df = predictions_df[
        predictions_df["position_main"].isin(["F", "M"])
    ].copy()

    top_df = top_df.sort_values(
        by=xg_column,
        ascending=False,
    ).head(top_n)

    selected_columns = [
        "model",
        "id",
        "player_name",
        "position",
        "position_main",
        "team_title",
        "league",
        "season",
        "games",
        "time",
        "minutes_per_game",
        "goals_90",
        "xG_90",
        "assists_90",
        "xA_90",
        "shots_90",
        "key_passes_90",
        "xGChain_90",
        "xGBuildup_90",
        "predicted_next_xG_90",
        "predicted_next_goals_90",
        "predicted_next_assists_90",
        "predicted_next_xA_90",
    ]

    available_columns = [
        column for column in selected_columns
        if column in top_df.columns
    ]

    return top_df[available_columns].reset_index(drop=True)


def plot_top_f_m_predictions(
    top_f_m_df: pd.DataFrame,
    predicted_column: str,
    metric_label: str,
    output_path: Path,
) -> None:
    """
    Genera figura horizontal del top F/M para una métrica.
    """
    plot_df = top_f_m_df.sort_values(
        by=predicted_column,
        ascending=True,
    ).copy()

    labels = (
        plot_df["player_name"]
        + " ("
        + plot_df["team_title"].astype(str)
        + ")"
    )

    plt.figure(figsize=(10, 8))
    plt.barh(labels, plot_df[predicted_column])

    plt.title(f"Top 25 F/M por predicción de {metric_label} - modelo Ridge")
    plt.xlabel(f"Predicción futura de {metric_label}")
    plt.ylabel("Jugador")

    for index, value in enumerate(plot_df[predicted_column]):
        plt.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_metric_figures(
    predictions_df: pd.DataFrame,
) -> None:
    """
    Genera figuras top 25 F/M para cada métrica prospectiva.
    """
    f_m_df = predictions_df[
        predictions_df["position_main"].isin(["F", "M"])
    ].copy()

    for metric_key, config in METRIC_CONFIG.items():
        predicted_column = config["predicted_column"]

        top_df = f_m_df.sort_values(
            by=predicted_column,
            ascending=False,
        ).head(25)

        output_path = (
            FIGURES_DIR
            / f"laliga_current_top_25_final_ridge_{metric_key}.png"
        )

        plot_top_f_m_predictions(
            top_f_m_df=top_df,
            predicted_column=predicted_column,
            metric_label=config["label"],
            output_path=output_path,
        )

    # Compatibilidad con la figura anterior centrada en xG_90.
    xg_top_df = f_m_df.sort_values(
        by="predicted_next_xG_90",
        ascending=False,
    ).head(25)

    plot_top_f_m_predictions(
        top_f_m_df=xg_top_df,
        predicted_column="predicted_next_xG_90",
        metric_label="Goles esperados",
        output_path=TOP_FM_XG_FIGURE_OUTPUT_PATH,
    )


# ---------------------------------------------------------------------------
# Informe Markdown
# ---------------------------------------------------------------------------


def build_metric_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume valores medios y máximos de predicción por métrica.
    """
    rows = []

    for metric_key, config in METRIC_CONFIG.items():
        predicted_column = config["predicted_column"]

        rows.append(
            {
                "metric_key": metric_key,
                "metric_label": config["label"],
                "players_evaluated": int(len(predictions_df)),
                "prediction_mean": float(predictions_df[predicted_column].mean()),
                "prediction_max": float(predictions_df[predicted_column].max()),
                "prediction_min": float(predictions_df[predicted_column].min()),
            }
        )

    return pd.DataFrame(rows)


def build_markdown_report(
    predictions_df: pd.DataFrame,
    top_f_m_xg_df: pd.DataFrame,
    metric_summary_df: pd.DataFrame,
) -> str:
    """
    Construye un informe Markdown con las predicciones prospectivas
    multi-métrica del modelo final Ridge.
    """
    report_summary_df = metric_summary_df.copy()

    for column in ["prediction_mean", "prediction_max", "prediction_min"]:
        report_summary_df[column] = report_summary_df[column].round(4)

    report_top_df = top_f_m_xg_df.copy()

    numeric_columns = [
        "minutes_per_game",
        "goals_90",
        "xG_90",
        "assists_90",
        "xA_90",
        "shots_90",
        "key_passes_90",
        "xGChain_90",
        "xGBuildup_90",
        "predicted_next_xG_90",
        "predicted_next_goals_90",
        "predicted_next_assists_90",
        "predicted_next_xA_90",
    ]

    for column in numeric_columns:
        if column in report_top_df.columns:
            report_top_df[column] = report_top_df[column].round(4)

    total_players = len(predictions_df)
    total_fm_players = len(
        predictions_df[predictions_df["position_main"].isin(["F", "M"])]
    )

    highest_xg_prediction = top_f_m_xg_df.iloc[0]

    lines = []

    lines.append("# Predicciones prospectivas multi-métrica con el modelo final Ridge")
    lines.append("")

    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este informe recoge las predicciones prospectivas generadas sobre los "
        "jugadores actuales de LaLiga utilizando los modelos Ridge finales. "
        "Cada métrica ofensiva se predice mediante un modelo independiente."
    )
    lines.append("")
    lines.append(
        "Estas predicciones tienen carácter prospectivo: todavía no pueden "
        "compararse con un valor real futuro, por lo que deben interpretarse "
        "como una aplicación práctica del sistema y no como evaluación externa."
    )
    lines.append("")

    lines.append("## Métricas predichas")
    lines.append("")
    lines.append("- `xG_90`: goles esperados por 90 minutos")
    lines.append("- `goals_90`: goles reales por 90 minutos")
    lines.append("- `assists_90`: asistencias reales por 90 minutos")
    lines.append("- `xA_90`: asistencias esperadas por 90 minutos")
    lines.append("")

    lines.append("## Configuración")
    lines.append("")
    lines.append(f"- Modelo base: `{MODEL_NAME}`")
    lines.append("- Enfoque: un modelo independiente por métrica")
    lines.append(f"- Jugadores no porteros evaluados: {total_players}")
    lines.append(f"- Jugadores F/M evaluados: {total_fm_players}")
    lines.append("")

    lines.append("## Resumen por métrica")
    lines.append("")
    lines.append(report_summary_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Top 25 F/M por goles esperados predichos")
    lines.append("")
    lines.append(report_top_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Figuras generadas")
    lines.append("")
    lines.append("![Top 25 F/M xG_90](figures/laliga_current_top_25_final_ridge_xG_90.png)")
    lines.append("")
    lines.append("![Top 25 F/M goals_90](figures/laliga_current_top_25_final_ridge_goals_90.png)")
    lines.append("")
    lines.append("![Top 25 F/M assists_90](figures/laliga_current_top_25_final_ridge_assists_90.png)")
    lines.append("")
    lines.append("![Top 25 F/M xA_90](figures/laliga_current_top_25_final_ridge_xA_90.png)")
    lines.append("")

    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        f"El jugador con mayor predicción prospectiva de `xG_90` es "
        f"`{highest_xg_prediction['player_name']}`, con un valor estimado de "
        f"{highest_xg_prediction['predicted_next_xG_90']:.4f}."
    )
    lines.append("")
    lines.append(
        "La predicción multi-métrica permite analizar distintas dimensiones "
        "del rendimiento ofensivo: finalización esperada, producción goleadora "
        "real, generación de asistencias y creación esperada de ocasiones para "
        "compañeros."
    )
    lines.append("")
    lines.append(
        "Estas salidas deben interpretarse como estimaciones estadísticas de "
        "apoyo al análisis, no como garantías deterministas de rendimiento "
        "futuro."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    current_players_df = pd.read_csv(CURRENT_PLAYERS_PATH)

    prediction_input_df = prepare_current_players_for_prediction(
        current_players_df
    )

    predictions_df = build_multi_metric_predictions(
        prediction_input_df
    )

    predictions_df = predictions_df.sort_values(
        by="predicted_next_xG_90",
        ascending=False,
    ).reset_index(drop=True)

    predictions_df.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    long_predictions_df = build_long_predictions(predictions_df)

    long_predictions_df.to_csv(
        PREDICTIONS_LONG_OUTPUT_PATH,
        index=False,
    )

    top_f_m_xg_df = select_top_f_m_xg_legacy(
        predictions_df=predictions_df,
        top_n=25,
    )

    top_f_m_xg_df.to_csv(
        TOP_FM_OUTPUT_PATH,
        index=False,
    )

    top_f_m_by_metric_df = select_top_f_m_predictions_by_metric(
        predictions_df=predictions_df,
        top_n=25,
    )

    top_f_m_by_metric_df.to_csv(
        TOP_FM_BY_METRIC_OUTPUT_PATH,
        index=False,
    )

    generate_metric_figures(predictions_df)

    metric_summary_df = build_metric_summary(predictions_df)

    report = build_markdown_report(
        predictions_df=predictions_df,
        top_f_m_xg_df=top_f_m_xg_df,
        metric_summary_df=metric_summary_df,
    )

    REPORT_OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Predicciones prospectivas multi-métrica con Ridge generadas correctamente.")
    print("")
    print(f"Jugadores evaluados: {len(predictions_df)}")
    print(
        "Jugadores F/M evaluados: "
        f"{len(predictions_df[predictions_df['position_main'].isin(['F', 'M'])])}"
    )
    print("")
    print("Resumen por métrica:")
    print(metric_summary_df)
    print("")
    print("Top 25 F/M por xG_90 predicho:")
    print(top_f_m_xg_df)
    print("")
    print(f"Predicciones wide generadas: {PREDICTIONS_OUTPUT_PATH}")
    print(f"Predicciones long generadas: {PREDICTIONS_LONG_OUTPUT_PATH}")
    print(f"Top 25 F/M legacy generado: {TOP_FM_OUTPUT_PATH}")
    print(f"Top 25 F/M por métrica generado: {TOP_FM_BY_METRIC_OUTPUT_PATH}")
    print(f"Informe generado: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
