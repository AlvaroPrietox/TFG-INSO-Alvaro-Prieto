from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_features import simplify_position  # noqa: E402
from football_predictor.player_temporal_modeling import get_temporal_feature_set  # noqa: E402


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = (
    MODELS_DIR / "ridge_temporal_without_previous_xg.joblib"
)

CURRENT_PLAYERS_PATH = (
    PROCESSED_DATA_DIR / "laliga_players_latest_clean.csv"
)

PREDICTIONS_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_player_predictions_final_ridge.csv"
)

TOP_FM_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_top_25_f_m_predictions_final_ridge.csv"
)

REPORT_OUTPUT_PATH = (
    REPORTS_DIR / "22_laliga_current_predictions_final_model.md"
)

TOP_FM_FIGURE_OUTPUT_PATH = (
    FIGURES_DIR / "laliga_current_top_25_final_ridge.png"
)


FEATURE_SET_NAME = "without_previous_xg"
MODEL_NAME = "Ridge"
PREDICTION_COLUMN = "predicted_next_xG_90"


def prepare_current_players_for_prediction(
    current_players_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara los jugadores actuales de LaLiga para aplicar el modelo final.

    Se generan las variables auxiliares usadas durante el entrenamiento:
    - position_main
    - minutes_per_game
    - high_participation

    Además, se eliminan porteros para mantener coherencia con el dataset
    temporal usado durante el entrenamiento.
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


def validate_prediction_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Comprueba que el dataset actual contiene todas las columnas necesarias
    para aplicar el modelo final.
    """
    missing_columns = [
        column for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para generar predicciones: "
            f"{missing_columns}"
        )


def build_predictions(
    model,
    prediction_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Genera predicciones de xG_90 futuro con el modelo final Ridge.
    """
    output_df = prediction_df.copy()

    output_df[PREDICTION_COLUMN] = model.predict(
        output_df[feature_columns]
    )

    output_df["model"] = MODEL_NAME
    output_df["feature_set"] = FEATURE_SET_NAME

    return output_df


def select_top_f_m_predictions(
    predictions_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Selecciona los 25 jugadores F/M con mayor predicción de xG_90 futuro.
    """
    top_df = predictions_df[
        predictions_df["position_main"].isin(["F", "M"])
    ].copy()

    top_df = top_df.sort_values(
        by=PREDICTION_COLUMN,
        ascending=False,
    ).head(top_n)

    selected_columns = [
        "model",
        "feature_set",
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
        "assists_90",
        "xA_90",
        "shots_90",
        "key_passes_90",
        "xGChain_90",
        "xGBuildup_90",
        PREDICTION_COLUMN,
    ]

    available_columns = [
        column for column in selected_columns
        if column in top_df.columns
    ]

    return top_df[available_columns].reset_index(drop=True)


def plot_top_f_m_predictions(
    top_f_m_df: pd.DataFrame,
) -> None:
    """
    Genera una figura con el top 25 F/M según la predicción del modelo final.
    """
    plot_df = top_f_m_df.sort_values(
        by=PREDICTION_COLUMN,
        ascending=True,
    ).copy()

    labels = (
        plot_df["player_name"]
        + " ("
        + plot_df["team_title"].astype(str)
        + ")"
    )

    plt.figure(figsize=(10, 8))
    plt.barh(labels, plot_df[PREDICTION_COLUMN])

    plt.title("Top 25 F/M por predicción de xG_90 futuro - modelo Ridge")
    plt.xlabel("Predicción de xG_90 futuro")
    plt.ylabel("Jugador")

    for index, value in enumerate(plot_df[PREDICTION_COLUMN]):
        plt.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        TOP_FM_FIGURE_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def build_markdown_report(
    predictions_df: pd.DataFrame,
    top_f_m_df: pd.DataFrame,
) -> str:
    """
    Construye un informe Markdown con las predicciones actuales del modelo
    final Ridge.
    """
    report_top_df = top_f_m_df.copy()

    numeric_columns = [
        "minutes_per_game",
        "goals_90",
        "assists_90",
        "xA_90",
        "shots_90",
        "key_passes_90",
        "xGChain_90",
        "xGBuildup_90",
        PREDICTION_COLUMN,
    ]

    for column in numeric_columns:
        if column in report_top_df.columns:
            report_top_df[column] = report_top_df[column].round(4)

    total_players = len(predictions_df)
    total_fm_players = len(
        predictions_df[predictions_df["position_main"].isin(["F", "M"])]
    )

    highest_prediction = top_f_m_df.iloc[0]

    lines = []

    lines.append("# Predicciones actuales de LaLiga con el modelo final Ridge")
    lines.append("")

    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este informe recoge las predicciones generadas sobre los jugadores "
        "actuales de LaLiga utilizando el modelo final seleccionado: Ridge "
        "con el conjunto de variables `without_previous_xg`."
    )
    lines.append("")
    lines.append(
        "La predicción representa una estimación del `xG_90` futuro del "
        "jugador a partir de sus métricas actuales de participación y "
        "producción ofensiva."
    )
    lines.append("")

    lines.append("## Configuración")
    lines.append("")
    lines.append(f"- Modelo: `{MODEL_NAME}`")
    lines.append(f"- Conjunto de variables: `{FEATURE_SET_NAME}`")
    lines.append("- Variable estimada: `predicted_next_xG_90`")
    lines.append(f"- Jugadores no porteros evaluados: {total_players}")
    lines.append(f"- Jugadores F/M evaluados: {total_fm_players}")
    lines.append("")

    lines.append("## Top 25 F/M")
    lines.append("")
    lines.append(
        report_top_df.to_markdown(index=False)
    )
    lines.append("")

    lines.append("## Figura generada")
    lines.append("")
    lines.append(
        "![Top 25 F/M modelo final Ridge](figures/laliga_current_top_25_final_ridge.png)"
    )
    lines.append("")

    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        f"El jugador con mayor predicción de `xG_90` futuro es "
        f"`{highest_prediction['player_name']}`, con un valor estimado de "
        f"{highest_prediction[PREDICTION_COLUMN]:.4f}."
    )
    lines.append("")
    lines.append(
        "El ranking se concentra principalmente en atacantes y mediapuntas, "
        "lo que es coherente con la naturaleza de la variable objetivo, ya que "
        "`xG_90` mide volumen y calidad esperada de ocasiones de gol por cada "
        "90 minutos."
    )
    lines.append("")
    lines.append(
        "Estas predicciones deben interpretarse como una estimación del "
        "rendimiento ofensivo esperado, no como una garantía determinista. "
        "El análisis experimental previo mostró que el modelo tiende a ser "
        "más estable en perfiles de producción baja o media y más conservador "
        "en perfiles de alto `xG_90`."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)

    current_players_df = pd.read_csv(CURRENT_PLAYERS_PATH)

    numeric_features, categorical_features = get_temporal_feature_set(
        FEATURE_SET_NAME
    )

    feature_columns = numeric_features + categorical_features

    prediction_input_df = prepare_current_players_for_prediction(
        current_players_df
    )

    validate_prediction_columns(
        df=prediction_input_df,
        feature_columns=feature_columns,
    )

    predictions_df = build_predictions(
        model=model,
        prediction_df=prediction_input_df,
        feature_columns=feature_columns,
    )

    predictions_df = predictions_df.sort_values(
        by=PREDICTION_COLUMN,
        ascending=False,
    ).reset_index(drop=True)

    predictions_df.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    top_f_m_df = select_top_f_m_predictions(
        predictions_df=predictions_df,
        top_n=25,
    )

    top_f_m_df.to_csv(
        TOP_FM_OUTPUT_PATH,
        index=False,
    )

    plot_top_f_m_predictions(
        top_f_m_df=top_f_m_df,
    )

    report = build_markdown_report(
        predictions_df=predictions_df,
        top_f_m_df=top_f_m_df,
    )

    REPORT_OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Predicciones actuales con el modelo final Ridge generadas correctamente.")
    print("")
    print(f"Jugadores evaluados: {len(predictions_df)}")
    print(
        "Jugadores F/M evaluados: "
        f"{len(predictions_df[predictions_df['position_main'].isin(['F', 'M'])])}"
    )
    print("")
    print("Top 25 F/M:")
    print(top_f_m_df)
    print("")
    print(f"Predicciones completas generadas: {PREDICTIONS_OUTPUT_PATH}")
    print(f"Top 25 F/M generado: {TOP_FM_OUTPUT_PATH}")
    print(f"Informe generado: {REPORT_OUTPUT_PATH}")
    print(f"Figura generada: {TOP_FM_FIGURE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()