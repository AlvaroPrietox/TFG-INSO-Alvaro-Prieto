import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.player_temporal_features import simplify_position
from football_predictor.player_temporal_modeling import get_temporal_feature_set
from football_predictor.text_normalization import normalize_player_key


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

HISTORICAL_PLAYERS_PATH = PROCESSED_DATA_DIR / "historical_players_clean.csv"
TARGET_2025_2026_PATH = PROCESSED_DATA_DIR / "laliga_2025_2026_target_clean.csv"

MODEL_PATH = MODELS_DIR / "random_forest_temporal_without_previous_xg.joblib"

COMPARISON_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation.csv"
)

SAMPLE_OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_sample_25.csv"
)

REPORT_OUTPUT_PATH = (
    REPORTS_DIR / "14_laliga_2025_2026_prediction_evaluation.md"
)


FEATURE_SET_NAME = "without_previous_xg"
INPUT_SEASON = "2024-2025"
TARGET_SEASON = "2025-2026"


def prepare_2024_2025_players_for_prediction(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara los jugadores de la temporada 2024-2025 como entrada del modelo.

    El modelo temporal predice xG_90 de la temporada siguiente usando
    métricas de la temporada actual.
    """
    input_df = historical_df[
        historical_df["season"] == INPUT_SEASON
    ].copy()

    input_df["position_main"] = input_df["position"].apply(simplify_position)

    rows_before_gk_filter = len(input_df)

    input_df = input_df[
        input_df["position_main"] != "GK"
    ].copy()

    removed_goalkeepers = rows_before_gk_filter - len(input_df)

    print(f"Porteros eliminados de 2024-2025: {removed_goalkeepers}")

    input_df["minutes_per_game"] = input_df["time"] / input_df["games"]
    input_df["high_participation"] = (
        input_df["minutes_per_game"] >= 60
    ).astype(int)

    input_df["player_key"] = input_df["player_name"].apply(
        normalize_player_key
    )

    input_df = input_df.sort_values(
        by="time",
        ascending=False,
    ).drop_duplicates(
        subset=["player_key"],
        keep="first",
    )

    return input_df.reset_index(drop=True)


def prepare_2025_2026_target(
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara el dataset objetivo real 2025-2026 para la comparación.
    """
    target_clean = target_df.copy()

    target_clean["player_key"] = target_clean["player_name"].apply(
        normalize_player_key
    )

    target_clean = target_clean.sort_values(
        by="time",
        ascending=False,
    ).drop_duplicates(
        subset=["player_key"],
        keep="first",
    )

    target_clean = target_clean.rename(
        columns={
            "player_name": "target_player_name",
            "team_title": "target_team_title",
            "games": "target_games",
            "time": "target_time",
            "xG_90": "actual_2025_2026_xG_90",
            "goals_90": "actual_2025_2026_goals_90",
            "assists_90": "actual_2025_2026_assists_90",
            "xA_90": "actual_2025_2026_xA_90",
        }
    )

    selected_columns = [
        "player_key",
        "target_player_name",
        "target_team_title",
        "target_games",
        "target_time",
        "actual_2025_2026_xG_90",
        "actual_2025_2026_goals_90",
        "actual_2025_2026_assists_90",
        "actual_2025_2026_xA_90",
    ]

    return target_clean[selected_columns].copy()


def build_predictions_from_2024_2025(
    input_df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Genera predicciones de xG_90 2025-2026 usando datos de 2024-2025.
    """
    numeric_features, categorical_features = get_temporal_feature_set(
        FEATURE_SET_NAME
    )

    feature_columns = numeric_features + categorical_features

    missing_columns = [
        column for column in feature_columns
        if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para predecir: {missing_columns}"
        )

    predictions = model.predict(input_df[feature_columns])

    prediction_df = input_df.copy()
    prediction_df["predicted_2025_2026_xG_90"] = predictions

    selected_columns = [
        "player_key",
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
        "predicted_2025_2026_xG_90",
    ]

    return prediction_df[selected_columns].copy()


def build_comparison_dataset(
    predictions_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une predicciones 2025-2026 con valores reales 2025-2026.
    """
    comparison_df = predictions_df.merge(
        target_df,
        how="inner",
        on="player_key",
    )

    comparison_df["signed_error"] = (
        comparison_df["predicted_2025_2026_xG_90"]
        - comparison_df["actual_2025_2026_xG_90"]
    )

    comparison_df["absolute_error"] = comparison_df["signed_error"].abs()

    comparison_df = comparison_df.sort_values(
        by="absolute_error",
        ascending=True,
    ).reset_index(drop=True)

    return comparison_df


def select_representative_sample(
    comparison_df: pd.DataFrame,
    sample_size: int = 25,
) -> pd.DataFrame:
    """
    Selecciona una muestra con aciertos, errores medios y errores altos.
    """
    if len(comparison_df) <= sample_size:
        return comparison_df.copy()

    sorted_df = comparison_df.sort_values(
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
        [best_cases, middle_cases, worst_cases],
        ignore_index=True,
    )

    sample_df = sample_df.sort_values(
        by="absolute_error",
        ascending=True,
    ).reset_index(drop=True)

    return sample_df


def build_markdown_report(
    comparison_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    total_input_players: int,
    total_target_players: int,
) -> str:
    """
    Construye un informe Markdown de evaluación 2025-2026.
    """
    mae = mean_absolute_error(
        comparison_df["actual_2025_2026_xG_90"],
        comparison_df["predicted_2025_2026_xG_90"],
    )

    r2 = r2_score(
        comparison_df["actual_2025_2026_xG_90"],
        comparison_df["predicted_2025_2026_xG_90"],
    )

    matched_players = len(comparison_df)

    report_columns = [
        "player_name",
        "team_title",
        "target_team_title",
        "position_main",
        "predicted_2025_2026_xG_90",
        "actual_2025_2026_xG_90",
        "signed_error",
        "absolute_error",
    ]

    rounded_sample = sample_df[report_columns].copy()

    numeric_columns = [
        "predicted_2025_2026_xG_90",
        "actual_2025_2026_xG_90",
        "signed_error",
        "absolute_error",
    ]

    for column in numeric_columns:
        rounded_sample[column] = rounded_sample[column].round(4)

    lines = []

    lines.append("# Evaluación de predicciones 2025-2026 en LaLiga")
    lines.append("")

    lines.append("## Planteamiento")
    lines.append("")
    lines.append(
        "Este experimento utiliza datos de jugadores de la temporada "
        "2024-2025 para predecir su `xG_90` en la temporada 2025-2026. "
        "Posteriormente, las predicciones se comparan con los valores reales "
        "de LaLiga 2025-2026 filtrados por un mínimo de 900 minutos."
    )
    lines.append("")

    lines.append("## Cobertura de emparejamiento")
    lines.append("")
    lines.append(f"- Jugadores disponibles en 2024-2025: {total_input_players}")
    lines.append(f"- Jugadores disponibles en objetivo 2025-2026: {total_target_players}")
    lines.append(f"- Jugadores emparejados por nombre normalizado: {matched_players}")
    lines.append("")

    lines.append("## Métricas")
    lines.append("")
    lines.append(f"- MAE: {mae:.4f}")
    lines.append(f"- R²: {r2:.4f}")
    lines.append("")

    lines.append("## Muestra representativa")
    lines.append("")
    lines.append(rounded_sample.to_markdown(index=False))
    lines.append("")

    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        "Esta evaluación permite comprobar el comportamiento del modelo sobre "
        "una temporada no utilizada como objetivo durante la fase inicial del "
        "pipeline. Debe interpretarse con cautela porque el emparejamiento se "
        "realiza por nombre normalizado, no por identificador único de jugador."
    )
    lines.append("")
    lines.append(
        "Los errores altos pueden deberse a cambios de equipo, cambios de rol, "
        "lesiones, variaciones tácticas o incrementos abruptos de rendimiento "
        "que el modelo no observa directamente en las variables de entrada."
    )

    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    historical_df = pd.read_csv(HISTORICAL_PLAYERS_PATH)
    target_2025_2026_df = pd.read_csv(TARGET_2025_2026_PATH)

    model = joblib.load(MODEL_PATH)

    input_2024_2025_df = prepare_2024_2025_players_for_prediction(
        historical_df
    )

    target_clean_df = prepare_2025_2026_target(
        target_2025_2026_df
    )

    predictions_df = build_predictions_from_2024_2025(
        input_df=input_2024_2025_df,
        model=model,
    )

    comparison_df = build_comparison_dataset(
        predictions_df=predictions_df,
        target_df=target_clean_df,
    )

    sample_df = select_representative_sample(
        comparison_df,
        sample_size=25,
    )

    comparison_df.to_csv(COMPARISON_OUTPUT_PATH, index=False)
    sample_df.to_csv(SAMPLE_OUTPUT_PATH, index=False)

    report = build_markdown_report(
        comparison_df=comparison_df,
        sample_df=sample_df,
        total_input_players=len(input_2024_2025_df),
        total_target_players=len(target_clean_df),
    )

    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print("Evaluación 2025-2026 completada.")
    print("")
    print(f"Jugadores 2024-2025 preparados: {len(input_2024_2025_df)}")
    print(f"Jugadores objetivo 2025-2026: {len(target_clean_df)}")
    print(f"Jugadores emparejados: {len(comparison_df)}")
    print("")
    print("Métricas:")
    print(
        {
            "mae": mean_absolute_error(
                comparison_df["actual_2025_2026_xG_90"],
                comparison_df["predicted_2025_2026_xG_90"],
            ),
            "r2": r2_score(
                comparison_df["actual_2025_2026_xG_90"],
                comparison_df["predicted_2025_2026_xG_90"],
            ),
        }
    )
    print("")
    print(f"Comparación completa: {COMPARISON_OUTPUT_PATH}")
    print(f"Muestra representativa: {SAMPLE_OUTPUT_PATH}")
    print(f"Informe generado: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()