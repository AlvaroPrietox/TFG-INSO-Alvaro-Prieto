import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from football_predictor.text_normalization import normalize_player_key  # noqa: E402


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


EVALUATED_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_predictions.csv"
)

CURRENT_FUTURE_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_player_predictions_final_ridge.csv"
)

HISTORICAL_PLAYERS_PATH = (
    PROCESSED_DATA_DIR / "historical_players_clean.csv"
)

MULTI_METRIC_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_comparison.csv"
)

MULTI_METRIC_COMPARISON_FM_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_comparison_f_m.csv"
)

FINAL_SUMMARY_PATH = (
    PROCESSED_DATA_DIR / "final_experiment_summary.csv"
)

FINAL_MULTI_METRIC_SUMMARY_PATH = (
    PROCESSED_DATA_DIR / "final_multi_metric_summary.csv"
)

FINAL_PROSPECTIVE_SUMMARY_PATH = (
    PROCESSED_DATA_DIR / "final_prospective_prediction_summary.csv"
)

EXTERNAL_RANGE_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_by_xg_range.csv"
)


FINAL_MODEL_NAME = "Ridge"
INPUT_SEASON = "2024-2025"
TARGET_SEASON = "2025-2026"

METRIC_CONFIG = {
    "Goles esperados": {
        "metric_key": "xG_90",
        "short_label": "xG_90",
        "human_label": "goles esperados por 90 minutos",
        "predicted_column": "predicted_2025_2026_xG_90",
        "actual_column": "actual_2025_2026_xG_90",
        "signed_error_column": "signed_error_xG_90",
        "absolute_error_column": "absolute_error_xG_90",
        "future_prediction_column": "predicted_next_xG_90",
    },
    "Goles": {
        "metric_key": "goals_90",
        "short_label": "goals_90",
        "human_label": "goles reales por 90 minutos",
        "predicted_column": "predicted_2025_2026_goals_90",
        "actual_column": "actual_2025_2026_goals_90",
        "signed_error_column": "signed_error_goals_90",
        "absolute_error_column": "absolute_error_goals_90",
        "future_prediction_column": "predicted_next_goals_90",
    },
    "Asistencias": {
        "metric_key": "assists_90",
        "short_label": "assists_90",
        "human_label": "asistencias reales por 90 minutos",
        "predicted_column": "predicted_2025_2026_assists_90",
        "actual_column": "actual_2025_2026_assists_90",
        "signed_error_column": "signed_error_assists_90",
        "absolute_error_column": "absolute_error_assists_90",
        "future_prediction_column": "predicted_next_assists_90",
    },
    "Asistencias esperadas": {
        "metric_key": "xA_90",
        "short_label": "xA_90",
        "human_label": "asistencias esperadas por 90 minutos",
        "predicted_column": "predicted_2025_2026_xA_90",
        "actual_column": "actual_2025_2026_xA_90",
        "signed_error_column": "signed_error_xA_90",
        "absolute_error_column": "absolute_error_xA_90",
        "future_prediction_column": "predicted_next_xA_90",
    },
}


st.set_page_config(
    page_title="Predicción de rendimiento ofensivo",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Carga y preparación de datos
# ---------------------------------------------------------------------------


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """
    Carga un CSV con caché de Streamlit.
    """
    if not path.exists():
        st.error(f"No se ha encontrado el archivo: {path}")
        st.stop()

    return pd.read_csv(path)


def load_data() -> dict[str, pd.DataFrame]:
    """
    Carga todos los datasets necesarios para la interfaz.
    """
    return {
        "evaluated_predictions": load_csv(EVALUATED_PREDICTIONS_PATH),
        "current_future_predictions": load_csv(CURRENT_FUTURE_PREDICTIONS_PATH),
        "historical_players": load_csv(HISTORICAL_PLAYERS_PATH),
        "multi_metric_comparison": load_csv(MULTI_METRIC_COMPARISON_PATH),
        "multi_metric_comparison_f_m": load_csv(MULTI_METRIC_COMPARISON_FM_PATH),
        "final_summary": load_csv(FINAL_SUMMARY_PATH),
        "final_multi_metric_summary": load_csv(FINAL_MULTI_METRIC_SUMMARY_PATH),
        "final_prospective_summary": load_csv(FINAL_PROSPECTIVE_SUMMARY_PATH),
        "external_range_comparison": load_csv(EXTERNAL_RANGE_COMPARISON_PATH),
    }


def add_error_type(
    df: pd.DataFrame,
    metric_config: dict,
) -> pd.DataFrame:
    """
    Añade una etiqueta cualitativa del tipo de error para la métrica activa.
    """
    output_df = df.copy()
    signed_error_column = metric_config["signed_error_column"]

    if signed_error_column not in output_df.columns:
        return output_df

    def classify_error(value: float) -> str:
        if value > 0:
            return "Sobreestimación"
        if value < 0:
            return "Infraestimación"
        return "Exacto"

    output_df["error_type"] = output_df[signed_error_column].apply(
        classify_error
    )

    return output_df


def format_number(value: float, decimals: int = 3) -> str:
    """
    Formatea un valor numérico.
    """
    if pd.isna(value):
        return "-"

    return f"{value:.{decimals}f}"


def get_metric_rows(
    comparison_df: pd.DataFrame,
    metric_key: str,
) -> pd.DataFrame:
    """
    Filtra una tabla de evaluación por métrica.
    """
    if "metric_key" not in comparison_df.columns:
        return pd.DataFrame()

    return comparison_df[comparison_df["metric_key"] == metric_key].copy()


def get_metric_row(
    comparison_df: pd.DataFrame,
    metric_key: str,
) -> pd.Series | None:
    """
    Devuelve la primera fila de una métrica en una tabla de comparación.
    """
    metric_df = get_metric_rows(comparison_df, metric_key)

    if metric_df.empty:
        return None

    return metric_df.iloc[0]


# ---------------------------------------------------------------------------
# Filtros y navegación
# ---------------------------------------------------------------------------


def get_filter_options(
    df: pd.DataFrame,
    column: str,
    all_label: str,
) -> list[str]:
    """
    Devuelve valores únicos ordenados para un filtro.
    """
    if column not in df.columns:
        return [all_label]

    values = (
        df[column]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    return [all_label] + values


def apply_evaluation_filters(
    df: pd.DataFrame,
    league: str,
    team: str,
    target_team: str,
    position: str,
    minimum_minutes: int,
    player_search: str,
    error_type: str,
) -> pd.DataFrame:
    """
    Aplica filtros comunes al dataset evaluado 2025-2026.
    """
    filtered_df = df.copy()

    if league != "Todas" and "league" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["league"].astype(str) == league]

    if team != "Todos" and "team_title" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["team_title"].astype(str) == team]

    if target_team != "Todos" and "target_team_title" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["target_team_title"].astype(str) == target_team
        ]

    if position != "Todas" and "position_main" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["position_main"].astype(str) == position
        ]

    if "time" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["time"] >= minimum_minutes]

    if player_search.strip():
        search_text = player_search.strip().lower()
        filtered_df = filtered_df[
            filtered_df["player_name"]
            .astype(str)
            .str.lower()
            .str.contains(search_text, na=False)
        ]

    if error_type != "Todos" and "error_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["error_type"] == error_type]

    return filtered_df.reset_index(drop=True)


def render_sidebar_navigation() -> str:
    """
    Renderiza la navegación principal.
    """
    st.sidebar.title("Menú")

    return st.sidebar.radio(
        "Selecciona una pantalla",
        [
            "Inicio",
            "Ranking evaluado 2025-2026",
            "Ficha individual",
            "Comparador de jugadores",
            "Evaluación del modelo",
        ],
    )


def render_metric_selector() -> tuple[str, dict]:
    """
    Renderiza el selector global de métrica en el sidebar.
    """
    selected_metric_label = st.sidebar.selectbox(
        "Métrica analizada",
        list(METRIC_CONFIG.keys()),
        index=0,
        key="global_metric_selector",
    )

    return selected_metric_label, METRIC_CONFIG[selected_metric_label]


def render_evaluation_filters(
    df: pd.DataFrame,
    key_prefix: str,
    include_top_n: bool = False,
    include_error_type: bool = False,
) -> dict:
    """
    Renderiza filtros comunes para las pantallas basadas en evaluación.
    """
    st.sidebar.subheader("Filtros")

    league = st.sidebar.selectbox(
        "Liga 2024-2025",
        get_filter_options(df, "league", all_label="Todas"),
        key=f"{key_prefix}_league",
    )

    team = st.sidebar.selectbox(
        "Equipo 2024-2025",
        get_filter_options(df, "team_title", all_label="Todos"),
        key=f"{key_prefix}_team",
    )

    target_team = st.sidebar.selectbox(
        "Equipo 2025-2026",
        get_filter_options(df, "target_team_title", all_label="Todos"),
        key=f"{key_prefix}_target_team",
    )

    position = st.sidebar.selectbox(
        "Posición principal",
        get_filter_options(df, "position_main", all_label="Todas"),
        key=f"{key_prefix}_position",
    )

    max_minutes = int(df["time"].max()) if "time" in df.columns else 3000

    minimum_minutes = st.sidebar.slider(
        "Mínimo de minutos en 2024-2025",
        min_value=0,
        max_value=max_minutes,
        value=0,
        step=100,
        key=f"{key_prefix}_minimum_minutes",
    )

    player_search = st.sidebar.text_input(
        "Buscar jugador",
        value="",
        key=f"{key_prefix}_player_search",
    )

    if include_error_type:
        error_type = st.sidebar.selectbox(
            "Tipo de error",
            ["Todos", "Infraestimación", "Sobreestimación", "Exacto"],
            key=f"{key_prefix}_error_type",
        )
    else:
        error_type = "Todos"

    filters = {
        "league": league,
        "team": team,
        "target_team": target_team,
        "position": position,
        "minimum_minutes": minimum_minutes,
        "player_search": player_search,
        "error_type": error_type,
    }

    if include_top_n:
        filters["top_n"] = st.sidebar.slider(
            "Top N",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            key=f"{key_prefix}_top_n",
        )

    return filters


def apply_filters_from_dict(
    df: pd.DataFrame,
    filters: dict,
) -> pd.DataFrame:
    """
    Aplica filtros desde un diccionario.
    """
    return apply_evaluation_filters(
        df=df,
        league=filters["league"],
        team=filters["team"],
        target_team=filters["target_team"],
        position=filters["position"],
        minimum_minutes=filters["minimum_minutes"],
        player_search=filters["player_search"],
        error_type=filters["error_type"],
    )


# ---------------------------------------------------------------------------
# Utilidades de jugadores y gráficos
# ---------------------------------------------------------------------------


def build_player_display_name(row: pd.Series) -> str:
    """
    Construye una etiqueta única para selectores de jugadores.
    """
    player_name = str(row.get("player_name", "Jugador"))
    team_title = str(row.get("team_title", "Equipo 2024-2025"))
    target_team = str(row.get("target_team_title", "Equipo 2025-2026"))
    player_id = str(row.get("id", ""))

    return f"{player_name} | {team_title} → {target_team} | {player_id}"


def add_display_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade una columna display_name.
    """
    output_df = df.copy()
    output_df["display_name"] = output_df.apply(
        build_player_display_name,
        axis=1,
    )

    return output_df


def get_historical_player_rows(
    selected_player: pd.Series,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recupera el histórico del jugador por id y, si no hay coincidencia, por
    nombre normalizado.
    """
    player_id = selected_player.get("id", None)
    player_key = normalize_player_key(selected_player["player_name"])

    history_by_id = pd.DataFrame()

    if "id" in historical_df.columns and pd.notna(player_id):
        history_by_id = historical_df[
            historical_df["id"].astype(str) == str(player_id)
        ].copy()

    if not history_by_id.empty:
        return history_by_id.copy()

    historical_copy = historical_df.copy()
    historical_copy["player_key"] = historical_copy["player_name"].apply(
        normalize_player_key
    )

    return historical_copy[historical_copy["player_key"] == player_key].copy()


def get_input_feature_row(
    selected_player: pd.Series,
    historical_df: pd.DataFrame,
) -> pd.Series | None:
    """
    Recupera la fila histórica de entrada 2024-2025.
    """
    player_history = get_historical_player_rows(
        selected_player=selected_player,
        historical_df=historical_df,
    )

    if player_history.empty:
        return None

    input_season = selected_player.get("season", INPUT_SEASON)

    season_df = player_history[
        player_history["season"].astype(str) == str(input_season)
    ].copy()

    if not season_df.empty:
        return season_df.sort_values(by="time", ascending=False).iloc[0]

    return player_history.sort_values(by="time", ascending=False).iloc[0]


def build_player_evolution(
    selected_player: pd.Series,
    historical_df: pd.DataFrame,
    metric_config: dict,
) -> pd.DataFrame:
    """
    Construye la evolución de la métrica activa:
    - temporadas históricas reales,
    - predicción 2025-2026,
    - valor real 2025-2026.
    """
    metric_key = metric_config["metric_key"]
    predicted_column = metric_config["predicted_column"]
    actual_column = metric_config["actual_column"]

    player_history = get_historical_player_rows(
        selected_player=selected_player,
        historical_df=historical_df,
    )

    rows = []

    if not player_history.empty and metric_key in player_history.columns:
        available_columns = [
            column for column in ["season", "team_title", "league", metric_key]
            if column in player_history.columns
        ]

        history_df = player_history[available_columns].copy()

        if "season" in history_df.columns:
            history_df = history_df.sort_values(by="season")

        for _, row in history_df.iterrows():
            if pd.notna(row.get(metric_key, None)):
                rows.append(
                    {
                        "label": str(row.get("season", "Histórico")),
                        "team_title": row.get("team_title", ""),
                        "league": row.get("league", ""),
                        "value": row.get(metric_key, None),
                        "source": "Real histórico",
                    }
                )

    rows.append(
        {
            "label": f"{TARGET_SEASON} predicho",
            "team_title": selected_player.get("team_title", ""),
            "league": selected_player.get("league", ""),
            "value": selected_player[predicted_column],
            "source": "Predicción Ridge",
        }
    )

    rows.append(
        {
            "label": f"{TARGET_SEASON} real",
            "team_title": selected_player.get("target_team_title", ""),
            "league": "La-Liga",
            "value": selected_player[actual_column],
            "source": "Real 2025-2026",
        }
    )

    evolution_df = pd.DataFrame(rows)
    evolution_df = evolution_df.dropna(subset=["value"]).reset_index(drop=True)

    return evolution_df


def plot_player_evolution(
    evolution_df: pd.DataFrame,
    metric_label: str,
) -> None:
    """
    Dibuja la evolución de la métrica activa.
    """
    if evolution_df.empty:
        st.info("No hay datos suficientes para representar la evolución.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    x_positions = list(range(len(evolution_df)))

    ax.plot(
        x_positions,
        evolution_df["value"],
        marker="o",
        linewidth=2,
    )

    for index, row in evolution_df.iterrows():
        if row["source"] == "Predicción Ridge":
            ax.scatter(index, row["value"], marker="x", s=120)
        elif row["source"] == "Real 2025-2026":
            ax.scatter(index, row["value"], marker="o", s=120)

        ax.text(
            index,
            row["value"],
            f" {row['value']:.3f}",
            va="bottom",
        )

    ax.set_title(f"Evolución de {metric_label}: histórico, predicción y real")
    ax.set_xlabel("Temporada / punto de evaluación")
    ax.set_ylabel(metric_label)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(evolution_df["label"], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_horizontal_bar(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    x_label: str,
) -> None:
    """
    Dibuja un gráfico horizontal de barras.
    """
    if df.empty:
        st.info("No hay datos disponibles para representar.")
        return

    plot_df = df.sort_values(by=value_column, ascending=True).copy()
    height = max(4, 0.38 * len(plot_df))

    fig, ax = plt.subplots(figsize=(10, height))

    ax.barh(
        plot_df[label_column].astype(str),
        plot_df[value_column],
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Jugador")

    for index, value in enumerate(plot_df[value_column]):
        ax.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def build_prediction_interpretation(
    row: pd.Series,
    selected_metric_label: str,
    metric_config: dict,
) -> str:
    """
    Genera interpretación textual de predicción frente a valor real.
    """
    predicted = row[metric_config["predicted_column"]]
    actual = row[metric_config["actual_column"]]
    signed_error = row[metric_config["signed_error_column"]]
    absolute_error = row[metric_config["absolute_error_column"]]

    if absolute_error < 0.03:
        precision_text = "muy ajustada"
    elif absolute_error < 0.08:
        precision_text = "razonablemente próxima"
    else:
        precision_text = "con desviación relevante"

    if signed_error > 0:
        direction = "sobreestimó"
    elif signed_error < 0:
        direction = "infraestimó"
    else:
        direction = "predijo exactamente"

    return (
        f"Para **{selected_metric_label}**, el modelo predijo un valor de "
        f"**{predicted:.3f}** en 2025-2026, mientras que el valor real "
        f"observado fue **{actual:.3f}**. El error absoluto fue "
        f"**{absolute_error:.3f}**, por lo que la predicción puede "
        f"considerarse **{precision_text}**. En este caso, el modelo "
        f"**{direction}** el rendimiento ofensivo del jugador."
    )


# ---------------------------------------------------------------------------
# Pantallas
# ---------------------------------------------------------------------------


def render_home_page(
    data: dict[str, pd.DataFrame],
    selected_metric_label: str,
    metric_config: dict,
) -> None:
    """
    Pantalla de inicio.
    """
    evaluated_df = add_error_type(
        data["evaluated_predictions"],
        metric_config,
    )

    current_future_df = data["current_future_predictions"]
    comparison_df = data["multi_metric_comparison"]
    comparison_fm_df = data["multi_metric_comparison_f_m"]
    prospective_summary_df = data["final_prospective_summary"]

    metric_key = metric_config["metric_key"]
    predicted_column = metric_config["predicted_column"]
    actual_column = metric_config["actual_column"]
    absolute_error_column = metric_config["absolute_error_column"]
    signed_error_column = metric_config["signed_error_column"]
    future_prediction_column = metric_config["future_prediction_column"]

    global_row = get_metric_row(comparison_df, metric_key)
    fm_row = get_metric_row(comparison_fm_df, metric_key)
    prospective_row = get_metric_row(prospective_summary_df, metric_key)

    best_predictions = (
        evaluated_df.sort_values(by=absolute_error_column, ascending=True)
        .head(5)
        .copy()
    )

    largest_errors = (
        evaluated_df.sort_values(by=absolute_error_column, ascending=False)
        .head(5)
        .copy()
    )

    st.title("Evaluación del rendimiento ofensivo 2025-2026")

    st.markdown(
        f"""
        Esta interfaz muestra predicciones de rendimiento ofensivo para la
        temporada 2025-2026 generadas a partir de información de 2024-2025.
        La métrica activa es **{selected_metric_label}** (`{metric_key}`).
        """
    )

    st.subheader("Configuración principal")

    col1, col2, col3 = st.columns(3)

    col1.metric("Modelo final", FINAL_MODEL_NAME)
    col2.metric("Métrica activa", metric_key)
    col3.metric("Jugadores evaluados", len(evaluated_df))

    st.subheader("Métricas externas del modelo final")

    col1, col2, col3, col4 = st.columns(4)

    if global_row is not None:
        col1.metric("MAE global", format_number(global_row["mae"], 4))
        col2.metric("R² global", format_number(global_row["r2"], 4))
    else:
        col1.metric("MAE global", "-")
        col2.metric("R² global", "-")

    if fm_row is not None:
        col3.metric("MAE F/M", format_number(fm_row["mae"], 4))
        col4.metric("R² F/M", format_number(fm_row["r2"], 4))
    else:
        col3.metric("MAE F/M", "-")
        col4.metric("R² F/M", "-")

    display_columns = [
        "player_name",
        "team_title",
        "target_team_title",
        "position_main",
        predicted_column,
        actual_column,
        absolute_error_column,
        signed_error_column,
        "error_type",
    ]

    available_columns = [
        column for column in display_columns
        if column in evaluated_df.columns
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicciones más ajustadas")
        st.dataframe(
            best_predictions[available_columns],
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.subheader("Mayores errores absolutos")
        st.dataframe(
            largest_errors[available_columns],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Aplicación prospectiva adicional")

    st.markdown(
        """
        Además de la evaluación validada 2025-2026, el sistema genera una
        predicción prospectiva sobre jugadores actuales. Esta predicción no se
        evalúa todavía porque no existe el valor real futuro correspondiente.
        """
    )

    if prospective_row is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Media prospectiva", format_number(prospective_row["prediction_mean"], 4))
        col2.metric("Máximo prospectivo", format_number(prospective_row["prediction_max"], 4))
        col3.metric("Jugador top", str(prospective_row["top_player"]))

    if not current_future_df.empty and future_prediction_column in current_future_df.columns:
        top_future = (
            current_future_df.sort_values(
                by=future_prediction_column,
                ascending=False,
            )
            .head(5)
            .copy()
        )

        future_columns = [
            "player_name",
            "team_title",
            "position_main",
            "time",
            metric_key,
            future_prediction_column,
        ]

        available_future_columns = [
            column for column in future_columns
            if column in top_future.columns
        ]

        st.dataframe(
            top_future[available_future_columns],
            use_container_width=True,
            hide_index=True,
        )


def render_ranking_page(
    data: dict[str, pd.DataFrame],
    selected_metric_label: str,
    metric_config: dict,
) -> None:
    """
    Ranking evaluado 2025-2026.
    """
    evaluated_df = add_error_type(
        data["evaluated_predictions"],
        metric_config,
    )

    predicted_column = metric_config["predicted_column"]
    actual_column = metric_config["actual_column"]
    absolute_error_column = metric_config["absolute_error_column"]
    signed_error_column = metric_config["signed_error_column"]

    st.title("Ranking evaluado 2025-2026")

    st.markdown(
        f"""
        Esta pantalla permite explorar las predicciones 2025-2026 del modelo
        Ridge para **{selected_metric_label}**, junto al valor real observado y
        el error cometido.
        """
    )

    filters = render_evaluation_filters(
        evaluated_df,
        key_prefix="ranking",
        include_top_n=True,
        include_error_type=True,
    )

    filtered_df = apply_filters_from_dict(evaluated_df, filters)

    ranking_mode = st.selectbox(
        "Tipo de ranking",
        [
            "Mayor valor predicho",
            "Mayor valor real",
            "Menor error absoluto",
            "Mayor error absoluto",
            "Más infraestimados",
            "Más sobreestimados",
        ],
    )

    if ranking_mode == "Mayor valor predicho":
        sorted_df = filtered_df.sort_values(by=predicted_column, ascending=False)
        chart_column = predicted_column
    elif ranking_mode == "Mayor valor real":
        sorted_df = filtered_df.sort_values(by=actual_column, ascending=False)
        chart_column = actual_column
    elif ranking_mode == "Menor error absoluto":
        sorted_df = filtered_df.sort_values(by=absolute_error_column, ascending=True)
        chart_column = absolute_error_column
    elif ranking_mode == "Mayor error absoluto":
        sorted_df = filtered_df.sort_values(by=absolute_error_column, ascending=False)
        chart_column = absolute_error_column
    elif ranking_mode == "Más infraestimados":
        sorted_df = filtered_df.sort_values(by=signed_error_column, ascending=True)
        chart_column = signed_error_column
    else:
        sorted_df = filtered_df.sort_values(by=signed_error_column, ascending=False)
        chart_column = signed_error_column

    sorted_df = sorted_df.reset_index(drop=True)
    top_df = sorted_df.head(filters["top_n"]).copy()

    st.subheader("Resumen del filtro")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Jugadores filtrados", len(filtered_df))
    col2.metric("Top mostrado", len(top_df))

    if not filtered_df.empty:
        col3.metric(
            "MAE filtrado",
            format_number(filtered_df[absolute_error_column].mean(), 4),
        )
        col4.metric(
            "Error medio con signo",
            format_number(filtered_df[signed_error_column].mean(), 4),
        )
    else:
        col3.metric("MAE filtrado", "-")
        col4.metric("Error medio con signo", "-")

    display_columns = [
        "player_name",
        "team_title",
        "target_team_title",
        "league",
        "position",
        "position_main",
        "games",
        "time",
        "minutes_per_game",
        predicted_column,
        actual_column,
        absolute_error_column,
        signed_error_column,
        "error_type",
    ]

    available_columns = [
        column for column in display_columns
        if column in top_df.columns
    ]

    st.subheader("Tabla evaluada")

    st.dataframe(
        top_df[available_columns],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = sorted_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar CSV filtrado",
        data=csv_data,
        file_name=f"ranking_evaluado_{metric_config['metric_key']}.csv",
        mime="text/csv",
    )

    st.subheader("Visualización")

    if not top_df.empty:
        chart_df = top_df.copy()
        chart_df["label"] = (
            chart_df["player_name"].astype(str)
            + " ("
            + chart_df["target_team_title"].astype(str)
            + ")"
        )

        plot_horizontal_bar(
            df=chart_df,
            label_column="label",
            value_column=chart_column,
            title=f"{ranking_mode}: {selected_metric_label}",
            x_label=chart_column,
        )
    else:
        st.info("No hay datos para visualizar.")


def render_player_profile_page(
    data: dict[str, pd.DataFrame],
    selected_metric_label: str,
    metric_config: dict,
) -> None:
    """
    Ficha individual.
    """
    evaluated_df = add_error_type(
        data["evaluated_predictions"],
        metric_config,
    )

    evaluated_df = add_display_name(evaluated_df)
    historical_df = data["historical_players"]

    predicted_column = metric_config["predicted_column"]
    actual_column = metric_config["actual_column"]
    absolute_error_column = metric_config["absolute_error_column"]
    signed_error_column = metric_config["signed_error_column"]
    metric_key = metric_config["metric_key"]

    st.title("Ficha individual del jugador")

    st.markdown(
        f"""
        Esta pantalla muestra la predicción 2025-2026, el valor real observado,
        el error cometido y la evolución histórica para **{selected_metric_label}**.
        """
    )

    filters = render_evaluation_filters(
        evaluated_df,
        key_prefix="profile",
        include_top_n=False,
        include_error_type=True,
    )

    filtered_df = apply_filters_from_dict(evaluated_df, filters)
    filtered_df = add_display_name(filtered_df)

    if filtered_df.empty:
        st.warning("No hay jugadores disponibles con los filtros seleccionados.")
        return

    selected_display_name = st.selectbox(
        "Selecciona un jugador",
        filtered_df["display_name"].tolist(),
    )

    selected_player = filtered_df[
        filtered_df["display_name"] == selected_display_name
    ].iloc[0]

    feature_row = get_input_feature_row(
        selected_player=selected_player,
        historical_df=historical_df,
    )

    st.subheader(selected_player["player_name"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Equipo 2024-2025", selected_player.get("team_title", "-"))
    col2.metric("Equipo 2025-2026", selected_player.get("target_team_title", "-"))
    col3.metric("Posición", selected_player.get("position", "-"))
    col4.metric("Tipo de error", selected_player.get("error_type", "-"))

    st.subheader("Predicción frente al valor real")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        f"Predicción {selected_metric_label}",
        format_number(selected_player[predicted_column], 3),
    )
    col2.metric(
        f"Valor real {selected_metric_label}",
        format_number(selected_player[actual_column], 3),
    )
    col3.metric(
        "Error absoluto",
        format_number(selected_player[absolute_error_column], 3),
    )
    col4.metric(
        "Error con signo",
        format_number(selected_player[signed_error_column], 3),
    )

    st.markdown(
        build_prediction_interpretation(
            row=selected_player,
            selected_metric_label=selected_metric_label,
            metric_config=metric_config,
        )
    )

    st.subheader("Información de entrada 2024-2025")

    input_data = {
        "Métrica": [
            "Partidos",
            "Minutos",
            "Minutos por partido",
            f"Valor 2024-2025 de {metric_key}",
        ],
        "Valor": [
            selected_player.get("games", "-"),
            selected_player.get("time", "-"),
            selected_player.get("minutes_per_game", "-"),
            selected_player.get(metric_key, "-"),
        ],
    }

    st.dataframe(
        pd.DataFrame(input_data),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Variables ofensivas 2024-2025")

    if feature_row is not None:
        feature_columns = [
            "goals_90",
            "xG_90",
            "assists_90",
            "xA_90",
            "shots_90",
            "key_passes_90",
            "npg_90",
            "xGChain_90",
            "xGBuildup_90",
        ]

        available_feature_columns = [
            column for column in feature_columns
            if column in feature_row.index
        ]

        feature_table = pd.DataFrame(
            {
                "Variable": available_feature_columns,
                "Valor": [
                    feature_row[column]
                    for column in available_feature_columns
                ],
            }
        )

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No se han encontrado variables ofensivas históricas del jugador.")

    st.subheader(f"Evolución de {selected_metric_label}")

    evolution_df = build_player_evolution(
        selected_player=selected_player,
        historical_df=historical_df,
        metric_config=metric_config,
    )

    plot_player_evolution(
        evolution_df=evolution_df,
        metric_label=selected_metric_label,
    )

    st.dataframe(
        evolution_df,
        use_container_width=True,
        hide_index=True,
    )


def render_player_comparison_page(
    data: dict[str, pd.DataFrame],
    selected_metric_label: str,
    metric_config: dict,
) -> None:
    """
    Comparador de jugadores.
    """
    evaluated_df = add_error_type(
        data["evaluated_predictions"],
        metric_config,
    )

    evaluated_df = add_display_name(evaluated_df)
    historical_df = data["historical_players"]

    predicted_column = metric_config["predicted_column"]
    actual_column = metric_config["actual_column"]
    absolute_error_column = metric_config["absolute_error_column"]
    signed_error_column = metric_config["signed_error_column"]

    st.title("Comparador de jugadores")

    st.markdown(
        f"""
        Esta pantalla compara dos jugadores según la predicción, el valor real
        y el error cometido para **{selected_metric_label}**.
        """
    )

    filters = render_evaluation_filters(
        evaluated_df,
        key_prefix="comparison",
        include_top_n=False,
        include_error_type=True,
    )

    filtered_df = apply_filters_from_dict(evaluated_df, filters)
    filtered_df = add_display_name(filtered_df)

    if len(filtered_df) < 2:
        st.warning("Se necesitan al menos dos jugadores con los filtros seleccionados.")
        return

    col1, col2 = st.columns(2)

    player_a_label = col1.selectbox(
        "Jugador A",
        filtered_df["display_name"].tolist(),
        index=0,
    )

    player_b_label = col2.selectbox(
        "Jugador B",
        filtered_df["display_name"].tolist(),
        index=min(1, len(filtered_df) - 1),
    )

    player_a = filtered_df[
        filtered_df["display_name"] == player_a_label
    ].iloc[0]

    player_b = filtered_df[
        filtered_df["display_name"] == player_b_label
    ].iloc[0]

    feature_a = get_input_feature_row(player_a, historical_df)
    feature_b = get_input_feature_row(player_b, historical_df)

    st.subheader("Comparación principal")

    col1, col2 = st.columns(2)

    col1.metric(
        f"{player_a['player_name']} | predicho / real",
        (
            f"{format_number(player_a[predicted_column], 3)} / "
            f"{format_number(player_a[actual_column], 3)}"
        ),
    )

    col2.metric(
        f"{player_b['player_name']} | predicho / real",
        (
            f"{format_number(player_b[predicted_column], 3)} / "
            f"{format_number(player_b[actual_column], 3)}"
        ),
    )

    base_metrics = [
        predicted_column,
        actual_column,
        absolute_error_column,
        signed_error_column,
    ]

    comparison_rows = []

    for metric in base_metrics:
        comparison_rows.append(
            {
                "Métrica": metric,
                player_a["player_name"]: player_a.get(metric, None),
                player_b["player_name"]: player_b.get(metric, None),
            }
        )

    feature_metrics = [
        "goals_90",
        "xG_90",
        "assists_90",
        "xA_90",
        "shots_90",
        "key_passes_90",
        "xGChain_90",
        "xGBuildup_90",
        "minutes_per_game",
    ]

    for metric in feature_metrics:
        value_a = feature_a.get(metric, None) if feature_a is not None else None
        value_b = feature_b.get(metric, None) if feature_b is not None else None

        if value_a is not None or value_b is not None:
            comparison_rows.append(
                {
                    "Métrica": metric,
                    player_a["player_name"]: value_a,
                    player_b["player_name"]: value_b,
                }
            )

    comparison_table = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Visualización comparativa")

    chart_metrics = [
        predicted_column,
        actual_column,
        absolute_error_column,
        "shots_90",
        "goals_90",
        "xG_90",
        "assists_90",
        "xA_90",
    ]

    chart_table = comparison_table[
        comparison_table["Métrica"].isin(chart_metrics)
    ].copy()

    if not chart_table.empty:
        st.bar_chart(chart_table.set_index("Métrica"))
    else:
        st.info("No hay métricas suficientes para construir el gráfico.")

    st.subheader("Lectura rápida")

    if player_a[absolute_error_column] < player_b[absolute_error_column]:
        st.markdown(
            f"El modelo predijo con menor error a **{player_a['player_name']}** "
            f"que a **{player_b['player_name']}** para **{selected_metric_label}**."
        )
    elif player_a[absolute_error_column] > player_b[absolute_error_column]:
        st.markdown(
            f"El modelo predijo con menor error a **{player_b['player_name']}** "
            f"que a **{player_a['player_name']}** para **{selected_metric_label}**."
        )
    else:
        st.markdown("Ambos jugadores presentan el mismo error absoluto.")


def render_model_evaluation_page(
    data: dict[str, pd.DataFrame],
    selected_metric_label: str,
    metric_config: dict,
) -> None:
    """
    Evaluación del modelo.
    """
    comparison_df = data["multi_metric_comparison"]
    comparison_fm_df = data["multi_metric_comparison_f_m"]
    final_summary_df = data["final_summary"]
    final_multi_metric_summary_df = data["final_multi_metric_summary"]
    prospective_summary_df = data["final_prospective_summary"]
    range_df = data["external_range_comparison"]

    metric_key = metric_config["metric_key"]

    global_row = get_metric_row(comparison_df, metric_key)
    fm_row = get_metric_row(comparison_fm_df, metric_key)

    st.title("Evaluación del modelo")

    st.markdown(
        f"""
        Esta pantalla resume el rendimiento experimental del modelo final Ridge.
        La métrica activa es **{selected_metric_label}** (`{metric_key}`).
        """
    )

    st.subheader("Modelo final seleccionado")

    col1, col2, col3 = st.columns(3)

    col1.metric("Modelo", FINAL_MODEL_NAME)
    col2.metric("Métrica activa", metric_key)
    col3.metric("Evaluación principal", f"{INPUT_SEASON} → {TARGET_SEASON}")

    st.subheader("Resultados externos de la métrica activa")

    col1, col2, col3, col4 = st.columns(4)

    if global_row is not None:
        col1.metric("MAE global", format_number(global_row["mae"], 4))
        col2.metric("R² global", format_number(global_row["r2"], 4))
    else:
        col1.metric("MAE global", "-")
        col2.metric("R² global", "-")

    if fm_row is not None:
        col3.metric("MAE F/M", format_number(fm_row["mae"], 4))
        col4.metric("R² F/M", format_number(fm_row["r2"], 4))
    else:
        col3.metric("MAE F/M", "-")
        col4.metric("R² F/M", "-")

    st.subheader("Evaluación multi-métrica global")

    st.dataframe(
        comparison_df.sort_values(by="metric_key", ascending=True),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Evaluación multi-métrica F/M")

    st.dataframe(
        comparison_fm_df.sort_values(by="metric_key", ascending=True),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Resumen final de selección de modelo")

    st.dataframe(
        final_summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Resumen final multi-métrica")

    st.dataframe(
        final_multi_metric_summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Resumen prospectivo")

    st.dataframe(
        prospective_summary_df,
        use_container_width=True,
        hide_index=True,
    )

    if metric_key == "xG_90":
        st.subheader("Análisis complementario de alto xG_90")

        high_xg_df = range_df[range_df["xg_range"] == "alto_>=0.50"].copy()

        if not high_xg_df.empty:
            st.dataframe(
                high_xg_df.sort_values(by="mae", ascending=True),
                use_container_width=True,
                hide_index=True,
            )

            best_high_xg = high_xg_df.sort_values(by="mae", ascending=True).iloc[0]

            st.markdown(
                f"En jugadores con `xG_90 >= 0.50`, el menor MAE corresponde a "
                f"**{best_high_xg['model']}**, con MAE = "
                f"**{best_high_xg['mae']:.4f}**. Este análisis se interpreta como "
                f"complementario por su menor tamaño muestral."
            )

    st.subheader("Figuras generadas")

    figure_paths = [
        FIGURES_DIR / "external_2025_2026_multi_metric_mae.png",
        FIGURES_DIR / "external_2025_2026_multi_metric_r2.png",
        FIGURES_DIR / f"laliga_current_top_25_final_ridge_{metric_key}.png",
    ]

    if metric_key == "xG_90":
        figure_paths.extend(
            [
                FIGURES_DIR / "external_2025_2026_model_comparison_mae.png",
                FIGURES_DIR / "external_2025_2026_model_comparison_r2.png",
            ]
        )

    for figure_path in figure_paths:
        if figure_path.exists():
            st.image(str(figure_path), caption=figure_path.name)

    st.subheader("Conclusión metodológica")

    if metric_key == "assists_90":
        st.markdown(
            """
            Las asistencias reales presentan menor capacidad explicativa que el
            resto de métricas, porque dependen no solo de la acción del pasador,
            sino también de la finalización posterior de sus compañeros.
            """
        )
    elif metric_key in ["xG_90", "xA_90"]:
        st.markdown(
            """
            Las métricas esperadas permiten representar calidad de ocasiones y
            creación ofensiva con menor dependencia del resultado final de la
            jugada. Por ello suelen ser especialmente útiles en análisis
            predictivo de rendimiento.
            """
        )
    else:
        st.markdown(
            """
            Los goles reales por 90 minutos son una medida directa de producción
            ofensiva. Su predicción es útil, aunque puede estar condicionada por
            variabilidad de finalización, rol táctico y disponibilidad de minutos.
            """
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    data = load_data()

    page = render_sidebar_navigation()
    selected_metric_label, metric_config = render_metric_selector()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Modelo final:** {FINAL_MODEL_NAME}")
    st.sidebar.markdown(f"**Evaluación:** {INPUT_SEASON} → {TARGET_SEASON}")

    if page == "Inicio":
        render_home_page(data, selected_metric_label, metric_config)
    elif page == "Ranking evaluado 2025-2026":
        render_ranking_page(data, selected_metric_label, metric_config)
    elif page == "Ficha individual":
        render_player_profile_page(data, selected_metric_label, metric_config)
    elif page == "Comparador de jugadores":
        render_player_comparison_page(data, selected_metric_label, metric_config)
    elif page == "Evaluación del modelo":
        render_model_evaluation_page(data, selected_metric_label, metric_config)


if __name__ == "__main__":
    main()
