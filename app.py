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


CURRENT_PREDICTIONS_PATH = (
    PROCESSED_DATA_DIR / "laliga_current_player_predictions_final_ridge.csv"
)

HISTORICAL_PLAYERS_PATH = (
    PROCESSED_DATA_DIR / "historical_players_clean.csv"
)

EXTERNAL_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison.csv"
)

EXTERNAL_FM_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_f_m.csv"
)

FINAL_SUMMARY_PATH = (
    PROCESSED_DATA_DIR / "final_experiment_summary.csv"
)

EXTERNAL_RANGE_COMPARISON_PATH = (
    PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_by_xg_range.csv"
)


PREDICTION_COLUMN = "predicted_next_xG_90"
FINAL_MODEL_NAME = "Ridge"
FEATURE_SET_NAME = "without_previous_xg"


st.set_page_config(
    page_title="Predicción de rendimiento ofensivo",
    page_icon="⚽",
    layout="wide",
)


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
        "current_predictions": load_csv(CURRENT_PREDICTIONS_PATH),
        "historical_players": load_csv(HISTORICAL_PLAYERS_PATH),
        "external_comparison": load_csv(EXTERNAL_COMPARISON_PATH),
        "external_fm_comparison": load_csv(EXTERNAL_FM_COMPARISON_PATH),
        "final_summary": load_csv(FINAL_SUMMARY_PATH),
        "external_range_comparison": load_csv(EXTERNAL_RANGE_COMPARISON_PATH),
    }


def format_number(value: float, decimals: int = 3) -> str:
    """
    Formatea números para mostrarlos en la interfaz.
    """
    if pd.isna(value):
        return "-"

    return f"{value:.{decimals}f}"


def get_filter_options(
    df: pd.DataFrame,
    column: str,
    all_label: str = "Todas",
) -> list[str]:
    """
    Devuelve valores únicos ordenados para filtros.
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


def apply_common_filters(
    df: pd.DataFrame,
    league: str,
    team: str,
    position: str,
    minimum_minutes: int,
    player_search: str,
) -> pd.DataFrame:
    """
    Aplica filtros comunes de liga, equipo, posición, minutos y nombre.
    """
    filtered_df = df.copy()

    if league != "Todas" and "league" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["league"].astype(str) == league]

    if team != "Todos" and "team_title" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["team_title"].astype(str) == team]

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

    return filtered_df.reset_index(drop=True)


def build_player_display_name(row: pd.Series) -> str:
    """
    Construye una etiqueta única para selectores de jugadores.
    """
    player_name = str(row.get("player_name", "Jugador"))
    team_title = str(row.get("team_title", "Equipo"))
    player_id = str(row.get("id", ""))

    return f"{player_name} | {team_title} | {player_id}"


def add_display_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade una columna auxiliar para mostrar jugadores en selectores.
    """
    output_df = df.copy()
    output_df["display_name"] = output_df.apply(build_player_display_name, axis=1)

    return output_df


def plot_horizontal_bar(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    x_label: str,
) -> None:
    """
    Genera un gráfico horizontal de barras.
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


def get_best_metric_row(
    df: pd.DataFrame,
    model_name: str = FINAL_MODEL_NAME,
) -> pd.Series:
    """
    Recupera la fila del modelo final en una tabla de comparación.
    """
    model_df = df[df["model"] == model_name].copy()

    if model_df.empty:
        return df.sort_values(by="mae", ascending=True).iloc[0]

    return model_df.iloc[0]


def classify_prediction_level(prediction_value: float) -> str:
    """
    Clasifica cualitativamente la predicción de xG_90 futuro.
    """
    if prediction_value >= 0.50:
        return "alta"
    if prediction_value >= 0.25:
        return "media-alta"
    if prediction_value >= 0.10:
        return "media-baja"

    return "baja"


def build_player_interpretation(player_row: pd.Series) -> str:
    """
    Genera una interpretación textual sencilla para la ficha individual.
    """
    prediction = float(player_row[PREDICTION_COLUMN])
    prediction_level = classify_prediction_level(prediction)

    shots = float(player_row.get("shots_90", 0))
    xA = float(player_row.get("xA_90", 0))
    xg_chain = float(player_row.get("xGChain_90", 0))

    interpretation = (
        f"El jugador presenta una predicción **{prediction_level}** de "
        f"`xG_90` futuro, con un valor estimado de "
        f"**{prediction:.3f}**."
    )

    explanatory_factors = []

    if shots >= 3:
        explanatory_factors.append(
            "un volumen elevado de tiros por 90 minutos"
        )

    if xA >= 0.25:
        explanatory_factors.append(
            "una contribución relevante en generación de ocasiones"
        )

    if xg_chain >= 0.8:
        explanatory_factors.append(
            "una participación alta en secuencias ofensivas"
        )

    if explanatory_factors:
        interpretation += (
            " Esta estimación se asocia principalmente con "
            + ", ".join(explanatory_factors)
            + "."
        )
    else:
        interpretation += (
            " Su perfil ofensivo es más moderado, sin destacar de forma extrema "
            "en las variables principales de producción ofensiva."
        )

    interpretation += (
        " La predicción debe interpretarse como una estimación estadística "
        "y no como una garantía determinista de rendimiento futuro."
    )

    return interpretation


def prepare_player_evolution(
    selected_player: pd.Series,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye la evolución histórica de xG_90 del jugador y añade la predicción
    final del modelo Ridge como último punto.
    """
    player_id = selected_player.get("id", None)
    player_key = normalize_player_key(selected_player["player_name"])

    history_by_id = pd.DataFrame()

    if "id" in historical_df.columns and pd.notna(player_id):
        history_by_id = historical_df[
            historical_df["id"].astype(str) == str(player_id)
        ].copy()

    if not history_by_id.empty:
        player_history = history_by_id.copy()
    else:
        historical_df = historical_df.copy()
        historical_df["player_key"] = historical_df["player_name"].apply(
            normalize_player_key
        )
        player_history = historical_df[
            historical_df["player_key"] == player_key
        ].copy()

    if player_history.empty:
        return pd.DataFrame()

    if "xG_90" not in player_history.columns:
        return pd.DataFrame()

    player_history = player_history[
        ["season", "team_title", "league", "xG_90"]
    ].copy()

    player_history["source"] = "Histórico"
    player_history["season_label"] = player_history["season"].astype(str)

    current_row = {
        "season": selected_player.get("season", "Temporada actual"),
        "team_title": selected_player.get("team_title", ""),
        "league": selected_player.get("league", ""),
        "xG_90": selected_player.get("xG_90", None),
        "source": "Actual/parcial",
        "season_label": f"{selected_player.get('season', 'Actual')} actual",
    }

    prediction_row = {
        "season": "Predicción",
        "team_title": selected_player.get("team_title", ""),
        "league": selected_player.get("league", ""),
        "xG_90": selected_player[PREDICTION_COLUMN],
        "source": "Predicción Ridge",
        "season_label": "Predicción futura",
    }

    evolution_df = pd.concat(
        [
            player_history,
            pd.DataFrame([current_row, prediction_row]),
        ],
        ignore_index=True,
    )

    evolution_df = evolution_df.dropna(subset=["xG_90"]).reset_index(drop=True)

    return evolution_df


def plot_player_evolution(evolution_df: pd.DataFrame) -> None:
    """
    Dibuja la evolución histórica del xG_90 y la predicción final.
    """
    if evolution_df.empty:
        st.info(
            "No hay datos históricos suficientes para representar la evolución "
            "del jugador seleccionado."
        )
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    observed_df = evolution_df[
        evolution_df["source"] != "Predicción Ridge"
    ].copy()

    prediction_df = evolution_df[
        evolution_df["source"] == "Predicción Ridge"
    ].copy()

    ax.plot(
        observed_df["season_label"],
        observed_df["xG_90"],
        marker="o",
        label="xG_90 observado / actual",
    )

    if not prediction_df.empty and not observed_df.empty:
        bridge_labels = [
            observed_df["season_label"].iloc[-1],
            prediction_df["season_label"].iloc[0],
        ]

        bridge_values = [
            observed_df["xG_90"].iloc[-1],
            prediction_df["xG_90"].iloc[0],
        ]

        ax.plot(
            bridge_labels,
            bridge_values,
            marker="o",
            linestyle="--",
            label="Predicción modelo Ridge",
        )

    ax.set_title("Evolución de xG_90 y predicción futura")
    ax.set_xlabel("Temporada")
    ax.set_ylabel("xG_90")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def render_sidebar_navigation() -> str:
    """
    Renderiza el menú principal.
    """
    st.sidebar.title("Menú")

    return st.sidebar.radio(
        "Selecciona una pantalla",
        [
            "Inicio",
            "Ranking de jugadores",
            "Ficha individual",
            "Comparador de jugadores",
            "Evaluación del modelo",
        ],
    )


def render_filter_controls(
    df: pd.DataFrame,
    key_prefix: str,
    include_top_n: bool = False,
) -> dict:
    """
    Renderiza filtros reutilizables.
    """
    st.sidebar.subheader("Filtros")

    league = st.sidebar.selectbox(
        "Liga",
        get_filter_options(df, "league", all_label="Todas"),
        key=f"{key_prefix}_league",
    )

    team = st.sidebar.selectbox(
        "Equipo",
        get_filter_options(df, "team_title", all_label="Todos"),
        key=f"{key_prefix}_team",
    )

    position = st.sidebar.selectbox(
        "Posición principal",
        get_filter_options(df, "position_main", all_label="Todas"),
        key=f"{key_prefix}_position",
    )

    max_minutes = int(df["time"].max()) if "time" in df.columns else 3000

    minimum_minutes = st.sidebar.slider(
        "Mínimo de minutos",
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

    filters = {
        "league": league,
        "team": team,
        "position": position,
        "minimum_minutes": minimum_minutes,
        "player_search": player_search,
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


def render_home_page(data: dict[str, pd.DataFrame]) -> None:
    """
    Pantalla de inicio.
    """
    current_df = data["current_predictions"]
    external_df = data["external_comparison"]
    external_fm_df = data["external_fm_comparison"]

    ridge_global = get_best_metric_row(external_df, FINAL_MODEL_NAME)
    ridge_fm = get_best_metric_row(external_fm_df, FINAL_MODEL_NAME)

    top_players = (
        current_df.sort_values(by=PREDICTION_COLUMN, ascending=False)
        .head(5)
        .copy()
    )

    st.title("Predicción del rendimiento ofensivo futuro de jugadores")

    st.markdown(
        """
        Esta interfaz permite consultar las predicciones generadas por el modelo
        final del proyecto. El sistema estima el `xG_90` futuro de jugadores de
        fútbol a partir de métricas actuales e históricas de participación,
        producción ofensiva y posición.
        """
    )

    st.subheader("Modelo final")

    col1, col2, col3 = st.columns(3)

    col1.metric("Modelo seleccionado", FINAL_MODEL_NAME)
    col2.metric("Variable predicha", PREDICTION_COLUMN)
    col3.metric("Jugadores actuales evaluados", len(current_df))

    st.subheader("Rendimiento experimental del modelo final")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "MAE externo global",
        format_number(ridge_global["mae"], 4),
    )
    col2.metric(
        "R² externo global",
        format_number(ridge_global["r2"], 4),
    )
    col3.metric(
        "MAE externo F/M",
        format_number(ridge_fm["mae"], 4),
    )
    col4.metric(
        "R² externo F/M",
        format_number(ridge_fm["r2"], 4),
    )

    st.subheader("Top 5 jugadores actuales según predicción")

    display_columns = [
        "player_name",
        "team_title",
        "position_main",
        "time",
        "shots_90",
        "goals_90",
        PREDICTION_COLUMN,
    ]

    available_columns = [
        column for column in display_columns
        if column in top_players.columns
    ]

    st.dataframe(
        top_players[available_columns],
        use_container_width=True,
        hide_index=True,
    )

    chart_df = top_players.copy()
    chart_df["label"] = (
        chart_df["player_name"].astype(str)
        + " ("
        + chart_df["team_title"].astype(str)
        + ")"
    )

    plot_horizontal_bar(
        df=chart_df,
        label_column="label",
        value_column=PREDICTION_COLUMN,
        title="Top 5 por predicción de xG_90 futuro",
        x_label="Predicción de xG_90 futuro",
    )


def render_ranking_page(data: dict[str, pd.DataFrame]) -> None:
    """
    Pantalla de ranking de jugadores.
    """
    current_df = data["current_predictions"]

    st.title("Ranking de jugadores")

    st.markdown(
        """
        Esta pantalla permite explorar el ranking de jugadores según la
        predicción de `xG_90` futuro generada por el modelo final Ridge.
        """
    )

    filters = render_filter_controls(
        current_df,
        key_prefix="ranking",
        include_top_n=True,
    )

    filtered_df = apply_common_filters(
        df=current_df,
        league=filters["league"],
        team=filters["team"],
        position=filters["position"],
        minimum_minutes=filters["minimum_minutes"],
        player_search=filters["player_search"],
    )

    filtered_df = filtered_df.sort_values(
        by=PREDICTION_COLUMN,
        ascending=False,
    ).reset_index(drop=True)

    top_df = filtered_df.head(filters["top_n"]).copy()

    st.subheader("Resumen del filtro")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jugadores filtrados", len(filtered_df))
    col2.metric("Top mostrado", len(top_df))

    if not filtered_df.empty:
        col3.metric(
            "Predicción media",
            format_number(filtered_df[PREDICTION_COLUMN].mean(), 3),
        )
    else:
        col3.metric("Predicción media", "-")

    display_columns = [
        "player_name",
        "team_title",
        "league",
        "position",
        "position_main",
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
        column for column in display_columns
        if column in top_df.columns
    ]

    st.subheader("Tabla de jugadores")

    st.dataframe(
        top_df[available_columns],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar CSV filtrado",
        data=csv_data,
        file_name="ranking_jugadores_filtrado.csv",
        mime="text/csv",
    )

    st.subheader("Visualización del ranking")

    chart_df = top_df.copy()

    if not chart_df.empty:
        chart_df["label"] = (
            chart_df["player_name"].astype(str)
            + " ("
            + chart_df["team_title"].astype(str)
            + ")"
        )

    plot_horizontal_bar(
        df=chart_df,
        label_column="label",
        value_column=PREDICTION_COLUMN,
        title=f"Top {filters['top_n']} jugadores por predicción",
        x_label="Predicción de xG_90 futuro",
    )


def render_player_profile_page(data: dict[str, pd.DataFrame]) -> None:
    """
    Pantalla de ficha individual del jugador.
    """
    current_df = add_display_name(data["current_predictions"])
    historical_df = data["historical_players"]

    st.title("Ficha individual del jugador")

    st.markdown(
        """
        Esta pantalla permite consultar el perfil completo de un jugador,
        sus métricas ofensivas actuales y su evolución histórica de `xG_90`,
        incorporando la predicción futura del modelo Ridge.
        """
    )

    filters = render_filter_controls(
        current_df,
        key_prefix="profile",
        include_top_n=False,
    )

    filtered_df = apply_common_filters(
        df=current_df,
        league=filters["league"],
        team=filters["team"],
        position=filters["position"],
        minimum_minutes=filters["minimum_minutes"],
        player_search=filters["player_search"],
    )

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

    st.subheader(selected_player["player_name"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Equipo", selected_player.get("team_title", "-"))
    col2.metric("Liga", selected_player.get("league", "-"))
    col3.metric("Posición", selected_player.get("position", "-"))
    col4.metric(
        "Predicción xG_90 futuro",
        format_number(selected_player[PREDICTION_COLUMN], 3),
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Partidos", int(selected_player.get("games", 0)))
    col2.metric("Minutos", int(selected_player.get("time", 0)))
    col3.metric(
        "Minutos / partido",
        format_number(selected_player.get("minutes_per_game", 0), 1),
    )
    col4.metric(
        "Tiros / 90",
        format_number(selected_player.get("shots_90", 0), 2),
    )

    st.subheader("Métricas ofensivas")

    metric_columns = [
        "goals_90",
        "assists_90",
        "xA_90",
        "shots_90",
        "key_passes_90",
        "xGChain_90",
        "xGBuildup_90",
        PREDICTION_COLUMN,
    ]

    available_metric_columns = [
        column for column in metric_columns
        if column in selected_player.index
    ]

    metrics_df = pd.DataFrame(
        {
            "Métrica": available_metric_columns,
            "Valor": [
                selected_player[column]
                for column in available_metric_columns
            ],
        }
    )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Interpretación automática")

    st.markdown(build_player_interpretation(selected_player))

    st.subheader("Evolución de xG_90")

    evolution_df = prepare_player_evolution(
        selected_player=selected_player,
        historical_df=historical_df,
    )

    plot_player_evolution(evolution_df)

    if not evolution_df.empty:
        st.dataframe(
            evolution_df[
                [
                    "season_label",
                    "team_title",
                    "league",
                    "xG_90",
                    "source",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_player_comparison_page(data: dict[str, pd.DataFrame]) -> None:
    """
    Pantalla de comparación entre dos jugadores.
    """
    current_df = add_display_name(data["current_predictions"])

    st.title("Comparador de jugadores")

    st.markdown(
        """
        Esta pantalla permite comparar dos jugadores a partir de sus métricas
        ofensivas actuales y su predicción de `xG_90` futuro.
        """
    )

    filters = render_filter_controls(
        current_df,
        key_prefix="comparison",
        include_top_n=False,
    )

    filtered_df = apply_common_filters(
        df=current_df,
        league=filters["league"],
        team=filters["team"],
        position=filters["position"],
        minimum_minutes=filters["minimum_minutes"],
        player_search=filters["player_search"],
    )

    filtered_df = add_display_name(filtered_df)

    if len(filtered_df) < 2:
        st.warning(
            "Se necesitan al menos dos jugadores con los filtros seleccionados."
        )
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

    st.subheader("Comparación principal")

    col1, col2 = st.columns(2)

    col1.metric(
        player_a["player_name"],
        format_number(player_a[PREDICTION_COLUMN], 3),
    )

    col2.metric(
        player_b["player_name"],
        format_number(player_b[PREDICTION_COLUMN], 3),
    )

    comparison_metrics = [
        PREDICTION_COLUMN,
        "goals_90",
        "assists_90",
        "xA_90",
        "shots_90",
        "key_passes_90",
        "xGChain_90",
        "xGBuildup_90",
        "minutes_per_game",
    ]

    comparison_metrics = [
        metric for metric in comparison_metrics
        if metric in filtered_df.columns
    ]

    comparison_table = pd.DataFrame(
        {
            "Métrica": comparison_metrics,
            player_a["player_name"]: [
                player_a[metric] for metric in comparison_metrics
            ],
            player_b["player_name"]: [
                player_b[metric] for metric in comparison_metrics
            ],
        }
    )

    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Visualización comparativa")

    chart_df = comparison_table.set_index("Métrica")

    st.bar_chart(chart_df)

    st.subheader("Lectura rápida")

    diff = player_a[PREDICTION_COLUMN] - player_b[PREDICTION_COLUMN]

    if diff > 0:
        st.markdown(
            f"**{player_a['player_name']}** presenta una predicción de "
            f"`xG_90` futuro superior a **{player_b['player_name']}** "
            f"por {diff:.3f} puntos."
        )
    elif diff < 0:
        st.markdown(
            f"**{player_b['player_name']}** presenta una predicción de "
            f"`xG_90` futuro superior a **{player_a['player_name']}** "
            f"por {abs(diff):.3f} puntos."
        )
    else:
        st.markdown(
            "Ambos jugadores presentan la misma predicción de `xG_90` futuro."
        )


def render_model_evaluation_page(data: dict[str, pd.DataFrame]) -> None:
    """
    Pantalla de evaluación del modelo.
    """
    external_df = data["external_comparison"]
    external_fm_df = data["external_fm_comparison"]
    final_summary_df = data["final_summary"]
    range_df = data["external_range_comparison"]

    st.title("Evaluación del modelo")

    st.markdown(
        """
        Esta pantalla resume el rendimiento experimental del sistema y justifica
        la selección de Ridge como modelo final.
        """
    )

    ridge_global = get_best_metric_row(external_df, FINAL_MODEL_NAME)
    ridge_fm = get_best_metric_row(external_fm_df, FINAL_MODEL_NAME)

    st.subheader("Modelo final seleccionado")

    col1, col2, col3 = st.columns(3)

    col1.metric("Modelo", FINAL_MODEL_NAME)
    col2.metric("Feature set", FEATURE_SET_NAME)
    col3.metric("Variable objetivo", "target_next_xG_90")

    st.subheader("Resultados externos 2025-2026")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE global", format_number(ridge_global["mae"], 4))
    col2.metric("R² global", format_number(ridge_global["r2"], 4))
    col3.metric("MAE F/M", format_number(ridge_fm["mae"], 4))
    col4.metric("R² F/M", format_number(ridge_fm["r2"], 4))

    st.subheader("Comparación externa global")

    st.dataframe(
        external_df.sort_values(by="mae", ascending=True),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Comparación externa F/M")

    st.dataframe(
        external_fm_df.sort_values(by="mae", ascending=True),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Resumen final de experimentos")

    st.dataframe(
        final_summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Análisis de jugadores con alto xG_90")

    high_xg_df = range_df[
        range_df["xg_range"] == "alto_>=0.50"
    ].copy()

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
            f"**{best_high_xg['mae']:.4f}**. Este subconjunto se interpreta "
            f"como análisis complementario, ya que contiene menos observaciones "
            f"que la evaluación global."
        )

    st.subheader("Figuras disponibles")

    figure_paths = [
        FIGURES_DIR / "external_2025_2026_model_comparison_mae.png",
        FIGURES_DIR / "external_2025_2026_model_comparison_r2.png",
        FIGURES_DIR / "external_2025_2026_model_comparison_f_m_mae.png",
        FIGURES_DIR / "error_by_xg_range.png",
    ]

    for figure_path in figure_paths:
        if figure_path.exists():
            st.image(str(figure_path), caption=figure_path.name)

    st.subheader("Conclusión metodológica")

    st.markdown(
        """
        Ridge se selecciona como modelo final porque obtiene el mejor rendimiento
        en la evaluación temporal interna, en la evaluación externa global y en
        el subconjunto F/M. Aunque otros modelos pueden mejorar en subconjuntos
        específicos, Ridge ofrece el mejor equilibrio entre rendimiento,
        simplicidad e interpretabilidad.
        """
    )

    st.markdown(
        """
        La principal limitación observada es que los jugadores con alto `xG_90`
        son más difíciles de predecir. Esto sugiere que los perfiles ofensivos
        extremos presentan mayor variabilidad y pueden depender de factores
        contextuales no incluidos en el modelo, como rol táctico, lesiones,
        cambios de equipo o volumen de penaltis.
        """
    )


def main() -> None:
    data = load_data()

    page = render_sidebar_navigation()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Modelo final:** Ridge")
    st.sidebar.markdown("**Variable:** `predicted_next_xG_90`")
    st.sidebar.markdown("**Feature set:** `without_previous_xg`")

    if page == "Inicio":
        render_home_page(data)
    elif page == "Ranking de jugadores":
        render_ranking_page(data)
    elif page == "Ficha individual":
        render_player_profile_page(data)
    elif page == "Comparador de jugadores":
        render_player_comparison_page(data)
    elif page == "Evaluación del modelo":
        render_model_evaluation_page(data)


if __name__ == "__main__":
    main()