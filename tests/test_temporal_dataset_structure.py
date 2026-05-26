import pandas as pd

from football_predictor.player_temporal_features import (
    build_player_temporal_modeling_dataset,
)


def test_build_player_temporal_modeling_dataset_creates_next_season_target() -> None:
    input_df = pd.DataFrame(
        [
            {
                "id": 1,
                "player_name": "Jugador Uno",
                "games": 30,
                "time": 2000,
                "position": "F",
                "team_title": "Equipo A",
                "league": "La-Liga",
                "season": "2023-2024",
                "goals_90": 0.30,
                "xG_90": 0.25,
                "assists_90": 0.10,
                "xA_90": 0.08,
                "shots_90": 2.50,
                "key_passes_90": 1.20,
                "yellow_cards_90": 0.10,
                "red_cards_90": 0.00,
                "npg_90": 0.25,
                "npxG_90": 0.20,
                "xGChain_90": 0.50,
                "xGBuildup_90": 0.20,
            },
            {
                "id": 1,
                "player_name": "Jugador Uno",
                "games": 31,
                "time": 2100,
                "position": "F",
                "team_title": "Equipo A",
                "league": "La-Liga",
                "season": "2024-2025",
                "goals_90": 0.40,
                "xG_90": 0.35,
                "assists_90": 0.12,
                "xA_90": 0.09,
                "shots_90": 2.80,
                "key_passes_90": 1.30,
                "yellow_cards_90": 0.08,
                "red_cards_90": 0.00,
                "npg_90": 0.35,
                "npxG_90": 0.30,
                "xGChain_90": 0.60,
                "xGBuildup_90": 0.25,
            },
        ]
    )

    temporal_df = build_player_temporal_modeling_dataset(
        input_df,
        target_column="xG_90",
    )

    assert len(temporal_df) == 1
    assert temporal_df.loc[0, "season"] == "2023-2024"
    assert temporal_df.loc[0, "next_season"] == "2024-2025"
    assert temporal_df.loc[0, "target_next_xG_90"] == 0.35