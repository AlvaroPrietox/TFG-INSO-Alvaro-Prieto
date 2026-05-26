import pytest

from football_predictor.player_temporal_features import (
    build_next_season_label,
    parse_season_start_year,
    simplify_position,
)


def test_simplify_position_returns_first_position() -> None:
    assert simplify_position("F M S") == "F"
    assert simplify_position("M S") == "M"
    assert simplify_position("D") == "D"


def test_simplify_position_handles_missing_or_empty_values() -> None:
    assert simplify_position("") == "Unknown"
    assert simplify_position(None) == "Unknown"


def test_parse_season_start_year_extracts_first_year() -> None:
    assert parse_season_start_year("2020-2021") == 2020
    assert parse_season_start_year("2024-2025") == 2024


def test_parse_season_start_year_raises_error_with_invalid_format() -> None:
    with pytest.raises(ValueError):
        parse_season_start_year("temporada_invalida")


def test_build_next_season_label() -> None:
    assert build_next_season_label("2020-2021") == "2021-2022"
    assert build_next_season_label("2024-2025") == "2025-2026"