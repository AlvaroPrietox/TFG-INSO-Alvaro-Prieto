from football_predictor.text_normalization import normalize_player_key


def test_normalize_player_key_lowercases_and_removes_accents() -> None:
    assert normalize_player_key("Álex Baena") == "alex baena"
    assert normalize_player_key("Rubén García") == "ruben garcia"


def test_normalize_player_key_removes_punctuation() -> None:
    assert normalize_player_key("Kylian Mbappé-Lottin") == "kylian mbappe lottin"
    assert normalize_player_key("Djené Dakonam.") == "djene dakonam"


def test_normalize_player_key_compacts_spaces() -> None:
    assert normalize_player_key("  Robert   Lewandowski  ") == "robert lewandowski"


def test_normalize_player_key_handles_missing_values() -> None:
    assert normalize_player_key(None) == ""