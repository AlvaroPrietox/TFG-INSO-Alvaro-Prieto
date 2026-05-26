import re
import unicodedata

import pandas as pd


def normalize_player_key(player_name: str) -> str:
    """
    Normaliza el nombre de un jugador para facilitar el emparejamiento
    entre datasets procedentes de fuentes distintas.

    Operaciones realizadas:
    - convierte a minúsculas,
    - elimina tildes,
    - elimina signos de puntuación,
    - compacta espacios múltiples.
    """
    if pd.isna(player_name):
        return ""

    text = str(player_name).strip().lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text