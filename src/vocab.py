"""Vocabulario de caracteres para CTC (el indice 0 es el blank)."""
from . import data_loading as dl

BLANK = 0

_CHAR_TO_IDX = {c: i + 1 for c, i in dl.load_char_to_prediction_index().items()}
_IDX_TO_CHAR = {i: c for c, i in _CHAR_TO_IDX.items()}

NUM_CLASSES = len(_CHAR_TO_IDX) + 1


def encode(frase: str) -> list[int]:
    """Frase -> indices; ignora caracteres fuera del vocabulario."""
    return [_CHAR_TO_IDX[c] for c in frase if c in _CHAR_TO_IDX]


def decode(indices) -> str:
    """Indices -> frase; ignora el blank."""
    return "".join(_IDX_TO_CHAR.get(int(i), "") for i in indices if int(i) != BLANK)
