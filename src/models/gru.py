"""Ablacion de la linea base: misma arquitectura que lstm.py, cambiando solo la celda por una GRU."""
from src.models.lstm import RecurrenteUnidireccional


def construir(**hiperparametros) -> RecurrenteUnidireccional:
    return RecurrenteUnidireccional(celda="gru", **hiperparametros)
