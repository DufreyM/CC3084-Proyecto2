"""Linea base: proyeccion lineal + LSTM unidireccional + CTC (gru.py reutiliza la clase con celda GRU)."""
import torch
from torch import nn
from torch.nn import functional as F

from src import features, vocab


class RecurrenteUnidireccional(nn.Module):
    """Lee la secuencia solo hacia adelante: cada letra se decide con lo que ya paso, no con lo que viene."""

    def __init__(self, hidden: int = 192, capas: int = 2, stride: int = 4, dropout: float = 0.3, celda: str = "lstm"):
        super().__init__()
        self.stride = stride
        self.proyeccion = nn.Sequential(
            nn.Linear(features.NUM_FEATURES, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        rnn = nn.LSTM if celda == "lstm" else nn.GRU
        self.rnn = rnn(hidden, hidden, num_layers=capas, batch_first=True, dropout=dropout if capas > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.salida = nn.Linear(hidden, vocab.NUM_CLASSES)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor):
        h = self.proyeccion(x)
        # Promedio cada `stride` cuadros para acortar la secuencia sin agregar parametros
        h = F.avg_pool1d(h.transpose(1, 2), self.stride, ceil_mode=True).transpose(1, 2)
        largo = torch.div(x_len + self.stride - 1, self.stride, rounding_mode="floor").clamp(1, h.size(1))
        # Al ser unidireccional, el padding del final no afecta las salidas validas: no hace falta empaquetar
        h, _ = self.rnn(h)
        return self.salida(self.drop(h)).log_softmax(-1), largo


def construir(**hiperparametros) -> RecurrenteUnidireccional:
    return RecurrenteUnidireccional(celda="lstm", **hiperparametros)
