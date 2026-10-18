"""Modelo de mayor techo: submuestreo con Conv1d + encoder Transformer pequeno + CTC."""
import math

import torch
from torch import nn

from src import features, vocab


class Submuestreo(nn.Module):
    """Conv1d con stride sobre el tiempo: acorta la secuencia antes de la atencion (costo O(T^2))."""

    def __init__(self, entrada: int, salida: int, stride: int, dropout: float):
        super().__init__()
        self.kernel, self.stride, self.padding = 5, stride, 2
        self.conv = nn.Conv1d(entrada, salida, self.kernel, stride=stride, padding=self.padding)
        self.norm = nn.BatchNorm1d(salida)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))

    def largo_salida(self, largo: torch.Tensor) -> torch.Tensor:
        return (largo + 2 * self.padding - self.kernel) // self.stride + 1


class PosicionSinusoidal(nn.Module):
    """Codificacion posicional fija de Vaswani et al. (2017); no agrega parametros."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TransformerCTC(nn.Module):
    """La atencion relaciona cualquier par de cuadros; la Conv1d previa capta movimientos cortos."""

    def __init__(self, d_model: int = 128, cabezas: int = 4, capas: int = 3, ff: int = 256,
                 stride: int = 2, dropout: float = 0.3):
        super().__init__()
        self.submuestreo = nn.ModuleList([
            Submuestreo(features.NUM_FEATURES, d_model, stride, dropout),
            Submuestreo(d_model, d_model, 1, dropout),
        ])
        self.posicion = PosicionSinusoidal(d_model)
        capa = nn.TransformerEncoderLayer(d_model, cabezas, ff, dropout, activation="gelu",
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.salida = nn.Linear(d_model, vocab.NUM_CLASSES)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor):
        h = x.transpose(1, 2)  # (B, F, T) para Conv1d
        largo = x_len
        for bloque in self.submuestreo:
            h = bloque(h)
            largo = bloque.largo_salida(largo).clamp(min=1, max=h.size(2))
            # Poner el padding en 0 evita que el siguiente kernel lo mezcle con los ultimos cuadros
            padding = torch.arange(h.size(2), device=h.device)[None, :] >= largo[:, None]
            h = h.masked_fill(padding[:, None, :], 0.0)
        h = h.transpose(1, 2)  # (B, T', d_model)

        # True en el padding para que la atencion no lo mire
        h = self.encoder(self.posicion(h), src_key_padding_mask=padding)
        return self.salida(self.drop(self.norm(h))).log_softmax(-1), largo


def construir(**hiperparametros) -> TransformerCTC:
    return TransformerCTC(**hiperparametros)
