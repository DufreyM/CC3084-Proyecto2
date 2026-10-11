"""Modelo principal: CNN 1D con stride + BiGRU + CTC (estructura de HandReader)."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src import features, vocab


class BloqueConv(nn.Module):
    """Conv1d sobre el tiempo + BatchNorm + GELU + Dropout."""

    def __init__(self, entrada: int, salida: int, kernel: int, stride: int, dropout: float):
        super().__init__()
        self.kernel, self.stride, self.padding = kernel, stride, kernel // 2
        self.conv = nn.Conv1d(entrada, salida, kernel, stride=stride, padding=self.padding)
        self.norm = nn.BatchNorm1d(salida)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))

    def largo_salida(self, largo: torch.Tensor) -> torch.Tensor:
        return (largo + 2 * self.padding - self.kernel) // self.stride + 1


class CNNBiGRU(nn.Module):
    """La CNN capta movimientos cortos y acorta la secuencia; la BiGRU usa contexto de ambos lados."""

    def __init__(self, canales: int = 192, kernel: int = 5, stride: int = 2, hidden: int = 128,
                 capas_gru: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            BloqueConv(features.NUM_FEATURES, canales, kernel, stride, dropout),
            BloqueConv(canales, canales, 3, 1, dropout),
        ])
        self.gru = nn.GRU(canales, hidden, num_layers=capas_gru, batch_first=True, bidirectional=True,
                          dropout=dropout if capas_gru > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.salida = nn.Linear(2 * hidden, vocab.NUM_CLASSES)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor):
        h = x.transpose(1, 2)  # (B, F, T) para Conv1d
        largo = x_len
        for bloque in self.convs:
            h = bloque(h)
            largo = bloque.largo_salida(largo)
        h = h.transpose(1, 2)  # (B, T', canales)
        largo = largo.clamp(min=1, max=h.size(1))

        # Empaquetar evita que la direccion hacia atras lea el padding
        empaquetado = pack_padded_sequence(h, largo.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.gru(empaquetado)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=largo.max().item())
        return self.salida(self.drop(h)).log_softmax(-1), largo


def construir(**hiperparametros) -> CNNBiGRU:
    return CNNBiGRU(**hiperparametros)
