"""Decodificacion CTC y metricas de error por caracter."""
import torch

from . import vocab


def greedy_decode(log_probs: torch.Tensor, out_len: torch.Tensor) -> list[str]:
    """Argmax por cuadro, colapsa repetidos y quita blanks. log_probs: (B, T, C)."""
    mejores = log_probs.argmax(dim=-1).cpu()
    frases = []
    for seq, n in zip(mejores, out_len.cpu()):
        seq = seq[: int(n)]
        colapsada = torch.unique_consecutive(seq)
        frases.append(vocab.decode(colapsada.tolist()))
    return frases


def levenshtein(ref: str, hyp: str) -> int:
    """Minimo de inserciones, borrados y sustituciones para pasar de hyp a ref."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    """Character error rate: ediciones totales / caracteres de referencia."""
    total = sum(len(r) for r in refs)
    return sum(levenshtein(r, h) for r, h in zip(refs, hyps)) / max(total, 1)


def score_kaggle(refs: list[str], hyps: list[str]) -> float:
    """Metrica de la competencia: (N - D) / N, equivale a 1 - CER."""
    return 1.0 - cer(refs, hyps)
