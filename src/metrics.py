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


def alinear(ref: str, hyp: str) -> list[tuple[str, str]]:
    """Alineacion de minima edicion como pares (ref, hyp); '' marca insercion o borrado."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pares, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pares.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pares.append((ref[i - 1], ""))
            i -= 1
        else:
            pares.append(("", hyp[j - 1]))
            j -= 1
    return pares[::-1]


def error_por_caracter(refs: list[str], hyps: list[str]) -> dict:
    """Por caracter de referencia: apariciones, errores y tasa; mas pares confundidos."""
    stats, confusiones = {}, {}
    for r, h in zip(refs, hyps):
        for c_ref, c_hyp in alinear(r, h):
            if not c_ref:
                continue
            s = stats.setdefault(c_ref, {"total": 0, "errores": 0})
            s["total"] += 1
            if c_ref != c_hyp:
                s["errores"] += 1
                clave = f"{c_ref}->{c_hyp or '<borrado>'}"
                confusiones[clave] = confusiones.get(clave, 0) + 1
    for s in stats.values():
        s["tasa"] = s["errores"] / s["total"]
    return {"por_caracter": stats, "confusiones": dict(sorted(confusiones.items(), key=lambda kv: -kv[1]))}


def score_kaggle(refs: list[str], hyps: list[str]) -> float:
    """Metrica de la competencia: (N - D) / N, equivale a 1 - CER."""
    return 1.0 - cer(refs, hyps)
