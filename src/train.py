"""Loop de entrenamiento comun con CTC para que todos los modelos sean comparables."""
import copy
import random
import time

import numpy as np
import torch
from torch import nn

from . import metrics, vocab


def fijar_semilla(semilla: int = 42):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _frases_reales(y, y_len) -> list[str]:
    return [vocab.decode(fila[:n].tolist()) for fila, n in zip(y.cpu(), y_len.cpu())]


@torch.no_grad()
def evaluar(model: nn.Module, loader, device=None) -> dict:
    """Loss, CER y predicciones de un loader completo."""
    device = device or dispositivo()
    model.eval()
    ctc = nn.CTCLoss(blank=vocab.BLANK, zero_infinity=True)
    refs, hyps, loss_total, n = [], [], 0.0, 0
    for x, x_len, y, y_len in loader:
        log_probs, out_len = model(x.to(device), x_len.to(device))
        loss = ctc(log_probs.transpose(0, 1), y.to(device), out_len, y_len.to(device))
        loss_total += loss.item() * len(x)
        n += len(x)
        refs += _frases_reales(y, y_len)
        hyps += metrics.greedy_decode(log_probs, out_len)
    return {"loss": loss_total / max(n, 1), "cer": metrics.cer(refs, hyps), "refs": refs, "hyps": hyps}


def fit(model: nn.Module, train_dl, val_dl, config: dict | None = None) -> dict:
    """Entrena con AdamW y early stopping por CER de validacion; deja en model los mejores pesos."""
    cfg = {"epochs": 50, "lr": 1e-3, "weight_decay": 1e-2, "paciencia": 8, "clip": 1.0, "semilla": 42}
    cfg.update(config or {})
    fijar_semilla(cfg["semilla"])
    device = dispositivo()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * len(train_dl))
    ctc = nn.CTCLoss(blank=vocab.BLANK, zero_infinity=True)

    historial = {"train_loss": [], "val_loss": [], "val_cer": []}
    mejor_cer, mejor_estado, sin_mejora = float("inf"), None, 0
    inicio = time.perf_counter()
    for epoca in range(1, cfg["epochs"] + 1):
        model.train()
        loss_total, n = 0.0, 0
        for x, x_len, y, y_len in train_dl:
            log_probs, out_len = model(x.to(device), x_len.to(device))
            loss = ctc(log_probs.transpose(0, 1), y.to(device), out_len, y_len.to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            sched.step()
            loss_total += loss.item() * len(x)
            n += len(x)

        val = evaluar(model, val_dl, device)
        historial["train_loss"].append(loss_total / n)
        historial["val_loss"].append(val["loss"])
        historial["val_cer"].append(val["cer"])
        print(f"epoca {epoca:3d} | train {loss_total / n:.3f} | val {val['loss']:.3f} | CER {val['cer']:.3f}")

        if val["cer"] < mejor_cer:
            mejor_cer, mejor_estado, sin_mejora = val["cer"], copy.deepcopy(model.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= cfg["paciencia"]:
                print(f"early stopping en la epoca {epoca}")
                break

    model.load_state_dict(mejor_estado)
    return {
        "historial": historial,
        "mejor_cer_val": mejor_cer,
        "epocas": len(historial["val_cer"]),
        "tiempo_entrenamiento_s": time.perf_counter() - inicio,
        "config": cfg,
    }
