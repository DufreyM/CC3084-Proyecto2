"""Loop de entrenamiento comun con CTC para que todos los modelos sean comparables."""
import copy
import importlib
import json
import random
import time

import numpy as np
import torch
from torch import nn

from . import config, metrics, vocab

MODELS_DIR = config.ROOT_DIR / "models"
METRICS_DIR = config.ROOT_DIR / "reports" / "metrics"


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


def n_parametros(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def guardar_resultados(nombre: str, model: nn.Module, hiperparametros: dict, resultado_fit: dict, loaders: dict) -> dict:
    """Evalua en val/test y guarda models/<nombre>.pt y reports/metrics/<nombre>.json."""
    device = dispositivo()
    val = evaluar(model, loaders["val"], device)
    inicio = time.perf_counter()
    test = evaluar(model, loaders["test"], device)
    ms_por_secuencia = 1000 * (time.perf_counter() - inicio) / max(len(test["refs"]), 1)

    ds_test = loaders["test"].dataset
    ids_test = ds_test.cache["sequence_id"][ds_test.idx].tolist()
    reporte = {
        "modelo": nombre,
        "cer_val": val["cer"],
        "cer_test": test["cer"],
        "score_kaggle_test": metrics.score_kaggle(test["refs"], test["hyps"]),
        "cer_por_caracter": metrics.error_por_caracter(test["refs"], test["hyps"]),
        "historial": resultado_fit["historial"],
        "epocas": resultado_fit["epocas"],
        "tiempo_entrenamiento_s": resultado_fit["tiempo_entrenamiento_s"],
        "ms_por_secuencia": ms_por_secuencia,
        "n_parametros": n_parametros(model),
        "hiperparametros": hiperparametros,
        "config_entrenamiento": resultado_fit["config"],
        "dispositivo": str(device),
        "predicciones_test": [
            {"sequence_id": sid, "real": r, "predicha": h} for sid, r, h in zip(ids_test, test["refs"], test["hyps"])
        ],
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"modelo": nombre, "hiperparametros": hiperparametros, "state_dict": model.cpu().state_dict()},
        MODELS_DIR / f"{nombre}.pt",
    )
    with open(METRICS_DIR / f"{nombre}.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    print(f"{nombre}: CER val {val['cer']:.3f} | CER test {test['cer']:.3f} | {ms_por_secuencia:.1f} ms/secuencia")
    return reporte


def cargar_modelo(nombre: str) -> nn.Module:
    """Reconstruye un modelo guardado con src/models/<nombre>.py y sus pesos."""
    ckpt = torch.load(MODELS_DIR / f"{nombre}.pt", map_location="cpu")
    modulo = importlib.import_module(f"src.models.{nombre}")
    model = modulo.construir(**ckpt["hiperparametros"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
