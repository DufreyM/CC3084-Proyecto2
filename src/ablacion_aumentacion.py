"""Mide el efecto de la aumentacion: CNN-BiGRU stride_4 con y sin src.augment, mismas semilla y config."""
import json

from . import dataset, metrics, train
from .augment import Aumentacion
from .models import cnn_bigru

HIPERPARAMETROS = {"canales": 192, "kernel": 5, "stride": 4, "hidden": 128, "capas_gru": 2, "dropout": 0.3}
CONFIG = {"epochs": 60, "lr": 3e-3, "paciencia": 10}  # igual que notebooks/07
SALIDA = train.METRICS_DIR / "ablacion_aumentacion.json"


def correr(con_aumentacion: bool) -> dict:
    transform = Aumentacion(semilla=train_semilla) if con_aumentacion else None
    loaders = dataset.crear_dataloaders(batch_size=32, max_frames=384, transform=transform)
    train.fijar_semilla(42)
    modelo = cnn_bigru.construir(**HIPERPARAMETROS)
    resultado = train.fit(modelo, loaders["train"], loaders["val"], CONFIG)
    test = train.evaluar(modelo, loaders["test"])
    return {
        "aumentacion": con_aumentacion,
        "cer_val": resultado["mejor_cer_val"],
        "cer_test": test["cer"],
        "cer_por_caracter": metrics.error_por_caracter(test["refs"], test["hyps"]),
        "epocas": resultado["epocas"],
        "tiempo_entrenamiento_s": resultado["tiempo_entrenamiento_s"],
        "historial": resultado["historial"],
    }


train_semilla = 42

if __name__ == "__main__":
    import sys

    # Uso: python -m src.ablacion_aumentacion sin|con  (cada corrida se agrega al JSON)
    con = sys.argv[1] == "con"
    corrida = correr(con)
    previo = json.loads(SALIDA.read_text(encoding="utf-8")) if SALIDA.exists() else {}
    previo["con_aumentacion" if con else "sin_aumentacion"] = {"hiperparametros": HIPERPARAMETROS, **corrida}
    train.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(previo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"aumentacion={con}: CER val {corrida['cer_val']:.3f} | CER test {corrida['cer_test']:.3f}")
