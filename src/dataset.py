"""Cache de secuencias preprocesadas, split por participante y Dataset de PyTorch."""
import numpy as np

from . import config, features
from . import data_loading as dl

CACHE_PATH = config.DATA_PROCESSED_DIR / "secuencias_modelado.npz"


def construir_cache(paths=None, destino=CACHE_PATH) -> dict:
    """Preprocesa los parquet de la muestra y guarda todo en un solo .npz."""
    paths = paths or config.SAMPLE_LANDMARK_PATHS
    meta = dl.load_train_index().set_index("sequence_id")
    secuencias, ids = [], []
    descartadas = 0
    for rel_path in paths:
        df = features.leer_parquet(dl.landmark_path(rel_path))
        for seq_id, seq_df in df.groupby(level=0):
            x = features.preprocesar_secuencia(seq_df)
            if seq_id not in meta.index or features.cobertura_mano(x[:, -2:]) < features.UMBRAL_MANO:
                descartadas += 1
                continue
            secuencias.append(x.astype(np.float16))
            ids.append(int(seq_id))

    largos = np.array([len(x) for x in secuencias])
    cache = {
        "X": np.concatenate(secuencias),
        "offsets": np.concatenate([[0], np.cumsum(largos)]),
        "sequence_id": np.array(ids),
        "participant_id": meta.loc[ids, "participant_id"].to_numpy(),
        "phrase": meta.loc[ids, "phrase"].to_numpy().astype(str),
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destino, **cache)
    print(f"{len(ids)} secuencias guardadas, {descartadas} descartadas -> {destino}")
    return cache


def cargar_cache(path=CACHE_PATH) -> dict:
    """Carga la cache; si no existe, la construye."""
    if not path.exists():
        return construir_cache(destino=path)
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


if __name__ == "__main__":
    construir_cache()
