"""Cache de secuencias preprocesadas, split por participante y Dataset de PyTorch."""
import json

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from . import config, features, vocab
from . import data_loading as dl

CACHE_PATH = config.DATA_PROCESSED_DIR / "secuencias_modelado.npz"
SPLIT_PATH = config.ROOT_DIR / "data" / "split.json"  # versionado: todos usan el mismo
SEMILLA = 42


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


def crear_split(participantes, proporciones=(0.70, 0.15, 0.15), semilla=SEMILLA, destino=SPLIT_PATH) -> dict:
    """Reparte participantes (no secuencias) en train/val/test y lo guarda en JSON."""
    unicos = np.array(sorted(set(int(p) for p in participantes)))
    np.random.default_rng(semilla).shuffle(unicos)
    n_train = round(proporciones[0] * len(unicos))
    n_val = round(proporciones[1] * len(unicos))
    split = {
        "semilla": semilla,
        "train": sorted(unicos[:n_train].tolist()),
        "val": sorted(unicos[n_train : n_train + n_val].tolist()),
        "test": sorted(unicos[n_train + n_val :].tolist()),
    }
    destino.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def cargar_split(path=SPLIT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def indices_de(cache: dict, particion: str, split: dict | None = None) -> np.ndarray:
    """Posiciones en la cache de las secuencias de una particion (train/val/test)."""
    split = split or cargar_split()
    return np.flatnonzero(np.isin(cache["participant_id"], split[particion]))


class LandmarkDataset(Dataset):
    """Devuelve (x (T, F), etiqueta) de una particion; transform recibe y devuelve x."""

    def __init__(self, cache: dict, particion: str, max_frames: int = 384, transform=None):
        self.cache = cache
        self.idx = indices_de(cache, particion)
        self.max_frames = max_frames
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def secuencia(self, i: int) -> np.ndarray:
        j = self.idx[i]
        ini, fin = self.cache["offsets"][j], self.cache["offsets"][j + 1]
        return self.cache["X"][ini:fin].astype(np.float32)

    def __getitem__(self, i):
        x = self.secuencia(i)
        if self.transform is not None:
            x = self.transform(x)
        x = features.submuestrear(x, self.max_frames)
        y = vocab.encode(self.cache["phrase"][self.idx[i]])
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.long)


def collate_ctc(batch):
    """Padding a (B, T, F) y etiquetas (B, L) con sus longitudes."""
    xs, ys = zip(*batch)
    x_len = torch.tensor([len(x) for x in xs])
    y_len = torch.tensor([len(y) for y in ys])
    return pad_sequence(xs, batch_first=True), x_len, pad_sequence(ys, batch_first=True), y_len


def crear_dataloaders(batch_size: int = 32, max_frames: int = 384, transform=None, num_workers: int = 0) -> dict:
    """DataLoaders de train (con transform y shuffle), val y test."""
    cache = cargar_cache()
    loaders = {}
    for particion in ("train", "val", "test"):
        es_train = particion == "train"
        ds = LandmarkDataset(cache, particion, max_frames, transform if es_train else None)
        loaders[particion] = DataLoader(
            ds, batch_size=batch_size, shuffle=es_train, collate_fn=collate_ctc, num_workers=num_workers
        )
    return loaders


if __name__ == "__main__":
    cache = construir_cache()
    if not SPLIT_PATH.exists():
        crear_split(cache["participant_id"])
