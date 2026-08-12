from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff, loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import config

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    n_clusters: int
    name: str
    parameters: str | None = None


def resolve_path(relative_path: str, join_base: bool = True) -> Path:
    if join_base:
        return config.BASE_DIR / relative_path

    return Path(relative_path)

def get_scaler(scaler_type: str = "standard"):
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    raise ValueError(f"scaler_type inválido: {scaler_type!r}. Use 'standard' ou 'minmax'.")

def apply_label_transform(values, transform):
    if transform == "sub1":
        return [v - 1 for v in values]
    if transform == "raw":
        return list(values)
    if isinstance(transform, dict):
        fallback = transform["else"]
        return [transform.get(v, fallback) for v in values]
    raise ValueError(f"label_transform inválido: {transform!r}")

def load_mat_dataset(cfg: dict):
    path = resolve_path(cfg["path"])
    mat = loadmat(path)
    X = mat["X"]
    y = [int(label) - 1 for label in mat["Y"].flatten().tolist()]
    return X, y

def load_csv_dataset(cfg: dict):
    path = resolve_path(cfg["path"], cfg.get("join_base", True))
    dataset = pd.read_csv(path, sep=cfg["sep"], header=None)

    raw_target = dataset.iloc[:, cfg["target_col"]].tolist()
    y = apply_label_transform(raw_target, cfg["label_transform"])

    drop_columns = dataset.columns[cfg["drop_cols"]]
    X = dataset.drop(columns=drop_columns).to_numpy()
    return X, y

def load_arff_dataset(cfg: dict):
    path = resolve_path(cfg["path"])
    raw_data, _ = arff.loadarff(path)
    dataset = pd.DataFrame(raw_data)
    dataset[dataset.columns[-1]] = dataset.iloc[:, -1].astype(int)

    extra_range = cfg.get("extra_drop_range")
    if extra_range is not None:
        start, end = extra_range
        dataset = dataset.drop(dataset.columns[start:end], axis=1)

    y = dataset.iloc[:, -1].tolist()
    X = dataset.drop(dataset.columns[-1], axis=1).to_numpy()
    return X, y

def load_dataset(
    dataset_id: int,
    scaler_type: str = "standard", 
    verbose: bool = True
) -> Dataset | None:

    data = None

    if dataset_id in config.MAT_DATASETS:
        cfg = config.MAT_DATASETS[dataset_id]
        X, y = load_mat_dataset(cfg)

    elif dataset_id in config.CSV_DATASETS:
        cfg = config.CSV_DATASETS[dataset_id]
        X, y = load_csv_dataset(cfg)

    elif dataset_id in config.ARFF_DATASETS:
        cfg = config.ARFF_DATASETS[dataset_id]
        X, y = load_arff_dataset(cfg)

    else:
        if verbose:
            print(f"Dataset ID {dataset_id} não encontrado.")
        return None

    X = get_scaler(scaler_type).fit_transform(X)

    data = Dataset(
        X=X,
        y=np.array(y),
        n_clusters=cfg["n_clusters"],
        name=cfg["name"],
        parameters=cfg.get("parameters")
    )

    if verbose:
        print(f"Dataset carregado: {data.name}")
        print(f"Shape: {data.X.shape}")
        print(f"Clusters: {data.n_clusters}")
        print(f"Parâmetros: {data.parameters}")
    
    return data
