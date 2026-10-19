import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


DATASET_ALIASES: Dict[str, str] = {
    "PEMS3": "PEMS03",
    "PEMS03": "PEMS03",
    "PEMS4": "PEMS04",
    "PEMS04": "PEMS04",
    "PEMS7": "PEMS07",
    "PEMS07": "PEMS07",
    "PEMS8": "PEMS08",
    "PEMS08": "PEMS08",
    "MANCHESTER": "MANCHESTER",
}


DATASET_META = {
    "PEMS03": {"num_nodes": 358, "folder": "PEMS03", "npz": "PEMS03.npz", "csv": "PEMS03.csv"},
    "PEMS04": {"num_nodes": 307, "folder": "PEMS04", "npz": "PEMS04.npz", "csv": "PEMS04.csv"},
    "PEMS07": {"num_nodes": 883, "folder": "PEMS07", "npz": "PEMS07.npz", "csv": "PEMS07.csv"},
    "PEMS08": {"num_nodes": 170, "folder": "PEMS08", "npz": "PEMS08.npz", "csv": "PEMS08.csv"},
    "MANCHESTER": {"num_nodes": 277, "folder": "Manchester", "npz": "ManchesterDataFinall.npz", "csv": "Manchester.csv"},
}


class StandardScaler:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if float(std) > 0 else 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"], state["std"])


def canonical_dataset_name(dataset: str) -> str:
    key = dataset.upper()
    if key not in DATASET_ALIASES:
        raise ValueError("Unsupported dataset '{}'. Use one of {}.".format(dataset, sorted(DATASET_ALIASES)))
    return DATASET_ALIASES[key]


def load_pems_data(data_root: str, dataset: str) -> np.ndarray:
    name = canonical_dataset_name(dataset)
    meta = DATASET_META[name]
    path = Path(data_root) / meta["folder"] / meta["npz"]
    if not path.exists():
        raise FileNotFoundError("Dataset file not found: {}".format(path))
    data = np.load(path)["data"]
    if data.ndim == 2:
        data = data[..., None]
    elif data.ndim == 3:
        data = data[:, :, :1]
    else:
        raise ValueError("Expected PEMS data with shape [T,N] or [T,N,D], got {}".format(data.shape))
    return data.astype(np.float32)


def split_by_ratio(data: np.ndarray, val_ratio: float, test_ratio: float):
    data_len = data.shape[0]
    test_len = int(data_len * test_ratio)
    val_len = int(data_len * val_ratio)
    train = data[: data_len - val_len - test_len]
    val = data[data_len - val_len - test_len : data_len - test_len]
    test = data[data_len - test_len :]
    return train, val, test


def add_window_horizon(data: np.ndarray, lag: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    end = len(data) - lag - horizon + 1
    if end <= 0:
        raise ValueError("Data length {} is too short for lag={} and horizon={}.".format(len(data), lag, horizon))
    x = np.stack([data[i : i + lag] for i in range(end)], axis=0)
    y = np.stack([data[i + lag : i + lag + horizon] for i in range(end)], axis=0)
    return x.astype(np.float32), y.astype(np.float32)


def make_loader(x, y, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int = 0):
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_dataloaders(
    data_root: str,
    dataset: str,
    lag: int,
    horizon: int,
    batch_size: int,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    scaler_scope: str = "all",
    num_workers: int = 0,
    drop_last_train: bool = True,
):
    raw = load_pems_data(data_root, dataset)
    train_raw, val_raw, test_raw = split_by_ratio(raw, val_ratio, test_ratio)

    if scaler_scope == "all":
        scaler_data = raw
    elif scaler_scope == "train":
        scaler_data = train_raw
    else:
        raise ValueError("scaler_scope must be 'all' or 'train'.")
    scaler = StandardScaler(scaler_data.mean(), scaler_data.std())

    train = scaler.transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)

    x_train, y_train = add_window_horizon(train, lag, horizon)
    x_val, y_val = add_window_horizon(val, lag, horizon)
    x_test, y_test = add_window_horizon(test, lag, horizon)

    loaders = {
        "train": make_loader(x_train, y_train, batch_size, shuffle=True, drop_last=drop_last_train, num_workers=num_workers),
        "val": make_loader(x_val, y_val, batch_size, shuffle=False, drop_last=False, num_workers=num_workers),
        "test": make_loader(x_test, y_test, batch_size, shuffle=False, drop_last=False, num_workers=num_workers),
    }
    info = {
        "raw_shape": tuple(raw.shape),
        "train_shape": tuple(x_train.shape),
        "val_shape": tuple(x_val.shape),
        "test_shape": tuple(x_test.shape),
        "flow_mean": float(raw[:, :, 0].mean()),
        "flow_max": float(raw[:, :, 0].max()),
        "flow_min": float(raw[:, :, 0].min()),
    }
    return loaders, scaler, info


def load_adjacency(data_root: str, dataset: str, num_nodes: int, add_self_loop: bool = True) -> np.ndarray:
    name = canonical_dataset_name(dataset)
    meta = DATASET_META[name]
    path = Path(data_root) / meta["folder"] / meta["csv"]
    if not path.exists():
        return np.eye(num_nodes, dtype=np.float32)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    with open(path, "r", newline="") as f:
        first = f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            try:
                i, j = int(row[0]), int(row[1])
            except ValueError:
                continue
            if 0 <= i < num_nodes and 0 <= j < num_nodes:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    if adj.sum() == 0:
        try:
            matrix = np.loadtxt(path, delimiter=",", dtype=np.float32)
            if matrix.shape == (num_nodes, num_nodes):
                adj = matrix
        except Exception:
            adj = np.eye(num_nodes, dtype=np.float32)

    if add_self_loop:
        np.fill_diagonal(adj, 1.0)
    degree = adj.sum(axis=1, keepdims=True)
    degree[degree == 0] = 1.0
    return (adj / degree).astype(np.float32)
