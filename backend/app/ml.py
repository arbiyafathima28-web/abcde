from __future__ import annotations
import copy
from typing import Literal, Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    """Convolutional Neural Network tailored for 28x28 grayscale PneumoniaMNIST images."""
    def __init__(self):
        super().__init__()
        # Conv Block 1: 1 -> 16 channels, (28, 28) -> (14, 14)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: 16 -> 32 channels, (14, 14) -> (7, 7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification Head
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.dropout2 = nn.Dropout(0.30)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(-1, 1, 28, 28)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x).squeeze(-1)
        return logits

def get_params_vector(model: nn.Module) -> np.ndarray:
    """Flatten all model parameters into a 1D NumPy array."""
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])

def set_params_vector(model: nn.Module, flat_params: np.ndarray) -> None:
    """Load a 1D NumPy array into model parameter tensors."""
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        chunk = flat_params[idx : idx + numel]
        p.data.copy_(torch.from_numpy(chunk.reshape(p.shape)))
        idx += numel

def partition_dataset(y: np.ndarray, n_clients: int = 10, partition_type: Literal["iid", "non_iid"] = "iid", seed: int = 42) -> List[np.ndarray]:
    """Partitions dataset indices across simulated hospital clients under IID or non-IID conditions."""
    rng = np.random.default_rng(seed)
    n_samples = len(y)
    if partition_type == "iid":
        indices = rng.permutation(n_samples)
        return list(np.array_split(indices, n_clients))

    # Non-IID Dirichlet distribution over binary classes (simulating realistic hospital demographics)
    idx_0 = rng.permutation(np.where(y == 0)[0])
    idx_1 = rng.permutation(np.where(y == 1)[0])

    prop_0 = rng.dirichlet(np.repeat(0.5, n_clients))
    prop_1 = rng.dirichlet(np.repeat(0.5, n_clients))

    counts_0 = (prop_0 * len(idx_0)).astype(int)
    counts_1 = (prop_1 * len(idx_1)).astype(int)
    counts_0[0] += len(idx_0) - counts_0.sum()
    counts_1[0] += len(idx_1) - counts_1.sum()

    client_partitions = []
    p0, p1 = 0, 0
    for c in range(n_clients):
        c0, c1 = counts_0[c], counts_1[c]
        cli_idx = np.concatenate([idx_0[p0 : p0 + c0], idx_1[p1 : p1 + c1]])
        rng.shuffle(cli_idx)
        client_partitions.append(cli_idx)
        p0 += c0
        p1 += c1

    return client_partitions

def train_local(
    global_model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 2,
    lr: float = 0.002,
    batch_size: int = 32,
    device: str = "cpu"
) -> Tuple[np.ndarray, List[float]]:
    """Local client training for federated rounds; returns parameter update delta and loss curve."""
    if len(x) == 0:
        return np.zeros(sum(p.numel() for p in global_model.parameters()), dtype=np.float32), [0.0]

    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)

    # Balance loss if class skew exists
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    pos_weight = torch.tensor([max(0.2, min(5.0, n_neg / max(1, n_pos)))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_tensor = torch.from_numpy(x).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for _ in range(epochs):
        ep_loss = 0.0
        total_samples = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = local_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item()) * len(by)
            total_samples += len(by)
        losses.append(round(ep_loss / max(1, total_samples), 4))

    w_initial = get_params_vector(global_model)
    w_updated = get_params_vector(local_model)
    delta = w_updated - w_initial
    return delta, losses

def metrics(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str = "cpu", batch_size: int = 64) -> Dict[str, Any]:
    """Computes genuine clinical performance metrics on independent held-out data."""
    if len(x) == 0:
        return {
            "loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0,
            "confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        }

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(x).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for bx, by in loader:
            logits = model(bx)
            loss = criterion(logits, by)
            total_loss += float(loss.item()) * len(by)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.cpu().numpy().astype(int))

  
    all_preds = np.asarray(all_preds)
    all_targets = np.asarray(all_targets)
    avg_loss = total_loss / max(1, len(all_targets))

    tp = int(((all_preds == 1) & (all_targets == 1)).sum())
    tn = int(((all_preds == 0) & (all_targets == 0)).sum())
    fp = int(((all_preds == 1) & (all_targets == 0)).sum())
    fn = int(((all_preds == 0) & (all_targets == 1)).sum())

    total = max(1, len(all_targets))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)

    return {
        "loss": round(float(avg_loss), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }

def fedavg(updates: List[np.ndarray], weights: List[int]) -> np.ndarray:
    """Standard Federated Averaging weighted by client sample volume."""
    return np.average(np.stack(updates), axis=0, weights=np.asarray(weights, dtype=np.float32))

def coordinate_median(updates: List[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median robust aggregation."""
    return np.median(np.stack(updates), axis=0)

def trimmed_mean(updates: List[np.ndarray], trim_ratio: float = 0.2) -> np.ndarray:
    """Trimmed mean robust aggregation, discarding high/low extreme quantiles."""
    arr = np.sort(np.stack(updates), axis=0)
    trim = min(int(len(updates) * trim_ratio), max(0, (len(updates) - 1) // 2))
    if trim > 0:
        arr = arr[trim:-trim]
    return arr.mean(axis=0)

def inspect_updates(updates: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multi-metric Byzantine update anomaly detection pipeline."""
    matrix = np.stack(updates)
    n_clients = len(updates)

    # 1. Finite-value verification (reject NaN / Inf poisoned payloads)
    is_finite = np.all(np.isfinite(matrix), axis=1)
    clean_matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)

    # 2. L2 update magnitude norm
    norms = np.linalg.norm(clean_matrix, axis=1)

    # 3. Pairwise cosine similarity alignment
    unit = clean_matrix / np.maximum(norms[:, None], 1e-12)
    cosine = unit @ unit.T
    mean_similarity = (cosine.sum(axis=1) - 1.0) / max(1, n_clients - 1)

    # 4. Geometric deviation from coordinate-wise median
    median_update = np.median(clean_matrix, axis=0)
    deviations = np.linalg.norm(clean_matrix - median_update, axis=1)

    # 5. Median Absolute Deviation (MAD) robust z-score
    med_dev = np.median(deviations)
    mad = np.median(np.abs(deviations - med_dev)) + 1e-9
    robust_z = 0.6745 * (deviations - med_dev) / mad

    # Byzantine quarantine trigger: non-finite OR statistical outlier (z > 3.0) OR anti-correlated direction
    flags = (~is_finite) | (robust_z > 3.0) | (mean_similarity < -0.1)
    return norms, mean_similarity, deviations, robust_z, flags, cosine
