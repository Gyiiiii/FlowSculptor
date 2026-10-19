import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from .metrics import compute_metrics, high_flow_interval_metrics, horizon_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def move_batch(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def inverse_scale(tensor, scaler):
    return tensor * scaler.std + scaler.mean


def run_epoch(model, loader, optimizer, loss_fn, scaler, device, static_adj=None, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    parts_sum = {"global": 0.0, "high_abs": 0.0, "high_rel": 0.0}
    seen = 0
    for batch in loader:
        x, y = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred_norm = model(x, static_adj=static_adj)
        pred = inverse_scale(pred_norm, scaler)
        true = inverse_scale(y, scaler)
        loss, parts = loss_fn(pred, true)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        for key in parts_sum:
            parts_sum[key] += float(parts[key].detach().cpu()) * batch_size
        seen += batch_size

    result = {"loss": total_loss / max(seen, 1)}
    result.update({k: v / max(seen, 1) for k, v in parts_sum.items()})
    return result


@torch.no_grad()
def predict(model, loader, scaler, device, static_adj=None) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, trues = [], []
    for batch in loader:
        x, y = move_batch(batch, device)
        pred_norm = model(x, static_adj=static_adj)
        preds.append(inverse_scale(pred_norm, scaler).cpu())
        trues.append(inverse_scale(y, scaler).cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


@torch.no_grad()
def evaluate(model, loader, scaler, device, static_adj=None, mae_thresh=0.0, mape_thresh=0.0):
    pred, true = predict(model, loader, scaler, device, static_adj)
    return compute_metrics(pred, true, mae_thresh, mape_thresh), horizon_metrics(pred, true, mae_thresh, mape_thresh), pred, true


def save_checkpoint(path, model, optimizer, config: Dict, scaler, best_metric: float, epoch: int, info: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "config": config,
            "scaler": scaler.state_dict(),
            "best_metric": best_metric,
            "epoch": epoch,
            "info": info,
        },
        path,
    )


def load_json(path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_interval_csv(path, rows: Iterable[Dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = ["interval", "lower", "upper", "count", "ratio", "true_mean", "mae", "rmse", "mape"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_high_flow_rows(pred, true, info, interval_width, mae_thresh=0.0, mape_thresh=0.0):
    return high_flow_interval_metrics(
        pred=pred,
        true=true,
        flow_mean=info["flow_mean"],
        flow_max=info["flow_max"],
        interval_width=interval_width,
        mae_thresh=mae_thresh,
        mape_thresh=mape_thresh,
    )
