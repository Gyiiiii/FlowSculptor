from typing import Dict, List

import torch


def masked_select(pred, true, threshold=None):
    if threshold is None:
        return pred, true
    mask = true > threshold
    return pred[mask], true[mask]


def mae(pred, true, threshold=None):
    pred, true = masked_select(pred, true, threshold)
    return torch.mean(torch.abs(pred - true))


def rmse(pred, true, threshold=None):
    pred, true = masked_select(pred, true, threshold)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def mape(pred, true, threshold=0.0, eps=1e-5):
    pred, true = masked_select(pred, true, threshold)
    return torch.mean(torch.abs((pred - true) / torch.clamp(true, min=eps)))


def compute_metrics(pred, true, mae_thresh=0.0, mape_thresh=0.0) -> Dict[str, float]:
    return {
        "mae": float(mae(pred, true, mae_thresh).detach().cpu()),
        "rmse": float(rmse(pred, true, mae_thresh).detach().cpu()),
        "mape": float(mape(pred, true, mape_thresh).detach().cpu()) * 100.0,
    }


def horizon_metrics(pred, true, mae_thresh=0.0, mape_thresh=0.0) -> List[Dict[str, float]]:
    rows = []
    for t in range(true.shape[1]):
        item = compute_metrics(pred[:, t], true[:, t], mae_thresh, mape_thresh)
        item["horizon"] = t + 1
        rows.append(item)
    return rows


def high_flow_interval_metrics(
    pred,
    true,
    flow_mean: float,
    flow_max: float,
    interval_width: float = 100.0,
    mae_thresh=0.0,
    mape_thresh=0.0,
):
    if interval_width <= 0:
        raise ValueError("interval_width must be greater than 0.")

    rows = []
    lower = float(flow_mean)
    total = true.numel()
    while lower <= flow_max:
        upper = lower + interval_width
        is_last = upper > flow_max
        if is_last:
            mask = true >= lower
            label = "[{:.2f}, +inf)".format(lower)
        else:
            mask = (true >= lower) & (true < upper)
            label = "[{:.2f}, {:.2f})".format(lower, upper)

        count = int(mask.sum().detach().cpu())
        row = {
            "interval": label,
            "lower": lower,
            "upper": "" if is_last else upper,
            "count": count,
            "ratio": count / total if total else 0.0,
            "true_mean": "",
            "mae": "",
            "rmse": "",
            "mape": "",
        }
        if count > 0:
            p = pred[mask]
            y = true[mask]
            values = compute_metrics(p, y, mae_thresh, mape_thresh)
            row.update(values)
            row["true_mean"] = float(y.mean().detach().cpu())
        rows.append(row)
        lower = upper
    return rows
