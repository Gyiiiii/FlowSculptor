import torch
import torch.nn as nn
import torch.nn.functional as F


class HighFlowAwareLoss(nn.Module):
    """Global error plus high-flow conditional risk.

    The model is optimized on the original traffic scale. The global term keeps
    average accuracy stable, while the high-flow terms increase gradients on
    samples whose true value is above the dataset mean.
    """

    def __init__(
        self,
        flow_mean: float,
        interval_width: float = 100.0,
        alpha: float = 0.2,
        lambda_high: float = 0.3,
        lambda_rel: float = 0.05,
        horizon_weight_gamma: float = 0.0,
        global_loss: str = "smooth_l1",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.flow_mean = float(flow_mean)
        self.interval_width = float(interval_width)
        self.alpha = float(alpha)
        self.lambda_high = float(lambda_high)
        self.lambda_rel = float(lambda_rel)
        self.horizon_weight_gamma = float(horizon_weight_gamma)
        self.global_loss = str(global_loss).lower()
        self.eps = float(eps)
        self.high_flow_scale = 1.0

    def set_high_flow_scale(self, scale: float):
        self.high_flow_scale = max(0.0, min(float(scale), 1.0))

    def horizon_weights(self, true):
        if self.horizon_weight_gamma <= 0 or true.dim() < 2:
            return None
        horizon = true.shape[1]
        weights = torch.linspace(
            1.0,
            1.0 + self.horizon_weight_gamma,
            steps=horizon,
            device=true.device,
            dtype=true.dtype,
        )
        weights = weights / weights.mean()
        return weights.view(1, horizon, 1, 1)

    def forward(self, pred, true):
        h_weights = self.horizon_weights(true)
        if self.global_loss == "mae":
            global_error = torch.abs(pred - true)
        elif self.global_loss == "mse":
            global_error = F.mse_loss(pred, true, reduction="none")
        elif self.global_loss == "smooth_l1":
            global_error = F.smooth_l1_loss(pred, true, reduction="none")
        else:
            raise ValueError("Unsupported global_loss '{}'. Use smooth_l1, mae, or mse.".format(self.global_loss))
        if h_weights is not None:
            global_error = global_error * h_weights
        global_loss = global_error.mean()

        high_mask = true >= self.flow_mean
        if high_mask.sum() == 0:
            return global_loss, {"global": global_loss.detach(), "high_abs": pred.new_tensor(0.0), "high_rel": pred.new_tensor(0.0)}

        level = torch.relu((true - self.flow_mean) / self.interval_width)
        weight = 1.0 + self.alpha * level
        if h_weights is not None:
            weight = weight * h_weights
        abs_error = torch.abs(pred - true)
        high_abs = (weight[high_mask] * abs_error[high_mask]).mean()
        high_rel = (weight[high_mask] * abs_error[high_mask] / torch.clamp(true[high_mask], min=self.eps)).mean() * 100.0
        loss = global_loss + self.high_flow_scale * (self.lambda_high * high_abs + self.lambda_rel * high_rel)
        parts = {"global": global_loss.detach(), "high_abs": high_abs.detach(), "high_rel": high_rel.detach()}
        return loss, parts
