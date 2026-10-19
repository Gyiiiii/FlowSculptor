import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hflitestnet.data import DATASET_META, canonical_dataset_name, get_dataloaders, load_adjacency
from hflitestnet.engine import (
    build_high_flow_rows,
    count_parameters,
    evaluate,
    load_json,
    run_epoch,
    save_checkpoint,
    save_interval_csv,
    save_json,
    set_seed,
)
from hflitestnet.losses import HighFlowAwareLoss
from hflitestnet.model import HFLiteSTNet, HFSTFormer


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Train HF-STFormer on PEMS datasets.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "pems04.json"))
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--log_file", type=str, default=None, help="Default: <output_dir>/train.log")
    parser.add_argument("--quick_run", action="store_true", help="Run only two mini-batches for shape/debug checks.")
    return parser.parse_args()


def merge_overrides(config, args):
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.output_dir is not None:
        config["train"]["output_dir"] = args.output_dir
    return config


def main():
    args = parse_args()
    config = merge_overrides(load_json(args.config), args)
    dataset = canonical_dataset_name(config["data"]["dataset"])
    meta = DATASET_META[dataset]
    set_seed(config["train"]["seed"])

    device_name = config["train"].get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    output_dir = Path(config["train"].get("output_dir") or (PROJECT_ROOT / "runs" / dataset))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else output_dir / "train.log"
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8")
    original_stdout = sys.__stdout__
    original_stderr = sys.__stderr__
    sys.stdout = TeeLogger(original_stdout, log_handle)
    sys.stderr = TeeLogger(original_stderr, log_handle)

    print("=" * 100)
    print("HF-STFormer training started at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Command:", " ".join(sys.argv))
    print("Config:", Path(args.config).resolve())
    print("Output dir:", output_dir)
    print("Log file:", log_path)
    save_json(output_dir / "config_used.json", config)

    loaders, scaler, info = get_dataloaders(
        data_root=args.data_root,
        dataset=dataset,
        lag=config["data"]["lag"],
        horizon=config["data"]["horizon"],
        batch_size=config["train"]["batch_size"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        scaler_scope=config["data"].get("scaler_scope", "all"),
        num_workers=config["train"].get("num_workers", 0),
    )
    num_nodes = meta["num_nodes"]
    static_adj = None
    if config["model"].get("use_static_adj", True):
        static_adj = torch.from_numpy(load_adjacency(args.data_root, dataset, num_nodes)).float().to(device)

    architecture = config["model"].get("architecture", "lite").lower()
    if architecture == "transformer":
        model = HFSTFormer(
            num_nodes=num_nodes,
            lag=config["data"]["lag"],
            input_dim=config["data"]["input_dim"],
            output_dim=config["data"]["output_dim"],
            horizon=config["data"]["horizon"],
            channels=config["model"]["channels"],
            temporal_layers=config["model"].get("temporal_layers", 2),
            spatial_layers=config["model"].get("spatial_layers", 2),
            attention_heads=config["model"].get("attention_heads", 4),
            ffn_ratio=config["model"].get("ffn_ratio", 2.0),
            dropout=config["model"]["dropout"],
            residual_prediction=config["model"].get("residual_prediction", True),
            high_flow_branch=config["model"].get("high_flow_branch", False),
            high_flow_gate_hidden=config["model"].get("high_flow_gate_hidden", 64),
            high_flow_gate_bias=config["model"].get("high_flow_gate_bias", -2.0),
            use_linear_baseline=config["model"].get("use_linear_baseline", True),
            use_temporal_encoder=config["model"].get("use_temporal_encoder", True),
            use_spatial_encoder=config["model"].get("use_spatial_encoder", True),
            use_readout_fusion=config["model"].get("use_readout_fusion", True),
            use_horizon_decoder=config["model"].get("use_horizon_decoder", True),
        ).to(device)
    else:
        model = HFLiteSTNet(
            num_nodes=num_nodes,
            lag=config["data"]["lag"],
            input_dim=config["data"]["input_dim"],
            output_dim=config["data"]["output_dim"],
            horizon=config["data"]["horizon"],
            channels=config["model"]["channels"],
            temporal_layers=config["model"]["temporal_layers"],
            graph_embed_dim=config["model"]["graph_embed_dim"],
            graph_topk=config["model"]["graph_topk"],
            dropout=config["model"]["dropout"],
            residual_prediction=config["model"].get("residual_prediction", True),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["train"].get("lr_factor", 0.5),
        patience=config["train"].get("lr_patience", 5),
    )
    loss_fn = HighFlowAwareLoss(
        flow_mean=info["flow_mean"],
        interval_width=config["high_flow"]["interval_width"],
        alpha=config["high_flow"]["loss_alpha"],
        lambda_high=config["high_flow"]["lambda_high"],
        lambda_rel=config["high_flow"]["lambda_rel"],
        horizon_weight_gamma=config["high_flow"].get("horizon_weight_gamma", 0.0),
        global_loss=config["high_flow"].get("global_loss", "smooth_l1"),
    )

    print("Dataset:", dataset, info)
    print("Device:", device)
    print("Parameters:", count_parameters(model))

    best_val = float("inf")
    best_path = output_dir / "best_model.pth"
    patience = config["train"]["early_stop_patience"]
    stale = 0
    start_time = time.time()

    for epoch in range(1, config["train"]["epochs"] + 1):
        warmup_epochs = int(config["high_flow"].get("warmup_epochs", 0))
        if warmup_epochs > 0:
            loss_fn.set_high_flow_scale(min(1.0, epoch / warmup_epochs))
        if args.quick_run:
            # Temporarily train on two mini-batches to validate dimensions quickly.
            small_batches = []
            for idx, batch in enumerate(loaders["train"]):
                small_batches.append(batch)
                if idx == 1:
                    break
            train_stats = run_epoch(
                model,
                small_batches,
                optimizer,
                loss_fn,
                scaler,
                device,
                static_adj,
                grad_clip=config["train"].get("grad_clip", 0.0),
            )
        else:
            train_stats = run_epoch(
                model,
                loaders["train"],
                optimizer,
                loss_fn,
                scaler,
                device,
                static_adj,
                grad_clip=config["train"].get("grad_clip", 0.0),
            )

        val_metrics, _, _, _ = evaluate(
            model,
            loaders["val"],
            scaler,
            device,
            static_adj,
            mae_thresh=config["eval"]["mae_thresh"],
            mape_thresh=config["eval"]["mape_thresh"],
        )
        scheduler.step(val_metrics["mae"])
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "Epoch {:03d} | loss {:.4f} | global {:.4f} | high_abs {:.4f} | high_rel {:.4f} | val MAE {:.4f} RMSE {:.4f} MAPE {:.4f}% | lr {:.8f}".format(
                epoch,
                train_stats["loss"],
                train_stats["global"],
                train_stats["high_abs"],
                train_stats["high_rel"],
                val_metrics["mae"],
                val_metrics["rmse"],
                val_metrics["mape"],
                current_lr,
            )
        )

        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            stale = 0
            save_checkpoint(best_path, model, optimizer, config, scaler, best_val, epoch, info)
        else:
            stale += 1
        if args.quick_run or stale >= patience:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, h_metrics, pred, true = evaluate(
        model,
        loaders["test"],
        scaler,
        device,
        static_adj,
        mae_thresh=config["eval"]["mae_thresh"],
        mape_thresh=config["eval"]["mape_thresh"],
    )
    high_rows = build_high_flow_rows(
        pred,
        true,
        info,
        config["high_flow"]["interval_width"],
        mae_thresh=config["eval"]["mae_thresh"],
        mape_thresh=config["eval"]["mape_thresh"],
    )
    save_interval_csv(output_dir / "high_flow_interval_metrics.csv", high_rows)
    result = {
        "test": test_metrics,
        "horizon": h_metrics,
        "high_flow": high_rows,
        "best_val_mae": best_val,
        "seconds": time.time() - start_time,
        "num_params": count_parameters(model),
    }
    save_json(output_dir / "metrics.json", result)

    print("Best checkpoint:", best_path)
    print("Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%".format(**test_metrics))
    print("High-flow interval metrics saved to:", output_dir / "high_flow_interval_metrics.csv")
    print("HF-STFormer training finished at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()


if __name__ == "__main__":
    main()
