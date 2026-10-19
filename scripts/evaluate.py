import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hflitestnet.data import DATASET_META, StandardScaler, canonical_dataset_name, get_dataloaders, load_adjacency
from hflitestnet.engine import build_high_flow_rows, evaluate, save_interval_csv, save_json
from hflitestnet.model import HFLiteSTNet, HFSTFormer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained HF-STFormer checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    dataset = canonical_dataset_name(config["data"]["dataset"])
    meta = DATASET_META[dataset]
    scaler = StandardScaler.from_state_dict(ckpt["scaler"])
    loaders, _, info = get_dataloaders(
        data_root=args.data_root,
        dataset=dataset,
        lag=config["data"]["lag"],
        horizon=config["data"]["horizon"],
        batch_size=config["train"]["batch_size"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        scaler_scope=config["data"].get("scaler_scope", "all"),
        num_workers=0,
    )
    static_adj = None
    if config["model"].get("use_static_adj", True):
        static_adj = torch.from_numpy(load_adjacency(args.data_root, dataset, meta["num_nodes"])).float().to(device)

    architecture = config["model"].get("architecture", "lite").lower()
    if architecture == "transformer":
        model = HFSTFormer(
            num_nodes=meta["num_nodes"],
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
            num_nodes=meta["num_nodes"],
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
    model.load_state_dict(ckpt["model_state_dict"])

    metrics, h_metrics, pred, true = evaluate(
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

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    save_interval_csv(output_dir / "eval_high_flow_interval_metrics.csv", high_rows)
    save_json(output_dir / "eval_metrics.json", {"test": metrics, "horizon": h_metrics, "high_flow": high_rows})

    print("Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%".format(**metrics))
    print("Saved:", output_dir / "eval_high_flow_interval_metrics.csv")


if __name__ == "__main__":
    main()
