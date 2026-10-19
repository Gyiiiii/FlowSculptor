import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("HF-STFormer", "AGCRN", "STDN")
SCALE_ORDER = {
    "data": {"PEMS04_data_25": 25, "PEMS04_data_50": 50, "PEMS04_data_75": 75, "PEMS04_data_100": 100},
    "graph": {"PEMS04_graph_25": 25, "PEMS04_graph_50": 50, "PEMS04_graph_75": 75, "PEMS04_graph_100": 100},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize scalability benchmark outputs.")
    parser.add_argument("--dataset", default="PEMS04", choices=["PEMS04"])
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows():
    rows = []
    for model in MODELS:
        root = PROJECT_ROOT / "runs" / "scalability" / model
        if not root.exists():
            continue
        for metrics_path in sorted(root.glob("*/efficiency_metrics.json")):
            item = read_json(metrics_path)
            row = {
                "Dataset": item.get("dataset", "PEMS04"),
                "Model": item.get("model", model),
                "Scale Type": item.get("scale_type"),
                "Scale": item.get("scale_name"),
                "Scale Percent": item.get("scale_percent"),
                "Nodes": item.get("num_nodes"),
                "Time Steps": item.get("time_steps"),
                "Params": item.get("num_params"),
                "Epochs": item.get("epochs"),
                "Train Time / Epoch (s)": item.get("avg_train_epoch_seconds"),
                "Inference Time / Batch (ms)": item.get("inference_batch_ms"),
                "Inference Time / Test Set (s)": item.get("inference_test_seconds"),
                "Peak GPU Memory (MB)": item.get("peak_gpu_memory_mb"),
                "Test MAE": (item.get("test_metrics") or {}).get("mae"),
                "Test RMSE": (item.get("test_metrics") or {}).get("rmse"),
                "Test MAPE": (item.get("test_metrics") or {}).get("mape"),
                "Device": item.get("device"),
                "Source": item.get("source"),
                "Metrics Path": str(metrics_path),
            }
            rows.append(row)
    rows.sort(key=lambda row: (row["Scale Type"], row["Scale Percent"], row["Model"]))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Dataset",
        "Model",
        "Scale Type",
        "Scale",
        "Scale Percent",
        "Nodes",
        "Time Steps",
        "Params",
        "Epochs",
        "Train Time / Epoch (s)",
        "Inference Time / Batch (ms)",
        "Inference Time / Test Set (s)",
        "Peak GPU Memory (MB)",
        "Test MAE",
        "Test RMSE",
        "Test MAPE",
        "Device",
        "Source",
        "Metrics Path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parse_args()
    rows = collect_rows()
    out_root = PROJECT_ROOT / "experiments" / "scalability"
    raw_path = out_root / "scalability_raw_results.csv"
    summary_path = out_root / "scalability_summary.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, rows)
    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
