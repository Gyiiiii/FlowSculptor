import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STDN_ROOT = Path(r"D:\baseline\STDN-main\STDN-main")
SCALES = ("data_25", "data_50", "data_75", "data_100", "graph_25", "graph_50", "graph_75", "graph_100")


def parse_args():
    parser = argparse.ArgumentParser(description="Run STDN scalability benchmark.")
    parser.add_argument("--stdn_root", default=str(DEFAULT_STDN_ROOT))
    parser.add_argument("--dataset", default="PEMS04", choices=["PEMS04"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only_scale", default=None, choices=SCALES)
    return parser.parse_args()


def scale_names(only_scale):
    names = [f"PEMS04_{item}" for item in SCALES]
    if only_scale:
        names = [f"PEMS04_{only_scale}"]
    return names


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path, row):
    fieldnames = [
        "Dataset",
        "Model",
        "Scale Type",
        "Scale",
        "Scale Percent",
        "Nodes",
        "Time Steps",
        "Params",
        "Train Time / Epoch (s)",
        "Inference Time / Batch (ms)",
        "Inference Time / Test Set (s)",
        "Peak GPU Memory (MB)",
        "Device",
        "Source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def update_train_config(config_path, epochs, batch_size, device):
    import configparser

    config = configparser.ConfigParser()
    read_files = config.read(config_path)
    if not read_files:
        raise FileNotFoundError(f"Cannot read STDN config: {config_path}")
    config["Training"]["epochs"] = str(epochs)
    config["Training"]["batch_size"] = str(batch_size)
    config["Training"]["patience"] = str(max(epochs + 1, 10))
    config["Training"]["gpu"] = "1" if device.startswith("cuda") else "0"
    config["Training"]["ctx"] = device.split(":", 1)[1] if ":" in device else "0"
    with open(config_path, "w", encoding="utf-8") as f:
        config.write(f)


def parse_metrics(text, batch_size, epochs):
    def find_float(pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    params = None
    match = re.search(r"trainable parameters:\s*([\d,]+)", text)
    if match:
        params = int(match.group(1).replace(",", ""))

    train_epoch = find_float(r"train one epoch for average:\s*([\d.]+)")
    inference_test = find_float(r"testing time:\s*([\d.]+)s")
    total_train = find_float(r"train time:\s*([\d.]+)\s*s")
    match = re.search(r"test\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", text)
    test_metrics = {"mae": None, "rmse": None, "mape": None}
    if match:
        test_metrics = {
            "mae": float(match.group(2)),
            "rmse": float(match.group(3)),
            "mape": float(match.group(4)),
        }

    test_samples = None
    match = re.search(r"test:\s*\((\d+),", text)
    if match:
        test_samples = int(match.group(1))
    test_batches = None
    if test_samples:
        test_batches = (test_samples + batch_size - 1) // batch_size

    inference_batch_ms = None
    if inference_test is not None and test_batches:
        inference_batch_ms = inference_test / test_batches * 1000.0

    return {
        "num_params": params,
        "avg_train_epoch_seconds": train_epoch,
        "total_train_seconds": total_train,
        "inference_test_seconds": inference_test,
        "inference_batch_ms": inference_batch_ms,
        "test_samples": test_samples,
        "test_batches": test_batches,
        "test_metrics": test_metrics,
        "epochs": epochs,
    }


def run_scale(scale_name, args):
    stdn_root = Path(args.stdn_root)
    meta_path = PROJECT_ROOT / "data" / "scalability" / scale_name / "scale_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing scale metadata: {meta_path}. Run prepare_scalability_data.py first.")
    meta = read_json(meta_path)
    config_path = stdn_root / "conf" / "scalability" / f"{scale_name}_1dim_12.conf"
    sample_path = stdn_root / "data" / "scalability" / scale_name / "samples_12_12_5.npz"
    if not sample_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Missing STDN samples/config for {scale_name}. Run prepare_stdn_scalability_data.py first."
        )

    update_train_config(config_path, args.epochs, args.batch_size, args.device)
    out_dir = PROJECT_ROOT / "runs" / "scalability" / "STDN" / scale_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdn_stdout.log"

    command = [sys.executable, "train.py", "--config", str(config_path)]
    print("Running:", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(stdn_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_chunks = []
    with open(stdout_path, "w", encoding="utf-8", errors="ignore") as log_file:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            output_chunks.append(line)
    return_code = process.wait()
    stdout_text = "".join(output_chunks)
    if return_code != 0:
        raise RuntimeError(f"STDN failed for {scale_name}. See {stdout_path}")

    metrics = parse_metrics(stdout_text, args.batch_size, args.epochs)
    result = {
        "dataset": args.dataset,
        "model": "STDN",
        "scale_name": scale_name,
        "scale_type": meta["scale_type"],
        "scale_percent": int(meta["scale_percent"]),
        "num_nodes": int(meta["num_nodes"]),
        "time_steps": int(meta["time_steps"]),
        "device": args.device,
        "source": "local scalability benchmark",
        "scale_meta": meta,
        "stdn_config": str(config_path),
        "stdout_log": str(stdout_path),
        "peak_gpu_memory_mb": None,
    }
    result.update(metrics)
    write_json(out_dir / "efficiency_metrics.json", result)
    write_csv(
        out_dir / "efficiency_metrics.csv",
        {
            "Dataset": args.dataset,
            "Model": "STDN",
            "Scale Type": result["scale_type"],
            "Scale": scale_name,
            "Scale Percent": result["scale_percent"],
            "Nodes": result["num_nodes"],
            "Time Steps": result["time_steps"],
            "Params": result["num_params"],
            "Train Time / Epoch (s)": result["avg_train_epoch_seconds"],
            "Inference Time / Batch (ms)": result["inference_batch_ms"],
            "Inference Time / Test Set (s)": result["inference_test_seconds"],
            "Peak GPU Memory (MB)": result["peak_gpu_memory_mb"],
            "Device": result["device"],
            "Source": result["source"],
        },
    )
    print(f"Saved STDN scalability result: {out_dir / 'efficiency_metrics.json'}")


def main():
    args = parse_args()
    for scale_name in scale_names(args.only_scale):
        run_scale(scale_name, args)


if __name__ == "__main__":
    main()
