import argparse
import csv
import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGCRN_ROOT = Path(r"D:\baseline\AGCRN-master\AGCRN-master")
SCALES = (25, 50, 75, 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PEMS04 scalability datasets for HF-STFormer and AGCRN.")
    parser.add_argument("--dataset", default="PEMS04", choices=["PEMS04"])
    parser.add_argument("--hf_root", default=str(PROJECT_ROOT))
    parser.add_argument("--agcrn_root", default=str(DEFAULT_AGCRN_ROOT))
    parser.add_argument("--seed", type=int, default=10)
    return parser.parse_args()


def read_edges(csv_path):
    edges = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                src = int(row[0])
                dst = int(row[1])
                cost = row[2]
            except ValueError:
                continue
            edges.append((src, dst, cost))
    return header or ["from", "to", "cost"], edges


def bfs_node_order(num_nodes, edges):
    graph = {idx: set() for idx in range(num_nodes)}
    for src, dst, _ in edges:
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            graph[src].add(dst)
            graph[dst].add(src)

    start = max(graph, key=lambda node: len(graph[node]))
    seen = {start}
    order = []
    queue = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in sorted(graph[node], key=lambda item: (-len(graph[item]), item)):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    remaining = [node for node in range(num_nodes) if node not in seen]
    remaining.sort(key=lambda item: (-len(graph[item]), item))
    return order + remaining


def write_edges(csv_path, header, edges, selected_nodes):
    selected = list(selected_nodes)
    mapping = {old: new for new, old in enumerate(selected)}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for src, dst, cost in edges:
            if src in mapping and dst in mapping:
                writer.writerow([mapping[src], mapping[dst], cost])


def write_scale(target_dir, data, header, edges, selected_nodes, meta):
    target_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_dir / "PEMS04.npz", data=data.astype(np.float32))
    write_edges(target_dir / "PEMS04.csv", header, edges, selected_nodes)
    with open(target_dir / "selected_nodes.json", "w", encoding="utf-8") as f:
        json.dump({"selected_nodes": list(map(int, selected_nodes))}, f, indent=2)
    with open(target_dir / "scale_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def mirror_to_agcrn(source_dir, agcrn_root, scale_name):
    target_dir = Path(agcrn_root) / "data" / "scalability" / scale_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("PEMS04.npz", "PEMS04.csv", "selected_nodes.json", "scale_meta.json"):
        shutil.copy2(source_dir / filename, target_dir / filename)


def main():
    args = parse_args()
    hf_root = Path(args.hf_root)
    agcrn_root = Path(args.agcrn_root)
    source_dir = hf_root / "data" / args.dataset
    data = np.load(source_dir / "PEMS04.npz")["data"].astype(np.float32)
    if data.ndim == 2:
        data = data[..., None]
    elif data.ndim == 3:
        data = data[:, :, :1]
    header, edges = read_edges(source_dir / "PEMS04.csv")
    num_steps, num_nodes = data.shape[:2]
    node_order = bfs_node_order(num_nodes, edges)

    output_root = hf_root / "data" / "scalability"
    output_root.mkdir(parents=True, exist_ok=True)

    for scale in SCALES:
        length = max(data.shape[0] * scale // 100, 24)
        scale_name = f"PEMS04_data_{scale}"
        scale_dir = output_root / scale_name
        selected_nodes = list(range(num_nodes))
        scaled_data = data[:length]
        meta = {
            "dataset": args.dataset,
            "scale_name": scale_name,
            "scale_type": "data",
            "scale_percent": scale,
            "time_steps": int(scaled_data.shape[0]),
            "num_nodes": int(scaled_data.shape[1]),
            "source_time_steps": int(num_steps),
            "source_num_nodes": int(num_nodes),
            "seed": args.seed,
        }
        write_scale(scale_dir, scaled_data, header, edges, selected_nodes, meta)
        mirror_to_agcrn(scale_dir, agcrn_root, scale_name)

    for scale in SCALES:
        keep = max(num_nodes * scale // 100, 1)
        scale_name = f"PEMS04_graph_{scale}"
        scale_dir = output_root / scale_name
        selected_nodes = sorted(node_order[:keep])
        scaled_data = data[:, selected_nodes, :]
        meta = {
            "dataset": args.dataset,
            "scale_name": scale_name,
            "scale_type": "graph",
            "scale_percent": scale,
            "time_steps": int(scaled_data.shape[0]),
            "num_nodes": int(scaled_data.shape[1]),
            "source_time_steps": int(num_steps),
            "source_num_nodes": int(num_nodes),
            "seed": args.seed,
        }
        write_scale(scale_dir, scaled_data, header, edges, selected_nodes, meta)
        mirror_to_agcrn(scale_dir, agcrn_root, scale_name)

    print(f"Prepared scalability datasets under {output_root}")
    print(f"Mirrored datasets under {agcrn_root / 'data' / 'scalability'}")


if __name__ == "__main__":
    main()
