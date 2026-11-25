#!/usr/bin/env python
"""Score a checkpoint and write the metrics JSON plus a PR curve.

Usage:
    python src/eval.py --model_name TrackNetFast --model_path path/to.ckpt \\
        [--batch_size 8] [--tol 4] [--val_games 1,2]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import HEIGHT, WIDTH, VAL_GAMES
from metrics import (FUSE_CLOSE_PX, both_protocols, detection_records,
                     point_distances, pr_points, save_pr_curve)
from models.registry import MODEL_NAMES
from util import get_dataset, get_model

def _empty_records():
    return {name: [] for name in
            ("score", "logit", "peak", "xy", "gt_xy", "visible")}


def _frame_keys_for(dataset, start, count):
    keys = []
    for offset in range(count):
        sample = dataset.samples[start + offset]
        keys.extend((sample['clip_folder'], name)
                    for name in sample['file_names'])
    return keys


@torch.no_grad()
def collect_records(model, dataset, device, batch_size=8, loader_kwargs=None,
                    use_amp=False, on_batch=None, desc="Evaluating",
                    height=None, width=None, loader=None):
    """Run the model over the dataset and collect per-slot records.

    ``on_batch(outputs, labels, batch_size)`` is called after each forward
    pass, so training can reuse this for validation loss.
    """
    model.eval()

    loader = loader or DataLoader(dataset, batch_size=batch_size,
                                  shuffle=False, **(loader_kwargs or {}))
    records = _empty_records()
    frame_keys = []
    total_frames = 0
    total_time = 0.0
    start = 0

    for batch in tqdm(loader, desc=desc):
        inputs, labels = batch
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        begin = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=use_amp):
            outputs = model(inputs)
        total_time += time.time() - begin
        total_frames += inputs.size(0) * 3

        batch_records = detection_records(outputs, labels)
        for name in records:
            records[name].append(batch_records[name])
        frame_keys.extend(_frame_keys_for(dataset, start, inputs.size(0)))
        start += inputs.size(0)

        if on_batch is not None:
            on_batch(outputs, labels, inputs.size(0))

    flat = {name: np.concatenate(values) for name, values in records.items()}
    return flat, frame_keys, total_frames, total_time


def parse_games(text):
    """``"1,2"`` / ``"game1,game2"`` -> ``["game1", "game2"]``."""
    if not text:
        return None
    return [g if g.startswith("game") else f"game{g}"
            for g in (item.strip() for item in text.split(",")) if g]


def load_model(model_name, model_path, device):
    model = get_model(model_name, HEIGHT, WIDTH, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {
        (k[10:] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def print_protocol(title, metrics):
    print(f"\n--- {title} ({metrics['n_frames']} samples, "
          f"{metrics['n_visible']} GT-visible) ---")
    print(f"TP={metrics['tp']}  FP={metrics['fp']} "
          f"(wrong-location {metrics['fp_wrong_location']} "
          f"+ no-ball {metrics['fp_invisible']})  FN={metrics['fn']} "
          f"(gate-closed {metrics['fn_gate_closed']} "
          f"+ wrong-location {metrics['fn_wrong_location']})")
    print("AP: " + " | ".join(
        f"{k}={metrics[k]:.4f}" for k in ("ap@4", "ap@8", "ap@12")))
    print(f"@0.5: P={metrics['precision@0.5']:.4f} "
          f"R={metrics['recall@0.5']:.4f} F1={metrics['f1@0.5']:.4f}")
    print(f"f1@best={metrics['f1@best']:.4f} "
          f"(P={metrics['precision@best']:.4f}, "
          f"R={metrics['recall@best']:.4f}, "
          f"th={metrics['threshold@best']:.3f})")
    if "f1@tuned" in metrics:
        print(f"@tuned th={metrics['tuned_threshold']:.3f}: "
              f"P={metrics['precision@tuned']:.4f} "
              f"R={metrics['recall@tuned']:.4f} "
              f"F1={metrics['f1@tuned']:.4f}")


def save_detection_metrics(model_path, result_dir, window_metrics,
                           frame_metrics, total_frames, total_time_taken):
    inference_speed = (total_frames / total_time_taken
                       if total_time_taken > 0 else 0.0)

    print_protocol("per-window", window_metrics)
    print_protocol("per-frame", frame_metrics)
    print(f"\nInference speed (frames per second): {inference_speed:.2f}")

    results = {
        "per_window": window_metrics,
        "per_frame": frame_metrics,
        "fusion_close_px": FUSE_CLOSE_PX,
        "inference_speed_fps": inference_speed,
        "total_frames": total_frames,
        "total_time_seconds": total_time_taken,
    }

    os.makedirs(result_dir, exist_ok=True)
    model_file_name = os.path.basename(model_path).replace(".pth", "")
    json_file_path = os.path.join(result_dir, f"{model_file_name}.json")
    with open(json_file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_file_path}")


def evaluate_model(model_name, model_path, batch_size, tol, result_dir,
                   threshold=None, val_games=None):
    """Run one checkpoint over the validation split and save the metrics."""
    config = {
        "model_name": model_name,
        "model_path": model_path,
        "batch_size": batch_size,
        "tol": tol,
        "threshold": threshold,
        "val_games": val_games or list(VAL_GAMES),
    }
    print("Evaluation Configurations:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_dataset = get_dataset(HEIGHT, WIDTH, val_games=val_games)
    if not len(test_dataset):
        raise SystemExit(
            f"No samples found for {config['val_games']} under "
            f"TENNIS_DATASET_ROOT. Check the dataset layout.")

    model = load_model(model_name, model_path, device)

    # The exact inference pass training uses, for every model contract.
    records, frame_keys, total_frames, total_time_taken = collect_records(
        model, test_dataset, device, batch_size=batch_size, desc="Evaluating",
        height=HEIGHT, width=WIDTH)

    window_metrics, frame_metrics, fused = both_protocols(
        records, frame_keys, tol, threshold=threshold)
    os.makedirs(result_dir, exist_ok=True)
    save_detection_metrics(model_path, result_dir, window_metrics,
                           frame_metrics, total_frames, total_time_taken)

    # Same PR curve as the final training evaluation, saved next to the JSON.
    recall_pts, precision_pts, _ = pr_points(
        fused["score"], point_distances(fused["xy"], fused["gt_xy"]),
        fused["visible"], tol=tol)
    tag = os.path.basename(model_path).replace(".pth", "")
    pr_path = os.path.join(result_dir, f"{tag}_pr_curve.png")
    figure = save_pr_curve(
        recall_pts, precision_pts, pr_path,
        title=f"{model_name} - PR curve (per frame, tol={tol}px)",
        best=(frame_metrics["recall@best"], frame_metrics["precision@best"]))
    import matplotlib.pyplot as plt
    plt.close(figure)
    print(f"PR curve saved to {pr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a TrackNet checkpoint with detection metrics."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, choices=list(MODEL_NAMES),
        help="Name of the model to use."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model weights file (.pth)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for evaluation (default: 8)."
    )
    parser.add_argument(
        "--tol", type=int, default=4,
        help="Pixel tolerance of the reported operating point; AP is always "
             "reported at 4, 8 and 12 (default: 4)."
    )
    parser.add_argument(
        "--result_dir", type=str, default="./",
        help="Directory to write the metrics JSON (default: './')."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Optional gate threshold calibrated elsewhere, reported as an "
             "extra operating point (does not change AP)."
    )
    parser.add_argument(
        "--val_games", type=str, default=None,
        help="Comma-separated games to evaluate, overriding "
             "constants.VAL_GAMES (e.g. '1,2')."
    )

    args = parser.parse_args()

    evaluate_model(
        args.model_name,
        args.model_path,
        args.batch_size,
        args.tol,
        args.result_dir,
        threshold=args.threshold,
        val_games=parse_games(args.val_games),
    )
