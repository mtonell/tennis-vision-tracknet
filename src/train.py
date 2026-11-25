#!/usr/bin/env python
"""
Training script.

Trains one of the registered TrackNet models on the game-level split and
runs threshold-free validation metrics every epoch.

Usage:
    python src/train.py --model_name TrackNetFast [options]

See ``--help`` for the available flags.
"""

import argparse
import datetime
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

from util import get_dataset, get_model
from constants import TRAIN_GAMES, VAL_GAMES
from losses import FastTrackNetLoss, model_loss
from eval import collect_records
from metrics import both_protocols, point_distances, pr_points, FUSE_CLOSE_PX
from metrics import save_pr_curve
from models.registry import MODEL_NAMES, contract_of


class FocalLoss(nn.Module):
    """Focal loss for dense heatmap prediction.

    ``alpha`` weights foreground pixels against the background.
    """
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        # Clamp for numerical stability
        pred = pred.clamp(1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = target * pred + (1 - target) * (1 - pred)
        # alpha_t: ball pixels get `alpha` weight, background pixels get `1-alpha`
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


def _print_protocol(label, m):
    """One compact metrics block for one evaluation protocol."""
    print(f"  [{label}] {m['n_frames']} samples, {m['n_visible']} GT-visible "
          f"| TP={m['tp']} FP={m['fp']} "
          f"(wrong-location {m['fp_wrong_location']}"
          f" + no-ball {m['fp_invisible']}) FN={m['fn']} "
          f"(gate-closed {m['fn_gate_closed']}"
          f" + wrong-location {m['fn_wrong_location']})")
    print(f"  [{label}] AP: "
          + " | ".join(f"{k}={m[k]:.4f}" for k in
                       ("ap@4", "ap@8", "ap@12", "ap_peak@4"))
          + f" || @0.5 P={m['precision@0.5']:.4f} R={m['recall@0.5']:.4f} "
            f"F1={m['f1@0.5']:.4f} || best F1={m['f1@best']:.4f} "
            f"(P={m['precision@best']:.4f}, R={m['recall@best']:.4f}, "
            f"th={m['threshold@best']:.3f})")
    if "f1@tuned" in m:
        print(f"  [{label}] @tuned th={m['tuned_threshold']:.3f}: "
              f"P={m['precision@tuned']:.4f} R={m['recall@tuned']:.4f} "
              f"F1={m['f1@tuned']:.4f}")


def _clip_or_measure(model, grad_clip):
    """Clip the gradient norm if asked, otherwise measure it.

    Don't probe with ``clip_grad_norm_(..., inf)``: inf / inf is nan and
    poisons every gradient, which AMP can then apply.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return None
    if grad_clip > 0:
        return float(torch.nn.utils.clip_grad_norm_(parameters, grad_clip))
    norms = [torch.linalg.vector_norm(p.grad.detach().float(), 2)
             for p in parameters]
    return float(torch.linalg.vector_norm(torch.stack(norms), 2))


def _parse_list(value, cast=str, default=()):
    """Parse a comma-separated CLI value, or return ``default`` when absent."""
    if value is None:
        return default
    if isinstance(value, str):
        items = [item.strip() for item in value.split(',') if item.strip()]
        return tuple(cast(item) for item in items) or default
    return tuple(cast(item) for item in value)


def _parse_games(value):
    """Game names may be given as ``1,2`` or ``game1,game2``."""
    games = _parse_list(value, cast=str, default=None)
    if games is None:
        return None
    return [g if g.startswith('game') else f'game{g}' for g in games]


def main(args):
    """
    Train the TrackNet model using specified configurations.
    """
    # Unpack arguments
    model_name = args.model_name
    contract = contract_of(model_name)
    batch_size = args.batch_size
    height = args.height
    width = args.width
    learning_rate = args.learning_rate
    optimizer_name = args.optimizer
    loss_name = args.loss
    epochs = args.epochs
    tol = args.tol
    save_freq = args.save_freq
    model_path = args.model_path
    work_dir = args.work_dir
    grad_clip = getattr(args, 'grad_clip', 0.5)

    # Augmentation / schedule / split knobs for this run.
    zoom_scales = _parse_list(args.zoom_scales, cast=float, default=(1.0,))
    train_games = _parse_games(args.train_games)
    val_games = _parse_games(args.val_games)
    scheduler_name = getattr(args, 'scheduler', 'constant')
    min_lr = getattr(args, 'min_lr', 0.0)
    tuned_threshold = getattr(args, 'threshold', None)
    final_checkpoint = getattr(args, 'final_checkpoint', 'best')

    # Augmentation: flip + zoom on the training split.  Off by default so a
    # plain baseline run stays plain; the reported run passes --augment.
    augment = args.augment if args.augment is not None else False

    # Build run_name: used for checkpoint filenames and Comet experiment name
    variation = args.variation
    run_name = f"{model_name}_{variation}" if variation else model_name

    # If using default work directory, append a timestamp for uniqueness
    if work_dir == "./models":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(work_dir, timestamp)

    # Create the work directory if it doesn't exist
    os.makedirs(work_dir, exist_ok=True)
    
    # Optional Comet.ml tracking.
    try:
        import comet_ml
        comet_key_file = os.path.join(work_dir, f"{run_name}_comet_key.txt")
        
        # Resume this run if it already has a key.
        if os.path.exists(comet_key_file):
            with open(comet_key_file, 'r') as f:
                exp_key = f.read().strip()
            experiment = comet_ml.ExistingExperiment(previous_experiment=exp_key)
            print(f"Resumed Comet.ml Experiment: {exp_key}")
        else:
            experiment = comet_ml.Experiment(project_name="tracknet-ablation", auto_output_logging="simple")
            experiment.set_name(run_name)
            with open(comet_key_file, 'w') as f:
                f.write(experiment.get_key())
            print(f"Started new Comet.ml Experiment: {experiment.get_key()}")
            
        experiment.log_parameters({
            "model_name": model_name,
            "variation": variation,
            "run_name": run_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "loss": loss_name,
            "resolution": f"{width}x{height}",
            "tol": tol,
            "scheduler": scheduler_name,
            "min_lr": min_lr,
            "zoom_scales": list(zoom_scales),
            "train_games": train_games or list(TRAIN_GAMES),
            "val_games": val_games or list(VAL_GAMES),
        })
    except ImportError:
        print("comet_ml not installed. Run `pip install comet_ml` to enable tracking.")
        experiment = None
    except Exception as exc:
        print(f"Comet.ml disabled ({type(exc).__name__}: {exc}). Continuing without tracking.")
        experiment = None

    experiment_config = {
        "model_name": model_name,
        "contract": contract,
        "run_name": run_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "height": height,
        "width": width,
        "epochs": epochs,
        "tol": tol,
        "model_path": model_path,
        "work_dir": work_dir,
        "save_freq": save_freq,
        "loss": loss_name,
        "augment": augment,
        "zoom_scales": list(zoom_scales),
        "scheduler": scheduler_name,
        "min_lr": min_lr,
        "train_games": train_games or list(TRAIN_GAMES),
        "val_games": val_games or list(VAL_GAMES),
        "final_checkpoint": final_checkpoint,
        "threshold": tuned_threshold,
    }
    print("Training Configurations:")
    for key, value in experiment_config.items():
        print(f"  {key}: {value}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reproducibility: seed init and batch order, and disable the cudnn
    # autotuner, which is not deterministic.
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        print(f"Seed: {args.seed} (cudnn.benchmark disabled)")
    else:
        # Enable benchmark for faster CNN operations with fixed input sizes
        torch.backends.cudnn.benchmark = True

    # Load dataset.
    train_dataset, val_dataset = get_dataset(
        height, width, augment=augment,
        zoom_scales=zoom_scales, train_games=train_games, val_games=val_games
    )
    print(f"Train windows: {len(train_dataset)} | Val windows: {len(val_dataset)}")
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # Load model.  Resume from the newest checkpoint in work_dir if there is
    # one, otherwise from --model_path.
    model = get_model(model_name, height, width)
    start_epoch = 0
    import glob
    import re
    
    def load_clean_state_dict(model_to_load, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Strip torch.compile's prefix.
        state_dict = {
            (key[10:] if key.startswith('_orig_mod.') else key): value
            for key, value in state_dict.items()
        }
        # Match by name and shape so one architecture can warm-start another.
        target = model_to_load.state_dict()
        loaded, skipped = {}, []
        for key, value in state_dict.items():
            if key in target and target[key].shape == value.shape:
                loaded[key] = value
            else:
                skipped.append(key)
        missing = [key for key in target if key not in loaded]
        model_to_load.load_state_dict(loaded, strict=False)
        print(f"Loaded {len(loaded)}/{len(target)} tensors from {checkpoint_path}")
        if skipped:
            print(f"  skipped {len(skipped)} shape/name mismatches "
                  f"(expected when switching architectures)")
        if missing:
            preview = ', '.join(missing[:5])
            suffix = ' ...' if len(missing) > 5 else ''
            print(f"  {len(missing)} new parameters left at init: {preview}{suffix}")

    existing_checkpoints = glob.glob(os.path.join(work_dir, f"{run_name}_epoch_*.pth"))
    if existing_checkpoints:
        # Find the highest epoch number
        highest_epoch = 0
        latest_checkpoint = ""
        for ckpt in existing_checkpoints:
            match = re.search(r'_epoch_(\d+)\.pth', ckpt)
            if match:
                ep = int(match.group(1))
                if ep > highest_epoch:
                    highest_epoch = ep
                    latest_checkpoint = ckpt
        
        if latest_checkpoint:
            print(f"Found existing checkpoint: {latest_checkpoint}")
            print(f"Resuming training from Epoch {highest_epoch}...")
            load_clean_state_dict(model, latest_checkpoint)
            start_epoch = highest_epoch

    if model_path and not existing_checkpoints:
        load_clean_state_dict(model, model_path)
        print(f"Loaded manual checkpoint: {model_path}")
        
    model.to(device)

    use_amp = bool(args.amp) and device.type == 'cuda'
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("Automatic mixed precision (fp16) enabled")
    else:
        scaler = None
    
    try:
        if not args.no_compile:
            model = torch.compile(model)
            print("PyTorch Compiler Enabled!")
    except Exception:
        pass

    # The dict contract has its own loss; legacy contracts use BCE or focal.
    if contract == "dict":
        criterion = FastTrackNetLoss()
        print("Using FastTrackNetLoss (focal heatmap + offset + visibility)")
    elif loss_name == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
        print(f"Using Focal Loss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    else:
        criterion = nn.BCELoss()
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    scheduler = None
    if scheduler_name == 'cosine':
        total_steps = max(1, epochs - start_epoch)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=min_lr)
        print(f"Cosine LR: {learning_rate:g} -> {min_lr:g} over "
              f"{total_steps} epochs")
    elif scheduler_name != 'constant':
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # Training loop
    best_ap4 = -1.0
    best_checkpoint = None
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_lr = optimizer.param_groups[0]['lr']
        running_loss = 0.0
        part_sums = {"heatmap": 0.0, "offset": 0.0, "visibility": 0.0}
        part_count = 0
        prob_sums = {"pos": 0.0, "bg": 0.0}
        prob_count = 0
        grad_norms = []
        nonfinite_steps = 0
        first_nonfinite = None
        print(f"======== Epoch {epoch + 1} ========")
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, batch in enumerate(train_pbar):
            inputs, labels = batch
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            loss_parts = {}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                model_output = model(inputs)
            loss, loss_parts = model_loss(criterion, model_output, labels)

            if loss_parts:
                for name, value in loss_parts.items():
                    part_sums[name] = part_sums.get(name, 0.0) + value.item()
                part_count += 1

            # Sample heatmap confidence on ball vs background every 50 steps;
            # a collapsed model has high background and low ball values.
            if i % 50 == 0 and isinstance(model_output, dict):
                with torch.no_grad():
                    prob = model_output["heatmap"].float().sigmoid()
                    heat_target = F.adaptive_max_pool2d(
                        labels, prob.shape[-2:]
                    )
                    prob_sums["pos"] += prob[heat_target > 0.5].mean().item() \
                        if (heat_target > 0.5).any() else 0.0
                    prob_sums["bg"] += prob[heat_target < 0.05].mean().item()
                    prob_count += 1

            grad_norm = None
            if scaler is not None:
                scaler.scale(loss).backward()
                # Unscale so the measured norm is the real one.
                scaler.unscale_(optimizer)
                grad_norm = _clip_or_measure(model, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = _clip_or_measure(model, grad_clip)
                optimizer.step()
            if grad_norm is not None:
                if np.isfinite(grad_norm):
                    grad_norms.append(float(grad_norm))
                else:
                    nonfinite_steps += 1
                    if first_nonfinite is None:
                        first_nonfinite = (i, float(loss.detach()),
                                           dict(loss_parts) if loss_parts else {})

            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': f"{running_loss / ((i + 1) * inputs.size(0)):.4f}"})

        if part_count:
            print("Train loss parts: "
                  + " | ".join(f"{k}={v / part_count:.4f}"
                               for k, v in part_sums.items()))
        if prob_count:
            print(f"Train heatmap prob: pos={prob_sums['pos'] / prob_count:.3f} "
                  f"bg={prob_sums['bg'] / prob_count:.4f} "
                  f"(want pos>0.7, bg<0.05)")
        if grad_norms or nonfinite_steps:
            finite = (f"mean={np.mean(grad_norms):.3f} max={np.max(grad_norms):.3f}"
                      if grad_norms else "mean=n/a max=n/a")
            print(f"Train grad norm: {finite} "
                  f"(clip={grad_clip if grad_clip > 0 else 'off'}) "
                  f"| non-finite steps={nonfinite_steps}")
            if first_nonfinite is not None:
                step, bad_loss, bad_parts = first_nonfinite
                print(f"  first non-finite at step {step}: loss={bad_loss:.4f} "
                      + " ".join(f"{k}={float(v):.4f}"
                                 for k, v in bad_parts.items()))
        if scaler is not None:
            print(f"Train AMP scaler scale: {scaler.get_scale():.0f} "
                  f"(halves on every overflow; falling = unstable steps)")

        if scheduler is not None:
            scheduler.step()

        # Save the checkpoint, keeping every 10th epoch and the current one.
        if (epoch + 1) % save_freq == 0:
            current_epoch = epoch + 1
            model_save_path = os.path.join(work_dir, f"{run_name}_epoch_{current_epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            
            existing_checkpoints = glob.glob(os.path.join(work_dir, f"{run_name}_epoch_*.pth"))
            for ckpt in existing_checkpoints:
                match = re.search(r'_epoch_(\d+)\.pth', ckpt)
                if match:
                    ep = int(match.group(1))
                    if ep != current_epoch and ep % 10 != 0:
                        try:
                            os.remove(ckpt)
                        except Exception:
                            pass

        # One shared validation pass: detection records and val loss together,
        # so the metrics match src/eval.py.
        model.eval()
        val_loss = 0.0
        val_parts = {"heatmap": 0.0, "offset": 0.0, "visibility": 0.0}
        val_part_count = 0
        val_probs = {"pos": 0.0, "bg": 0.0}
        val_prob_count = 0
        records, frame_keys = {}, []

        def _val_hook(model_output, labels, count):
            nonlocal val_loss, val_part_count, val_prob_count
            batch_loss, batch_parts = model_loss(
                criterion, model_output, labels)
            val_loss += batch_loss.item() * count
            for name, value in batch_parts.items():
                val_parts[name] = val_parts.get(name, 0.0) + value.item()
            val_part_count += 1
            if val_part_count % 50 == 0 and isinstance(model_output, dict):
                prob = model_output["heatmap"].float().sigmoid()
                heat_target = F.adaptive_max_pool2d(labels, prob.shape[-2:])
                if (heat_target > 0.5).any():
                    val_probs["pos"] += prob[heat_target > 0.5].mean().item()
                val_probs["bg"] += prob[heat_target < 0.05].mean().item()
                val_prob_count += 1

        records, frame_keys, _, _ = collect_records(
            model, val_dataset, device, loader=val_loader,
            use_amp=use_amp, on_batch=_val_hook,
            desc=f"Epoch {epoch+1}/{epochs} [Val]",
        )

        val_loss /= len(val_loader.dataset) # Average loss per sample
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}")
        if val_part_count:
            print("Val loss parts:   "
                  + " | ".join(f"{k}={v / val_part_count:.4f}"
                               for k, v in val_parts.items()))
        if val_prob_count:
            print(f"Val heatmap prob:   pos={val_probs['pos'] / val_prob_count:.3f} "
                  f"bg={val_probs['bg'] / val_prob_count:.4f}")

        # Both protocols over the same records.
        window_metrics, frame_metrics, _ = both_protocols(
            records, frame_keys, tol)

        _print_protocol("per-window", window_metrics)
        _print_protocol("per-frame ", frame_metrics)
        precision = (frame_metrics["tp"] /
                     (frame_metrics["tp"] + frame_metrics["fp"])
                     if (frame_metrics["tp"] + frame_metrics["fp"]) else 0.0)
        recall = (frame_metrics["tp"] / frame_metrics["n_visible"]
                  if frame_metrics["n_visible"] else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)

        # Select on per-frame AP@4 (the deployment protocol).
        if frame_metrics and frame_metrics["ap@4"] > best_ap4:
            best_ap4 = frame_metrics["ap@4"]
            best_checkpoint = os.path.join(work_dir, f"{run_name}_best.pth")
            torch.save(model.state_dict(), best_checkpoint)
            print(f"New best per-frame AP@4={best_ap4:.4f} -> {best_checkpoint}")

        # Comet.ml logging.
        if experiment:
            avg_train_loss = running_loss / len(train_dataset)
            log_payload = {
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "learning_rate": epoch_lr,
            }
            log_payload.update(
                {f"win_{k}": v for k, v in window_metrics.items()})
            log_payload.update(
                {f"frame_{k}": v for k, v in frame_metrics.items()})
            experiment.log_metrics(log_payload, epoch=epoch+1)
        
        end_time = time.time()
        print(f"Time taken for epoch {epoch + 1}: {end_time - start_time:.2f} seconds")
        start_time = end_time

    # Final evaluation of one checkpoint: ``best`` or ``last``.
    last_checkpoint = os.path.join(
        work_dir, f"{run_name}_epoch_{epochs}.pth")
    report_checkpoint = None
    if final_checkpoint == 'last':
        if os.path.exists(last_checkpoint):
            report_checkpoint = last_checkpoint
        else:
            print(f"\nWARNING: {last_checkpoint} not found; "
                  f"falling back to the best checkpoint.")
    if report_checkpoint is None:
        report_checkpoint = best_checkpoint

    if report_checkpoint and os.path.exists(report_checkpoint):
        if best_checkpoint and os.path.exists(best_checkpoint):
            print(f"\nBest per-frame AP@4 on val: {best_ap4:.4f} "
                  f"({os.path.basename(best_checkpoint)})")
        print(f"Reporting checkpoint: {os.path.basename(report_checkpoint)}")
        final_evaluation(
            model_name=model_name,
            checkpoint=report_checkpoint,
            height=height,
            width=width,
            tol=tol,
            result_dir=work_dir,
            experiment=experiment,
            threshold=tuned_threshold,
            train_games=train_games,
            val_games=val_games,
        )
    else:
        print("\nNo validation run recorded a checkpoint, "
              "skipping the final evaluation.")


def final_evaluation(model_name, checkpoint, height, width, tol,
                     result_dir, experiment=None, threshold=None,
                     train_games=None, val_games=None):
    """Evaluate one checkpoint and write its PR curve and metrics JSON."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tag = os.path.basename(checkpoint).replace(".pth", "")
    print(f"\n===== Final evaluation: {checkpoint} =====")

    _, test_dataset = get_dataset(
        height, width, train_games=train_games, val_games=val_games)
    model = get_model(model_name, height, width, pretrained=False)
    state_dict = torch.load(checkpoint, map_location=device)
    state_dict = {
        (k[10:] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  note: {len(missing)} unexpected/missing keys while loading")
    model.to(device).eval()

    # Same shared pass as src/eval.py.
    flat, frame_keys, _, _ = collect_records(
        model, test_dataset, device, batch_size=8, desc="Final eval",
        height=height, width=width)

    window_metrics, frame_metrics, fused = both_protocols(
        flat, frame_keys, tol, threshold=threshold)

    _print_protocol("per-window", window_metrics)
    _print_protocol("per-frame ", frame_metrics)

    # The PR curve is drawn for the deployment protocol (one point per frame).
    recall_pts, precision_pts, _ = pr_points(
        fused["score"], point_distances(fused["xy"], fused["gt_xy"]),
        fused["visible"], tol=tol)
    metrics = frame_metrics

    plot_path = os.path.join(result_dir, f"{tag}_pr_curve.png")
    figure = save_pr_curve(
        recall_pts, precision_pts, plot_path,
        title=f"{model_name} - PR curve (per frame, tol={tol}px)",
        best=(metrics["recall@best"], metrics["precision@best"]))
    print(f"  PR curve saved to {plot_path}")

    json_path = os.path.join(result_dir, f"{tag}_final_eval.json")
    with open(json_path, "w") as handle:
        json.dump({"per_window": window_metrics,
                   "per_frame": frame_metrics,
                   "fusion_close_px": FUSE_CLOSE_PX}, handle, indent=4)
    print(f"  metrics saved to {json_path}")

    if experiment:
        payload = {}
        for protocol, values in (("final/win", window_metrics),
                                 ("final/frame", frame_metrics)):
            payload.update({f"{protocol}/{k}": v for k, v in values.items()})
        experiment.log_metrics(payload)
        try:
            experiment.log_figure(
                figure_name=f"{tag}_pr_curve", figure=figure
            )
            experiment.log_image(plot_path, name=f"{tag}_pr_curve")
        except Exception as exc:
            print(f"  (comet figure logging skipped: {exc})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TrackNet model with configurable parameters."
    )
    parser.add_argument('--model_name', type=str, default='TrackNetFast',
                        choices=list(MODEL_NAMES),
                        help="Name of the model to use.")
    parser.add_argument("--variation", type=str, default=None, help="Optional variation tag (e.g. 'pt_focal'). Appended to model_name for checkpoint filenames and Comet experiment name.")
    parser.add_argument("--optimizer", type=str, default="Adadelta", choices=["Adadelta", "AdamW", "Adam"], help="Optimizer to use for training")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate for the optimizer")
    parser.add_argument("--height", type=int, default=288, help="Target height of the images")
    parser.add_argument("--width", type=int, default=512, help="Target width of the images")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--tol", type=int, default=4, help="Pixel tolerance of the reported operating point (AP is always reported at 4, 8 and 12).")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model (.pth) to load before training"
    )
    parser.add_argument("--work_dir", type=str, default="./models", help="Directory to save the trained models")
    parser.add_argument("--save_freq", type=int, default=1, help="Frequency (in epochs) to save model checkpoints")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"], help="Loss function: 'bce' (standard) or 'focal' (better for class-imbalanced heatmaps)")
    parser.add_argument("--amp", action="store_true", help="Enable fp16 mixed precision training (CUDA only). Recommended on T4.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile (useful for debugging or first runs).")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers (default: 2).")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Training augmentation: horizontal flip and zoom-in crops. "
                             "Use --augment to enable, --no-augment to disable.")
    parser.add_argument("--zoom_scales", type=str, default="1.0",
                        help="Comma-separated zoom-in crop scales, sampled per window when "
                             "--augment is on. '1.0' disables zoom; '1.0,0.9,0.8' crops to "
                             "90%%/80%% and resizes back (ball 1.11x/1.25x larger).")
    parser.add_argument("--scheduler", type=str, default="constant",
                        choices=["constant", "cosine"],
                        help="LR schedule stepped once per epoch (default: constant).")
    parser.add_argument("--min_lr", type=float, default=0.0,
                        help="Final LR for --scheduler cosine (default: 0).")
    parser.add_argument("--train_games", type=str, default=None,
                        help="Comma-separated training games, e.g. '3,4,5,6,7,8,9,10' "
                             "(default: constants.TRAIN_GAMES).")
    parser.add_argument("--val_games", type=str, default=None,
                        help="Comma-separated validation games, e.g. '1,2' "
                             "(default: constants.VAL_GAMES).")
    parser.add_argument("--final_checkpoint", type=str, default="best",
                        choices=["best", "last"],
                        help="Checkpoint to run the final evaluation on: the per-frame "
                             "AP@4 'best' (default) or the fixed 'last' epoch.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional gate threshold calibrated elsewhere; reported as an "
                             "extra operating point. Does not change AP.")
    parser.add_argument("--focal_alpha", type=float, default=0.75, help="Focal Loss: foreground (ball) pixel weight. >0.5 = more emphasis on ball. Use 0.75-0.99 for extreme imbalance. (default: 0.75)")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Global gradient-norm clipping (default: 0 = off, which is the configuration the TrackNetFast baseline was trained with). Set e.g. 1.0 to protect a warm-started fine-tune.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for init, batch order and cuDNN kernels. Set it to make two runs comparable; leave unset for maximum throughput (default: None).")
    
    args = parser.parse_args()
    main(args)
