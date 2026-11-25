"""Detection metrics: decoding, overlap fusion, AP and P/R/F1.

Frames are ranked by the confidence score, so AP is threshold-free.
Two protocols score the same predictions: per-window (three slots per
window) and per-frame (one fused point per frame).  Both call
``summarize``; only the sample population differs.

A frame never has more than one prediction.  On a visible frame a point
inside ``tol`` is a TP, no point is an FN, and a point outside ``tol`` is
both an FN and an FP.  On an invisible frame only a point is scored, as an
FP.  There is no true negative: frames without a ball are not counted.
"""

from __future__ import annotations

import numpy as np
import torch

from constants import TOLERANCES
from models.TrackNetFast import decode_outputs
from losses import FastTrackNetLoss


@torch.no_grad()
def detection_points(outputs, height, width):
    """Decode model output into ``(xy, logit, peak)`` per slot.

    Accepts the dict contract or a legacy heatmap (or ``(heatmap, motion)``
    tuple).  Legacy heatmaps are rescaled to the input size.
    """
    if isinstance(outputs, dict):
        coords, _ = decode_outputs(outputs, height, width)
        logit = outputs["visibility"].float()
        peak = outputs["heatmap"].float().sigmoid().amax(dim=(2, 3))
        return coords, logit, peak

    heatmaps = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    heatmaps = heatmaps.float()
    batch, frames, heat_h, heat_w = heatmaps.shape
    stride_x = width / heat_w
    stride_y = height / heat_h

    peak, index = heatmaps.reshape(batch * frames, heat_h * heat_w).max(dim=1)
    xs = ((index % heat_w).to(peak.dtype) + 0.5) * stride_x
    ys = (torch.div(index, heat_w, rounding_mode="floor").to(peak.dtype)
          + 0.5) * stride_y
    xy = torch.stack([xs, ys], dim=-1).reshape(batch, frames, 2)
    peak = peak.reshape(batch, frames)
    logit = torch.logit(peak.clamp(1e-6, 1.0 - 1e-6))
    return xy, logit, peak


@torch.no_grad()
def detection_records(outputs, labels):
    """Flatten one batch into per-slot detection records.

    Keeps coordinates rather than distances so overlapping windows can be
    fused before the error is measured.
    """
    height, width = labels.shape[-2], labels.shape[-1]
    xy, logit, peak = detection_points(outputs, height, width)
    gt_coords, gt_visible = FastTrackNetLoss._target_coords(labels.float())
    return {
        "score": logit.sigmoid().reshape(-1).cpu().numpy().astype(np.float64),
        "logit": logit.reshape(-1).cpu().numpy().astype(np.float64),
        "peak": peak.reshape(-1).cpu().numpy().astype(np.float64),
        "xy": xy.reshape(-1, 2).cpu().numpy().astype(np.float64),
        "gt_xy": gt_coords.reshape(-1, 2).cpu().numpy().astype(np.float64),
        "visible": gt_visible.reshape(-1).cpu().numpy().astype(bool),
    }


def point_distances(xy, gt_xy):
    """Euclidean pixel error between decoded points and their GT centroids."""
    return np.linalg.norm(np.asarray(xy) - np.asarray(gt_xy), axis=-1)


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


#: Candidates closer than this (pixels) are averaged instead of competing.
#: Twice the ``tol=4`` operating tolerance; override per call if needed.
FUSE_CLOSE_PX = 8.0


def _ranked(scores, dists, visible, tol):
    """Sort by score and return the running precision/recall curve."""
    scores = np.asarray(scores, dtype=np.float64)
    dists = np.asarray(dists, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    n_positive = int(visible.sum())

    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    correct = (dists[order] <= tol) & visible[order]

    tp = np.cumsum(correct)
    fp = np.cumsum(~correct)
    recall = tp / max(n_positive, 1)
    precision = tp / np.maximum(tp + fp, 1)
    return ordered_scores, precision, recall, n_positive


def _ap(precision, recall):
    """Step-wise area under the PR curve (recall is non-decreasing)."""
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def _counts(scores, dists, visible, tol, threshold):
    """TP/FN/FP counts at a fixed threshold, both FP kinds split out."""
    detected = scores > threshold
    correct = (dists <= tol) & visible
    true_positive = detected & correct

    fp_wrong = detected & visible & ~correct   # ball present, point too far
    fp_ghost = detected & ~visible             # no ball, still fired

    return {
        "tp": int(np.sum(true_positive)),
        "fp_wrong_location": int(np.sum(fp_wrong)),
        "fp_invisible": int(np.sum(fp_ghost)),
        "fp": int(np.sum(fp_wrong) + np.sum(fp_ghost)),
        # A visible frame is TP or FN, with no third option.
        "fn": int(np.sum(visible & ~true_positive)),
        "fn_gate_closed": int(np.sum(~detected & visible)),
        # Same frames as fp_wrong_location: they cost one of each.
        "fn_wrong_location": int(np.sum(fp_wrong)),
        "n_visible": int(visible.sum()),
        "n_frames": int(len(visible)),
    }


def _pr_at(precision, recall, index):
    if len(precision) == 0:
        return 0.0, 0.0, 0.0
    p, r = float(precision[index]), float(recall[index])
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def select_indices(points, peaks, close_px=FUSE_CLOSE_PX):
    """Pick which of one frame's candidates to combine.

    Returns indices into ``points``: the closest pair if it agrees within
    ``close_px``, all points if they all agree, else the highest peak.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    peaks = np.asarray(peaks, dtype=np.float64).reshape(-1)
    keep = np.arange(len(points))

    if len(points) > 2:
        spacing = np.linalg.norm(
            points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(spacing, np.inf)
        first, second = np.unravel_index(
            int(np.argmin(spacing)), spacing.shape)
        if spacing[first, second] <= close_px:
            keep = keep[[first, second]]

    if len(keep) > 1:
        pool = points[keep]
        widest = np.linalg.norm(
            pool[:, None, :] - pool[None, :, :], axis=-1).max()
        if widest <= close_px:
            return keep
    if len(keep) == 1:
        return keep
    return keep[[int(np.argmax(peaks[keep]))]]


def fuse_frames(records, frame_keys, mode="fused", close_px=FUSE_CLOSE_PX):
    """Collapse overlapping windows to one detection per unique frame.

    ``mode`` is ``fused`` (default, uses ``select_indices``), ``head0``
    (newest window, two frames of lookahead) or ``head2`` (oldest window,
    zero latency).  Results are ordered by frame key.
    """
    if len(frame_keys) == 0:
        empty = {name: np.zeros(0) for name in
                 ("score", "logit", "peak", "xy", "gt_xy", "visible")}
        empty["xy"] = empty["gt_xy"] = np.zeros((0, 2))
        return empty

    if mode not in ("fused", "head0", "head2"):
        raise ValueError(f"unknown frame mode {mode!r}")

    buckets = {}
    for index, key in enumerate(frame_keys):
        buckets.setdefault(key, []).append(index)

    fused = {
        "score": [], "logit": [], "peak": [], "xy": [], "gt_xy": [],
        "visible": [],
    }
    for key in sorted(buckets):
        indices = np.asarray(buckets[key], dtype=np.int64)

        if mode == "head0":
            chosen = indices[-1:]
        elif mode == "head2":
            chosen = indices[:1]
        else:
            chosen = indices[select_indices(
                records["xy"][indices], records["peak"][indices], close_px)]

        logit = float(records["logit"][chosen].mean())
        fused["logit"].append(logit)
        fused["score"].append(float(_sigmoid(logit)))
        fused["peak"].append(float(records["peak"][chosen].mean()))
        fused["xy"].append(records["xy"][chosen].mean(axis=0))
        # Every copy of a frame shares the same labels.
        fused["gt_xy"].append(records["gt_xy"][indices[0]])
        fused["visible"].append(bool(records["visible"][indices[0]]))

    result = {}
    for name in ("score", "logit", "peak", "visible"):
        result[name] = np.asarray(fused[name])
    for name in ("xy", "gt_xy"):
        result[name] = np.asarray(fused[name]).reshape(-1, 2)
    result["visible"] = result["visible"].astype(bool)
    return result

def summarize(scores, dists, visible, tolerances=TOLERANCES,
              operating_tol=4, fixed_threshold=0.5, tuned_threshold=None):
    """One-pass summary: AP at several tolerances plus both operating points.

    AP is threshold-free.  Counts and P/R/F1 are reported at
    ``fixed_threshold``; if ``tuned_threshold`` is given, a second operating
    point is added next to it.
    """
    metrics = {}

    for tol in tolerances:
        _, precision, recall, _ = _ranked(scores, dists, visible, tol)
        metrics[f"ap@{tol}"] = _ap(precision, recall)

    # Operating point: threshold that maximises F1 at the deployment tolerance.
    ordered, precision, recall, _ = _ranked(scores, dists, visible, operating_tol)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1)) if len(f1) else 0
    p, r, f = _pr_at(precision, recall, best)
    metrics["precision@best"] = p
    metrics["recall@best"] = r
    metrics["f1@best"] = f
    metrics["threshold@best"] = float(ordered[best]) if len(ordered) else 0.0

    counts = _counts(scores, dists, visible, operating_tol, fixed_threshold)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    metrics[f"precision@{fixed_threshold}"] = prec
    metrics[f"recall@{fixed_threshold}"] = rec
    metrics[f"f1@{fixed_threshold}"] = (
        2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    )
    metrics.update(counts)

    if tuned_threshold is not None:
        tuned = _counts(scores, dists, visible, operating_tol, tuned_threshold)
        tp, fp, fn = tuned["tp"], tuned["fp"], tuned["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        metrics["tuned_threshold"] = float(tuned_threshold)
        metrics["precision@tuned"] = prec
        metrics["recall@tuned"] = rec
        metrics["f1@tuned"] = (
            2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        )
    return metrics


def pr_points(scores, dists, visible, tol=4, n_points=200):
    """Evenly spaced PR curve points, for plotting and Comet curves."""
    ordered, precision, recall, _ = _ranked(scores, dists, visible, tol)
    if len(ordered) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    # Walk the curve in recall order and sample it uniformly.
    indices = np.unique(
        np.linspace(0, len(ordered) - 1, min(n_points, len(ordered))).astype(int)
    )
    # Precision must be non-increasing along a PR curve; enforce it so the
    # plot is a curve rather than a zig-zag.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return (
        recall[indices],
        precision[indices],
        ordered[indices],
    )


def pr_axis_limits(recall, precision, best=None, margin=0.06, min_span=0.15):
    """Data-driven axis limits for a PR plot.

    A PR curve usually lives in a narrow band (e.g. precision 0.93-1.0), so a
    fixed 0-1 window wastes most of the figure.  Both axes are cropped to the
    plotted values plus a small margin, clamped to [0, 1].
    """
    def _limits(values):
        vals = np.asarray(list(values), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        low, high = float(vals.min()), float(vals.max())
        pad = max(margin * max(high - low, 1e-6), 0.01)
        lower = max(0.0, low - pad)
        upper = min(1.0, high + pad)
        if upper - lower < min_span:
            lower = max(0.0, upper - min_span)
            if upper - lower < min_span:
                upper = min(1.0, lower + min_span)
        return lower, upper

    recall_values, precision_values = list(recall), list(precision)
    if best is not None:
        recall_values.append(float(best[0]))
        precision_values.append(float(best[1]))
    return _limits(recall_values), _limits(precision_values)


def save_pr_curve(recall, precision, path, title, best=None, dpi=200):
    """Save a PR curve cropped to its data range; returns the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, lw=2, label="reported checkpoint")
    if best is not None:
        axis.scatter([best[0]], [best[1]], zorder=5, color="crimson",
                     label="best F1")
    (x_low, x_high), (y_low, y_high) = pr_axis_limits(
        recall, precision, best=best)
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    return figure


def _protocol_metrics(score, xy, gt_xy, visible, peak, tol, threshold=None):
    """Metric dict for one prediction population."""
    dist = point_distances(xy, gt_xy)
    metrics = summarize(score, dist, visible, tolerances=TOLERANCES,
                        operating_tol=tol, tuned_threshold=threshold)
    # Localisation-only ranking, independent of the gate: if this is high while
    # ap@4 is low the gate is the problem, not the heatmap.
    metrics["ap_peak@4"] = summarize(
        peak, dist, visible, tolerances=(4,), operating_tol=tol)["ap@4"]
    return metrics


def both_protocols(records, frame_keys, tol, mode="fused",
                   close_px=FUSE_CLOSE_PX, threshold=None):
    """Score the same predictions per-window and per-frame.

    Returns ``(window_metrics, frame_metrics, fused)``.  ``threshold`` adds
    an operating point and never changes AP.
    """
    window = _protocol_metrics(
        records["score"], records["xy"], records["gt_xy"],
        records["visible"], records["peak"], tol, threshold)
    fused = fuse_frames(records, frame_keys, mode=mode, close_px=close_px)
    frame = _protocol_metrics(
        fused["score"], fused["xy"], fused["gt_xy"],
        fused["visible"], fused["peak"], tol, threshold)
    return window, frame, fused
