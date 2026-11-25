"""Track a ball through a video and save an annotated copy.

Causal, batch-1 pipeline: every 3-frame window yields three heatmaps, and a
frame is fused from the up-to-three windows that see it, using the same rule
as the evaluation (``metrics.select_indices``).  The output has one frame per
input frame; the last two are fused from fewer candidates.

Speed (model only and full pipeline) is printed at the end.  ``--warmup N``
skips the first N windows so lazy CUDA init and autotuning do not skew it.

Usage:
    python src/predict_video.py --model_path model.pth --video_path clip.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm

from constants import HEIGHT, WIDTH
from metrics import FUSE_CLOSE_PX, detection_points, select_indices

MODEL_CHOICES = ["TrackNetFast"]  # only the dict-contract model has ONNX/TRT exports

# One colour per head, so the same candidate is recognisable across frames.
HEAD_COLOURS = [(0, 0, 255), (0, 200, 0), (255, 0, 0)]   # BGR: red, green, blue
FUSED_COLOUR = (0, 255, 255)                             # yellow


def detect_backend(model_path):
    lower = model_path.lower()
    if lower.endswith((".trt", ".engine")):
        return "engine"
    if lower.endswith(".onnx"):
        return "onnx"
    return "pt"


def load_model(model_name, model_path, device, height, width):
    """Load a PyTorch checkpoint, an ONNX graph, or a TensorRT engine.

    Returns a callable ``run((1, 9, H, W) tensor) -> dict`` with the
    heatmap/offset/visibility keys, regardless of backend.
    """
    backend = detect_backend(model_path)
    print(f"Backend: {backend}  ({model_path})")

    if backend == "pt":
        from util import get_model
        model = get_model(model_name, height, width, pretrained=False)
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {
            (key[10:] if key.startswith('_orig_mod.') else key): value
            for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"  note: {len(missing)} missing / {len(unexpected)} unexpected "
                  f"keys while loading the checkpoint")
        model.to(device)
        model.eval()

        @torch.no_grad()
        def run(inputs):
            return model(inputs)

        return run

    if backend == "onnx":
        import onnxruntime as ort

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device.type == "cuda" else ["CPUExecutionProvider"])
        session = ort.InferenceSession(model_path, providers=providers)
        names = [out.name for out in session.get_outputs()]

        def run(inputs):
            arrays = session.run(names, {"input": inputs.cpu().numpy()})
            return dict(zip(("heatmap", "offset", "visibility"),
                            [torch.from_numpy(a) for a in arrays]))

        return run

    # TensorRT engine, TRT 10+ name-based API.
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(model_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    # Match the engine's declared input/output dtypes (FP16 engines take half).
    trt_to_torch = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
    in_dtype = trt_to_torch.get(engine.get_tensor_dtype(input_names[0]), torch.float32)
    out_dtypes = {name: trt_to_torch.get(engine.get_tensor_dtype(name), torch.float32)
                  for name in output_names}
    print(f"  engine input dtype: {engine.get_tensor_dtype(input_names[0]).name}")

    def run(inputs):
        device_input = inputs.to(device=device, dtype=in_dtype).contiguous()
        context.set_input_shape(input_names[0], device_input.shape)
        context.set_tensor_address(input_names[0], device_input.data_ptr())
        buffers = {}
        for name in output_names:
            shape = tuple(context.get_tensor_shape(name))
            out = torch.empty(shape, dtype=out_dtypes[name], device=device)
            context.set_tensor_address(name, out.data_ptr())
            buffers[name] = out
        stream = torch.cuda.current_stream(device)
        ok = context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize(device)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return {"heatmap": buffers["heatmap"].float(), "offset": buffers["offset"].float(),
                "visibility": buffers["visibility"].float()}

    return run


def preprocess(frames, height, width):
    """Stack 3 BGR frames into the (1, 9, H, W) RGB input the dataset uses."""
    channels = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        channels.append(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)
    return torch.from_numpy(np.concatenate(channels, axis=0)).unsqueeze(0)


def _sigmoid(logit):
    # Stable for extreme logits (FP16 engines can emit very large |values|).
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp = math.exp(logit)
    return exp / (1.0 + exp)


@torch.no_grad()
def parse_candidates(outputs, height, width):
    """One candidate per head, in input-image pixels.

    Candidates are kept even when the gate is closed; the gate only decides
    what gets drawn.
    """
    coords, logits, peaks = detection_points(outputs, height, width)
    coords = coords.cpu().numpy()
    logits = logits.cpu().numpy().reshape(-1)
    peaks = peaks.cpu().numpy().reshape(-1)
    candidates = []
    for k in range(coords.shape[1]):
        xy = (float(coords[0, k, 0]), float(coords[0, k, 1]))
        logit = float(logits[k])
        peak = float(peaks[k])
        if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
            # FP16 engines can emit NaN coordinates; clamp into the image.
            xy = (min(max(0.0 if not np.isfinite(xy[0]) else xy[0], 0.0), float(width)),
                  min(max(0.0 if not np.isfinite(xy[1]) else xy[1], 0.0), float(height)))
        if not np.isfinite(logit):
            logit = -1e30 if np.isnan(logit) else logit  # NaN -> gate closed
        if not np.isfinite(peak):
            peak = 0.0
        candidates.append({"xy": xy, "logit": logit, "peak": peak})
    return candidates


def fuse_candidates(candidates, final="fused", close_px=FUSE_CLOSE_PX):
    """Collapse one frame's candidates into a single detection.

    ``final="fused"`` uses the evaluation rule; ``final="head2"`` keeps only
    the newest causal head.  Returns ``(detection, used)``.
    """
    if not candidates:
        return None, []

    if final == "head2":
        newest = [c for c in candidates if c["head"] == 2]
        used = newest[:1] if newest else [max(candidates, key=lambda c: c["head"])]
    else:
        points = np.asarray([c["xy"] for c in candidates], dtype=np.float64)
        peaks = np.asarray([c["peak"] for c in candidates], dtype=np.float64)
        used = [candidates[i] for i in select_indices(points, peaks, close_px)]

    logit = float(np.mean([c["logit"] for c in used]))
    detection = {
        "xy": tuple(np.mean([c["xy"] for c in used], axis=0)),
        "score": _sigmoid(logit),
        "peak": float(np.mean([c["peak"] for c in used])),
        "source": final,
    }
    return detection, used


def draw_candidates(frame, candidates, used, scale_x, scale_y):
    """Draw every candidate (thin, colour per head) and mark the used ones."""
    used_ids = {id(candidate) for candidate in used}
    for index, candidate in enumerate(candidates):
        x = int(round(candidate["xy"][0] * scale_x))
        y = int(round(candidate["xy"][1] * scale_y))
        colour = HEAD_COLOURS[candidate["head"] % len(HEAD_COLOURS)]
        if id(candidate) in used_ids:
            cv2.circle(frame, (x, y), 7, colour, 2)
        else:
            cv2.circle(frame, (x, y), 4, colour, 1)
        cv2.putText(frame, f"h{candidate['head']}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def draw_fused(frame, detection, scale_x, scale_y, trail):
    """Draw the final fused point, a short trail, and its confidence."""
    if detection is None:
        return
    x = int(round(detection["xy"][0] * scale_x))
    y = int(round(detection["xy"][1] * scale_y))
    trail.appendleft((x, y))
    for index, point in enumerate(trail):
        cv2.circle(frame, point, 4 if index else 6, (0, 255, 0),
                   1 if index else 2)
    cv2.drawMarker(frame, (x, y), FUSED_COLOUR, cv2.MARKER_CROSS, 14, 2)
    cv2.putText(frame, f"{detection['score']:.2f}", (x + 10, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, FUSED_COLOUR, 1, cv2.LINE_AA)


def output_fourcc(path):
    return cv2.VideoWriter_fourcc(*('XVID' if path.lower().endswith('.avi')
                                    else 'mp4v'))


def main(args):
    device = torch.device(args.device if args.device
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    height, width = args.height, args.width
    print(f"Device: {device}")

    run = load_model(args.model_name, args.model_path, device, height, width)

    capture = cv2.VideoCapture(args.video_path)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # 0 or nan
        fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.video_path)),
        f"{os.path.splitext(os.path.basename(args.video_path))[0]}_tracked.mp4",
    )
    if args.max_frames is not None and total_frames is not None:
        total_frames = min(total_frames, args.max_frames)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    writer = cv2.VideoWriter(output_path, output_fourcc(output_path), fps,
                             (frame_width, frame_height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for: {output_path}")

    print(f"Video: {frame_width}x{frame_height} @ {fps:.2f} fps"
          + (f" ({total_frames} frames)" if total_frames else ""))
    print(f"Model: {args.model_path}  input {width}x{height}")
    print(f"Output: {output_path}")
    print(f"Final: {args.final}" + ("  (showing candidates)"
                                    if args.show_candidates else ""))

    scale_x = frame_width / width
    scale_y = frame_height / height

    # frame index -> {"frame": BGR frame, "candidates": [...]}
    pending = {}
    next_write = 0
    trail = deque(maxlen=max(args.trail, 0))

    frames_read = 0
    windows = 0
    timed_windows = 0
    forward_time = 0.0
    pipeline_time = 0.0

    stats = {"single": 0, "consensus": 0, "conflict": 0,
             "gate_closed": 0, "written": 0}

    def emit(index):
        """Fuse, annotate and write one buffered frame."""
        entry = pending.pop(index)
        frame = entry["frame"]
        candidates = entry["candidates"]
        detection, used = fuse_candidates(
            candidates, final=args.final, close_px=args.close_px)

        if len(candidates) == 1:
            stats["single"] += 1
        elif len(used) == len(candidates):
            stats["consensus"] += 1
        else:
            stats["conflict"] += 1
        if detection is not None and detection["score"] < 0.5:
            stats["gate_closed"] += 1

        if args.show_candidates:
            draw_candidates(frame, candidates, used, scale_x, scale_y)
        if detection is not None and detection["score"] >= 0.5:
            draw_fused(frame, detection, scale_x, scale_y, trail)

        writer.write(frame)
        stats["written"] += 1

    start_time = time.time()
    progress = tqdm(total=total_frames, desc="Tracking", unit="frame")

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if args.max_frames is not None and frames_read >= args.max_frames:
            break

        frame_index = frames_read
        frames_read += 1
        progress.update(1)
        pending[frame_index] = {"frame": frame, "candidates": []}

        if frames_read < 3:
            continue

        window_start = frames_read - 3
        window = [pending[window_start + k]["frame"] for k in range(3)]

        pipeline_begin = time.perf_counter()

        # Preprocessing and the host->device copy are I/O, not model time.
        inputs = preprocess(window, height, width).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.no_grad():
            outputs = run(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        candidates = parse_candidates(outputs, height, width)

        windows += 1
        if windows > args.warmup:
            forward_time += elapsed
            timed_windows += 1

        # Head k of window_start predicts frame window_start + k.
        for k, candidate in enumerate(candidates):
            candidate["head"] = k
            candidate["window"] = window_start
            pending[window_start + k]["candidates"].append(candidate)

        # Frame f is complete once window f has run (it has head 0 from f,
        # head 1 from f-1 and head 2 from f-2).  That delays writing by two
        # frames relative to reading.
        while next_write <= window_start:
            emit(next_write)
            next_write += 1

        if windows > args.warmup:
            pipeline_time += time.perf_counter() - pipeline_begin

    progress.close()

    if windows == 0:
        # Fewer than three frames: no window can be formed, so write them raw.
        for index in sorted(pending):
            writer.write(pending[index]["frame"])
            stats["written"] += 1
    else:
        # The last two frames have fewer candidates (their later windows do
        # not exist); fuse from what is available.
        for index in sorted(pending):
            emit(index)

    capture.release()
    writer.release()

    wall = time.time() - start_time
    print(f"\nFrames read: {frames_read}, windows: {windows}, "
          f"warmup: {args.warmup}")
    if windows:
        print(f"Candidates: single {stats['single']}, all agree "
              f"{stats['consensus']}, disagreement {stats['conflict']}; "
              f"gate closed {stats['gate_closed']}; written {stats['written']}")
    if timed_windows:
        per_window = forward_time / timed_windows
        print(f"Model only:  {per_window * 1000:.2f} ms/window  ->  "
              f"{1.0 / per_window:.1f} fps  (batch 1, {timed_windows} windows)")
        pipeline = pipeline_time / timed_windows
        print(f"Pipeline:    {pipeline * 1000:.2f} ms/window  ->  "
              f"{1.0 / pipeline:.1f} fps  "
              f"(preprocess + model + draw + write)")
        print(f"Total wall time: {wall:.2f} s for {frames_read} frames")
    elif windows:
        print("All windows were consumed by --warmup; no timed measurement.")
    else:
        print("Not enough frames for a 3-frame window; nothing inferred.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track a ball through a video and save an annotated copy."
    )
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input video.")
    parser.add_argument("--model_name", type=str, default="TrackNetFast",
                        choices=MODEL_CHOICES,
                        help="Model architecture (only used for .pth "
                             "checkpoints; .onnx/.engine files carry their "
                             "own graph).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Model file: .pth checkpoint, .onnx graph, or "
                             ".trt/.engine TensorRT engine (backend is "
                             "detected from the extension).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: <input>_tracked.mp4).")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Windows to run untimed before measuring speed.")
    parser.add_argument("--trail", type=int, default=10,
                        help="How many past detections to draw (0 disables).")
    parser.add_argument("--show_candidates", action="store_true",
                        help="Draw every head's candidate and the fused point.")
    parser.add_argument("--final", type=str, default="fused",
                        choices=["fused", "head2"],
                        help="Final detection: fused (evaluation rule) or "
                             "head2 (newest causal head, zero latency).")
    parser.add_argument("--close_px", type=float, default=FUSE_CLOSE_PX,
                        help="Fusion agreement radius in pixels.")
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Stop after this many frames (for quick checks).")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda:0, cpu, ... (default: auto).")

    main(parser.parse_args())
