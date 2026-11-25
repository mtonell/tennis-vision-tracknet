#!/usr/bin/env python
"""Per-stage latency benchmark for TrackNetFast on pt, onnx or TensorRT.

Times preprocess, inference and postprocess separately, plus the total, over
a fixed number of windows after a warmup.  Preprocess and fusion are the
functions the video pipeline uses, so the numbers are the pipeline and not a
bare forward pass.

Usage:
    python src/benchmark.py --model_path tracknet_fast.onnx --instances 1000
"""

import argparse
import time

import numpy as np
import torch

from constants import HEIGHT, WIDTH
from metrics import FUSE_CLOSE_PX, select_indices
from predict_video import preprocess

BACKENDS = ("pt", "onnx", "engine")

# Only the resize cost depends on this; 720p matches the dataset.
SOURCE_HEIGHT, SOURCE_WIDTH = 720, 1280


def detect_backend(model_path):
    lower = model_path.lower()
    if lower.endswith((".trt", ".engine")):
        return "engine"
    if lower.endswith(".onnx"):
        return "onnx"
    return "pt"


# Each backend returns ``run(batch) -> outputs`` in the model contract.

def make_pytorch_backend(model_path, device):
    from util import get_model

    model = get_model("TrackNetFast", HEIGHT, WIDTH, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {
        (k[10:] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    @torch.no_grad()
    def run(inputs):
        return model(inputs.to(device))

    return run


def make_onnx_backend(model_path, device):
    import onnxruntime as ort

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device.type == "cuda" else ["CPUExecutionProvider"])
    session = ort.InferenceSession(model_path, providers=providers)
    names = [out.name for out in session.get_outputs()]

    def run(inputs):
        arrays = session.run(names, {"input": inputs.cpu().numpy()})
        return dict(zip(names, [torch.from_numpy(a) for a in arrays]))

    return run


def make_tensorrt_backend(model_path, device):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(model_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # TRT 10: name-based I/O tensors instead of binding indices.
    input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    if not input_names:
        raise SystemExit("Engine has no input tensors")

    # Match the engine's declared input/output dtypes (FP16 engines take half).
    trt_to_torch = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
    in_dtype = trt_to_torch.get(engine.get_tensor_dtype(input_names[0]), torch.float32)
    out_dtypes = {name: trt_to_torch.get(engine.get_tensor_dtype(name), torch.float32)
                  for name in output_names}

    def run(inputs):
        # Named tensor addresses are read on the GPU stream, so these buffers
        # must stay alive until the synchronize below.
        device_input = inputs.to(device=device, dtype=in_dtype).contiguous()
        context.set_input_shape(input_names[0], device_input.shape)
        for name in input_names:
            context.set_tensor_address(name, device_input.data_ptr())
        outputs = {}
        for name in output_names:
            shape = tuple(context.get_tensor_shape(name))
            out = torch.empty(shape, dtype=out_dtypes[name], device=device)
            context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out
        stream = torch.cuda.current_stream(device)
        ok = context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize(device)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return {
            "heatmap": outputs.get("heatmap"),
            "offset": outputs.get("offset"),
            "visibility": outputs.get("visibility"),
        }

    return run


def make_backend(model_path, backend, device):
    if backend == "pt":
        return make_pytorch_backend(model_path, device)
    if backend == "onnx":
        return make_onnx_backend(model_path, device)
    return make_tensorrt_backend(model_path, device)


def fuse_frame(candidates, close_px=FUSE_CLOSE_PX):
    """Fuse one frame's candidates; returns the chosen point or None."""
    if not candidates:
        return None
    points = np.asarray([c for c in candidates], dtype=np.float64)
    # Equal peaks reduce the rule to the geometric majority.
    used = select_indices(points, np.ones(len(points)), close_px)
    return points[used].mean(axis=0)


def decode_candidates(outputs, height, width):
    """Decode any backend output into per-slot ``(x, y)`` candidates."""
    from metrics import detection_points

    xy, _, _ = detection_points(outputs, height, width)
    return [tuple(p) for p in xy[0].cpu().numpy()]


def main(args):
    device = torch.device(args.device if args.device
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    backend = args.backend or detect_backend(args.model_path)
    if backend not in BACKENDS:
        raise SystemExit(f"Unknown backend {backend!r}; expected one of {BACKENDS}")
    print(f"Backend: {backend}  device: {device}")
    print(f"Model:   {args.model_path}")
    print(f"Instances: {args.instances} (warmup {args.warmup}), "
          f"input {WIDTH}x{HEIGHT}, batch 1")

    run = make_backend(args.model_path, backend, device)

    # Reused every instance, so only conversion/resize/copy is timed.
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8)
              for _ in range(3)]

    # Pending candidates per frame index, mirroring the video pipeline.
    pending = {}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def one_instance(index):
        begin = time.perf_counter()
        inputs = preprocess(frames, HEIGHT, WIDTH).to(device)
        synchronize()
        model_begin = time.perf_counter()

        outputs = run(inputs)
        synchronize()
        post_begin = time.perf_counter()

        candidates = decode_candidates(outputs, HEIGHT, WIDTH)
        window_start = index
        for k, xy in enumerate(candidates):
            pending.setdefault(window_start + k, []).append(xy)
        for frame_index in sorted(pending):
            if frame_index < window_start:
                fuse_frame(pending.pop(frame_index), args.close_px)
        finish = time.perf_counter()

        return (model_begin - begin, post_begin - model_begin, finish - post_begin,
                finish - begin)

    # Warmup (untimed).
    for i in range(args.warmup):
        one_instance(i % args.instances)

    samples = [one_instance(i) for i in range(args.instances)]

    # Flush the last two frames.
    for frame_index in sorted(pending):
        fuse_frame(pending.pop(frame_index), args.close_px)

    columns = np.asarray(samples) * 1000.0     # instances x 4 stages, in ms

    def fps(ms):
        return 1000.0 / ms if ms > 0 else float("inf")

    stages = (
        ("preprocess (+H2D)", columns[:, 0]),
        ("inference", columns[:, 1]),
        ("postprocess", columns[:, 2]),
        ("total", columns[:, 3]),
    )
    print(f"\n{'stage':<20}{'mean':>10}{'p50':>10}{'p95':>10}")
    for name, values in stages:
        print(f"{name:<20}{values.mean():>8.2f}ms{np.percentile(values, 50):>8.2f}ms"
              f"{np.percentile(values, 95):>8.2f}ms")

    total = columns[:, 3].mean()
    print(f"\nThroughput: {fps(total):.1f} windows/s"
          f"  ->  {fps(total) * 3:.1f} frames/s (3 frames per window)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark TrackNetFast latency (pt / onnx / engine)."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Checkpoint (.pth), ONNX graph (.onnx) or "
                             "TensorRT engine (.trt/.engine).")
    parser.add_argument("--backend", type=str, default=None, choices=BACKENDS,
                        help="Force a backend; default is inferred from the file "
                             "extension.")
    parser.add_argument("--instances", type=int, default=1000,
                        help="Timed inference instances (default: 1000).")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Untimed warmup instances (default: 50).")
    parser.add_argument("--close_px", type=float, default=FUSE_CLOSE_PX,
                        help="Fusion agreement radius in pixels.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda:0, cpu, ... (default: auto).")
    main(parser.parse_args())
