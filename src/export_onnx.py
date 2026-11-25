#!/usr/bin/env python
"""Export TrackNetFast to a static FP16 ONNX graph for TensorRT.

Unlike the dynamic-batch export, this runs the model in half precision on
CUDA and exports a batch-1, FP16, fully-static graph.  TensorRT 10/11 builds
engines from it with zero profile/precision configuration, which is the most
reliable path on Colab.

Usage:
    python src/export_onnx.py --model_path runs/tracknet_fast/TrackNetFast_epoch_20.pth \
        --output_path tracknet_fast_fp16_b1.onnx --fp16
"""

import argparse

import torch

from constants import HEIGHT, WIDTH
from models.registry import build_model


def export_onnx(model_path, output_path, opset=16, fp16=False):
    if fp16 and not torch.cuda.is_available():
        raise SystemExit("--fp16 export needs a CUDA device (run on Colab/GPU).")
    device = torch.device("cuda" if fp16 else "cpu")

    model = build_model("TrackNetFast", HEIGHT, WIDTH, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {
        (k[10:] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    if fp16:
        model.half()

    dtype = torch.float16 if fp16 else torch.float32
    dummy_input = torch.randn(1, 9, HEIGHT, WIDTH, dtype=dtype, device=device)

    print(f"Exporting TrackNetFast to ONNX: {output_path} "
          f"(opset {opset}, {'fp16' if fp16 else 'fp32'}, static batch 1)...")
    # dynamo=False: the TorchScript exporter produces weights TensorRT's ONNX
    # parser can import; the dynamo exporter trips on MobileNetV3's
    # squeeze-excite initializers.
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            input_names=["input"],
            output_names=["heatmap", "offset", "visibility"],
        )
    print(f"Successfully exported ONNX model to {output_path}")

    engine = output_path[:-5] + ".engine" if output_path.endswith(".onnx") else output_path + ".engine"
    print("To build the TensorRT engine (see src/export_tensorrt.py):")
    print(f"  python src/export_tensorrt.py --onnx {output_path} --engine {engine}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TrackNetFast to ONNX.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the PyTorch checkpoint (.pth).")
    parser.add_argument("--output_path", type=str, default="tracknet_fast.onnx",
                        help="Output ONNX file path (default: tracknet_fast.onnx).")
    parser.add_argument("--opset", type=int, default=16,
                        help="ONNX opset version (default: 16, TensorRT-safe).")
    parser.add_argument("--fp16", action="store_true",
                        help="Export a static batch-1 FP16 graph on CUDA "
                             "(recommended for TensorRT).")
    args = parser.parse_args()

    export_onnx(args.model_path, args.output_path, opset=args.opset, fp16=args.fp16)
