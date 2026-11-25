#!/usr/bin/env python
"""Build a TensorRT engine from an ONNX graph (no trtexec needed).

For a static batch-1 graph (the ``--fp16`` export of export_onnx.py) no
optimization profile is needed and no precision flag is required: TRT 10/11
infers FP16 from the graph itself.  For a dynamic-batch graph, an optimization
profile is added automatically.

Usage:
    python src/export_tensorrt.py --onnx tracknet_fast_fp16_b1.onnx \
        --engine tracknet_fast_fp16_b1.engine
"""

import argparse

try:
    import tensorrt as trt
except ImportError:  # local dev machine without TensorRT
    trt = None

from constants import HEIGHT, WIDTH


def build_engine(onnx_path, engine_path, workspace_gb=2):
    if trt is None:
        raise SystemExit(
            "tensorrt is not installed. On Colab: "
            "pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com"
        )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 10+: explicit batch is implicit, pass 0.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * (1 << 30)))

    # Dynamic batch axis -> TRT needs an optimization profile.
    input_tensor = network.get_input(0)
    if any(d < 0 for d in input_tensor.shape):
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, (1, 9, HEIGHT, WIDTH),
                          (4, 9, HEIGHT, WIDTH), (16, 9, HEIGHT, WIDTH))
        config.add_optimization_profile(profile)
        print("Dynamic input detected; added optimization profile (batch 1/4/16)")
    else:
        print(f"Static input {tuple(input_tensor.shape)}, dtype "
              f"{input_tensor.dtype.name}; no profile needed")

    print(f"Building engine from {onnx_path} ...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("Engine build failed (see log above)")

    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"Engine saved to {engine_path}")

    # Inspect the saved engine (serialized bytes -> ICudaEngine) to report the
    # precision TRT actually chose per I/O tensor.
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        saved = runtime.deserialize_cuda_engine(f.read())
    for i in range(saved.num_io_tensors):
        name = saved.get_tensor_name(i)
        print(f"  {name}: {saved.get_tensor_dtype(name).name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from an ONNX model."
    )
    parser.add_argument("--onnx", type=str, required=True,
                        help="Input ONNX file.")
    parser.add_argument("--engine", type=str, required=True,
                        help="Output engine file (.trt/.engine).")
    parser.add_argument("--workspace_gb", type=float, default=2.0,
                        help="Builder workspace in GB (default: 2).")
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, workspace_gb=args.workspace_gb)
