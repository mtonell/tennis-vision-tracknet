# Methodology

Detailed description of how TrackNetFast is built, trained, decoded and
scored. 

Contents:

1. [Task and protocols](#task-and-protocols)
2. [Data pipeline](#data-pipeline)
3. [Architecture](#architecture)
4. [Targets and loss](#targets-and-loss)
5. [Training recipe](#training-recipe)
6. [Decoding and overlap fusion](#decoding-and-overlap-fusion)
7. [Deployment: ONNX and TensorRT](#deployment-onnx-and-tensorrt)
8. [Reproducing the reported numbers](#reproducing-the-reported-numbers)

---

## Task and protocols

Input is three consecutive RGB frames at 288×512. Output is, for each of those
three frames, one 2-D point in input-image pixels plus a confidence score.

Training and evaluation both use **stride-1 windows**: window `i` contains
frames `i`, `i+1`, `i+2` and predicts targets for all three. A frame is
therefore seen by up to three windows, which gives two ways to score the same
predictions:

- **standard** - every window contributes three frame-slots.
- **fusion** - one prediction per unique frame, obtained by fusing the
  candidates from every window that sees it. 

## Data pipeline

`src/dataset.py` indexes clips from `Dataset/gameX/ClipY/` and reads
`Label.csv` (`file name`, `visibility`, `x-coordinate`, `y-coordinate`).

For each clip:

1. The label scale is derived from the first frame: `ratio = source_height /
   288`. Labels are divided by `ratio` to live in model pixels.
2. Consecutive frames are stacked into a `(9, H, W)` RGB tensor in `[0, 1]`.
3. The three label frames produce `(3, H, W)` targets.
4. A visible point is rendered as a Gaussian with `sigma = 2.5` px and peak
   `1.0`. 

Augmentation is training-only and applied **per window**, so all three frames
share the same transform and the motion cue stays intact:

- **Horizontal flip** with probability 0.5, with paired label mirroring.
- **Zoom-in crop** at scale `1.0`, `0.9` or `0.8`, resized back to 288×512.
  A ball that leaves the crop is relabelled invisible.

## Architecture

### Input preparation

The dataset already yields `(B, 9, H, W)` in `[0, 1]`. Inside the model
(`TrackNetFast._prepare_input`):

1. The 9 channels are reshaped to three RGB frames.
2. Each channel is normalised with ImageNet mean/std, repeated per frame.
3. If `use_motion_diffs`, luma is computed per frame with
   `0.299 R + 0.587 G + 0.114 B`, and two absolute differences are appended:
   `|L_t − L_{t−1}|` and `|L_{t−1} − L_{t−2}|`.

The stem therefore sees 11 channels. 

### Encoder

MobileNetV3-Large (`features` from torchvision), split into stages that expose
feature maps at stride 2, 4, 8, 16 and 32.

### Neck

A minimal FPN that output a 64-channel map at stride 2 (144×256 for a 288×512 input).

### Heads

- **Heatmap**: 1×1 convolution to 3 channels; one logit map per frame.
- **Offset**: 1×1 convolution to 6 channels, reshaped to `(3, 2, H/2, W/2)`;
  sigmoid gives a subpixel position inside the argmax cell.
- **Visibility**: per-frame maximum heatmap logit, transformed by one learned
  scalar scale and bias.
  This keeps the gate frame-specific without a separate pooled head.

Total: **3.05 M parameters**.

## Targets and loss

`src/losses.py` derives everything from the full-resolution Gaussian target.

### Heatmap

The target is max-pooled to the stride-2 heatmap grid. The loss is a
CenterNet-style focal loss:

```
p        = sigmoid(heatmap_logit)
positive = target
negative = (1 - target)^4
loss     = [ (1-p)^2 * -log(p) * positive
           + p^2    * -log(1-p) * negative ] / sum(positive)
```

with `sum(positive)` clamped to at least 1. 

### Subpixel offset

The target coordinate is the intensity-weighted centroid of the Gaussian. Only
the cell containing that coordinate is supervised:

```
cell = floor(coord / stride)
target_offset = coord / stride - cell
offset_error = |pred_x - target_offset_x| + |pred_y - target_offset_y|
```

The L1 loss is averaged over visible frames only.

### Visibility gate

Binary cross-entropy with logits. The default weight is 0.1, small enough not to disturb the
localisation objective.

```
total = heatmap + 1.0 * offset + 0.1 * visibility
```

## Training recipe

The published checkpoint was trained for 20
epochs cosine run at `1e-3` and annealed to zero.

| item | value |
| --- | --- |
| optimizer | Adam |
| learning rate | `1e-3`, cosine to 0 |
| epochs | 20 |
| batch size | 8 |
| precision | AMP (CUDA) |
| augmentation | flip 0.5 + one zoom crop from 1.0 / 0.9 / 0.8 |

The reported checkpoint is the **last** epoch, because the schedule is fixed.

## Decoding and overlap fusion

### Single prediction

For each of the three slots (`metrics.detection_points`):

1. `p = sigmoid(heatmap)`; take `argmax` over the heatmap.
2. Read the sigmoid offset at that cell.
3. Convert to pixels:
   ```
   x = (cell_x + offset_x) * (W / heat_w)
   y = (cell_y + offset_y) * (H / heat_h)
   ```
4. Score = `sigmoid(visibility_logit)`.
5. Peak = maximum heatmap probability (a localisation diagnostic).

A point is emitted at inference time when score ≥ 0.5 (the canonical operating
point; the threshold is tunable).

### Fusing overlapping windows

Because windows overlap, a frame can have one, two or three candidates.

- **One candidate** → keep it.
- **Two candidates** → average them if within 8 px; otherwise keep the one with
  the higher heatmap peak.
- **Three candidates** → find the closest pair. If that pair agrees within
  8 px, use the pair (a majority vote against the third). If all three agree
  within 8 px, average all three. Otherwise keep the single highest peak.

### Accounting

Each frame produces at most one prediction. With tolerance `tol`:

| GT | prediction | result |
| --- | --- | --- |
| visible | predicted within `tol` | TP |
| visible | no prediction | FN (`fn_gate_closed`) |
| visible | predicted outside `tol` | FN (`fn_wrong_location`) **and** FP (`fp_wrong_location`) |
| invisible | no prediction | not scored |
| invisible | predicted | FP (`fp_invisible`) |

So:

```
TP + FN = number of GT-visible frames
recall    = TP / (TP + FN)
precision = TP / (TP + FP),   FP = fp_wrong_location + fp_invisible
F1        = 2PR / (P + R)
```

### AP

For a tolerance, frames are ranked by their score (gate probability). Sweeping
the threshold gives a PR curve; AP is the step-wise area:

```
AP = Σ (R_i − R_{i−1}) · P_i
```

AP is threshold-free, so the gate cannot be tuned to inflate it. AP@4 is the
headline; AP@8 and AP@12 are loose-localisation context.

### Operating points

- **@0.5**: precision, recall and F1 at the fixed gate of 0.5.
- **best F1**: the threshold that maximises F1 on test, reported with its
  threshold so it is clear it was tuned on these games.

## Deployment: ONNX and TensorRT

`export_onnx.py`:

- **`--fp16`**: run the model in `half()` on CUDA and export a
  static batch-1 graph (opset 16, `dynamo=False`, output names `heatmap`,
  `offset`, `visibility`). No dynamic axes and no precision configuration are
  needed at engine-build time.

`export_tensorrt.py`:

- TRT 10+ network creation with `create_network(0)`;
- an optimization profile is added **only** if the ONNX input has a dynamic
  dimension (batch 1/4/16);
- the saved engine is re-deserialised to print each I/O tensor's actual dtype,
  so FP16 can be verified (`HALF`) instead of assumed.

On a Tesla T4 with the static FP16 engine, one window costs 2.59 ms of inference.

## Reproducing the reported numbers

```bash
# train
python src/train.py --model_name TrackNetFast --epochs 20 --batch_size 8 \
    --optimizer Adam --learning_rate 1e-3 --amp --augment \
    --zoom_scales 1.0,0.9,0.8 --scheduler cosine --min_lr 0 \
    --final_checkpoint last --work_dir ./runs/tracknet_fast
```

The exact checkpoint used for the published numbers is the release asset.
```bash
python src/export_onnx.py --model_path TrackNetFast.pt --output_path tracknet_fast.onnx --fp16

python src/export_tensorrt.py --onnx tracknet_fast.onnx --engine tracknet_fast.engine

python src/benchmark.py --model_path tracknet_fast.engine --instances 1000 --warmup 50
```


