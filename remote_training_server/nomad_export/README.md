# NoMaD export pipeline (DGX side)

PyTorch → ONNX export for the pretrained NoMaD checkpoint from
`https://github.com/robodhruv/visualnav-transformer`. The rover then
runs its existing `convert_onnx_to_rknn.sh` on each ONNX to produce the
two `.rknn` files consumed by `nomad_rknn_runner`.

## One-time setup (on the DGX)

```bash
# 1. Clone visualnav-transformer alongside this checkout
cd ~/
git clone https://github.com/robodhruv/visualnav-transformer.git
cd visualnav-transformer
pip install -e train/   # installs the vint_train package

# 2. Download the pretrained NoMaD checkpoint
#    (link is in the repo's README under "Model Weights")
mkdir -p ~/visualnav-transformer/checkpoints
# Drop nomad.pth into the directory above. The fetch script below
# can do this if a direct URL is set in fetch_checkpoint.py.
```

## Per-export workflow

```bash
cd ~/ros2-rover/remote_training_server/nomad_export

# Export PyTorch → ONNX (writes onnx/vision_encoder.onnx, onnx/noise_pred_net.onnx)
python export_nomad_onnx.py \
    --checkpoint ~/visualnav-transformer/checkpoints/nomad.pth \
    --vint_repo ~/visualnav-transformer

# (Optional) parity-check ONNX vs PyTorch
python verify_onnx.py

# Deploy ONNX to rover and trigger on-rover RKNN conversion
./deploy_nomad_model.sh 192.168.1.50 benson
```

## What gets exported

Two ONNX files, each static-shape, opset 12:

- `vision_encoder.onnx`
  - inputs: `obs_img (1, 12, 96, 96) f32` (context_size+1 = 4 frames stacked),
    `goal_img (1, 3, 96, 96) f32`, `input_goal_mask (1,) i64`
  - output: `obs_cond (1, 256) f32`
- `noise_pred_net.onnx` (N = `--num_samples`, default 8)
  - inputs: `sample (N, 8, 2) f32`, `timestep (N,) i64`,
    `global_cond (N, 256) f32`
  - output: `noise_pred (N, 8, 2) f32`

The diffusion loop (10 denoising steps with the Square Cosine schedule)
runs in Python on the rover — `noise_pred_net.rknn` is invoked 10 times
per inference with the scheduler step done in pure numpy. No loop is
compiled into the RKNN graph.

`noise_pred_net` is exported with a fixed batch of N so the rover samples
N candidate trajectories per inference (the fan of paths shown in the
NoMaD demo). `vision_encoder` stays batch-1; its 256-d output is tiled to
N rows on the rover. **`--num_samples` must match `NUM_SAMPLES` in
`nomad_rknn_runner.py`.**
