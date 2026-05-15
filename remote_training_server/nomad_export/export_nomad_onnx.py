#!/usr/bin/env python3
"""Export pretrained NoMaD checkpoint to two ONNX submodules.

Splits the monolithic NoMaD model into:
  1. vision_encoder.onnx — one-shot per inference, produces a 256-dim
     conditioning vector from 3 RGB context frames (+ optional goal image).
  2. noise_pred_net.onnx — invoked 10 times per inference by the rover's
     numpy diffusion loop, predicting noise on the 8x2 waypoint tensor.

Run on the DGX (or any x86 host with PyTorch + the visualnav-transformer
repo installed). The rover only sees the resulting ONNX files and then runs
its existing `convert_onnx_to_rknn.sh` on each.

Usage:
    python export_nomad_onnx.py \
        --checkpoint ~/visualnav-transformer/checkpoints/nomad.pth \
        --vint_repo ~/visualnav-transformer
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


# context_size=3 past frames; obs_img stacks those + the current frame.
CONTEXT_SIZE = 3
NUM_OBS_FRAMES = CONTEXT_SIZE + 1  # -> obs_img has 3 * 4 = 12 channels
IMAGE_SIZE = 96
PRED_HORIZON = 8
ACTION_DIM = 2
ENCODING_SIZE = 256


class VisionEncoderWrapper(nn.Module):
    """Standalone wrapper around NoMaD's vision_encoder for ONNX export.

    NoMaD's reference forward dispatches on a string mode argument; ONNX
    does not handle that pattern. This wrapper calls the same underlying
    sub-network with positional tensors so the export is clean and the
    op graph is fully static.

    Inputs:
        obs_img         (1, 3 * (context_size+1), H, W) float32  — context + current frame
        goal_img        (1, 3, H, W)                    float32  — goal RGB (zeros in exploration)
        input_goal_mask (1,)                         int64    — 0=use goal, 1=masked (exploration)
    Output:
        obs_cond        (1, encoding_size)           float32
    """

    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, obs_img, goal_img, input_goal_mask):
        return self.vision_encoder(
            obs_img=obs_img,
            goal_img=goal_img,
            input_goal_mask=input_goal_mask,
        )


class NoisePredWrapper(nn.Module):
    """Standalone wrapper around NoMaD's noise_pred_net (ConditionalUnet1D).

    Inputs:
        sample      (1, pred_horizon, action_dim) float32  — current noisy waypoints
        timestep    (1,)                          int64    — current diffusion step
        global_cond (1, encoding_size)            float32  — output of vision_encoder
    Output:
        noise_pred  (1, pred_horizon, action_dim) float32
    """

    def __init__(self, noise_pred_net):
        super().__init__()
        self.noise_pred_net = noise_pred_net

    def forward(self, sample, timestep, global_cond):
        return self.noise_pred_net(
            sample=sample,
            timestep=timestep,
            global_cond=global_cond,
        )


def load_nomad(checkpoint_path: str, vint_repo: str):
    """Build a NoMaD model and load the pretrained weights.

    Uses the model construction helpers from `vint_train`. The exact API
    surface tracks the upstream repo; if visualnav-transformer changes the
    constructor signature, adjust here.
    """
    sys.path.insert(0, os.path.join(vint_repo, "train"))
    sys.path.insert(0, os.path.join(vint_repo, "deployment", "src"))

    # visualnav-transformer exposes a `load_model` helper in deployment/src/utils.py
    # that reconstructs the model from a yaml config and loads weights. The
    # pretrained checkpoint ships alongside a config file (nomad.yaml).
    from utils import load_model  # noqa: E402  (path injected above)

    config_path = os.path.join(vint_repo, "train", "config", "nomad.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find NoMaD config at {config_path}. "
            "Verify --vint_repo points at the visualnav-transformer checkout."
        )

    import yaml
    with open(config_path) as f:
        params = yaml.safe_load(f)

    device = torch.device("cpu")
    model = load_model(
        model_path=checkpoint_path,
        config=params,
        device=device,
    )
    model.eval()
    return model, params


def export_vision_encoder(model, output_path: str):
    print(f"Exporting vision_encoder -> {output_path}")
    wrapper = VisionEncoderWrapper(model.vision_encoder).eval()

    obs_img = torch.randn(1, 3 * NUM_OBS_FRAMES, IMAGE_SIZE, IMAGE_SIZE)
    goal_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_goal_mask = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        out = wrapper(obs_img, goal_img, input_goal_mask)
    print(f"  sanity: vision_encoder output shape = {tuple(out.shape)}")
    assert out.shape == (1, ENCODING_SIZE), (
        f"Expected (1, {ENCODING_SIZE}), got {tuple(out.shape)}. "
        "ConditionalUnet1D global_cond dimension may have changed."
    )

    torch.onnx.export(
        wrapper,
        (obs_img, goal_img, input_goal_mask),
        output_path,
        opset_version=12,
        do_constant_folding=True,
        input_names=["obs_img", "goal_img", "input_goal_mask"],
        output_names=["obs_cond"],
        dynamic_axes=None,  # all static
    )
    print(f"  wrote {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


def export_noise_pred_net(model, output_path: str):
    print(f"Exporting noise_pred_net -> {output_path}")
    wrapper = NoisePredWrapper(model.noise_pred_net).eval()

    sample = torch.randn(1, PRED_HORIZON, ACTION_DIM)
    timestep = torch.zeros(1, dtype=torch.long)
    global_cond = torch.randn(1, ENCODING_SIZE)

    with torch.no_grad():
        out = wrapper(sample, timestep, global_cond)
    print(f"  sanity: noise_pred_net output shape = {tuple(out.shape)}")
    assert out.shape == (1, PRED_HORIZON, ACTION_DIM), (
        f"Expected (1, {PRED_HORIZON}, {ACTION_DIM}), got {tuple(out.shape)}."
    )

    torch.onnx.export(
        wrapper,
        (sample, timestep, global_cond),
        output_path,
        opset_version=12,
        do_constant_folding=True,
        input_names=["sample", "timestep", "global_cond"],
        output_names=["noise_pred"],
        dynamic_axes=None,
    )
    print(f"  wrote {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to nomad.pth")
    parser.add_argument(
        "--vint_repo", required=True,
        help="Path to a visualnav-transformer git checkout",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write the two ONNX files (default: ./onnx/ next to this script)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else (here / "onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, params = load_nomad(args.checkpoint, args.vint_repo)
    print("Loaded NoMaD checkpoint. Config summary:")
    for k in ("image_size", "context_size", "len_traj_pred",
              "num_diffusion_iters", "encoding_size"):
        if k in params:
            print(f"  {k}: {params[k]}")

    export_vision_encoder(model, str(output_dir / "vision_encoder.onnx"))
    export_noise_pred_net(model, str(output_dir / "noise_pred_net.onnx"))

    print("\nDone. Next step: deploy_nomad_model.sh <rover_ip> <rover_user>")


if __name__ == "__main__":
    main()
