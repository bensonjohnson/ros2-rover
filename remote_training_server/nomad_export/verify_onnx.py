#!/usr/bin/env python3
"""Parity-check: compare PyTorch vs ONNX after a full 10-step diffusion loop.

Run after export_nomad_onnx.py. Fails if waypoint MSE between the two
backends exceeds the threshold — that means the ONNX export silently
dropped or rewired an op and the rover would see different behavior.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


CONTEXT_SIZE = 3
IMAGE_SIZE = 96
PRED_HORIZON = 8
ACTION_DIM = 2
ENCODING_SIZE = 256
NUM_DIFFUSION_ITERS = 10
NUM_TRIALS = 5
MSE_THRESHOLD = 0.05


def run_pytorch(model, obs_img, goal_img, mask, init_noise):
    scheduler = DDPMScheduler(
        num_train_timesteps=NUM_DIFFUSION_ITERS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    with torch.no_grad():
        cond = model.vision_encoder(
            obs_img=torch.from_numpy(obs_img),
            goal_img=torch.from_numpy(goal_img),
            input_goal_mask=torch.from_numpy(mask),
        )
        naction = torch.from_numpy(init_noise)
        for k in scheduler.timesteps:
            noise_pred = model.noise_pred_net(
                sample=naction,
                timestep=k.unsqueeze(0) if k.ndim == 0 else k,
                global_cond=cond,
            )
            naction = scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
    return naction.numpy()


def run_onnx(vis_sess, noise_sess, obs_img, goal_img, mask, init_noise):
    scheduler = DDPMScheduler(
        num_train_timesteps=NUM_DIFFUSION_ITERS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    cond = vis_sess.run(
        ["obs_cond"],
        {"obs_img": obs_img, "goal_img": goal_img, "input_goal_mask": mask},
    )[0]
    naction = init_noise.copy()
    for k in scheduler.timesteps:
        timestep = np.array([int(k)], dtype=np.int64)
        noise_pred = noise_sess.run(
            ["noise_pred"],
            {"sample": naction, "timestep": timestep, "global_cond": cond},
        )[0]
        naction = scheduler.step(
            model_output=torch.from_numpy(noise_pred),
            timestep=k,
            sample=torch.from_numpy(naction),
        ).prev_sample.numpy()
    return naction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vint_repo", required=True)
    parser.add_argument("--onnx_dir", default=None)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else (here / "onnx")

    from export_nomad_onnx import load_nomad
    model, _ = load_nomad(args.checkpoint, args.vint_repo)

    vis_sess = ort.InferenceSession(
        str(onnx_dir / "vision_encoder.onnx"), providers=["CPUExecutionProvider"]
    )
    noise_sess = ort.InferenceSession(
        str(onnx_dir / "noise_pred_net.onnx"), providers=["CPUExecutionProvider"]
    )

    rng = np.random.default_rng(42)
    max_mse = 0.0
    for trial in range(NUM_TRIALS):
        obs_img = rng.standard_normal(
            (1, 3 * CONTEXT_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        ).astype(np.float32)
        goal_img = rng.standard_normal((1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        mask = np.zeros(1, dtype=np.int64)
        init_noise = rng.standard_normal((1, PRED_HORIZON, ACTION_DIM)).astype(np.float32)

        torch_out = run_pytorch(model, obs_img, goal_img, mask, init_noise)
        onnx_out = run_onnx(vis_sess, noise_sess, obs_img, goal_img, mask, init_noise)
        mse = float(np.mean((torch_out - onnx_out) ** 2))
        max_mse = max(max_mse, mse)
        print(f"trial {trial}: MSE={mse:.6f}")

    print(f"\nmax MSE across {NUM_TRIALS} trials: {max_mse:.6f}")
    if max_mse > MSE_THRESHOLD:
        print(f"FAIL: max MSE {max_mse:.4f} > {MSE_THRESHOLD}")
        sys.exit(1)
    print(f"PASS: max MSE {max_mse:.4f} <= {MSE_THRESHOLD}")


if __name__ == "__main__":
    main()
