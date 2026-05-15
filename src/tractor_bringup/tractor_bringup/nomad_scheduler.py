"""Pure-numpy port of diffusers.DDPMScheduler for NoMaD's 10-step diffusion loop.

The rover only needs the inference-time `step` operation; training code is not
ported. Constants match the upstream NoMaD config:

    num_train_timesteps = 10
    beta_schedule       = "squaredcos_cap_v2"  (Improved DDPM, Nichol & Dhariwal)
    prediction_type     = "epsilon"
    clip_sample         = True
    variance_type       = "fixed_small_log" (matches diffusers default for this schedule)

Keeping this in numpy avoids dragging torch + diffusers onto the RK3588 board.
The scheduler is stateless across calls aside from the per-trajectory noise the
caller injects, so a single instance is safe to reuse.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _betas_for_alpha_bar(num_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
    """Squared cosine beta schedule (NoMaD's `squaredcos_cap_v2`)."""
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2
    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


class NoMaDDDPMScheduler:
    """Inference-only DDPM scheduler with NoMaD's defaults baked in."""

    def __init__(self, num_train_timesteps: int = 10):
        self.num_train_timesteps = num_train_timesteps
        self.betas = _betas_for_alpha_bar(num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.timesteps = np.arange(num_train_timesteps - 1, -1, -1, dtype=np.int64)

    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """One reverse-diffusion step. Mirrors diffusers.DDPMScheduler.step.

        Args:
            model_output: predicted noise epsilon, same shape as `sample`.
            timestep:     scalar int in [0, num_train_timesteps - 1].
            sample:       current noisy sample x_t.
            noise:        optional gaussian noise the same shape as `sample`.
                          If omitted, a fresh standard normal is drawn — fine
                          for inference where reproducibility is not required.

        Returns:
            x_{t-1}, the next less-noisy sample.
        """
        t = int(timestep)
        prev_t = t - 1

        alpha_prod_t = float(self.alphas_cumprod[t])
        alpha_prod_t_prev = float(self.alphas_cumprod[prev_t]) if prev_t >= 0 else 1.0
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        # epsilon -> x_0 prediction, then clip to [-1, 1] (clip_sample=True).
        pred_original = (sample - math.sqrt(beta_prod_t) * model_output) / math.sqrt(alpha_prod_t)
        pred_original = np.clip(pred_original, -1.0, 1.0)

        # Posterior mean coefficients (Eq. 7, DDPM).
        coef_orig = (math.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
        coef_curr = (math.sqrt(current_alpha_t) * beta_prod_t_prev) / beta_prod_t
        prev_mean = coef_orig * pred_original + coef_curr * sample

        if t == 0:
            return prev_mean.astype(np.float32, copy=False)

        # variance_type="fixed_small_log" -> use small variance from posterior.
        variance = (beta_prod_t_prev / beta_prod_t) * current_beta_t
        variance = max(variance, 1e-20)
        if noise is None:
            noise = np.random.randn(*sample.shape).astype(sample.dtype)
        return (prev_mean + math.sqrt(variance) * noise).astype(np.float32, copy=False)
