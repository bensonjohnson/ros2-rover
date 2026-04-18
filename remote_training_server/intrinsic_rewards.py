"""Server-side intrinsic motivation utilities for DreamerV3.

Two complementary novelty signals are produced here, both consumed inside the
imagination loop in `v620_dreamer_trainer.py`:

1. **Plan2Explore (P2E)** — variance of an ensemble that predicts the next
   stochastic latent. The ensemble itself lives in `model_architectures.py`
   (class `P2EEnsemble`); training is driven from the trainer. This module
   only provides the loss helper.

2. **Lifetime SimHash novelty** — a locality-sensitive hash count over encoder
   embeddings, persisted to disk so that "have I ever seen anything like this
   embedding before" survives across runs and across environments. Used as a
   *modulator* on the rover-supplied episodic novelty channel, never as a
   standalone reward (NGU pattern: lifetime × episodic).

Reward channels carried in the chunk are computed on the rover. P2E and the
lifetime modulator are added here at the actor objective so that the rover never
pays NPU/inference cost for them and the ONNX export graph is unaffected.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Plan2Explore loss helper
# ---------------------------------------------------------------------------


def p2e_ensemble_loss(
    ensemble,                       # P2EEnsemble
    feat_seq: torch.Tensor,         # (B, T, feat_dim)
    action_seq: torch.Tensor,       # (B, T, action_dim)
    target_z_seq: torch.Tensor,     # (B, T, stoch_dim) — posterior z at next step (detached)
    bootstrap_p: float = 0.5,
) -> torch.Tensor:
    """Train each ensemble head on independently-bootstrapped (input, target) pairs.

    `feat_seq[t]` is concat(h_t, z_t); the target is `target_z_seq[t]` which is the
    next-step posterior z (detached) — i.e. we predict z_{t+1} from (h_t, z_t, a_t).
    Caller is responsible for the time-shift; pass already-aligned tensors here.
    """
    # Predict (n_heads, B, T, stoch_dim)
    preds = ensemble(feat_seq, action_seq)
    n_heads = preds.shape[0]
    target = target_z_seq.detach()

    # Per-head bootstrap mask: drop ~ (1 - bootstrap_p) of timesteps
    B, T = feat_seq.shape[:2]
    device = feat_seq.device
    masks = (torch.rand(n_heads, B, T, device=device) < bootstrap_p).float()
    # Avoid degenerate all-zero mask per head
    masks = masks + 1e-6
    masks = masks / masks.mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)

    # Cross-entropy against soft target via MSE on logits is fine — DreamerV3
    # uses categorical logits and the ensemble is just a regressor, so MSE on
    # the flattened representation gives a clean disagreement signal.
    per_step = F.mse_loss(preds, target.unsqueeze(0).expand_as(preds), reduction='none')
    per_step = per_step.mean(dim=-1)  # (n_heads, B, T)
    loss = (per_step * masks).mean()
    return loss


# ---------------------------------------------------------------------------
# Lifetime SimHash novelty
# ---------------------------------------------------------------------------


class LifetimeSimHash:
    """Locality-sensitive hash counter over encoder embeddings.

    Why SimHash: O(1) lookup, O(D × bits) update, no learned model that can
    drift, robust to small embedding noise. We store a fixed random projection
    matrix and a `dict[hash_int, count]`.

    Usage pattern (server, inside the actor objective):

        modulator = lifetime.modulator(embeds_flat)   # tensor (N,)
        episodic_signal = batch_episodic * modulator  # NGU two-timescale combo
        intrinsic = w_episodic * episodic_signal + w_p2e * p2e_disagreement

    The store is keyed on encoder embeddings (not world coordinates), so the
    "have I seen this latent" signal is *environment agnostic*: re-mapping the
    same room produces hits, but driving into a brand-new yard does not.
    """

    def __init__(
        self,
        embed_dim: int,
        n_bits: int = 16,
        modulator_strength: float = 1.0,
        seed: int = 0xD1EAFA,
    ):
        self.embed_dim = embed_dim
        self.n_bits = n_bits
        self.modulator_strength = modulator_strength
        rng = np.random.default_rng(seed)
        # Random projection — drawn once and persisted with the counts so
        # restored hashes line up with new ones.
        self.projection = rng.standard_normal((embed_dim, n_bits)).astype(np.float32)
        self.counts: dict[int, int] = {}

    # ----- core ops -----

    def _hash_np(self, embeds: np.ndarray) -> np.ndarray:
        """Hash a (N, D) array of embeddings → (N,) int64 buckets."""
        if embeds.ndim == 1:
            embeds = embeds[None, :]
        bits = (embeds @ self.projection) > 0  # (N, n_bits)
        # Pack bool bits → int. n_bits ≤ 64 so a single int per row is fine.
        weights = (1 << np.arange(self.n_bits, dtype=np.int64))
        return (bits.astype(np.int64) * weights).sum(axis=1)

    def update_and_query(self, embeds: np.ndarray) -> np.ndarray:
        """Increment counts for these embeddings and return their *post-update* counts."""
        keys = self._hash_np(embeds)
        out = np.empty(keys.shape[0], dtype=np.float32)
        for i, k in enumerate(keys):
            k = int(k)
            self.counts[k] = self.counts.get(k, 0) + 1
            out[i] = self.counts[k]
        return out

    def query(self, embeds: np.ndarray) -> np.ndarray:
        """Return counts without updating (use during imagination scoring)."""
        keys = self._hash_np(embeds)
        return np.array([self.counts.get(int(k), 0) for k in keys], dtype=np.float32)

    # ----- modulator API consumed by the trainer -----

    def modulator(self, embeds: torch.Tensor) -> torch.Tensor:
        """`1 + c / sqrt(1 + count(embed))`. Tensor in, tensor out (no grad).

        Multiplies the rover-supplied episodic channel inside the imagination
        loss to give a two-timescale NGU-style intrinsic.
        """
        flat = embeds.reshape(-1, embeds.shape[-1]).detach().cpu().numpy()
        counts = self.query(flat)
        mod = 1.0 + self.modulator_strength / np.sqrt(1.0 + counts)
        return torch.from_numpy(mod.astype(np.float32)).to(embeds.device).reshape(embeds.shape[:-1])

    # ----- persistence -----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'embed_dim': self.embed_dim,
                'n_bits': self.n_bits,
                'modulator_strength': self.modulator_strength,
                'projection': self.projection,
                'counts': self.counts,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Optional["LifetimeSimHash"]:
        path = Path(path)
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            d = pickle.load(f)
        obj = cls.__new__(cls)
        obj.embed_dim = d['embed_dim']
        obj.n_bits = d['n_bits']
        obj.modulator_strength = d['modulator_strength']
        obj.projection = d['projection']
        obj.counts = d['counts']
        return obj

    def hit_rate(self) -> float:
        """Fraction of buckets currently populated. Saturates as exploration completes."""
        return len(self.counts) / float(1 << self.n_bits)
