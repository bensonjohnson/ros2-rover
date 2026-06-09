"""Action selection by expected free energy — pure epistemic value.

Active inference picks actions that minimize expected free energy. With no
pragmatic (goal) term, that reduces to MAXIMIZING expected information gain:
do the thing you'd learn the most from. We estimate information gain as the
world model's transition-ensemble disagreement about the resulting latent.

We don't run a planner. Each control step we propose a batch of candidate
track commands (a structured set + random samples), score each by one-step
epistemic value, and softmax-sample one. Sampling (rather than hard argmax)
keeps early behavior from locking onto a single direction and lets the brain
keep probing. One forward pass over the ensemble — cheap enough for the CPU
control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch


# Structured candidates always evaluated: stop, forward, reverse, spin L/R,
# gentle arcs. [left, right] in [-1, 1].
_STRUCTURED = [
    [0.0, 0.0],
    [1.0, 1.0], [0.5, 0.5],
    [-1.0, -1.0], [-0.5, -0.5],
    [1.0, -1.0], [-1.0, 1.0],
    [0.5, -0.5], [-0.5, 0.5],
    [1.0, 0.3], [0.3, 1.0],
]


@dataclass
class ActorConfig:
    action_dim: int = 2
    n_random: int = 48          # random candidates per step
    temperature: float = 0.5    # softmax temperature over the blended score
    deterministic: bool = False # if True, argmax instead of sampling
    forward_bias: float = 0.3   # 0 = pure epistemic, 1 = pure forward translation
    seed: int = 0
    structured: list = field(default_factory=lambda: list(_STRUCTURED))


class EFEActor:
    def __init__(self, cfg: ActorConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self._structured = torch.tensor(cfg.structured, dtype=torch.float32,
                                        device=self.device)
        self._g = torch.Generator(device="cpu").manual_seed(cfg.seed)

    def _candidates(self) -> torch.Tensor:
        rand = (torch.rand(self.cfg.n_random, self.cfg.action_dim,
                           generator=self._g) * 2.0 - 1.0).to(self.device)
        return torch.cat([self._structured, rand], dim=0)

    @torch.no_grad()
    def select(self, model, z_from: torch.Tensor, forward_bias: float | None = None):
        """Return (action[2], info) for the current latent state.

        Score blends normalized epistemic value with a forward-translation
        preference: `score = (1-fb)*epi_norm + fb*fwd_norm`. The translation
        term breaks pure epistemic's rotational degeneracy (spinning in place is
        the most 'informative' move per unit motion) so the rover commits to
        roaming instead of pirouetting.
        """
        fb = self.cfg.forward_bias if forward_bias is None else forward_bias
        cands = self._candidates()
        epi = model.epistemic_value(z_from, cands)        # [N], higher = better

        if fb > 0.0:
            epi_n = (epi - epi.min()) / (epi.max() - epi.min() + 1e-6)
            fwd = (cands[:, 0] + cands[:, 1]) * 0.5        # net translation [-1,1]
            fwd_n = (fwd + 1.0) * 0.5                      # [0,1]
            score = (1.0 - fb) * epi_n + fb * fwd_n
        else:
            score = epi

        if self.cfg.deterministic:
            idx = int(torch.argmax(score))
        else:
            logits = score / max(self.cfg.temperature, 1e-6)
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=0)
            idx = int(torch.multinomial(probs, 1, generator=self._g))

        action = cands[idx]
        info = {
            "epistemic": float(epi[idx]),
            "epistemic_max": float(epi.max()),
            "epistemic_mean": float(epi.mean()),
        }
        return action, info
