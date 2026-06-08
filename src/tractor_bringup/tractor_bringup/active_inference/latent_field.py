"""Growing latent map field — the rover's allocentric 'cognitive map'.

Not an occupancy grid. A sparse field of learned latent vectors, one per coarse
cell in the odom frame, allocated lazily as the rover discovers space. Each
cell's latent encodes the *allocentric* local openness signature seen from that
cell (lidar scans are de-rotated by heading so the same place reads the same
regardless of approach direction). Three shared, locally-learned pieces:

    decoder  g(m)      : latent -> predicted 72-bin allocentric openness
    spatial  h(nbrs)   : neighbour latents -> predicted cell latent  (fill-in)
    (per-cell confidence grows with observation; frontier = low-conf edge)

Everything trains with the same pure predictive-coding local rules as
pc_world_model — no backprop. The decoder/spatial weights are shared across all
cells (the reusable 'structure'); the per-cell latents are the map 'content'.

Querying g on a cell whose latent came from the spatial prior (never directly
observed) is the amodal fill-in: "what would I see from over there?" The
uncertainty of those predictions is what the epistemic actor steers toward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass
class FieldConfig:
    obs_dim: int = 72
    latent_dim: int = 24
    cell_size: float = 0.5        # metres per cell
    n_infer_iters: int = 30
    infer_lr: float = 0.35
    lr_decode: float = 0.08
    lr_spatial: float = 0.03
    precision_obs: float = 1.0
    precision_spatial: float = 0.5
    conf_cap: float = 8.0         # observations after which a cell is "known"
    seed: int = 0


class LatentField:
    def __init__(self, cfg: FieldConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        O, D = cfg.obs_dim, cfg.latent_dim

        def randn(*shape, scale):
            return (torch.randn(*shape, generator=g) * scale).to(self.device)

        # Shared decoder: o_allo = σ(W_g tanh(m) + b_g)
        self.W_g = randn(O, D, scale=0.1)
        self.b_g = torch.zeros(O, device=self.device)
        # Shared spatial prior: m ≈ W_h tanh(mean_neighbours) + b_h
        self.W_h = randn(D, D, scale=0.1)
        self.b_h = torch.zeros(D, device=self.device)

        self.pi_o = cfg.precision_obs
        self.pi_s = cfg.precision_spatial
        self._bin_w = 2.0 * math.pi / O

        # Sparse map: (ix, iy) -> {"m": tensor[D], "conf": float}
        self.cells: dict[tuple[int, int], dict] = {}

    # ---- geometry ----------------------------------------------------------

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        cs = self.cfg.cell_size
        return (int(math.floor(x / cs)), int(math.floor(y / cs)))

    def _neighbours(self, cell):
        ix, iy = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    yield (ix + dx, iy + dy)

    def _neighbour_mean(self, cell) -> torch.Tensor:
        ms = [self.cells[n]["m"] for n in self._neighbours(cell) if n in self.cells]
        if not ms:
            return torch.zeros(self.cfg.latent_dim, device=self.device)
        return torch.stack(ms).mean(dim=0)

    def _spatial_pred(self, cell) -> torch.Tensor:
        nb = torch.tanh(self._neighbour_mean(cell))
        return self.W_h @ nb + self.b_h

    # ---- generative pieces -------------------------------------------------

    def _decode(self, m: torch.Tensor):
        s = torch.tanh(m)
        o_hat = torch.sigmoid(self.W_g @ s + self.b_g)
        return o_hat, s

    def _allocentric(self, obs: torch.Tensor, theta: float) -> torch.Tensor:
        """De-rotate an egocentric scan (bin 0 = robot forward) into the world
        frame (bin 0 = odom +x), so a cell's latent is heading-invariant."""
        shift = int(round(theta / self._bin_w))
        return torch.roll(obs, shifts=shift, dims=0)

    # ---- the per-step update: observe, infer the cell, learn ---------------

    @torch.no_grad()
    def observe(self, x: float, y: float, theta: float, obs: torch.Tensor) -> float:
        """Fold one scan into the cell at (x, y). Returns the decode error norm."""
        cell = self.world_to_cell(x, y)
        allo = self._allocentric(obs, theta)
        m_pred = self._spatial_pred(cell)              # prior from neighbours

        if cell not in self.cells:
            # New cell: seed from the spatial prior (structure-guided fill-in).
            self.cells[cell] = {"m": m_pred.clone(), "conf": 0.0}

        m = self.cells[cell]["m"].clone()
        cfg = self.cfg

        # Settle the latent: minimize sensory error + spatial-prior error.
        for _ in range(cfg.n_infer_iters):
            o_hat, s = self._decode(m)
            e_o = allo - o_hat
            dF_du = -self.pi_o * e_o * (o_hat * (1.0 - o_hat))
            g_obs = (self.W_g.t() @ dF_du) * (1.0 - s * s)
            g_sp = self.pi_s * (m - m_pred)
            m = m - cfg.infer_lr * (g_obs + g_sp)

        # Local learning: decoder weights (shared 'structure').
        o_hat, s = self._decode(m)
        e_o = allo - o_hat
        dF_du = -self.pi_o * e_o * (o_hat * (1.0 - o_hat))
        self.W_g -= cfg.lr_decode * torch.outer(dF_du, s)
        self.b_g -= cfg.lr_decode * dF_du

        # Local learning: spatial prior (neighbours predict this cell).
        nb = torch.tanh(self._neighbour_mean(cell))
        e_s = m - (self.W_h @ nb + self.b_h)
        self.W_h += cfg.lr_spatial * self.pi_s * torch.outer(e_s, nb)
        self.b_h += cfg.lr_spatial * self.pi_s * e_s

        self.cells[cell]["m"] = m
        self.cells[cell]["conf"] = min(self.cells[cell]["conf"] + 1.0, cfg.conf_cap)
        return float(torch.linalg.norm(e_o))

    # ---- queries used by the actor and the dashboard -----------------------

    def confidence(self, cell) -> float:
        c = self.cells.get(cell)
        return (c["conf"] / self.cfg.conf_cap) if c else 0.0

    def novelty_at(self, x: float, y: float) -> float:
        """1.0 = totally unknown (frontier), 0.0 = fully known."""
        return 1.0 - self.confidence(self.world_to_cell(x, y))

    @torch.no_grad()
    def predicted_openness(self, cell) -> float:
        """Mean predicted openness for a cell (observed latent, else spatial
        prior). Used to render the filled-in map."""
        if cell in self.cells:
            m = self.cells[cell]["m"]
        else:
            m = self._spatial_pred(cell)
        o_hat, _ = self._decode(m)
        return float(o_hat.mean())

    def frontier_cells(self) -> list[tuple[int, int]]:
        """Allocated cells that border at least one unallocated cell."""
        out = []
        for cell in self.cells:
            if any(n not in self.cells for n in self._neighbours(cell)):
                out.append(cell)
        return out

    def map_snapshot(self) -> list[tuple[int, int, float, float]]:
        """(ix, iy, confidence, mean_openness) for every known cell — dashboard."""
        return [(ix, iy, self.confidence((ix, iy)), self.predicted_openness((ix, iy)))
                for (ix, iy) in self.cells]

    # ---- persistence -------------------------------------------------------

    def state_dict(self):
        return {"W_g": self.W_g, "b_g": self.b_g, "W_h": self.W_h, "b_h": self.b_h,
                "cells": self.cells, "cfg": self.cfg}

    def load_state_dict(self, sd):
        self.W_g, self.b_g = sd["W_g"], sd["b_g"]
        self.W_h, self.b_h = sd["W_h"], sd["b_h"]
        self.cells = sd["cells"]
