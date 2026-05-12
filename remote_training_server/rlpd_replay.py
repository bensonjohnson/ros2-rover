"""RLPD replay buffers — online (FIFO ring) + demo (append-only, persistent).

Stores per-step transitions and reconstructs frame-stacked observations at
sample time using `is_first` as the episode boundary. Past frames that cross
an `is_first` boundary are zero-padded (matching the rover-side reset
semantics). This keeps the wire format unstacked (saves bandwidth) and lets
the trainer pick stack depth without rover changes.

A chunk of T transitions produces (T-1) replay entries: each entry pairs
transition t with t+1 as its next-state. The final transition of every chunk
is dropped because we don't have its next-state until the following chunk
arrives — at ~1.5% data loss for T=64, not worth implementing a pending buffer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _stack_window(buf: np.ndarray, idx: int, k: int, is_first: np.ndarray) -> np.ndarray:
    """Return the k-frame stack ending at `idx`, zero-padded across episode boundaries.

    `buf` has shape (capacity, *frame_shape); the stack is concatenated along
    the channel axis (axis 0 of `frame_shape`) to match the ONNX wrapper's
    expected input layout (rgb_stack: 3*K channels, bev_stack: 2*K channels,
    proprio_stack: 6*K flat).

    Walks backwards from `idx`; once we hit a transition with `is_first=True`
    we stop and zero-pad the rest (oldest frames).
    """
    frames = [None] * k
    # The current frame goes last (most recent)
    frames[-1] = buf[idx]
    blocked = False
    cap = buf.shape[0]
    for offset in range(1, k):
        if blocked:
            frames[k - 1 - offset] = np.zeros_like(buf[idx])
            continue
        prev = (idx - offset) % cap
        # The "current" frame's is_first marker only blocks frames *older* than it
        if is_first[prev]:
            blocked = True
            frames[k - 1 - offset] = np.zeros_like(buf[idx])
        else:
            frames[k - 1 - offset] = buf[prev]
    return np.concatenate(frames, axis=0)


# ---------------------------------------------------------------------------
# OnlineReplay — FIFO ring buffer
# ---------------------------------------------------------------------------


class OnlineReplay:
    """Circular buffer of per-step transitions.

    All observation streams kept in their wire-format dtype (uint8 for rgb/bev)
    to save host RAM; the trainer casts to float32 and divides by 255 at
    sample time.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        rgb_shape: tuple = (3, 84, 84),
        bev_shape: tuple = (2, 128, 128),
        proprio_dim: int = 6,
        action_dim: int = 2,
        n_reward_channels: int = 5,
    ):
        self.capacity = capacity
        self.rgb_shape = rgb_shape
        self.bev_shape = bev_shape
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.n_reward_channels = n_reward_channels

        self.rgb = np.zeros((capacity, *rgb_shape), dtype=np.uint8)
        self.bev = np.zeros((capacity, *bev_shape), dtype=np.uint8)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, n_reward_channels), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.is_first = np.zeros(capacity, dtype=bool)
        self.is_demo = np.zeros(capacity, dtype=bool)
        self.is_intervention = np.zeros(capacity, dtype=bool)
        # Slot is "valid for sampling as state-t" iff its successor in the
        # circular buffer is also populated and belongs to the same chunk
        # (next_is_first == False AND not just overwritten).
        self.has_next = np.zeros(capacity, dtype=bool)

        self.idx = 0      # next write position
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_chunk(self, chunk: dict) -> int:
        """Append a chunk's transitions, dropping the final transition.

        Returns number of transitions actually inserted.
        """
        T = len(chunk['rewards'])
        if T < 2:
            return 0

        # Per chunk we add T-1 transitions (last one has no next-state available
        # within the chunk; we don't bother stitching cross-chunk).
        n_add = T - 1
        rgb = chunk['rgb']
        bev = chunk['bev']
        proprio = chunk['proprio']
        action = chunk['action'] if 'action' in chunk else chunk['actions']
        reward = chunk['reward'] if 'reward' in chunk else chunk['rewards']
        done = chunk['done'] if 'done' in chunk else chunk['dones']
        is_first = chunk['is_first']
        is_demo = chunk.get('is_demo', np.zeros(T, dtype=bool))
        is_intervention = chunk.get('is_intervention', np.zeros(T, dtype=bool))

        for t in range(n_add):
            i = self.idx
            self.rgb[i] = rgb[t]
            self.bev[i] = bev[t]
            self.proprio[i] = proprio[t]
            self.action[i] = action[t]
            self.reward[i] = reward[t]
            self.done[i] = bool(done[t])
            self.is_first[i] = bool(is_first[t])
            self.is_demo[i] = bool(is_demo[t])
            self.is_intervention[i] = bool(is_intervention[t])
            # has_next is True iff t+1 in the chunk shares the episode AND is
            # not flagged as a fresh episode start.
            self.has_next[i] = (not bool(done[t])) and (not bool(is_first[t + 1]))
            # Whoever previously occupied i+1 had its has_next mark relative to
            # the OLD i+1's old neighbor; that neighbor is now this newly-written
            # i, so the OLD i+1's has_next is no longer valid. We mark it False
            # to be safe — a marginal data loss but avoids bogus next-state
            # pairings across the ring boundary.
            nxt = (i + 1) % self.capacity
            if self.size > 0 or self.idx > 0:
                self.has_next[nxt] = False
            self.idx = (self.idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        # Stash the FINAL chunk frame as obs only (so sampling the second-to-last
        # transition can fetch its next-state from this slot). We mark has_next
        # False on it so it's not sampled as a state-t itself.
        i = self.idx
        last = T - 1
        self.rgb[i] = rgb[last]
        self.bev[i] = bev[last]
        self.proprio[i] = proprio[last]
        self.action[i] = action[last]
        self.reward[i] = reward[last]
        self.done[i] = bool(done[last])
        self.is_first[i] = bool(is_first[last])
        self.is_demo[i] = bool(is_demo[last])
        self.is_intervention[i] = bool(is_intervention[last])
        self.has_next[i] = False
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return n_add

    def sample(self, batch: int, frame_stack: int, rng: np.random.Generator) -> dict:
        """Sample `batch` transitions with frame-stacked observations.

        Returns a dict of stacked arrays:
          rgb_stack       : (B, 3*K, 84, 84) uint8
          bev_stack       : (B, 2*K, 128, 128) uint8
          proprio_stack   : (B, 6*K) float32
          action          : (B, 2) float32
          reward          : (B, 5) float32
          done            : (B,) bool
          is_demo         : (B,) bool
          is_intervention : (B,) bool
          next_rgb_stack  : (B, 3*K, 84, 84) uint8
          next_bev_stack  : (B, 2*K, 128, 128) uint8
          next_proprio_stack : (B, 6*K) float32
        """
        if self.size == 0:
            raise RuntimeError("OnlineReplay is empty")

        # Pick valid sample indices — those flagged has_next=True.
        valid_idx = np.flatnonzero(self.has_next[: self.size]) if self.size < self.capacity \
            else np.flatnonzero(self.has_next)
        if len(valid_idx) == 0:
            raise RuntimeError("No valid (has_next) transitions available")
        chosen = valid_idx[rng.integers(0, len(valid_idx), size=batch)]

        rgb_stack = np.empty((batch, 3 * frame_stack, *self.rgb_shape[1:]), dtype=np.uint8)
        bev_stack = np.empty((batch, 2 * frame_stack, *self.bev_shape[1:]), dtype=np.uint8)
        proprio_stack = np.empty((batch, self.proprio_dim * frame_stack), dtype=np.float32)
        next_rgb_stack = np.empty_like(rgb_stack)
        next_bev_stack = np.empty_like(bev_stack)
        next_proprio_stack = np.empty_like(proprio_stack)
        actions = np.empty((batch, self.action_dim), dtype=np.float32)
        rewards = np.empty((batch, self.n_reward_channels), dtype=np.float32)
        dones = np.empty(batch, dtype=bool)
        is_demo = np.empty(batch, dtype=bool)
        is_intervention = np.empty(batch, dtype=bool)

        for b, i in enumerate(chosen):
            i = int(i)
            nxt = (i + 1) % self.capacity
            rgb_stack[b] = _stack_window(self.rgb, i, frame_stack, self.is_first)
            bev_stack[b] = _stack_window(self.bev, i, frame_stack, self.is_first)
            pro_window = _stack_window(
                self.proprio.reshape(self.capacity, self.proprio_dim, 1),
                i, frame_stack, self.is_first,
            ).reshape(-1)
            proprio_stack[b] = pro_window
            next_rgb_stack[b] = _stack_window(self.rgb, nxt, frame_stack, self.is_first)
            next_bev_stack[b] = _stack_window(self.bev, nxt, frame_stack, self.is_first)
            next_pro_window = _stack_window(
                self.proprio.reshape(self.capacity, self.proprio_dim, 1),
                nxt, frame_stack, self.is_first,
            ).reshape(-1)
            next_proprio_stack[b] = next_pro_window
            actions[b] = self.action[i]
            rewards[b] = self.reward[i]
            dones[b] = self.done[i]
            is_demo[b] = self.is_demo[i]
            is_intervention[b] = self.is_intervention[i]

        return {
            'rgb_stack': rgb_stack,
            'bev_stack': bev_stack,
            'proprio_stack': proprio_stack,
            'action': actions,
            'reward': rewards,
            'done': dones,
            'is_demo': is_demo,
            'is_intervention': is_intervention,
            'next_rgb_stack': next_rgb_stack,
            'next_bev_stack': next_bev_stack,
            'next_proprio_stack': next_proprio_stack,
        }


# ---------------------------------------------------------------------------
# DemoReplay — append-only, persists to disk
# ---------------------------------------------------------------------------


class DemoReplay:
    """Append-only transition store with disk persistence.

    Layout mirrors OnlineReplay (same fields, same dtypes) but grows
    dynamically rather than ring-cycling. Saved to `demos.npz` after every
    `add_chunk` call so demos survive server restarts. The save is atomic
    (write to `.tmp`, fsync, rename).
    """

    def __init__(
        self,
        rgb_shape: tuple = (3, 84, 84),
        bev_shape: tuple = (2, 128, 128),
        proprio_dim: int = 6,
        action_dim: int = 2,
        n_reward_channels: int = 5,
        initial_capacity: int = 16384,
    ):
        self.rgb_shape = rgb_shape
        self.bev_shape = bev_shape
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.n_reward_channels = n_reward_channels

        self.capacity = initial_capacity
        self.size = 0
        self._alloc()

    def _alloc(self):
        self.rgb = np.zeros((self.capacity, *self.rgb_shape), dtype=np.uint8)
        self.bev = np.zeros((self.capacity, *self.bev_shape), dtype=np.uint8)
        self.proprio = np.zeros((self.capacity, self.proprio_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, self.n_reward_channels), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=bool)
        self.is_first = np.zeros(self.capacity, dtype=bool)
        self.is_intervention = np.zeros(self.capacity, dtype=bool)
        self.has_next = np.zeros(self.capacity, dtype=bool)

    def _ensure_capacity(self, need: int):
        if self.size + need <= self.capacity:
            return
        new_cap = self.capacity
        while self.size + need > new_cap:
            new_cap *= 2

        def _grow(arr: np.ndarray) -> np.ndarray:
            new_arr = np.zeros((new_cap, *arr.shape[1:]), dtype=arr.dtype)
            new_arr[: self.size] = arr[: self.size]
            return new_arr

        self.rgb = _grow(self.rgb)
        self.bev = _grow(self.bev)
        self.proprio = _grow(self.proprio)
        self.action = _grow(self.action)
        self.reward = _grow(self.reward)
        self.done = _grow(self.done)
        self.is_first = _grow(self.is_first)
        self.is_intervention = _grow(self.is_intervention)
        self.has_next = _grow(self.has_next)
        self.capacity = new_cap

    def __len__(self) -> int:
        return self.size

    def add_chunk(self, chunk: dict) -> int:
        T = len(chunk['rewards'])
        if T < 2:
            return 0
        self._ensure_capacity(T)  # we also stash the trailing frame
        rgb = chunk['rgb']
        bev = chunk['bev']
        proprio = chunk['proprio']
        action = chunk['action'] if 'action' in chunk else chunk['actions']
        reward = chunk['reward'] if 'reward' in chunk else chunk['rewards']
        done = chunk['done'] if 'done' in chunk else chunk['dones']
        is_first = chunk['is_first']
        is_intervention = chunk.get('is_intervention', np.zeros(T, dtype=bool))

        n_add = T - 1
        start = self.size
        for t in range(n_add):
            i = start + t
            self.rgb[i] = rgb[t]
            self.bev[i] = bev[t]
            self.proprio[i] = proprio[t]
            self.action[i] = action[t]
            self.reward[i] = reward[t]
            self.done[i] = bool(done[t])
            self.is_first[i] = bool(is_first[t])
            self.is_intervention[i] = bool(is_intervention[t])
            self.has_next[i] = (not bool(done[t])) and (not bool(is_first[t + 1]))
        # Trailing frame (no has_next)
        i = start + n_add
        last = T - 1
        self.rgb[i] = rgb[last]
        self.bev[i] = bev[last]
        self.proprio[i] = proprio[last]
        self.action[i] = action[last]
        self.reward[i] = reward[last]
        self.done[i] = bool(done[last])
        self.is_first[i] = bool(is_first[last])
        self.is_intervention[i] = bool(is_intervention[last])
        self.has_next[i] = False
        self.size = start + T

        return n_add

    def sample(self, batch: int, frame_stack: int, rng: np.random.Generator) -> dict:
        if self.size == 0:
            raise RuntimeError("DemoReplay is empty")
        valid_idx = np.flatnonzero(self.has_next[: self.size])
        if len(valid_idx) == 0:
            raise RuntimeError("No valid (has_next) demo transitions")
        chosen = valid_idx[rng.integers(0, len(valid_idx), size=batch)]

        rgb_stack = np.empty((batch, 3 * frame_stack, *self.rgb_shape[1:]), dtype=np.uint8)
        bev_stack = np.empty((batch, 2 * frame_stack, *self.bev_shape[1:]), dtype=np.uint8)
        proprio_stack = np.empty((batch, self.proprio_dim * frame_stack), dtype=np.float32)
        next_rgb_stack = np.empty_like(rgb_stack)
        next_bev_stack = np.empty_like(bev_stack)
        next_proprio_stack = np.empty_like(proprio_stack)
        actions = np.empty((batch, self.action_dim), dtype=np.float32)
        rewards = np.empty((batch, self.n_reward_channels), dtype=np.float32)
        dones = np.empty(batch, dtype=bool)
        is_demo = np.ones(batch, dtype=bool)  # all demo
        is_intervention = np.empty(batch, dtype=bool)

        # `_stack_window` expects a circular buffer; for the linear demo store
        # we just clamp negative indices to zero-pad. Reuse helper by passing
        # `is_first` of the full buffer — index 0 is always treated as a start.
        cap = self.size
        is_first_eff = self.is_first[: cap].copy()
        is_first_eff[0] = True  # force zero-pad before index 0
        rgb_view = self.rgb[: cap]
        bev_view = self.bev[: cap]
        proprio_view = self.proprio[: cap]

        def _stack_linear(buf: np.ndarray, idx: int, k: int) -> np.ndarray:
            frames = [None] * k
            frames[-1] = buf[idx]
            blocked = False
            for offset in range(1, k):
                if blocked or idx - offset < 0:
                    frames[k - 1 - offset] = np.zeros_like(buf[idx])
                    blocked = True
                    continue
                prev = idx - offset
                if is_first_eff[prev]:
                    blocked = True
                    frames[k - 1 - offset] = np.zeros_like(buf[idx])
                else:
                    frames[k - 1 - offset] = buf[prev]
            return np.concatenate(frames, axis=0)

        for b, i in enumerate(chosen):
            i = int(i)
            nxt = i + 1
            rgb_stack[b] = _stack_linear(rgb_view, i, frame_stack)
            bev_stack[b] = _stack_linear(bev_view, i, frame_stack)
            proprio_stack[b] = _stack_linear(
                proprio_view.reshape(cap, self.proprio_dim, 1), i, frame_stack
            ).reshape(-1)
            next_rgb_stack[b] = _stack_linear(rgb_view, nxt, frame_stack)
            next_bev_stack[b] = _stack_linear(bev_view, nxt, frame_stack)
            next_proprio_stack[b] = _stack_linear(
                proprio_view.reshape(cap, self.proprio_dim, 1), nxt, frame_stack
            ).reshape(-1)
            actions[b] = self.action[i]
            rewards[b] = self.reward[i]
            dones[b] = self.done[i]
            is_intervention[b] = self.is_intervention[i]

        return {
            'rgb_stack': rgb_stack,
            'bev_stack': bev_stack,
            'proprio_stack': proprio_stack,
            'action': actions,
            'reward': rewards,
            'done': dones,
            'is_demo': is_demo,
            'is_intervention': is_intervention,
            'next_rgb_stack': next_rgb_stack,
            'next_bev_stack': next_bev_stack,
            'next_proprio_stack': next_proprio_stack,
        }

    # ---- persistence ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pass an explicit file handle so np.savez doesn't append `.npz` to our
        # `*.tmp` path. Open in binary write + immediately rename for atomicity.
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'wb') as f:
            np.savez(
                f,
                size=self.size,
                rgb=self.rgb[: self.size],
                bev=self.bev[: self.size],
                proprio=self.proprio[: self.size],
                action=self.action[: self.size],
                reward=self.reward[: self.size],
                done=self.done[: self.size],
                is_first=self.is_first[: self.size],
                is_intervention=self.is_intervention[: self.size],
                has_next=self.has_next[: self.size],
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        rgb_shape: tuple = (3, 84, 84),
        bev_shape: tuple = (2, 128, 128),
        proprio_dim: int = 6,
        action_dim: int = 2,
        n_reward_channels: int = 5,
    ) -> Optional["DemoReplay"]:
        path = Path(path)
        if not path.exists():
            return None
        data = np.load(path)
        n = int(data['size'])
        if n == 0:
            return None
        obj = cls(
            rgb_shape=rgb_shape,
            bev_shape=bev_shape,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            n_reward_channels=n_reward_channels,
            initial_capacity=max(n, 16384),
        )
        obj.size = n
        obj.rgb[:n] = data['rgb']
        obj.bev[:n] = data['bev']
        obj.proprio[:n] = data['proprio']
        obj.action[:n] = data['action']
        obj.reward[:n] = data['reward']
        obj.done[:n] = data['done']
        obj.is_first[:n] = data['is_first']
        obj.is_intervention[:n] = data['is_intervention']
        obj.has_next[:n] = data['has_next']
        return obj


# ---------------------------------------------------------------------------
# v3 replay buffers — depth (uint8, 72×96) + 1D LiDAR (float32, 360) + proprio
# ---------------------------------------------------------------------------
#
# Storage layout mirrors the legacy buffers but swaps the visual modalities:
#   depth: (cap, 1, 72, 96) uint8  — 0..255, trainer divides by 255
#   lidar: (cap, 360)       float32 — already normalized to [0, 1]
# `_stack_window` is modality-agnostic so we reuse it directly.


class OnlineReplayV3:
    """FIFO ring buffer for v3 observations (depth + lidar + proprio)."""

    def __init__(
        self,
        capacity: int = 100_000,
        depth_shape: tuple = (1, 72, 96),
        lidar_beams: int = 360,
        proprio_dim: int = 6,
        action_dim: int = 2,
        n_reward_channels: int = 5,
    ):
        self.capacity = capacity
        self.depth_shape = depth_shape
        self.lidar_beams = lidar_beams
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.n_reward_channels = n_reward_channels

        self.depth = np.zeros((capacity, *depth_shape), dtype=np.uint8)
        self.lidar = np.zeros((capacity, lidar_beams), dtype=np.float32)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, n_reward_channels), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.is_first = np.zeros(capacity, dtype=bool)
        self.is_demo = np.zeros(capacity, dtype=bool)
        self.is_intervention = np.zeros(capacity, dtype=bool)
        self.has_next = np.zeros(capacity, dtype=bool)

        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_chunk(self, chunk: dict) -> int:
        T = len(chunk['rewards'])
        if T < 2:
            return 0
        n_add = T - 1
        depth = chunk['depth']
        lidar = chunk['lidar']
        proprio = chunk['proprio']
        action = chunk.get('action', chunk.get('actions'))
        reward = chunk.get('reward', chunk.get('rewards'))
        done = chunk.get('done', chunk.get('dones'))
        is_first = chunk['is_first']
        is_demo = chunk.get('is_demo', np.zeros(T, dtype=bool))
        is_intervention = chunk.get('is_intervention', np.zeros(T, dtype=bool))

        for t in range(n_add):
            i = self.idx
            self.depth[i] = depth[t]
            self.lidar[i] = lidar[t]
            self.proprio[i] = proprio[t]
            self.action[i] = action[t]
            self.reward[i] = reward[t]
            self.done[i] = bool(done[t])
            self.is_first[i] = bool(is_first[t])
            self.is_demo[i] = bool(is_demo[t])
            self.is_intervention[i] = bool(is_intervention[t])
            self.has_next[i] = (not bool(done[t])) and (not bool(is_first[t + 1]))
            nxt = (i + 1) % self.capacity
            if self.size > 0 or self.idx > 0:
                self.has_next[nxt] = False
            self.idx = (self.idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        # Trailing frame stash (for next-state lookup of the (T-2)-th transition).
        i = self.idx
        last = T - 1
        self.depth[i] = depth[last]
        self.lidar[i] = lidar[last]
        self.proprio[i] = proprio[last]
        self.action[i] = action[last]
        self.reward[i] = reward[last]
        self.done[i] = bool(done[last])
        self.is_first[i] = bool(is_first[last])
        self.is_demo[i] = bool(is_demo[last])
        self.is_intervention[i] = bool(is_intervention[last])
        self.has_next[i] = False
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return n_add

    def sample(self, batch: int, frame_stack: int, rng: np.random.Generator) -> dict:
        if self.size == 0:
            raise RuntimeError("OnlineReplayV3 is empty")
        valid_idx = (np.flatnonzero(self.has_next[: self.size])
                     if self.size < self.capacity else np.flatnonzero(self.has_next))
        if len(valid_idx) == 0:
            raise RuntimeError("No valid (has_next) transitions available")
        chosen = valid_idx[rng.integers(0, len(valid_idx), size=batch)]

        depth_stack = np.empty((batch, frame_stack, *self.depth_shape[1:]), dtype=np.uint8)
        lidar_stack = np.empty((batch, frame_stack, self.lidar_beams), dtype=np.float32)
        proprio_stack = np.empty((batch, self.proprio_dim * frame_stack), dtype=np.float32)
        next_depth_stack = np.empty_like(depth_stack)
        next_lidar_stack = np.empty_like(lidar_stack)
        next_proprio_stack = np.empty_like(proprio_stack)
        actions = np.empty((batch, self.action_dim), dtype=np.float32)
        rewards = np.empty((batch, self.n_reward_channels), dtype=np.float32)
        dones = np.empty(batch, dtype=bool)
        is_demo = np.empty(batch, dtype=bool)
        is_intervention = np.empty(batch, dtype=bool)

        for b, i in enumerate(chosen):
            i = int(i)
            nxt = (i + 1) % self.capacity
            # depth: stored as (1, H, W); stacking on axis 0 yields (K, H, W).
            depth_stack[b] = _stack_window(self.depth, i, frame_stack, self.is_first)
            next_depth_stack[b] = _stack_window(self.depth, nxt, frame_stack, self.is_first)
            # lidar: stored as (360,); reshape to (1, 360) so stacking yields (K, 360).
            lidar_stack[b] = _stack_window(
                self.lidar.reshape(self.capacity, 1, self.lidar_beams),
                i, frame_stack, self.is_first,
            )
            next_lidar_stack[b] = _stack_window(
                self.lidar.reshape(self.capacity, 1, self.lidar_beams),
                nxt, frame_stack, self.is_first,
            )
            proprio_stack[b] = _stack_window(
                self.proprio.reshape(self.capacity, self.proprio_dim, 1),
                i, frame_stack, self.is_first,
            ).reshape(-1)
            next_proprio_stack[b] = _stack_window(
                self.proprio.reshape(self.capacity, self.proprio_dim, 1),
                nxt, frame_stack, self.is_first,
            ).reshape(-1)
            actions[b] = self.action[i]
            rewards[b] = self.reward[i]
            dones[b] = self.done[i]
            is_demo[b] = self.is_demo[i]
            is_intervention[b] = self.is_intervention[i]

        return {
            'depth_stack': depth_stack,
            'lidar_stack': lidar_stack,
            'proprio_stack': proprio_stack,
            'action': actions,
            'reward': rewards,
            'done': dones,
            'is_demo': is_demo,
            'is_intervention': is_intervention,
            'next_depth_stack': next_depth_stack,
            'next_lidar_stack': next_lidar_stack,
            'next_proprio_stack': next_proprio_stack,
        }


class DemoReplayV3:
    """Append-only persistent demo store for v3 observations."""

    SCHEMA_VERSION = 3

    def __init__(
        self,
        depth_shape: tuple = (1, 72, 96),
        lidar_beams: int = 360,
        proprio_dim: int = 6,
        action_dim: int = 2,
        n_reward_channels: int = 5,
        initial_capacity: int = 16384,
    ):
        self.depth_shape = depth_shape
        self.lidar_beams = lidar_beams
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.n_reward_channels = n_reward_channels

        self.capacity = initial_capacity
        self.size = 0
        self._alloc()

    def _alloc(self):
        self.depth = np.zeros((self.capacity, *self.depth_shape), dtype=np.uint8)
        self.lidar = np.zeros((self.capacity, self.lidar_beams), dtype=np.float32)
        self.proprio = np.zeros((self.capacity, self.proprio_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, self.n_reward_channels), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=bool)
        self.is_first = np.zeros(self.capacity, dtype=bool)
        self.is_intervention = np.zeros(self.capacity, dtype=bool)
        self.has_next = np.zeros(self.capacity, dtype=bool)

    def _ensure_capacity(self, need: int):
        if self.size + need <= self.capacity:
            return
        new_cap = self.capacity
        while self.size + need > new_cap:
            new_cap *= 2

        def _grow(arr: np.ndarray) -> np.ndarray:
            new_arr = np.zeros((new_cap, *arr.shape[1:]), dtype=arr.dtype)
            new_arr[: self.size] = arr[: self.size]
            return new_arr

        self.depth = _grow(self.depth)
        self.lidar = _grow(self.lidar)
        self.proprio = _grow(self.proprio)
        self.action = _grow(self.action)
        self.reward = _grow(self.reward)
        self.done = _grow(self.done)
        self.is_first = _grow(self.is_first)
        self.is_intervention = _grow(self.is_intervention)
        self.has_next = _grow(self.has_next)
        self.capacity = new_cap

    def __len__(self) -> int:
        return self.size

    def add_chunk(self, chunk: dict) -> int:
        T = len(chunk['rewards'])
        if T < 2:
            return 0
        self._ensure_capacity(T)
        depth = chunk['depth']
        lidar = chunk['lidar']
        proprio = chunk['proprio']
        action = chunk.get('action', chunk.get('actions'))
        reward = chunk.get('reward', chunk.get('rewards'))
        done = chunk.get('done', chunk.get('dones'))
        is_first = chunk['is_first']
        is_intervention = chunk.get('is_intervention', np.zeros(T, dtype=bool))

        n_add = T - 1
        start = self.size
        for t in range(n_add):
            i = start + t
            self.depth[i] = depth[t]
            self.lidar[i] = lidar[t]
            self.proprio[i] = proprio[t]
            self.action[i] = action[t]
            self.reward[i] = reward[t]
            self.done[i] = bool(done[t])
            self.is_first[i] = bool(is_first[t])
            self.is_intervention[i] = bool(is_intervention[t])
            self.has_next[i] = (not bool(done[t])) and (not bool(is_first[t + 1]))
        i = start + n_add
        last = T - 1
        self.depth[i] = depth[last]
        self.lidar[i] = lidar[last]
        self.proprio[i] = proprio[last]
        self.action[i] = action[last]
        self.reward[i] = reward[last]
        self.done[i] = bool(done[last])
        self.is_first[i] = bool(is_first[last])
        self.is_intervention[i] = bool(is_intervention[last])
        self.has_next[i] = False
        self.size = start + T
        return n_add

    def sample(self, batch: int, frame_stack: int, rng: np.random.Generator) -> dict:
        if self.size == 0:
            raise RuntimeError("DemoReplayV3 is empty")
        valid_idx = np.flatnonzero(self.has_next[: self.size])
        if len(valid_idx) == 0:
            raise RuntimeError("No valid (has_next) demo transitions")
        chosen = valid_idx[rng.integers(0, len(valid_idx), size=batch)]

        cap = self.size
        is_first_eff = self.is_first[: cap].copy()
        is_first_eff[0] = True

        def _stack_linear(buf: np.ndarray, idx: int, k: int) -> np.ndarray:
            frames = [None] * k
            frames[-1] = buf[idx]
            blocked = False
            for offset in range(1, k):
                if blocked or idx - offset < 0:
                    frames[k - 1 - offset] = np.zeros_like(buf[idx])
                    blocked = True
                    continue
                prev = idx - offset
                if is_first_eff[prev]:
                    blocked = True
                    frames[k - 1 - offset] = np.zeros_like(buf[idx])
                else:
                    frames[k - 1 - offset] = buf[prev]
            return np.concatenate(frames, axis=0)

        depth_view = self.depth[: cap]
        lidar_view = self.lidar[: cap].reshape(cap, 1, self.lidar_beams)
        proprio_view = self.proprio[: cap].reshape(cap, self.proprio_dim, 1)

        depth_stack = np.empty((batch, frame_stack, *self.depth_shape[1:]), dtype=np.uint8)
        lidar_stack = np.empty((batch, frame_stack, self.lidar_beams), dtype=np.float32)
        proprio_stack = np.empty((batch, self.proprio_dim * frame_stack), dtype=np.float32)
        next_depth_stack = np.empty_like(depth_stack)
        next_lidar_stack = np.empty_like(lidar_stack)
        next_proprio_stack = np.empty_like(proprio_stack)
        actions = np.empty((batch, self.action_dim), dtype=np.float32)
        rewards = np.empty((batch, self.n_reward_channels), dtype=np.float32)
        dones = np.empty(batch, dtype=bool)
        is_demo = np.ones(batch, dtype=bool)
        is_intervention = np.empty(batch, dtype=bool)

        for b, i in enumerate(chosen):
            i = int(i)
            nxt = i + 1
            depth_stack[b] = _stack_linear(depth_view, i, frame_stack)
            next_depth_stack[b] = _stack_linear(depth_view, nxt, frame_stack)
            lidar_stack[b] = _stack_linear(lidar_view, i, frame_stack)
            next_lidar_stack[b] = _stack_linear(lidar_view, nxt, frame_stack)
            proprio_stack[b] = _stack_linear(proprio_view, i, frame_stack).reshape(-1)
            next_proprio_stack[b] = _stack_linear(proprio_view, nxt, frame_stack).reshape(-1)
            actions[b] = self.action[i]
            rewards[b] = self.reward[i]
            dones[b] = self.done[i]
            is_intervention[b] = self.is_intervention[i]

        return {
            'depth_stack': depth_stack,
            'lidar_stack': lidar_stack,
            'proprio_stack': proprio_stack,
            'action': actions,
            'reward': rewards,
            'done': dones,
            'is_demo': is_demo,
            'is_intervention': is_intervention,
            'next_depth_stack': next_depth_stack,
            'next_lidar_stack': next_lidar_stack,
            'next_proprio_stack': next_proprio_stack,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'wb') as f:
            np.savez(
                f,
                schema_version=np.int32(self.SCHEMA_VERSION),
                size=self.size,
                depth=self.depth[: self.size],
                lidar=self.lidar[: self.size],
                proprio=self.proprio[: self.size],
                action=self.action[: self.size],
                reward=self.reward[: self.size],
                done=self.done[: self.size],
                is_first=self.is_first[: self.size],
                is_intervention=self.is_intervention[: self.size],
                has_next=self.has_next[: self.size],
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        depth_shape: tuple = (1, 72, 96),
        lidar_beams: int = 360,
        proprio_dim: int = 6,
        action_dim: int = 2,
        n_reward_channels: int = 5,
    ) -> Optional["DemoReplayV3"]:
        """Load v3 demos. Returns None if the file is missing, empty, or a
        legacy (schema != 3) demos.npz — the caller wipes/replaces in that case.
        """
        path = Path(path)
        if not path.exists():
            return None
        data = np.load(path)
        # Reject legacy demos.npz silently — schema_version key is absent in v1/v2.
        sv = int(data['schema_version']) if 'schema_version' in data.files else 0
        if sv != cls.SCHEMA_VERSION:
            return None
        n = int(data['size'])
        if n == 0:
            return None
        obj = cls(
            depth_shape=depth_shape,
            lidar_beams=lidar_beams,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            n_reward_channels=n_reward_channels,
            initial_capacity=max(n, 16384),
        )
        obj.size = n
        obj.depth[:n] = data['depth']
        obj.lidar[:n] = data['lidar']
        obj.proprio[:n] = data['proprio']
        obj.action[:n] = data['action']
        obj.reward[:n] = data['reward']
        obj.done[:n] = data['done']
        obj.is_first[:n] = data['is_first']
        obj.is_intervention[:n] = data['is_intervention']
        obj.has_next[:n] = data['has_next']
        return obj
