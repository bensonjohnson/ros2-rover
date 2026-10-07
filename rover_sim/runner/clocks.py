"""Multi-rate clocks: the sensor/actuator timing the real rover actually has.

SIM_PLATFORM fidelity item 1 (docs/SIM_PLATFORM.md §8). The legacy sim runs
everything off ONE clock at ``CONTROL_HZ`` (15 Hz): a fresh lidar revolution,
a safety-gate update, a policy tick and a motor step, all at the same
instant. The rover does not work that way, and the difference is CAUSAL, not
cosmetic:

* **Lidar 10 Hz vs brain 15 Hz.** The LD19 publishes ~10.0 Hz (measured
  10.004 Hz on the rover, field bag 2026-09-10) while the brain ticks at
  15 Hz (``pc_active_inference_runner`` ``control_rate_hz: 15.0``, and
  ``evo/deploy/evo_runner.py --rate 15.0``). The brain therefore acts on a
  scan that is 0-100 ms old (mean ~48 ms): roughly every third control tick
  re-uses the same revolution. The deploy runner says so in its own
  docstring ("the latest scan is reused between"); the sim was handing the
  policy a perfectly fresh scan on every single tick.
* **The safety monitor runs in the /scan callback.** Its streak counters and
  hold timers advance once per REVOLUTION, not once per control tick. Feeding
  it the same revolution twice makes a 2-scan block trip up to one control
  tick (67 ms) earlier than the hardware would.
* **The motor driver rate-limits commands** (``motor_command_rate_limit_secs``:
  0.04 s = 25 Hz today, 0.1 s = 10 Hz in the config that produced the
  historical clock-mismatch failure, REVIVAL_PLAN "why prior attempts
  failed" #2). At 15 Hz control / 25 Hz driver nothing is dropped *today* —
  the knob is modelled so the mismatch is expressible (and visible in a
  smoke's metrics) instead of being assumed away.

A policy trained on the single-clock sim learns a causally wrong map: sensing
is always fresh, every action lands, and the gate reacts within one control
tick. Under ES that is not a small offset either — the population is selected
for whatever exploits the exact timing it is given.

Everything here is device-side: static buffers, in-place updates, no host
sync, no per-tick allocation, so it is legal inside a CUDA-graph capture.
Phases come from a fixed low-discrepancy sequence over the env index
(golden ratio), NOT from RNG, so a capture replay and a re-run agree; the
stagger exists because on hardware the phase between the 10 Hz lidar and the
15 Hz brain is arbitrary and drifts, so no single phase may become the
trained convention.

Off (``clocks=None``, the default) the engine keeps the single-clock path
byte for byte — the whole module is inert until asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Low-discrepancy fraction sequence: phase_i = frac(i * golden). Deterministic
# across processes/devices, no RNG state, no host sync.
_GOLDEN = 0.6180339887498949


@dataclass(frozen=True)
class ClocksConfig:
    """Rate schedule for the engine's sensors and actuators.

    lidar_hz   — lidar revolution rate (LD19 ~10.0 Hz). None = one revolution
                 per control tick (the legacy single-clock sim).
    driver_hz  — motor driver command acceptance rate
                 (``motor_command_rate_limit_secs`` -> 25 Hz today, 10 Hz in
                 the historical config). Commands arriving faster than this
                 are dropped by the driver and the motors keep running the
                 last accepted one. None = every command lands.
    phase_stagger — spread the initial phases of these clocks uniformly over
                 the env batch (deterministic golden-ratio sequence) so the
                 policy cannot learn one lucky phase relationship. On
                 hardware the phase is arbitrary and drifts.
    """

    lidar_hz: float | None = None
    driver_hz: float | None = None
    phase_stagger: bool = True

    def __post_init__(self):
        for name in ("lidar_hz", "driver_hz"):
            hz = getattr(self, name)
            if hz is not None and not hz > 0:
                raise ValueError(f"{name} must be > 0 or None, got {hz!r}")

    @property
    def active(self) -> bool:
        """True when this config changes anything at all."""
        return self.lidar_hz is not None or self.driver_hz is not None

    def describe(self) -> str:
        return (f"lidar={'off' if self.lidar_hz is None else f'{self.lidar_hz:g}Hz'}"
                f" driver={'off' if self.driver_hz is None else f'{self.driver_hz:g}Hz'}"
                f" stagger={'on' if self.phase_stagger else 'off'}")


class MultiRateClock:
    """Per-env device-side rate schedule + the held sample buffers.

    Usage inside the (captured) rollout loop::

        mc.tick()                       # advance, recompute `due` flags
        fresh = env.scan()              # the sensor keeps spinning
        ranges = mc.hold_scan(fresh)    # ...the brain only gets new data on due
        gate.process_scan(ranges, ..., due=mc.lidar_due)
        mc.scan_taken()                 # schedule the next revolution
        ...
        applied = mc.latch_cmd(gated)   # the driver drops commands it can't take
        env.step(applied, dt)

    All buffers are allocated once, in ``__init__``; ``reset()`` only writes
    (copy_/zero_), so both are capture-legal.
    """

    def __init__(self, cfg: ClocksConfig, dt: float, B: int, n_beams: int,
                 device="cpu"):
        self.cfg = cfg
        self.dt = float(dt)
        self.B = int(B)
        self.device = torch.device(device)
        dev = self.device
        self.lidar_on = cfg.lidar_hz is not None
        self.driver_on = cfg.driver_hz is not None
        self._clk = torch.zeros(self.B, device=dev)
        self.lidar_period = 1.0 / float(cfg.lidar_hz) if self.lidar_on else None
        self.driver_period = (1.0 / float(cfg.driver_hz) if self.driver_on
                              else None)
        if self.lidar_on:
            self._lidar_phase0 = self._phases(self.lidar_period)
            self._lidar_next = self._lidar_phase0.clone()
            self.lidar_due = torch.zeros(self.B, dtype=torch.bool, device=dev)
            # Static held revolution: scan dtypes are fp32 on every path
            # (Fp16Env casts its fp16 raycast back to fp32), so one buffer
            # serves eager, fp16 and fused engines.
            self.held_scan = torch.zeros(self.B, int(n_beams),
                                         dtype=torch.float32, device=dev)
        if self.driver_on:
            self._drv_phase0 = self._phases(self.driver_period)
            self._drv_next = self._drv_phase0.clone()
            self.driver_due = torch.zeros(self.B, dtype=torch.bool, device=dev)
            self.held_cmd = torch.zeros(self.B, 2, dtype=torch.float32,
                                        device=dev)

    # ------------------------------------------------------------------ setup
    def _phases(self, period: float) -> torch.Tensor:
        """Initial 'next due' time per env, in [0, period)."""
        i = torch.arange(self.B, device=self.device, dtype=torch.float32)
        frac = (torch.remainder(i * _GOLDEN, 1.0) if self.cfg.phase_stagger
                else torch.zeros(self.B, device=self.device))
        return frac * float(period)

    def reset(self) -> None:
        """New game: clock to zero, phases back to their initial offsets.
        The caller re-seeds ``held_scan`` with a real revolution (the rover
        starts with whatever the lidar last published)."""
        self._clk.zero_()
        if self.lidar_on:
            self._lidar_next.copy_(self._lidar_phase0)
        if self.driver_on:
            self._drv_next.copy_(self._drv_phase0)
            self.held_cmd.zero_()

    # ------------------------------------------------------------------ rates
    def tick(self) -> None:
        """Advance the shared clock by one control tick and refresh `due`."""
        self._clk.add_(self.dt)
        if self.lidar_on:
            torch.ge(self._clk, self._lidar_next, out=self.lidar_due)
        if self.driver_on:
            torch.ge(self._clk, self._drv_next, out=self.driver_due)

    def hold_scan(self, fresh: torch.Tensor) -> torch.Tensor:
        """Give the brain `fresh` only where the lidar is due; hold the last
        revolution everywhere else. Returns the shared buffer (no alloc)."""
        torch.where(self.lidar_due[:, None], fresh, self.held_scan,
                    out=self.held_scan)
        return self.held_scan

    def scan_taken(self) -> None:
        """Schedule the next revolution for the envs that just delivered."""
        torch.where(self.lidar_due, self._lidar_next + self.lidar_period,
                    self._lidar_next, out=self._lidar_next)

    def latch_cmd(self, cmd: torch.Tensor) -> torch.Tensor:
        """The motor driver accepts `cmd` only where it is due; elsewhere the
        motors keep running the last accepted command. Returns the shared
        buffer (no alloc)."""
        torch.where(self.driver_due[:, None], cmd, self.held_cmd,
                    out=self.held_cmd)
        return self.held_cmd

    def cmd_taken(self) -> None:
        torch.where(self.driver_due, self._drv_next + self.driver_period,
                    self._drv_next, out=self._drv_next)

    # ------------------------------------------------------------------ probes
    def ages(self) -> torch.Tensor:
        """Current age (s) of each env's held revolution — forensics only
        (host-side read; never call inside a capture)."""
        return self._clk - (self._lidar_next - self.lidar_period)

    def due_counts(self) -> int:
        """How many envs are due right now (host-side read; forensics)."""
        n = 0
        if self.lidar_on:
            n += int(self.lidar_due.sum())
        return n
