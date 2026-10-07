"""Meta-search over ES configurations: the unattended experiment manager.

SIM_PLATFORM's payoff is that a *model idea* costs one adapter file; this
package is the other half — a *search* costs one config, and nobody has to sit
and babysit it.

What it is FOR, stated as the failure it prevents: runs 1-25 were hand-driven
hypothesis tests. Individually they were sound, but they were serial, and the
selection was done by a human reading a gens.jsonl window and choosing a run to
extend. Two things went wrong repeatedly and both are mechanical:

* **Selection on noise.** Seed spread on the cross-rate metric measured ~0.10
  while the effects under test were 0.03-0.08 (run-12 verdicts). A single-seed
  comparison is a coin flip dressed as a result.
* **Selection on the wrong thing.** `best_genome` is an argmax over a
  per-generation resampled draw and has repeatedly been a collision
  brute-forcer; door-crossing counts are gameable by ping-ponging one doorway
  (metric flaw #2). Only rooms/game survived every audit.

So this package automates the loop and hard-codes the discipline the audits
bought: score on the VAL suite across >=2 seeds, gate on the collision and
reverse-fraction constraints, promote by successive halving, and touch the
holdout suite exactly once, at the end, with `final=True`.

Everything is resumable: the ledger is append-only JSONL and every promotion
decision is recomputed from it, so a crash, an ssh drop or a reboot costs at
most the trials that were in flight.
"""

from .ledger import Ledger, config_hash, trial_key            # noqa: F401
from .space import KNOBS, FIXED, SPACE_VERSION, sample_configs  # noqa: F401

__all__ = ["Ledger", "config_hash", "trial_key", "KNOBS", "FIXED",
           "SPACE_VERSION", "sample_configs"]
