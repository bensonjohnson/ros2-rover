"""Append-only trial ledger: the thing that makes the search unattended.

One JSONL record per (config, seed, attempt). Nothing is ever mutated or
deleted: state transitions are new records, and the current state of a trial is
"the last record for that key wins". That buys three properties that matter for
a job meant to run for days behind an ssh session:

* **Crash-safe.** A record is one line appended with a single write; a torn
  final line is skipped on load. No database to corrupt, no partial writes.
* **Resumable.** `completed()` is derived from the records, so re-running the
  manager after a reboot skips every finished trial instead of paying for it
  again. (Cost of getting this wrong: a full stage of 40-min runs.)
* **Auditable.** Every promotion decision can be recomputed from the file
  alone, and every trial carries its FULL config — so a champion can be
  rebuilt exactly, months later, without reading anyone's memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict

RECORD_FIELDS = ("trial", "config_hash", "stage", "seed", "status", "config",
                 "metrics", "score", "feasible", "reasons", "out_dir",
                 "gens", "detail", "t_start", "t_end", "elapsed_s")

#: Keys that are NOT part of a config's identity. `gens` is the stage BUDGET
#: (successive halving changes it between stages); hashing it would make the
#: same config hash differently per stage, and since promotion matches trials
#: by hash, a stage would find 0 survivors despite having run its trials
#: (observed: "stage dry done -> 0 survivors" with 2 done records).
BUDGET_KEYS = ("gens",)


def config_hash(cfg: dict) -> str:
    blob = json.dumps({k: v for k, v in cfg.items() if k not in BUDGET_KEYS},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def trial_key(cfg: dict, seed: int) -> str:
    return f"{config_hash(cfg)}-s{int(seed)}"


class Ledger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ------------------------------------------------------------- writing
    def append(self, **rec) -> dict:
        rec = {k: rec[k] for k in RECORD_FIELDS if k in rec}
        rec.setdefault("t_end", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
        return rec

    # ------------------------------------------------------------- reading
    def load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # torn write from a hard kill: skip it
        return out

    def state(self) -> "OrderedDict[str, dict]":
        """trial -> most recent record."""
        st: OrderedDict[str, dict] = OrderedDict()
        for rec in self.load():
            key = rec.get("trial")
            if key:
                st[key] = rec
        return st

    def completed(self, cfg: dict, seed: int) -> bool:
        rec = self.state().get(trial_key(cfg, seed))
        return bool(rec and rec.get("status") == "done")

    def done_trials(self) -> dict:
        """trial -> record, for every DONE trial (what scoring reads)."""
        return {k: v for k, v in self.state().items()
                if v.get("status") == "done"}

    def configs(self) -> dict:
        """config_hash -> config, from any record that carried one."""
        out = {}
        for rec in self.load():
            if rec.get("config_hash") and rec.get("config"):
                out[rec["config_hash"]] = rec["config"]
        return out
