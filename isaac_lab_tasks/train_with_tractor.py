"""Wrapper that registers the tractor task, then defers to Isaac Lab's
skrl train.py.

Isaac Lab's train.py doesn't import our out-of-tree package, so the
gym.register call in tractor_nav/__init__.py never runs. This wrapper
imports the package first, then execs the upstream script with argv
forwarded.
"""

import os
import runpy
import sys

# Make `tractor_nav` importable, then register the task.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tractor_nav  # noqa: F401  (side-effect: gym.register)

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
TRAIN_SCRIPT = os.path.join(
    ISAACLAB_PATH, "scripts", "reinforcement_learning", "skrl", "train.py"
)

# train.py uses `import cli_args` (local), so its dir must be on sys.path.
sys.path.insert(0, os.path.dirname(TRAIN_SCRIPT))

runpy.run_path(TRAIN_SCRIPT, run_name="__main__")
