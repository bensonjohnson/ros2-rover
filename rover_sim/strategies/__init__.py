"""Strategy plug-ins for the rover ES platform (SIM_PLATFORM stage 3b).

The evolutionary strategy lives here, off the CLI: `reproduce` (truncation
GA + immigrants, from the legacy `evo.evolve.reproduce`) and `OES`
(OpenAI-ES, mirrored + rank-shaped + Adam, from `evo.evolve.OES`), both
moved byte-for-byte. `STRATEGIES` is the name -> strategy catalog the
runner will read as more strategies are added. `evo.evolve` imports these
names back, so legacy `from evo.evolve import reproduce` still resolves.
"""

from __future__ import annotations

from .ga import reproduce
from .oes import OES
from .registry import STRATEGIES

__all__ = ["reproduce", "OES", "STRATEGIES"]
