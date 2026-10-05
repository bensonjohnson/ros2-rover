"""Strategy registry (SIM_PLATFORM stage 3b).

The plug-in catalog: a name -> strategy mapping built from the strategy
modules themselves. Kept deliberately simple — `ga` exposes the
`reproduce` operator, `oes` exposes the `OES` class. Future strategies
(sep-CMA-ES, PGPE, MAP-Elites archives) are one file plus an entry here.
"""

from __future__ import annotations

from .ga import reproduce
from .oes import OES

STRATEGIES = {
    "ga": {"reproduce": reproduce},
    "oes": {"class": OES},
}
