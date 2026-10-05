"""rover_sim.adapters — platform Policy adapters around legacy controllers.

Stage 3a of the SIM_PLATFORM migration (docs/SIM_PLATFORM.md §6): a native
adapter that presents the ES genome (`rover_sim.policies.es_genome.
PopulationNet`) through the `rover_sim.runner.policy.Policy` protocol.
"""

from .es import ESGenomePolicy, GraphRunner

__all__ = ["ESGenomePolicy", "GraphRunner"]
