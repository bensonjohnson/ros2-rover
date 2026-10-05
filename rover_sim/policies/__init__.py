"""rover_sim.policies — evolved policy genomes as platform building blocks.

Stage 3a of the SIM_PLATFORM migration (docs/SIM_PLATFORM.md §6): the ES
genome (recurrent MLP + optional explicit memory, flat float32) moves out of
`evo.policy` into `rover_sim/policies/es_genome.py`, content verbatim.
`evo.policy` is now a re-export shim so legacy importers keep working
unchanged. Explicit re-exports below are the public surface; `_param_shapes`
(private) is still importable directly from `rover_sim.policies.es_genome`.
"""

from .es_genome import (
    ACTION_MODES,
    OBS_MODES,
    PopulationNet,
    genome_meta,
    genome_size,
    per_gene_scale,
    read_meta,
    sample_population,
)

__all__ = [
    "PopulationNet", "genome_meta", "genome_size", "per_gene_scale",
    "read_meta", "sample_population", "OBS_MODES", "ACTION_MODES",
]
