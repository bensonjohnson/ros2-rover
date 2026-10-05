"""Compatibility shim — the genome moved to rover_sim.policies.es_genome.

Stage 3a of the SIM_PLATFORM migration (docs/SIM_PLATFORM.md §6): the ES
genome module (`PopulationNet`, the packing/meta helpers, the obs/action
mode tuples) now lives in `rover_sim/policies/es_genome.py`, content
verbatim. This module re-exports every name legacy consumers import —
including the private `_param_shapes`, used by evo.test_policy and
evo.bc_seed — so existing scripts keep working unchanged.
"""

from __future__ import annotations

from rover_sim.policies.es_genome import (  # noqa: F401
    ACTION_MODES, OBS_MODES, PopulationNet, _param_shapes, genome_meta,
    genome_size, per_gene_scale, read_meta, sample_population)
