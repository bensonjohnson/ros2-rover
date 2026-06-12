"""Batched trainer: ONE shared brain learning from B parallel worlds.

The reference modules in tractor_bringup.active_inference are single-stream
(one env, one tiny update per tick) — too small for a GPU. Here the same
math is reimplemented with a batch dimension: per-env recurrent state and
behavior, shared weights, local PC updates averaged across the batch each
tick. Same stationary point, B× the experience per weight update, and
tensors big enough that a GPU finally pays for its kernel launches.

batched/verify.py proves equivalence against the reference implementation;
checkpoints keep the reference state_dict schema, so the trained brain
loads on the rover unchanged.
"""
