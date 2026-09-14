"""Greenfield evolutionary rover controller.

A from-scratch attempt (2026-09-14): no PC world model, no EFE actor, no
local learning — a population of small recurrent MLP policies is evolved
directly against the batched room-exploration sim (pnn_sim.batched.env),
scored on GROUND-TRUTH rooms visited in unseen procedural houses.

The sim (physics, lidar raycast, safety gate) is reused verbatim; the
controller and its learning algorithm are new.

    python3 -m evo.evolve --pop 64 --games 32 --device cuda
    python3 -m evo.baselines --device cuda
"""
