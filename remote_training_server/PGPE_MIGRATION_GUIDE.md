# PGPE Migration Guide

## What Changed

### ✅ Resolution Upgrade
- **Old:** 240×424 (102k pixels)
- **New:** 480×640 (307k pixels)
- **Impact:** 3× better perception, same 322k parameters

### ✅ Algorithm Upgrade
- **Old:** Custom MAP-Elites implementation (~1,500 lines)
- **New:** EvoTorch PGPE (~400 lines)
- **Reduction:** 75% less custom code!

## Quick Start

### 1. Start PGPE Server (V620)

```bash
cd /root/ros2-rover/remote_training_server
source /root/rocm-train/bin/activate

# Basic usage (20 population, 1000 evaluations)
./start_pgpe_server.sh

# Advanced usage
./start_pgpe_server.sh [port] [num_evals] [checkpoint_dir] [popsize] [center_lr] [stdev_lr]

# Example: 30 population, faster learning
./start_pgpe_server.sh 5556 2000 ./checkpoints 30 0.02 0.002
```

### 2. Start Rover (Unchanged)

```bash
cd /root/ros2-rover
./start_map_elites_rover.sh
```

## Advanced Features

### Checkpoint & Resume (Auto-Resume)

PGPE automatically saves checkpoints every 10 generations and **auto-resumes** from the latest checkpoint when you restart:

```bash
# Start training
./start_pgpe_server.sh

# Training runs... saves checkpoints every 10 generations
# (pgpe_gen_10.pt, pgpe_gen_20.pt, etc.)

# Stop training (Ctrl+C)

# Restart - automatically resumes from latest checkpoint!
./start_pgpe_server.sh
```

**What gets saved:**
- ✅ PGPE center (current best policy)
- ✅ PGPE sigma (exploration parameters)
- ✅ Adam optimizer state (momentum, variance)
- ✅ Best model found so far
- ✅ Generation count & evaluation count

### Manual Resume

Resume from a specific checkpoint:

```bash
python3 v620_pgpe_trainer.py --resume gen_100
```

### Start Fresh (Ignore Checkpoints)

Force a fresh start:

```bash
./start_pgpe_server.sh
# Then pass --fresh when prompted, or:

python3 v620_pgpe_trainer.py --fresh
```

### Warmstart from MAP-Elites Model

Continue training from your best MAP-Elites model:

```bash
cd /root/ros2-rover/remote_training_server
source /root/rocm-train/bin/activate

python3 v620_pgpe_trainer.py \
  --warmstart checkpoints/best_models/best_final.pt \
  --population-size 20 \
  --center-lr 0.005 \
  --stdev-lr 0.0005
```

This initializes PGPE's center distribution at your best MAP-Elites policy, then continues optimizing from there.

## Parameter Tuning

### Population Size (`--population-size`)
- **Smaller (10-15):** Faster iterations, noisier gradients
- **Default (20):** Good balance
- **Larger (30-50):** Better gradient estimates, slower

### Center Learning Rate (`--center-lr`)
- **Lower (0.001-0.005):** Conservative, stable
- **Default (0.01):** Moderate
- **Higher (0.02-0.05):** Aggressive, may diverge

### Stdev Learning Rate (`--stdev-lr`)
- **Lower (0.0001-0.0005):** Exploration changes slowly
- **Default (0.001):** Moderate adaptation
- **Higher (0.002-0.005):** Fast adaptation, may collapse

### Initial Stdev (`--stdev-init`)
- **Lower (0.01):** Small initial exploration
- **Default (0.02):** Moderate
- **Higher (0.05):** Large initial exploration

## Comparison: MAP-Elites vs PGPE

| Feature | MAP-Elites | PGPE |
|---------|------------|------|
| **Code complexity** | ~1,500 lines | ~400 lines |
| **Algorithm** | Custom evolutionary | EvoTorch (production) |
| **Population** | 10→25 adaptive | 20 fixed |
| **Selection** | Tournament (75-500) | Gradient-based |
| **Mutation** | Manual Gaussian | Learned distribution |
| **Learning** | Behavioral cloning | Natural gradients |
| **Convergence** | Slower | Faster |
| **Diversity** | Archive of behaviors | Single best policy |
| **Maintenance** | High | Low |
| **Checkpointing** | ✅ Full state | ✅ Full state + optimizer |
| **Auto-resume** | ✅ Yes | ✅ Yes |

## Expected Performance

PGPE should:
- ✅ Converge 2-3× faster (gradient-based vs random search)
- ✅ Find better final policies
- ✅ Use less memory (no tournament cache)
- ✅ Be easier to tune (2 learning rates vs many hyperparams)

## Monitoring Training

### Watch Logs
```bash
tail -f /root/ros2-rover/remote_training_server/logs/pgpe_*.log
```

### Check Checkpoints
```bash
# List all checkpoints (saved every 10 generations)
ls -lh checkpoints/pgpe_gen_*.pt

# Check latest checkpoint
ls -t checkpoints/pgpe_gen_*.pt | head -1

# Inspect checkpoint (requires Python)
python3 -c "
import torch
cp = torch.load('checkpoints/pgpe_gen_10.pt')
print(f\"Generation: {cp['generation']}\")
print(f\"Evaluations: {cp['evaluations']}\")
print(f\"Best fitness: {cp['best_fitness_ever']:.2f}\")
"
```

### Check Best Model
```bash
# Best model saved every 10 generations
ls -lh checkpoints/best_models/

# Test best model (convert to ONNX for rover)
# Uses the model from best_final.pt
```

### Compare to MAP-Elites Baseline
```bash
# Load old MAP-Elites checkpoint
cat checkpoints/evolution_final.json | jq '.population[0].fitness'

# Compare to PGPE (inspect latest checkpoint)
python3 -c "
import torch
import glob
latest = sorted(glob.glob('checkpoints/pgpe_gen_*.pt'))[-1]
cp = torch.load(latest)
print(f\"PGPE Best: {cp['best_fitness_ever']:.2f}\")
"
```

## Troubleshooting

### Training is too slow
- Decrease `--population-size` (e.g., 15)
- Increase `--center-lr` (e.g., 0.02)

### Training is unstable/diverging
- Decrease `--center-lr` (e.g., 0.005)
- Decrease `--stdev-lr` (e.g., 0.0005)
- Use warmstart from MAP-Elites model

### Stuck in local optimum
- Increase `--stdev-init` (e.g., 0.05)
- Increase `--stdev-lr` (e.g., 0.002)
- Start fresh without warmstart

## Migration Checklist

- [x] Resolution upgraded (240×424 → 480×640)
- [x] EvoTorch installed
- [x] PGPE trainer created
- [x] Startup script created
- [ ] Test resolution on rover
- [ ] Run first PGPE training session
- [ ] Compare results to MAP-Elites baseline

## Rollback to MAP-Elites

If you need to go back to the old system:

```bash
# Revert resolution changes
cd /root/ros2-rover/src/tractor_bringup
git diff launch/map_elites_autonomous.launch.py
git checkout launch/map_elites_autonomous.launch.py

cd /root/ros2-rover/src/tractor_bringup/tractor_bringup
git diff export_actor_to_onnx.py
git checkout export_actor_to_onnx.py

# Use old MAP-Elites server
cd /root/ros2-rover/remote_training_server
./start_map_elites_server.sh
```

## Files Modified

**Modified:**
- `src/tractor_bringup/launch/map_elites_autonomous.launch.py` (resolution)
- `src/tractor_bringup/tractor_bringup/export_actor_to_onnx.py` (resolution)

**Created:**
- `remote_training_server/v620_pgpe_trainer.py` (new trainer)
- `remote_training_server/start_pgpe_server.sh` (startup script)
- `remote_training_server/PGPE_MIGRATION_GUIDE.md` (this file)

## Support

For issues or questions:
1. Check logs in `remote_training_server/logs/`
2. Verify GPU is being used (should see CUDA messages)
3. Compare training curves between MAP-Elites and PGPE
4. Try warmstart if PGPE struggles with cold start
