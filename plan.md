# PPO BEV Rover - Local Training Implementation Plan

## Overview
Create a local PPO training system for the rover using the Unified BEV architecture (LiDAR + Depth fusion), with checkpoint saving every 200 steps and RKNN export for NPU inference.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ROS2 Rover (Rock64/RK3588)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  start_ppo_bev_rover.sh                                 │   │
│  │  ┌─────────────────────┐  ┌─────────────────────────┐  │   │
│  │  │ Mode 1: TRAIN       │  │ Mode 2: INFERENCE     │  │   │
│  │  │ - PPO Training      │  │ - Load RKNN Model     │  │   │
│  │  │ - Save checkpoints  │  │ - NPU Inference       │  │   │
│  │  │ - Export ONNX       │  │ - Autonomous Driving  │  │   │
│  │  │ - Convert to RKNN   │  │                       │  │   │
│  │  └─────────────────────┘  └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ppo_bev_trainer.py (Local Training)                    │   │
│  │  - Unified BEV Encoder (2×128×128)                      │   │
│  │  - PPO Policy + Value Heads                             │   │
│  │  - Checkpoint every 200 steps                           │   │
│  │  - Export ONNX → RKNN pipeline                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  bev_inference.py (Inference Mode)                      │   │
│  │  - Load .rknn model                                     │   │
│  │  - NPU inference at 30Hz                                │   │
│  │  - Safety monitoring                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Files Created

| File | Location | Purpose |
|------|----------|---------|
| `model_architectures.py` | `remote_training_server/` | Added `UnifiedBEVPPOPolicy` class for PPO with BEV encoder |
| `ppo_bev_trainer.py` | `remote_training_server/` | Local PPO trainer with BEV support, checkpoint/RKNN export |
| `export_bev_to_rknn.py` | `remote_training_server/` | ONNX → RKNN conversion utility |
| `bev_inference.py` | `remote_training_server/` | NPU inference engine for autonomous driving |
| `start_ppo_bev_rover.sh` | `remote_training_server/` | Dual-mode startup script (TRAIN / INFERENCE) |

## Checkpoint Format

```
checkpoints_ppo/
├── ppo_step_200.pt          # PyTorch checkpoint
├── ppo_step_200.onnx        # ONNX export
├── ppo_step_200.rknn       # RKNN for NPU
├── ppo_step_400.pt
├── ppo_step_400.onnx
├── ppo_step_400.rknn
...
```

## Usage

### Training Mode
```bash
cd remote_training_server
./start_ppo_bev_rover.sh
# Select option 1: TRAIN
```

Or directly:
```bash
python3 ppo_bev_trainer.py \
  --nats_server nats://nats.gokickrocks.org:4222 \
  --checkpoint_dir ./checkpoints_ppo \
  --log_dir ./logs_ppo \
  --checkpoint_interval 200
```

### Inference Mode
```bash
cd remote_training_server
./start_ppo_bev_rover.sh
# Select option 2: INFERENCE
```

Or directly:
```bash
python3 bev_inference.py \
  --rknn checkpoints_ppo/latest_actor.rknn \
  --max_speed 0.18
```

### Manual RKNN Export
```bash
python3 export_bev_to_rknn.py \
  --checkpoint checkpoints_ppo/ppo_step_200.pt \
  --output_dir checkpoints_ppo \
  --target_platform rk3588
```

## Implementation Steps (COMPLETED)

- [x] Analyze SAC training architecture (v620_sac_trainer.py)
- [x] Analyze PPO training architecture (v620_ppo_trainer.py)
- [x] Review BEV model architectures (model_architectures.py)
- [x] Assess feasibility of SAC BEV → PPO adaptation
- [x] Create implementation plan for UnifiedBEV-PPO
- [x] Write plan.md
- [x] Add UnifiedBEVPPOPolicy to model_architectures.py
- [x] Create ppo_bev_trainer.py with checkpoint/RKNN export
- [x] Create export_bev_to_rknn.py for ONNX→RKNN conversion
- [x] Create bev_inference.py for NPU inference
- [x] Create start_ppo_bev_rover.sh startup script

## Key Features

1. **Unified BEV Architecture**: Uses the same 2-channel 128×128 BEV grid from SAC training (LiDAR + Depth fusion)

2. **PPO Algorithm**: On-policy training with GAE, suitable for local rover training

3. **Automatic Checkpointing**: Saves every 200 steps with automatic ONNX and RKNN export

4. **Dual-Mode Operation**: 
   - TRAIN: Collects experience via NATS, trains PPO, exports models
   - INFERENCE: Loads RKNN model for NPU-accelerated autonomous driving

5. **Stateless Policy**: No LSTM required for PPO, making it simpler and more efficient for ONNX/RKNN export

## Next Steps (Optional)

- [ ] Integrate with rover's ROS2 nodes for BEV generation
- [ ] Add reward shaping functions specific to rover navigation
- [ ] Create calibration dataset for better RKNN quantization
- [ ] Add safety monitoring and emergency stop integration