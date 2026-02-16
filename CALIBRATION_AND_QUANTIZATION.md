# Calibration Data & INT8 Quantization Guide

## Overview

For optimal performance, RKNN models can be quantized to INT8, reducing model size by ~4x and increasing inference speed. This requires **calibration data** - representative samples from your rover's sensors.

## Architecture: Unified BEV

The current SAC rover uses a **Unified BEV (Bird's Eye View)** architecture:
- **BEV Input**: 2-channel 128×128 grid (LiDAR + Depth occupancy)
- **Proprioception**: 6-dimensional vector

### Proprioception Normalization

The proprioception vector is normalized for INT8 quantization:

```python
proprio = [lidar_min, prev_lin, prev_ang, cur_lin, cur_ang, gap_heading]
           # Raw ranges:
           # lidar_min: 0-4m
           # prev_lin:  -1 to 1 (track command)
           # prev_ang:  -1 to 1 (track command)
           # cur_lin:   ~-0.2 to 0.2 m/s (velocity)
           # cur_ang:   ~-1 to 1 rad/s (angular velocity)
           # gap_heading: -1 to 1 (normalized heading)

# Normalization (applied in sac_episode_runner.py and convert_onnx_to_rknn.py):
PROPRIO_MEAN = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PROPRIO_STD  = [2.0, 1.0, 1.0, 0.2, 1.0, 1.0]

normalized_proprio = (proprio - PROPRIO_MEAN) / PROPRIO_STD
# Result: All dimensions approximately in [-3, 3] range
```

**IMPORTANT**: The same normalization MUST be applied during:
1. Calibration data collection (in `sac_episode_runner.py`)
2. RKNN inference (in `sac_episode_runner.py`)
3. ONNX to RKNN conversion (in `convert_onnx_to_rknn.py`)

## Workflow

```
1. Data Collection  →  2. Model Training  →  3. Quantized Conversion
   (Save calibration)     (Train on V620)       (Use calibration on rover)
```

## Step 1: Collect Calibration Data

During SAC operation, the rover automatically saves calibration samples.

### Automatic Collection

```bash
# On rover: Start SAC autonomous training
./start_sac_rover.sh nats://nats.gokickrocks.org:4222
```

The system will automatically:
- Save ~100 calibration samples (10% of steps, randomly sampled)
- Store in `calibration_data/` directory
- Include BEV grid and proprioception data

### Calibration Data Format

Each `.npz` file contains:
- `bev`: (2, 128, 128) float32 - Unified BEV grid
  - Channel 0: LiDAR occupancy (0=free, 1=occupied)
  - Channel 1: Depth occupancy (0=free, 1=occupied)
- `proprio`: (6,) float32 - Proprioception vector (unnormalized raw values)

### Verify Calibration Data

```bash
# Check samples
ls -lh calibration_data/
# Should show: calib_XXXX.npz files

# Count samples
find calibration_data -name "*.npz" | wc -l

# Inspect a sample
python3 -c "
import numpy as np
data = np.load('calibration_data/calib_1234567890.npz')
print(f'BEV shape: {data[\"bev\"].shape}')  # Should be (2, 128, 128)
print(f'BEV range: [{data[\"bev\"].min():.3f}, {data[\"bev\"].max():.3f}]')  # Should be [0, 1]
print(f'Proprio shape: {data[\"proprio\"].shape}')  # Should be (6,)
print(f'Proprio: {data[\"proprio\"]}')
"
```

## Step 2: Train Model on V620

```bash
# On V620
cd ~/remote_training_server
./start_sac_server.sh
```

Training proceeds normally. The V620 exports ONNX models with the unified BEV architecture.

## Step 3: Convert with Quantization

### Automatic (Recommended)

```bash
# On rover (after calibration data is collected)
./convert_onnx_to_rknn.sh models/sac_actor_v50.onnx calibration_data
```

This will:
1. Load ONNX model (Unified BEV architecture)
2. Load calibration samples from `calibration_data/`
3. Normalize proprioception for quantization
4. Convert ONNX → RKNN with INT8 quantization
5. Test inference and validate output
6. Output quantized `.rknn` file

### Manual

```bash
# On rover
python3 src/tractor_bringup/tractor_bringup/convert_onnx_to_rknn.py \
  models/sac_actor_v50.onnx \
  --output models/sac_actor_v50_int8.rknn \
  --quantize \
  --calibration-dir calibration_data
```

### Without Quantization (Float16)

If you don't have calibration data:

```bash
# On rover
./convert_onnx_to_rknn.sh models/sac_actor_v50.onnx
```

This creates a float16 model (larger, slightly slower, but no calibration needed).

## Performance Comparison

| Model Type | Size   | Inference Time | Accuracy |
|------------|--------|----------------|----------|
| Float16    | ~2 MB  | 50-80ms        | 100%     |
| INT8       | ~600KB | 30-50ms        | ~98-99%  |

**INT8 is recommended** for production use.

## Quantization Accuracy Validation

After conversion, you can verify quantization accuracy:

```bash
# Run accuracy test
python3 -c "
import numpy as np
from rknnlite.api import RKNNLite

# Load RKNN model
rknn = RKNNLite()
rknn.load_rknn('models/sac_actor_v50.rknn')
rknn.init_runtime()

# Create test inputs
np.random.seed(42)
test_bev = np.random.rand(1, 2, 128, 128).astype(np.float32)
test_proprio = np.random.rand(1, 6).astype(np.float32)

# Run multiple inferences
outputs = []
for _ in range(10):
    out = rknn.inference(inputs=[test_bev, test_proprio])
    outputs.append(out[0])

# Check for NaN/Inf
if any(np.isnan(o).any() or np.isinf(o).any() for o in outputs):
    print('❌ Quantization error: NaN/Inf in outputs')
else:
    print('✅ Quantization OK: No NaN/Inf')
    
# Check output consistency (should be deterministic)
std = np.std(outputs, axis=0)
if np.max(std) < 0.01:
    print('✅ Output is deterministic')
else:
    print(f'⚠️ Output variance: max_std={np.max(std):.4f}')

print(f'Output range: [{np.min(outputs):.4f}, {np.max(outputs):.4f}]')
print(f'Output mean: {np.mean(outputs):.4f}')

rknn.release()
"
```

## Calibration Dataset Quality

### Good Calibration Data

✅ Diverse environments (indoor, outdoor, different lighting)
✅ Various obstacles and terrain types
✅ Full range of rover speeds and maneuvers
✅ Representative of deployment conditions
✅ At least 100 samples

### Poor Calibration Data

❌ All from same environment
❌ Limited lighting conditions
❌ Only straight-line driving
❌ Not representative of real use
❌ Fewer than 50 samples

### Tips

1. **Collect during actual operation**: Let the rover drive around naturally
2. **Multiple sessions**: Collect calibration data from different runs
3. **100+ samples**: More samples = better quantization accuracy
4. **Update periodically**: Recollect if deployment environment changes

## Troubleshooting

### "No calibration samples found"

```bash
# Check directory exists
ls -la calibration_data/

# Run SAC rover to collect data
./start_sac_rover.sh
# Let it run for a few minutes

# Verify samples were created
ls -la calibration_data/*.npz | wc -l
```

### "Quantization produces NaN/Inf outputs"

This usually indicates:
1. **Out-of-range proprioception values**: Check that `lidar_min` is in [0, 4] range
2. **BEV values outside [0, 1]**: Check occupancy grid processing
3. **Insufficient calibration data**: Collect more samples

```bash
# Debug: Check calibration data ranges
python3 -c "
import numpy as np
import os

for f in sorted(os.listdir('calibration_data'))[:10]:
    if f.endswith('.npz'):
        data = np.load(f'calibration_data/{f}')
        bev = data['bev']
        proprio = data['proprio']
        print(f'{f}:')
        print(f'  BEV: [{bev.min():.3f}, {bev.max():.3f}]')
        print(f'  Proprio: {proprio}')
        print(f'  lidar_min: {proprio[0]:.3f} (should be 0-4)')
"
```

### "Model output is all zeros or constant"

This indicates the quantization range is incorrect. Check:
1. Proprioception normalization is applied correctly
2. Calibration data covers the full input range
3. BEV grid is in [0, 1] range

## File Structure

```
ros2-rover/
├── calibration_data/              # Calibration samples
│   ├── calib_1234567890.npz
│   ├── calib_1234567891.npz
│   ├── rknn_dataset/              # Generated by conversion script
│   │   ├── bev_0.npy
│   │   ├── proprio_0.npy
│   │   └── dataset.txt
│   └── ...
├── models/
│   ├── sac_actor_v50.onnx         # From V620
│   ├── sac_actor_v50.rknn         # Quantized (INT8)
│   └── remote_trained.rknn        # Symlink to latest
└── convert_onnx_to_rknn.sh        # Conversion script
```

## Summary

1. **Run SAC rover** to collect calibration data (automatic, ~100 samples)
2. **Convert with `--quantize`** flag and calibration directory
3. **Verify quantization** with test inference
4. **Deploy** with `./start_sac_rover.sh`

The normalization constants in `sac_episode_runner.py` and `convert_onnx_to_rknn.py` must match for consistent quantization.