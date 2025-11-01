# Calibration Data & INT8 Quantization Guide

## Overview

For optimal performance, RKNN models can be quantized to INT8, reducing model size by ~4x and increasing inference speed. This requires **calibration data** - representative samples from your rover's sensors.

## Workflow

```
1. Data Collection  →  2. Model Training  →  3. Quantized Conversion
   (Save calibration)     (Train on V620)       (Use calibration on rover)
```

## Step 1: Collect Calibration Data

During data collection, the rover automatically saves calibration samples.

### Automatic Collection

```bash
# On rover: Start data collection
./start_remote_training_collection.sh tcp://V620_IP:5555
```

The system will automatically:
- Save 100 calibration samples (one every 10 seconds)
- Store in `calibration_data/` directory
- Include RGB, depth, and proprioception data

### Configuration

Edit `src/tractor_bringup/launch/remote_training_collection.launch.py`:

```python
'save_calibration_samples': True,       # Enable/disable
'calibration_sample_interval': 10.0,    # Seconds between samples
'calibration_sample_count': 100,        # Total samples to save
'calibration_data_dir': 'calibration_data'  # Output directory
```

### Verify Calibration Data

```bash
# Check samples
ls -lh calibration_data/
# Should show: calibration_0000.npz ... calibration_0099.npz

# Count samples
find calibration_data -name "*.npz" | wc -l
```

Each `.npz` file contains:
- `rgb`: (240, 424, 3) uint8 - RGB image
- `depth`: (240, 424) float32 - Depth map
- `proprio`: (6,) float32 - Proprioception [lin_vel, ang_vel, roll, pitch, accel, min_dist]

## Step 2: Train Model on V620

```bash
# On V620
cd ~/remote_training_server
./start_v620_server.sh
```

Training proceeds normally. The V620 exports ONNX models.

## Step 3: Convert with Quantization

### Automatic (Recommended)

```bash
# On rover (after calibration data is collected)
./convert_onnx_to_rknn.sh models/ppo_v620_update_50.onnx calibration_data
```

This will:
1. Check for calibration data in `calibration_data/`
2. Load calibration samples
3. Convert ONNX → RKNN with INT8 quantization
4. Output quantized `.rknn` file

### Manual

```bash
# On rover
python3 src/tractor_bringup/tractor_bringup/convert_onnx_to_rknn.py \
  models/ppo_v620_update_50.onnx \
  --output models/ppo_v620_update_50_int8.rknn \
  --quantize \
  --calibration-dir calibration_data
```

### Without Quantization (Float16)

If you don't have calibration data:

```bash
# On rover
./convert_onnx_to_rknn.sh models/ppo_v620_update_50.onnx
```

This creates a float16 model (larger, slightly slower, but no calibration needed).

## Performance Comparison

| Model Type | Size   | Inference Time | Accuracy |
|------------|--------|----------------|----------|
| Float16    | ~2 MB  | 50-80ms        | 100%     |
| INT8       | ~600KB | 30-50ms        | ~98-99%  |

**INT8 is recommended** for production use.

## Calibration Dataset Quality

### Good Calibration Data

✅ Diverse environments (indoor, outdoor, different lighting)
✅ Various obstacles and terrain types
✅ Full range of rover speeds and maneuvers
✅ Representative of deployment conditions

### Poor Calibration Data

❌ All from same environment
❌ Limited lighting conditions
❌ Only straight-line driving
❌ Not representative of real use

### Tips

1. **Collect during actual operation**: Let the rover drive around naturally
2. **Multiple sessions**: Collect calibration data from different runs
3. **100+ samples**: More samples = better quantization accuracy
4. **Update periodically**: Recollect if deployment environment changes

## Deployment Workflow

### With Quantization (Recommended)

```bash
# 1. On rover: Collect data (includes calibration samples)
./start_remote_training_collection.sh tcp://V620_IP:5555
# Drive for 10-20 minutes

# 2. On V620: Train and deploy
cd ~/remote_training_server
./deploy_model.sh checkpoints/ppo_v620_update_50.pt ROVER_IP USER

# 3. On rover: Conversion happens automatically with quantization
# (deploy script detects calibration_data/ and uses it)

# 4. On rover: Run inference
./start_remote_trained_inference.sh
```

### Without Quantization (Float16 Only)

```bash
# 1. Disable calibration saving
# Edit launch file: 'save_calibration_samples': False

# 2. Collect data as normal
./start_remote_training_collection.sh tcp://V620_IP:5555

# 3. Deploy (will create float16 model)
# On V620:
./deploy_model.sh checkpoints/ppo_v620_update_50.pt ROVER_IP USER
```

## Transfer Calibration Data

If you want to use calibration data from one rover on another:

```bash
# Package calibration data
tar -czf calibration_data.tar.gz calibration_data/

# Copy to another rover
scp calibration_data.tar.gz other_rover:~/Documents/ros2-rover/

# Extract on other rover
ssh other_rover
cd ~/Documents/ros2-rover
tar -xzf calibration_data.tar.gz

# Convert with shared calibration data
./convert_onnx_to_rknn.sh models/model.onnx calibration_data
```

## Troubleshooting

### "No calibration samples found"

```bash
# Check directory exists
ls -la calibration_data/

# Check if collection was enabled
ros2 param get /remote_training_collector save_calibration_samples

# Manually enable
ros2 param set /remote_training_collector save_calibration_samples True
```

### "Quantization accuracy poor"

1. Collect more diverse calibration data (200+ samples)
2. Ensure calibration data matches deployment environment
3. Try float16 mode if accuracy is critical

### "Conversion fails with calibration data"

```bash
# Test loading calibration samples
python3 -c "
import numpy as np
import os
cal_dir = 'calibration_data'
files = [f for f in os.listdir(cal_dir) if f.endswith('.npz')]
print(f'Found {len(files)} files')
for f in files[:5]:
    data = np.load(os.path.join(cal_dir, f))
    print(f'{f}: rgb={data[\"rgb\"].shape}, depth={data[\"depth\"].shape}, proprio={data[\"proprio\"].shape}')
"
```

## File Structure

```
ros2-rover/
├── calibration_data/              # Calibration samples
│   ├── calibration_0000.npz
│   ├── calibration_0001.npz
│   └── ...
├── models/
│   ├── ppo_v620_update_50.onnx    # From V620
│   ├── ppo_v620_update_50.rknn    # Quantized (INT8)
│   └── remote_trained.rknn        # Symlink to latest
└── convert_onnx_to_rknn.sh        # Conversion script
```

## Summary

1. **Enable calibration** during data collection (default: enabled)
2. **Collect 100+ samples** from diverse environments
3. **Convert with `--quantize`** flag and calibration directory
4. **Enjoy 4x smaller models** and faster inference!

Calibration data is automatically collected - you just need to make sure it's present when converting ONNX → RKNN.
