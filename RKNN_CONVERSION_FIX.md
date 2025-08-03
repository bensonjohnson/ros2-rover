# RKNN Conversion Fix - Adaptive Pooling Issue

## Problem Summary

The NPU exploration system was failing during RKNN conversion with the error:
```
RKNN conversion failed: Unsupported: ONNX export of operator adaptive_avg_pool2d, 
output size that are not factor of input size.
```

## Root Cause Analysis

The issue was in the `DepthImageExplorationNet` model architecture in `rknn_trainer_depth.py`. The model used:

```python
nn.AdaptiveAvgPool2d((7, 7))  # 20x15 -> 7x7
```

This was problematic because:
1. The input tensor had dimensions 20×15 after convolution layers
2. `AdaptiveAvgPool2d` was trying to pool to 7×7
3. 7 doesn't divide evenly into both 20 and 15
4. ONNX export (required for RKNN) doesn't support adaptive pooling with non-factor output sizes

## Solution Applied

### Before (Problematic Architecture)
```python
# Depth image branch (CNN)
self.depth_conv = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 424x240 -> 212x120
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 160x120 -> 80x60
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 80x60 -> 40x30
    nn.ReLU(),
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 40x30 -> 20x15
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((7, 7)),  # 20x15 -> 7x7 (PROBLEMATIC)
    nn.Flatten()
)

self.depth_fc = nn.Linear(256 * 7 * 7, 512)
```

### After (Fixed Architecture)
```python
# Depth image branch (CNN)
self.depth_conv = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 424x240 -> 212x120
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 212x120 -> 106x60
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 106x60 -> 53x30
    nn.ReLU(),
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 53x30 -> 27x15
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 27x15 -> 27x15
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3)),  # 27x15 -> 9x5
    nn.Flatten()
)

self.depth_fc = nn.Linear(256 * 9 * 5, 512)
```

## Key Changes Made

1. **Replaced AdaptiveAvgPool2d with AvgPool2d**: 
   - `AdaptiveAvgPool2d((7, 7))` → `AvgPool2d(kernel_size=(3, 3), stride=(3, 3))`
   
2. **Added Extra Convolution Layer**:
   - Added `nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)` to better process features
   
3. **Updated Linear Layer Input Size**:
   - `nn.Linear(256 * 7 * 7, 512)` → `nn.Linear(256 * 9 * 5, 512)`
   - From 12,544 features to 11,520 features

4. **Fixed Dimension Comments**:
   - Corrected the dimension calculations in comments to reflect actual tensor sizes

## Technical Benefits

1. **ONNX Compatibility**: Fixed pooling uses exact kernel/stride ratios that ONNX can handle
2. **RKNN Support**: The model can now be converted to RKNN format for NPU inference
3. **Preserved Functionality**: Output dimensions remain the same (3 outputs: linear_vel, angular_vel, confidence)
4. **Better Feature Processing**: Additional convolution layer provides more feature refinement

## Testing Requirements

To verify the fix works:

1. **Model Forward Pass**: Ensure the model can process 240×424 depth images + 10D sensor data
2. **ONNX Export**: Verify `torch.onnx.export()` completes without errors
3. **RKNN Conversion**: Test conversion using RKNN toolkit on target hardware
4. **Inference Verification**: Confirm outputs are in expected ranges

## Expected Results

After this fix, the system should:
- ✅ Complete ONNX export successfully
- ✅ Convert to RKNN format without adaptive pooling errors
- ✅ Run NPU inference on RK3588 hardware
- ✅ Continue learning and exploration with depth-based navigation

## Files Modified

- `src/tractor_bringup/tractor_bringup/rknn_trainer_depth.py`: Updated `DepthImageExplorationNet` architecture

## Next Steps

1. **Test on Target Hardware**: Run the NPU exploration system on the actual robot
2. **Monitor Performance**: Check inference speed and accuracy with the new architecture
3. **Retrain if Needed**: If using pre-trained models, they may need retraining with the new architecture
4. **Validate Exploration**: Ensure the rover still performs effective autonomous exploration

## Related Issues

This fix resolves the core RKNN conversion problem that was preventing NPU deployment. The system should now successfully:
- Process depth images for obstacle avoidance
- Learn from exploration experiences
- Convert trained models to NPU-optimized format
- Run real-time inference on embedded hardware
