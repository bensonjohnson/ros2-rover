#!/bin/bash

echo "==================================="
echo "Orange Pi 5 Plus Performance Optimizer"
echo "==================================="

# Set CPU governor to performance mode
echo "Setting CPU governor to performance mode..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -f "$cpu" ]; then
        echo "performance" | sudo tee "$cpu" > /dev/null
        echo "  $(basename $(dirname $cpu)): performance mode"
    fi
done

# Optimize NPU frequency
echo "Optimizing NPU frequency..."
if [ -f "/sys/class/devfreq/fdab0000.npu/governor" ]; then
    echo "performance" | sudo tee /sys/class/devfreq/fdab0000.npu/governor > /dev/null
    echo "  NPU governor: performance"
fi

if [ -f "/sys/class/devfreq/fdab0000.npu/cur_freq" ]; then
    echo "1000000000" | sudo tee /sys/class/devfreq/fdab0000.npu/max_freq > /dev/null 2>&1
    npu_freq=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq)
    echo "  NPU frequency: $((npu_freq / 1000000))MHz"
fi

# Optimize GPU frequency
echo "Optimizing GPU frequency..."
if [ -f "/sys/class/devfreq/fb000000.gpu/governor" ]; then
    echo "performance" | sudo tee /sys/class/devfreq/fb000000.gpu/governor > /dev/null
    echo "  GPU governor: performance"
fi

# Set process priorities for SLAM
echo "Setting process priorities..."
sudo sysctl -w kernel.sched_rt_runtime_us=950000 > /dev/null

# Optimize memory settings
echo "Optimizing memory settings..."
sudo sysctl -w vm.swappiness=10 > /dev/null
sudo sysctl -w vm.dirty_ratio=15 > /dev/null
sudo sysctl -w vm.dirty_background_ratio=5 > /dev/null

# USB optimization for RealSense
echo "Optimizing USB for RealSense..."
echo 1024 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb > /dev/null

# Set CPU affinity for SLAM (bind to big cores)
echo "CPU affinity will be set during SLAM execution..."

echo "==================================="
echo "Performance optimization complete!"
echo "==================================="

# Show current status
echo "Current frequencies:"
echo "  CPU: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 'N/A') kHz"
echo "  NPU: $(($(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || echo 0) / 1000000)) MHz"
echo "  GPU: $(($(cat /sys/class/devfreq/fb000000.gpu/cur_freq 2>/dev/null || echo 0) / 1000000)) MHz"