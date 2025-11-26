#!/usr/bin/env python3
"""Hybrid velocity estimator with encoder + IMU fusion.

This module provides robust velocity estimation for tank-steer rovers by:
1. Using encoder velocity as primary estimate
2. Comparing with IMU gyroscope for slip detection
3. Falling back to IMU when encoder slip detected
4. Providing confidence metric for proprioception
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
import time


@dataclass
class VelocityEstimate:
    """Velocity estimate with confidence metric."""
    linear: float          # Linear velocity (m/s)
    angular: float         # Angular velocity (rad/s)
    confidence: float      # 0.0 - 1.0
    source: str           # 'encoder', 'imu', 'fused'
    slip_detected: bool   # True if wheel slip detected


class HybridVelocityEstimator:
    """Fuses encoder and IMU data for robust velocity estimation.

    Features:
    - Primary: Encoder-based velocity (accurate in ideal conditions)
    - Backup: IMU gyroscope for angular velocity validation
    - Slip detection: Compares encoder vs IMU angular velocity
    - Graceful degradation: Falls back to IMU when slip detected
    - Confidence metric: Indicates estimate reliability
    """

    def __init__(self, wheel_radius=0.15, wheel_separation=0.5, encoder_ppr=1980):
        """Initialize hybrid velocity estimator.

        Args:
            wheel_radius: Wheel radius in meters
            wheel_separation: Distance between wheel centers in meters
            encoder_ppr: Encoder pulses per revolution
        """
        self.wheel_radius = wheel_radius
        self.wheel_separation = wheel_separation
        self.encoder_ppr = encoder_ppr

        # IMU integration state
        self.imu_linear_vel = 0.0
        self.imu_angular_vel = 0.0
        self.last_imu_time = None

        # History for slip detection and filtering
        self.encoder_history = deque(maxlen=10)  # Last 10 encoder estimates
        self.imu_history = deque(maxlen=10)      # Last 10 IMU estimates

        # Confidence thresholds
        self.slip_threshold = 0.15  # rad/s difference triggers slip detection
        self.imu_noise_std = 0.01   # rad/s, LSM9DS1 spec

        # Velocity estimate cache
        self.last_estimate = None

    def update(self, w_left, w_right, imu_ax=None, imu_wz=None, dt=0.01):
        """Update velocity estimate with latest sensor data.

        Args:
            w_left: Left wheel angular velocity (rad/s)
            w_right: Right wheel angular velocity (rad/s)
            imu_ax: IMU linear acceleration X (m/s²), optional
            imu_wz: IMU angular velocity Z (rad/s), optional
            dt: Time delta since last update (seconds)

        Returns:
            VelocityEstimate with confidence metric
        """
        # 1. Encoder-based estimate (primary)
        encoder_linear = (w_left + w_right) * self.wheel_radius / 2.0
        encoder_angular = -(w_left - w_right) * self.wheel_radius / self.wheel_separation

        # 2. IMU-based estimate (backup/validation)
        imu_linear = None
        imu_angular = None

        if imu_wz is not None:
            imu_angular = imu_wz

        if imu_ax is not None and self.last_imu_time is not None:
            # Integrate acceleration for linear velocity (simple dead reckoning)
            # Note: This accumulates error quickly, only use as short-term backup
            self.imu_linear_vel += imu_ax * dt
            # Apply decay to prevent unbounded drift
            self.imu_linear_vel *= 0.95
            imu_linear = self.imu_linear_vel

        # 3. Slip Detection
        slip_detected = False
        if imu_angular is not None:
            angular_diff = abs(encoder_angular - imu_angular)
            if angular_diff > self.slip_threshold:
                slip_detected = True

        # 4. Fusion and Confidence Estimation
        if slip_detected:
            # Encoder slip detected - trust IMU more
            linear_vel = imu_linear if imu_linear is not None else encoder_linear
            angular_vel = imu_angular if imu_angular is not None else encoder_angular
            confidence = 0.6  # Medium confidence (IMU has drift)
            source = 'imu'
        elif imu_angular is not None:
            # Both sensors available and agree - fuse them
            # Weighted average: prefer encoders when no slip
            alpha = 0.8  # 80% encoder, 20% IMU
            angular_vel = alpha * encoder_angular + (1 - alpha) * imu_angular
            linear_vel = encoder_linear  # Encoders more reliable for linear
            confidence = 0.95  # High confidence
            source = 'fused'
        else:
            # Encoder only (no IMU data available)
            linear_vel = encoder_linear
            angular_vel = encoder_angular
            confidence = 0.9  # Good confidence (encoder-only is reliable in most cases)
            source = 'encoder'

        # 5. Store history for trend analysis
        self.encoder_history.append((encoder_linear, encoder_angular))
        if imu_angular is not None:
            self.imu_history.append((imu_linear, imu_angular))

        self.last_imu_time = time.time()

        # 6. Create estimate
        estimate = VelocityEstimate(
            linear=linear_vel,
            angular=angular_vel,
            confidence=confidence,
            source=source,
            slip_detected=slip_detected
        )

        self.last_estimate = estimate
        return estimate

    def get_slip_statistics(self):
        """Get slip detection statistics over recent history.

        Returns:
            dict with slip frequency and magnitude
        """
        if len(self.encoder_history) < 2 or len(self.imu_history) < 2:
            return {'slip_frequency': 0.0, 'avg_slip_magnitude': 0.0}

        # Compare recent encoder vs IMU angular velocities
        slip_count = 0
        slip_magnitudes = []

        for enc, imu in zip(self.encoder_history, self.imu_history):
            enc_ang = enc[1]
            imu_ang = imu[1] if imu[1] is not None else enc_ang
            diff = abs(enc_ang - imu_ang)
            if diff > self.slip_threshold:
                slip_count += 1
                slip_magnitudes.append(diff)

        slip_frequency = slip_count / len(self.encoder_history)
        avg_slip_mag = np.mean(slip_magnitudes) if slip_magnitudes else 0.0

        return {
            'slip_frequency': slip_frequency,
            'avg_slip_magnitude': float(avg_slip_mag)
        }

    def reset(self):
        """Reset IMU integration state (e.g., between episodes)."""
        self.imu_linear_vel = 0.0
        self.imu_angular_vel = 0.0
        self.last_imu_time = None
        self.encoder_history.clear()
        self.imu_history.clear()
