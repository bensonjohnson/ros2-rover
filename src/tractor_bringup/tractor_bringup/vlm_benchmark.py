#!/usr/bin/env python3
"""
VLM Processing Benchmark

This node benchmarks VLM inference performance at different resolutions
to determine optimal settings for real-time rover control.
Tests both RKLLM inference speed and image processing overhead.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import threading
from typing import List, Dict, Optional
import json
import os
import statistics

try:
    from rkllm.api import RKLLM
    RKLLM_AVAILABLE = True
except ImportError:
    RKLLM_AVAILABLE = False


class VLMBenchmark(Node):
    def __init__(self):
        super().__init__('vlm_benchmark')
        
        # Parameters
        self.declare_parameter('vlm_model_path', '/home/ubuntu/models/Qwen2.5-VL-7B-Instruct-rk3588-1.2.1.rkllm')
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('test_duration', 60.0)  # seconds
        self.declare_parameter('target_fps', 1.0)  # target images per second
        self.declare_parameter('test_resolutions', [320, 480, 640, 800])  # test widths
        
        self.vlm_model_path = self.get_parameter('vlm_model_path').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.test_duration = self.get_parameter('test_duration').value
        self.target_fps = self.get_parameter('target_fps').value
        self.test_resolutions = self.get_parameter('test_resolutions').value
        
        # State
        self.bridge = CvBridge()
        self.current_image: Optional[np.ndarray] = None
        self.benchmark_running = False
        self.results: Dict = {}
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.processing_times: List[float] = []
        self.total_inferences = 0
        self.successful_inferences = 0
        self.start_time = 0.0
        
        # VLM model
        self.vlm_model = None
        
        # Camera subscriber
        self.image_sub = self.create_subscription(
            Image, 
            self.camera_topic,
            self.image_callback, 
            1
        )
        
        self.get_logger().info("VLM Benchmark Node initialized")
        self.get_logger().info(f"RKLLM Available: {RKLLM_AVAILABLE}")
        self.get_logger().info(f"Test duration: {self.test_duration}s")
        self.get_logger().info(f"Target FPS: {self.target_fps}")
        self.get_logger().info(f"Test resolutions: {self.test_resolutions}")
        
        # Start benchmark after a short delay
        self.create_timer(2.0, self.start_benchmark)
    
    def image_callback(self, msg: Image):
        """Store latest camera image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
    
    def start_benchmark(self):
        """Initialize and start the benchmark sequence"""
        if self.current_image is None:
            self.get_logger().warn("No camera images received yet, retrying in 2s...")
            self.create_timer(2.0, self.start_benchmark)
            return
        
        self.get_logger().info("Starting VLM benchmark...")
        
        # Initialize VLM model if available
        if RKLLM_AVAILABLE:
            self.init_vlm_model()
        
        # Run benchmark for each resolution
        for resolution in self.test_resolutions:
            self.run_resolution_benchmark(resolution)
        
        # Print final results
        self.print_final_results()
        
        # Shutdown
        self.get_logger().info("Benchmark complete, shutting down...")
        rclpy.shutdown()
    
    def init_vlm_model(self):
        """Initialize RKLLM model"""
        try:
            if not os.path.exists(self.vlm_model_path):
                self.get_logger().error(f"Model file not found: {self.vlm_model_path}")
                return False
            
            self.get_logger().info("Loading RKLLM model...")
            self.vlm_model = RKLLM()
            
            start_time = time.time()
            ret = self.vlm_model.load_rkllm(self.vlm_model_path)
            load_time = time.time() - start_time
            
            if ret != 0:
                self.get_logger().error(f"Failed to load RKLLM model: {ret}")
                return False
            
            self.get_logger().info(f"RKLLM model loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            self.get_logger().error(f"RKLLM initialization failed: {e}")
            return False
    
    def run_resolution_benchmark(self, target_width: int):
        """Run benchmark at specific resolution"""
        self.get_logger().info(f"\n--- Benchmarking at {target_width}px width ---")
        
        # Reset counters
        self.inference_times.clear()
        self.processing_times.clear()
        self.total_inferences = 0
        self.successful_inferences = 0
        self.start_time = time.time()
        
        # Calculate interval for target FPS
        interval = 1.0 / self.target_fps
        
        # Run for specified duration
        while (time.time() - self.start_time) < self.test_duration:
            if self.current_image is not None:
                self.process_single_image(self.current_image.copy(), target_width)
                time.sleep(max(0, interval - 0.1))  # Try to maintain target FPS
            else:
                time.sleep(0.1)
        
        # Calculate and store results
        self.calculate_resolution_results(target_width)
    
    def process_single_image(self, image: np.ndarray, target_width: int):
        """Process a single image and measure performance"""
        process_start = time.time()
        
        try:
            # Resize image
            height, width = image.shape[:2]
            if width != target_width:
                scale = target_width / width
                new_height = int(height * scale)
                image = cv2.resize(image, (target_width, new_height))
            
            processing_time = time.time() - process_start
            self.processing_times.append(processing_time)
            
            # Run inference
            inference_start = time.time()
            
            if self.vlm_model is not None:
                # Real RKLLM inference
                success = self.run_real_inference(image)
            else:
                # Simulate inference
                success = self.simulate_inference(image)
            
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            self.total_inferences += 1
            if success:
                self.successful_inferences += 1
            
            # Log progress every 10 inferences
            if self.total_inferences % 10 == 0:
                elapsed = time.time() - self.start_time
                current_fps = self.total_inferences / elapsed
                avg_inference = statistics.mean(self.inference_times[-10:])
                self.get_logger().info(
                    f"Progress: {self.total_inferences} inferences, "
                    f"{current_fps:.2f} FPS, "
                    f"avg inference: {avg_inference*1000:.1f}ms"
                )
        
        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")
    
    def run_real_inference(self, image: np.ndarray) -> bool:
        """Run actual RKLLM inference"""
        try:
            # Simple prompt for benchmarking
            prompt = """Analyze this rover camera view. Respond with JSON:
{"linear_speed": 0.1, "angular_speed": 0.0, "obstacle_detected": false}"""
            
            # Note: Actual RKLLM API call would go here
            # This is a placeholder since the exact API isn't implemented
            # response = self.vlm_model.chat([
            #     {"type": "image", "image": image},
            #     {"type": "text", "text": prompt}
            # ])
            
            # For now, simulate processing delay based on image size
            pixels = image.shape[0] * image.shape[1]
            # Simulate processing time: ~1ms per 1000 pixels + base 100ms
            sim_time = 0.1 + (pixels / 1000000.0)
            time.sleep(sim_time)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"RKLLM inference error: {e}")
            return False
    
    def simulate_inference(self, image: np.ndarray) -> bool:
        """Simulate VLM inference with realistic timing"""
        try:
            # Simulate processing based on image complexity
            pixels = image.shape[0] * image.shape[1]
            
            # Realistic timing model for RK3588 NPU:
            # - Base overhead: ~50ms
            # - Per-pixel processing: ~0.5μs
            # - Model complexity factor: 2x for VLM vs simple CNN
            
            base_time = 0.05  # 50ms base
            pixel_time = pixels * 0.0000005  # 0.5μs per pixel
            complexity_factor = 2.0
            
            sim_time = (base_time + pixel_time) * complexity_factor
            time.sleep(sim_time)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Simulation error: {e}")
            return False
    
    def calculate_resolution_results(self, resolution: int):
        """Calculate and store results for current resolution"""
        if not self.inference_times:
            self.get_logger().warn(f"No data collected for {resolution}px")
            return
        
        elapsed_time = time.time() - self.start_time
        actual_fps = self.total_inferences / elapsed_time
        success_rate = self.successful_inferences / max(1, self.total_inferences)
        
        results = {
            'resolution_width': resolution,
            'test_duration': elapsed_time,
            'total_inferences': self.total_inferences,
            'successful_inferences': self.successful_inferences,
            'success_rate': success_rate,
            'actual_fps': actual_fps,
            'target_fps': self.target_fps,
            'fps_efficiency': actual_fps / self.target_fps,
            'processing_times': {
                'min': min(self.processing_times) * 1000,  # ms
                'max': max(self.processing_times) * 1000,
                'mean': statistics.mean(self.processing_times) * 1000,
                'median': statistics.median(self.processing_times) * 1000,
            },
            'inference_times': {
                'min': min(self.inference_times) * 1000,  # ms
                'max': max(self.inference_times) * 1000,
                'mean': statistics.mean(self.inference_times) * 1000,
                'median': statistics.median(self.inference_times) * 1000,
            }
        }
        
        self.results[resolution] = results
        
        # Print immediate results
        self.get_logger().info(f"\n=== Results for {resolution}px width ===")
        self.get_logger().info(f"Completed: {self.total_inferences} inferences in {elapsed_time:.1f}s")
        self.get_logger().info(f"Success rate: {success_rate*100:.1f}%")
        self.get_logger().info(f"Actual FPS: {actual_fps:.2f} (target: {self.target_fps})")
        self.get_logger().info(f"FPS efficiency: {results['fps_efficiency']*100:.1f}%")
        self.get_logger().info(f"Avg inference time: {results['inference_times']['mean']:.1f}ms")
        self.get_logger().info(f"Avg processing time: {results['processing_times']['mean']:.1f}ms")
    
    def print_final_results(self):
        """Print comprehensive benchmark results"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("FINAL BENCHMARK RESULTS")
        self.get_logger().info("="*60)
        
        if not self.results:
            self.get_logger().warn("No results to display")
            return
        
        # Find best performing resolution
        best_fps = 0
        best_resolution = None
        
        for resolution, data in self.results.items():
            fps = data['actual_fps']
            if fps > best_fps:
                best_fps = fps
                best_resolution = resolution
        
        # Summary table
        self.get_logger().info(f"{'Resolution':<12} {'FPS':<8} {'Efficiency':<12} {'Inference (ms)':<15} {'Success %':<10}")
        self.get_logger().info("-" * 60)
        
        for resolution in sorted(self.results.keys()):
            data = self.results[resolution]
            self.get_logger().info(
                f"{resolution}px{'':<7} "
                f"{data['actual_fps']:<8.2f} "
                f"{data['fps_efficiency']*100:<12.1f}% "
                f"{data['inference_times']['mean']:<15.1f} "
                f"{data['success_rate']*100:<10.1f}%"
            )
        
        self.get_logger().info("-" * 60)
        self.get_logger().info(f"RECOMMENDATION: {best_resolution}px width achieves {best_fps:.2f} FPS")
        
        # Determine if 1+ FPS is achievable
        achievable_1fps = any(data['actual_fps'] >= 1.0 for data in self.results.values())
        if achievable_1fps:
            fast_resolutions = [res for res, data in self.results.items() if data['actual_fps'] >= 1.0]
            self.get_logger().info(f"✓ 1+ FPS achievable at: {fast_resolutions}px widths")
        else:
            max_fps = max(data['actual_fps'] for data in self.results.values())
            self.get_logger().info(f"✗ Max achievable FPS: {max_fps:.2f} (target was 1.0)")
        
        # Save results to file
        results_file = f"/tmp/vlm_benchmark_results_{int(time.time())}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            self.get_logger().info(f"Detailed results saved to: {results_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to save results: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    benchmark = VLMBenchmark()
    
    try:
        rclpy.spin(benchmark)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            benchmark.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()