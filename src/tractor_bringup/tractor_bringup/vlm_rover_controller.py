#!/usr/bin/env python3
"""
Vision Language Model Rover Controller

This node uses an Ollama-compatible REST API server with vision language models
to control the rover based on visual input from the RealSense camera. The VLM
analyzes the camera feed and generates movement commands to navigate autonomously.

Compatible with: Ollama, rkllama, or any Ollama-compatible API
Optimized for: Nvidia Cosmos-Reason1-7B with chain-of-thought reasoning
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import threading
from typing import Optional, List
from collections import deque
import json
import base64
import requests
import tempfile
import os
from enum import Enum


class ControlMode(Enum):
    """Control loop modes"""
    TIME_BASED = "time_based"  # Original: inference every N seconds
    SYNCHRONIZED = "synchronized"  # New: inference triggered by control response


class VLMRoverController(Node):
    def __init__(self):
        super().__init__('vlm_rover_controller')

        # Parameters
        self.declare_parameter('rkllama_url', 'https://ollama.gokickrocks.org')  # Ollama-compatible API URL
        self.declare_parameter('model_name', 'mistral-small3.2:24b')
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('max_linear_speed', 0.2)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('inference_interval', 2.0)  # seconds between VLM inferences (time_based mode)
        self.declare_parameter('command_timeout', 5.0)  # seconds before stopping if no new command
        self.declare_parameter('simulation_mode', False)
        self.declare_parameter('request_timeout', 30.0)  # HTTP request timeout (higher for remote)
        self.declare_parameter('use_video_mode', False)  # Disable multi-frame mode for testing
        self.declare_parameter('video_duration', 0.5)  # Duration of video clips in seconds (2 frames @ 4 FPS)
        self.declare_parameter('video_fps', 4)  # FPS for video clips (Cosmos trained on 4 FPS)
        self.declare_parameter('num_ctx', 16384)  # Context window size for Ollama (16K for 2-frame mode)
        self.declare_parameter('control_mode', 'synchronized')  # 'time_based' or 'synchronized'
        self.declare_parameter('control_duration', 1.0)  # seconds to apply control before next inference (synchronized mode)

        self.rkllama_url = self.get_parameter('rkllama_url').value
        self.model_name = self.get_parameter('model_name').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.inference_interval = self.get_parameter('inference_interval').value
        self.command_timeout = self.get_parameter('command_timeout').value
        self.simulation_mode = self.get_parameter('simulation_mode').value
        self.request_timeout = self.get_parameter('request_timeout').value
        self.use_video_mode = self.get_parameter('use_video_mode').value
        self.video_duration = self.get_parameter('video_duration').value
        self.video_fps = self.get_parameter('video_fps').value
        self.num_ctx = self.get_parameter('num_ctx').value

        # Parse control mode
        control_mode_str = self.get_parameter('control_mode').value
        try:
            self.control_mode = ControlMode(control_mode_str)
        except ValueError:
            self.get_logger().warn(f"Invalid control_mode '{control_mode_str}', defaulting to 'synchronized'")
            self.control_mode = ControlMode.SYNCHRONIZED

        self.control_duration = self.get_parameter('control_duration').value

        # State
        self.bridge = CvBridge()
        self.current_image: Optional[np.ndarray] = None
        self.last_inference_time = 0.0
        self.last_command_time = 0.0
        self.current_command = Twist()
        self.inference_lock = threading.Lock()
        self.inference_in_progress = False  # Track if inference is running

        # Frame buffer for video mode (store frames with timestamps)
        # Buffer enough frames for video_duration * video_fps * 2 (with headroom)
        max_buffer_size = int(self.video_duration * self.video_fps * 4)  # 4x buffer
        self.frame_buffer: deque = deque(maxlen=max_buffer_size)
        self.frame_lock = threading.Lock()

        # Synchronized control mode state
        self.control_cycle_start_time = 0.0  # When control command started being applied
        self.frame_for_next_inference: Optional[np.ndarray] = None  # Frame captured after control response
        self.waiting_for_control_completion = False  # True when applying control commands

        # Test rkllama server connection
        if not self.simulation_mode:
            self.test_rkllama_connection()
        else:
            self.get_logger().warn("Running in simulation mode - VLM commands will be randomized")
        
        # Subscribers
        if self.use_compressed:
            self.image_sub = self.create_subscription(
                CompressedImage, 
                f"{self.camera_topic}/compressed",
                self.compressed_image_callback, 
                1
            )
        else:
            self.image_sub = self.create_subscription(
                Image, 
                self.camera_topic,
                self.image_callback, 
                1
            )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel_vlm', 10)
        self.vlm_response_pub = self.create_publisher(String, 'vlm_response', 10)

        # Timers
        self.inference_timer = self.create_timer(0.1, self.inference_timer_callback)
        self.command_timer = self.create_timer(0.1, self.command_timer_callback)

        self.get_logger().info(f"VLM Rover Controller initialized")
        self.get_logger().info(f"  Ollama URL: {self.rkllama_url}")
        self.get_logger().info(f"  Model name: {self.model_name}")
        self.get_logger().info(f"  Camera topic: {self.camera_topic}")
        self.get_logger().info(f"  Control mode: {self.control_mode.value}")
        if self.control_mode == ControlMode.SYNCHRONIZED:
            self.get_logger().info(f"  Control duration: {self.control_duration}s (apply control then capture next frame)")
        else:
            self.get_logger().info(f"  Inference interval: {self.inference_interval}s (time-based triggering)")
        self.get_logger().info(f"  Video mode: {self.use_video_mode} ({self.video_fps} FPS, {self.video_duration}s clips)")
        self.get_logger().info(f"  Context window: {self.num_ctx} tokens")
        self.get_logger().info(f"  Simulation mode: {self.simulation_mode}")

    def test_rkllama_connection(self):
        """Test connection to Ollama-compatible server"""
        try:
            response = requests.get(f"{self.rkllama_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json()
                self.get_logger().info(f"Successfully connected to Ollama server")
                self.get_logger().info(f"Available models: {len(models.get('models', []))} found")
            else:
                self.get_logger().warn(f"Server responded with status {response.status_code}")
                self.get_logger().warn("Continuing anyway - will retry on inference")
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Failed to connect to Ollama server: {e}")
            self.get_logger().error(f"Make sure server is accessible at {self.rkllama_url}")
            self.get_logger().warn("Falling back to simulation mode")
            self.simulation_mode = True
    
    def image_callback(self, msg: Image):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image

            # Add to frame buffer for video mode
            if self.use_video_mode:
                with self.frame_lock:
                    self.frame_buffer.append((time.time(), cv_image.copy()))
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def compressed_image_callback(self, msg: CompressedImage):
        """Process incoming compressed camera images"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.current_image = cv_image

            # Add to frame buffer for video mode
            if self.use_video_mode:
                with self.frame_lock:
                    self.frame_buffer.append((time.time(), cv_image.copy()))
        except Exception as e:
            self.get_logger().error(f"Compressed image conversion failed: {e}")
    
    def extract_frames_from_buffer(self) -> List[np.ndarray]:
        """Extract frames from buffer for multi-frame inference"""
        with self.frame_lock:
            # Calculate how many frames we need
            num_frames_needed = int(self.video_duration * self.video_fps)
            # Cap at 10 frames (Ollama/Cosmos limit)
            num_frames_needed = min(num_frames_needed, 10)

            if len(self.frame_buffer) < num_frames_needed:
                return []

            # Sample frames at target FPS from the buffer
            buffer_list = list(self.frame_buffer)

            if len(buffer_list) < num_frames_needed:
                return []

            # Take the most recent frames and sample them evenly
            sample_interval = len(buffer_list) / num_frames_needed
            sampled_frames = []

            for i in range(num_frames_needed):
                idx = int(i * sample_interval)
                if idx < len(buffer_list):
                    _, frame = buffer_list[idx]
                    sampled_frames.append(frame)

            return sampled_frames

    def create_video_clip(self) -> Optional[str]:
        """Create a video clip from buffered frames at target FPS"""
        with self.frame_lock:
            if len(self.frame_buffer) < self.video_fps:
                self.get_logger().warn(f"Not enough frames in buffer: {len(self.frame_buffer)} < {self.video_fps}")
                return None

            # Get frames for the video duration
            num_frames_needed = int(self.video_duration * self.video_fps)

            # Sample frames at target FPS from the buffer
            # The buffer contains frames at camera rate (15 FPS), we need to downsample to 4 FPS
            buffer_list = list(self.frame_buffer)

            # Calculate sampling interval
            # If we have 30 frames (2 seconds at 15 FPS) and need 8 frames (2 seconds at 4 FPS)
            # We should sample every 3.75 frames
            if len(buffer_list) < num_frames_needed:
                self.get_logger().warn(f"Not enough frames for full clip: {len(buffer_list)} < {num_frames_needed}")
                return None

            # Take the most recent frames and sample them
            sample_interval = len(buffer_list) / num_frames_needed
            sampled_frames = []

            for i in range(num_frames_needed):
                idx = int(i * sample_interval)
                if idx < len(buffer_list):
                    _, frame = buffer_list[idx]
                    sampled_frames.append(frame)

        if not sampled_frames:
            return None

        # Create temporary video file
        try:
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_fd)  # Close the file descriptor

            # Get frame dimensions
            height, width = sampled_frames[0].shape[:2]

            # Create video writer (H.264 codec)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_path, fourcc, float(self.video_fps), (width, height))

            if not video_writer.isOpened():
                self.get_logger().error("Failed to open video writer")
                os.unlink(temp_path)
                return None

            # Write frames
            for frame in sampled_frames:
                video_writer.write(frame)

            video_writer.release()

            self.get_logger().debug(f"Created video clip: {len(sampled_frames)} frames at {self.video_fps} FPS")
            return temp_path

        except Exception as e:
            self.get_logger().error(f"Failed to create video clip: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            return None

    def inference_timer_callback(self):
        """Timer callback for VLM inference - handles both time_based and synchronized modes"""
        # Check if inference is already running - wait for it to complete
        if self.inference_in_progress:
            self.get_logger().debug("Inference already in progress, waiting...")
            return

        current_time = time.time()

        if self.control_mode == ControlMode.SYNCHRONIZED:
            # Synchronized mode: wait for control completion before next inference
            if self.waiting_for_control_completion:
                # Check if control duration has elapsed
                if current_time - self.control_cycle_start_time < self.control_duration:
                    return

                # Control duration complete - capture current live frame for next inference
                self.get_logger().info("Control duration elapsed, capturing frame for next inference")
                if self.current_image is not None:
                    self.frame_for_next_inference = self.current_image.copy()
                    self.waiting_for_control_completion = False
                else:
                    self.get_logger().warn("No camera frame available after control completion")
                    return

            # Ready for next inference if we have a frame
            if self.frame_for_next_inference is None:
                # First iteration - use current frame to start
                if self.current_image is not None:
                    self.frame_for_next_inference = self.current_image.copy()
                    self.get_logger().info("Starting synchronized control loop with initial frame")
                else:
                    return

            # Start inference with the captured frame
            self.inference_in_progress = True
            if self.use_video_mode:
                inference_thread = threading.Thread(target=self.run_vlm_inference_video)
            else:
                inference_thread = threading.Thread(
                    target=self.run_vlm_inference,
                    args=(self.frame_for_next_inference.copy(),)
                )
            inference_thread.daemon = True
            inference_thread.start()
            self.frame_for_next_inference = None  # Clear after use

        else:
            # TIME_BASED mode: original behavior - inference at regular intervals
            # Check if it's time for a new inference (minimum interval)
            if current_time - self.last_inference_time < self.inference_interval:
                return

            if self.use_video_mode:
                # Check if we have enough frames in buffer
                with self.frame_lock:
                    buffer_size = len(self.frame_buffer)

                min_frames = int(self.video_duration * self.video_fps)
                if buffer_size < min_frames:
                    self.get_logger().debug(f"Waiting for frames: {buffer_size}/{min_frames}")
                    return

                # Mark inference as in progress and run in a separate thread
                self.inference_in_progress = True
                inference_thread = threading.Thread(target=self.run_vlm_inference_video)
                inference_thread.daemon = True
                inference_thread.start()
            else:
                # Single image mode
                if self.current_image is None:
                    return

                # Mark inference as in progress and run in a separate thread
                self.inference_in_progress = True
                inference_thread = threading.Thread(target=self.run_vlm_inference, args=(self.current_image.copy(),))
                inference_thread.daemon = True
                inference_thread.start()

            self.last_inference_time = current_time
    
    def run_vlm_inference_video(self):
        """Run VLM inference on multiple frames (temporal context)"""
        try:
            with self.inference_lock:
                if self.simulation_mode:
                    # Simulation mode - generate random but reasonable commands
                    response = self.simulate_vlm_response()
                else:
                    # Real VLM inference via Ollama API with multiple frames
                    # No need to pass video_path anymore - the function extracts frames itself
                    response = self.query_rkllama_api_video(None)

                # Parse the response and extract movement commands
                command = self.parse_vlm_response(response)

                # Update current command
                self.current_command = command
                self.last_command_time = time.time()

                # Publish VLM response for monitoring
                response_msg = String()
                response_msg.data = response
                self.vlm_response_pub.publish(response_msg)

                self.get_logger().info(f"VLM command (multi-frame): linear={command.linear.x:.2f}, angular={command.angular.z:.2f}")

                # If in synchronized mode, start control cycle
                if self.control_mode == ControlMode.SYNCHRONIZED:
                    self.control_cycle_start_time = time.time()
                    self.waiting_for_control_completion = True
                    self.get_logger().info(f"Starting control cycle, will capture next frame in {self.control_duration}s")

        except Exception as e:
            self.get_logger().error(f"VLM multi-frame inference failed: {e}")
        finally:
            # Always mark inference as complete, even if there was an error
            self.inference_in_progress = False

    def run_vlm_inference(self, image: np.ndarray):
        """Run VLM inference on the current image"""
        inference_start = time.time()
        self.get_logger().info("Starting VLM inference...")
        try:
            with self.inference_lock:
                if self.simulation_mode:
                    # Simulation mode - generate random but reasonable commands
                    response = self.simulate_vlm_response()
                else:
                    # Real VLM inference via Ollama API
                    response = self.query_rkllama_api(image)

                # Parse the response and extract movement commands
                command = self.parse_vlm_response(response)

                # Update current command
                self.current_command = command
                self.last_command_time = time.time()

                # Publish VLM response for monitoring
                response_msg = String()
                response_msg.data = response
                self.vlm_response_pub.publish(response_msg)

                inference_time = time.time() - inference_start
                self.get_logger().info(f"VLM command: linear={command.linear.x:.2f}, angular={command.angular.z:.2f} (took {inference_time:.2f}s)")

                # If in synchronized mode, start control cycle
                if self.control_mode == ControlMode.SYNCHRONIZED:
                    self.control_cycle_start_time = time.time()
                    self.waiting_for_control_completion = True
                    self.get_logger().info(f"Starting control cycle, will capture next frame in {self.control_duration}s")

        except Exception as e:
            self.get_logger().error(f"VLM inference failed: {e}")
        finally:
            # Always mark inference as complete, even if there was an error
            self.inference_in_progress = False
            inference_time = time.time() - inference_start
            self.get_logger().info(f"Inference completed in {inference_time:.2f}s, ready for next request")
    
    def query_rkllama_api(self, image: np.ndarray) -> str:
        """Query the Ollama-compatible VLM API with the current image"""
        try:
            # Qwen2.5-VL optimal range: 200K-1M pixels
            # Camera is configured for 640x480 (307K pixels) - perfect!
            # Only resize if someone changes camera config to higher resolution
            height, width = image.shape[:2]
            if width > 640:
                scale = 640.0 / width
                new_width = 640
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                self.get_logger().debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

            # Convert image to base64 JPEG
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Prepare the prompt for rover navigation
            user_prompt = """You are controlling an autonomous rover. Analyze this camera image and decide how to move.

Rules:
- linear_speed: 0.0 to 0.15 (meters/second forward)
- angular_speed: -0.3 to 0.3 (radians/second turning, negative=left, positive=right)
- If path is clear ahead: move forward at 0.10-0.15
- If you see obstacles: slow down or turn to avoid
- Prefer exploration and movement over standing still

Output your movement decision as JSON."""

            # JSON schema for structured output (Ollama format parameter)
            json_schema = {
                "type": "object",
                "properties": {
                    "linear_speed": {
                        "type": "number",
                        "description": "Forward speed in m/s (0.0 to 0.15)",
                        "minimum": 0.0,
                        "maximum": 0.15
                    },
                    "angular_speed": {
                        "type": "number",
                        "description": "Turn rate in rad/s (-0.3 to 0.3, negative=left)",
                        "minimum": -0.3,
                        "maximum": 0.3
                    }
                },
                "required": ["linear_speed", "angular_speed"]
            }

            # Prepare Ollama API request
            api_url = f"{self.rkllama_url}/api/chat"

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": [image_base64]
                    }
                ],
                "stream": False,
                "format": json_schema,  # Structured output enforcement
                "options": {
                    "temperature": 0.0,  # Deterministic for consistent JSON
                    "top_p": 0.95,
                    "num_predict": 256,  # Smaller since we only need JSON
                    "num_ctx": self.num_ctx
                }
            }

            # Make API request
            self.get_logger().info(f"Sending request to {api_url}")
            self.get_logger().info(f"Model: {self.model_name}")
            # Find which message has images
            for i, msg in enumerate(payload['messages']):
                if 'images' in msg:
                    self.get_logger().info(f"Messages: {len(payload['messages'])} messages, {len(msg['images'])} image(s)")
                    break
            self.get_logger().debug(f"User prompt: {user_prompt[:200]}...")

            response = requests.post(
                api_url,
                json=payload,
                timeout=self.request_timeout
            )

            if response.status_code != 200:
                self.get_logger().error(f"Ollama API error: {response.status_code} - {response.text[:200]}")
                return self.simulate_vlm_response()

            # Parse response
            result = response.json()

            # Extract the message content from Ollama-format response
            if "message" in result and "content" in result["message"]:
                content = result["message"]["content"]
                self.get_logger().debug(f"VLM response: {content}")
                return content
            else:
                self.get_logger().error(f"Unexpected API response format: {result}")
                return self.simulate_vlm_response()

        except requests.exceptions.Timeout:
            self.get_logger().error(f"API request timed out after {self.request_timeout}s")
            return self.simulate_vlm_response()
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"API request failed: {e}")
            return self.simulate_vlm_response()
        except Exception as e:
            self.get_logger().error(f"VLM query failed: {e}")
            return self.simulate_vlm_response()

    def query_rkllama_api_video(self, video_path: Optional[str] = None) -> str:
        """Query the Ollama-compatible VLM API with multiple frames (temporal context)"""
        try:
            # Instead of sending video file, extract frames and send as multiple images
            # Ollama/Cosmos supports up to 10 images per request
            frames = self.extract_frames_from_buffer()

            if not frames:
                self.get_logger().error("No frames available for multi-frame inference")
                return self.simulate_vlm_response()

            # Encode frames as base64 JPEG images
            frame_base64_list = []
            for frame in frames:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frame_base64_list.append(frame_base64)

            self.get_logger().debug(f"Sending {len(frame_base64_list)} frames to VLM")

            # Prepare the prompt for rover navigation with temporal context
            system_prompt = """You are a helpful assistant controlling an autonomous rover. Answer the question in the following format:
<think>
your reasoning process here
</think>

<answer>
your answer here
</answer>"""

            user_prompt = f"""Analyze these {len(frame_base64_list)} sequential frames (taken over {self.video_duration} seconds) from my indoor rover and decide how to move.

The frames are in chronological order showing the rover's recent view.

MOVEMENT RULES:
- linear_speed: 0.0 to 0.15 (meters/second forward) - use 0.10-0.15 when path is clear
- angular_speed: -0.3 to 0.3 (radians/second turning) - negative=left, positive=right
- When you see open floor/hallway ahead: move forward at 0.10-0.15 m/s
- When you see obstacles nearby: slow down to 0.05 or turn
- Look at the sequence to understand if the rover is moving and in what direction
- EXPLORE - prefer movement over standing still

In your <think> section, analyze what you see across the frame sequence and reason about the best movement.
In your <answer> section, provide ONLY this exact JSON format (no markdown):
{{"analysis": "what I see", "linear_speed": 0.10, "angular_speed": 0.0, "reasoning": "why"}}"""

            # Prepare Ollama API request
            api_url = f"{self.rkllama_url}/api/chat"

            # Send multiple frames in the images array - Ollama/Cosmos supports up to 10 images
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": frame_base64_list  # Multiple frames as separate images
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "repeat_penalty": 1.05,
                    "num_predict": 4096,
                    "num_ctx": self.num_ctx
                }
            }

            # Make API request
            self.get_logger().debug(f"Sending {len(frame_base64_list)}-frame request to {api_url}")
            response = requests.post(
                api_url,
                json=payload,
                timeout=self.request_timeout
            )

            if response.status_code != 200:
                self.get_logger().error(f"Ollama API error: {response.status_code} - {response.text[:200]}")
                return self.simulate_vlm_response()

            # Parse response
            result = response.json()

            # Extract the message content from Ollama-format response
            if "message" in result and "content" in result["message"]:
                content = result["message"]["content"]
                self.get_logger().debug(f"VLM response: {content}")
                return content
            else:
                self.get_logger().error(f"Unexpected API response format: {result}")
                return self.simulate_vlm_response()

        except requests.exceptions.Timeout:
            self.get_logger().error(f"Multi-frame API request timed out after {self.request_timeout}s")
            return self.simulate_vlm_response()
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Multi-frame API request failed: {e}")
            return self.simulate_vlm_response()
        except Exception as e:
            self.get_logger().error(f"VLM multi-frame query failed: {e}")
            return self.simulate_vlm_response()

    def simulate_vlm_response(self) -> str:
        """Generate a simulated VLM response for testing"""
        import random
        
        # Simple simulation logic
        linear_speed = random.uniform(0.0, self.max_linear_speed)
        angular_speed = random.uniform(-self.max_angular_speed, self.max_angular_speed)
        
        # Bias towards forward movement
        if random.random() > 0.3:
            angular_speed *= 0.3
        
        response = {
            "analysis": "Simulated rover view - testing VLM integration",
            "linear_speed": round(linear_speed, 2),
            "angular_speed": round(angular_speed, 2),
            "reasoning": "Simulation mode - random but safe movement"
        }
        
        return json.dumps(response)
    
    def parse_vlm_response(self, response: str) -> Twist:
        """Parse VLM response and convert to Twist command"""
        command = Twist()

        try:
            response = response.strip()
            self.get_logger().info(f"Raw response: {response[:300]}")

            # Try to find JSON in the response
            # Look for JSON object pattern
            json_start = response.find('{')
            json_end = response.rfind('}')

            if json_start >= 0 and json_end >= 0 and json_end > json_start:
                json_str = response[json_start:json_end+1]
                self.get_logger().debug(f"Extracted JSON: {json_str}")
            else:
                # Try markdown code blocks
                if "```json" in response:
                    start = response.find("```json") + 7
                    end = response.find("```", start)
                    json_str = response[start:end].strip()
                elif "```" in response:
                    start = response.find("```") + 3
                    end = response.find("```", start)
                    json_str = response[start:end].strip()
                else:
                    json_str = response

            # Try to parse JSON response
            data = json.loads(json_str)

            # Extract movement commands
            linear_speed = data.get('linear_speed', 0.0)
            angular_speed = data.get('angular_speed', 0.0)

            # Clamp speeds to safe limits
            linear_speed = max(0.0, min(self.max_linear_speed, float(linear_speed)))
            angular_speed = max(-self.max_angular_speed, min(self.max_angular_speed, float(angular_speed)))

            command.linear.x = linear_speed
            command.angular.z = angular_speed

            self.get_logger().info(f"Parsed successfully: linear={linear_speed:.2f}, angular={angular_speed:.2f}")

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().warn(f"Failed to parse VLM response: {e}")
            self.get_logger().warn(f"Response was: {response[:200]}")
            # Default to safe stop command
            command.linear.x = 0.0
            command.angular.z = 0.0

        return command
    
    def command_timer_callback(self):
        """Timer callback for publishing movement commands"""
        current_time = time.time()

        if self.control_mode == ControlMode.SYNCHRONIZED:
            # In synchronized mode, publish commands during control cycle
            if self.waiting_for_control_completion:
                # We're in the control application phase - publish the command
                self.cmd_vel_pub.publish(self.current_command)
            else:
                # Not in control phase - could be during inference
                # Publish current command (or stop if timeout exceeded)
                if current_time - self.last_command_time > self.command_timeout:
                    stop_command = Twist()
                    self.cmd_vel_pub.publish(stop_command)
                else:
                    self.cmd_vel_pub.publish(self.current_command)
        else:
            # TIME_BASED mode: original behavior
            # Check if command has timed out
            if current_time - self.last_command_time > self.command_timeout:
                # Stop the rover if no recent commands
                stop_command = Twist()
                self.cmd_vel_pub.publish(stop_command)
                return

            # Publish current command
            self.cmd_vel_pub.publish(self.current_command)
    
    def destroy_node(self):
        """Clean shutdown"""
        # Stop the rover
        stop_command = Twist()
        self.cmd_vel_pub.publish(stop_command)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    controller = VLMRoverController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            controller.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()