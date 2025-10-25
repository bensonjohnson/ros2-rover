#!/usr/bin/env python3
"""
Vision Language Model Rover Controller

This node uses an Ollama-compatible REST API server with vision language models
to control the rover based on visual input from the RealSense camera. The VLM
analyzes the camera feed and generates movement commands to navigate autonomously.

Compatible with: Ollama, rkllama, or any Ollama-compatible API
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
from typing import Optional
import json
import base64
import requests


class VLMRoverController(Node):
    def __init__(self):
        super().__init__('vlm_rover_controller')

        # Parameters
        self.declare_parameter('rkllama_url', 'https://ollama.gokickrocks.org')  # Ollama-compatible API URL
        self.declare_parameter('model_name', 'qwen2.5vl:7b')
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('max_linear_speed', 0.2)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('inference_interval', 2.0)  # seconds between VLM inferences
        self.declare_parameter('command_timeout', 5.0)  # seconds before stopping if no new command
        self.declare_parameter('simulation_mode', False)
        self.declare_parameter('request_timeout', 30.0)  # HTTP request timeout (higher for remote)

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

        # State
        self.bridge = CvBridge()
        self.current_image: Optional[np.ndarray] = None
        self.last_inference_time = 0.0
        self.last_command_time = 0.0
        self.current_command = Twist()
        self.inference_lock = threading.Lock()

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
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
    
    def compressed_image_callback(self, msg: CompressedImage):
        """Process incoming compressed camera images"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Compressed image conversion failed: {e}")
    
    def inference_timer_callback(self):
        """Timer callback for VLM inference"""
        current_time = time.time()
        
        # Check if it's time for a new inference
        if current_time - self.last_inference_time < self.inference_interval:
            return
        
        # Check if we have a current image
        if self.current_image is None:
            return
        
        # Run inference in a separate thread to avoid blocking
        inference_thread = threading.Thread(target=self.run_vlm_inference, args=(self.current_image.copy(),))
        inference_thread.daemon = True
        inference_thread.start()
        
        self.last_inference_time = current_time
    
    def run_vlm_inference(self, image: np.ndarray):
        """Run VLM inference on the current image"""
        with self.inference_lock:
            try:
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

                self.get_logger().info(f"VLM command: linear={command.linear.x:.2f}, angular={command.angular.z:.2f}")

            except Exception as e:
                self.get_logger().error(f"VLM inference failed: {e}")
    
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
            prompt = """You are controlling a rover with a camera. Analyze this image and provide movement commands.

Your task:
1. Look for obstacles, paths, and safe navigation areas
2. Decide on forward movement (0.0 to 0.2 m/s) and turning (-0.5 to 0.5 rad/s)
3. Avoid obstacles and walls
4. Prefer forward movement when path is clear
5. Turn to avoid obstacles or find better paths

Respond ONLY in JSON format (no additional text):
{
  "analysis": "brief description of what you see",
  "linear_speed": 0.0,
  "angular_speed": 0.0,
  "reasoning": "why you chose this command"
}"""

            # Prepare Ollama API request
            api_url = f"{self.rkllama_url}/api/chat"

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_base64]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }

            # Make API request
            self.get_logger().debug(f"Sending request to {api_url}")
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
            # The response might have markdown code blocks, extract JSON
            response = response.strip()

            # Try to find JSON in markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            # Try to parse JSON response
            data = json.loads(response)

            # Extract movement commands
            linear_speed = data.get('linear_speed', 0.0)
            angular_speed = data.get('angular_speed', 0.0)

            # Clamp speeds to safe limits
            linear_speed = max(0.0, min(self.max_linear_speed, float(linear_speed)))
            angular_speed = max(-self.max_angular_speed, min(self.max_angular_speed, float(angular_speed)))

            command.linear.x = linear_speed
            command.angular.z = angular_speed

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