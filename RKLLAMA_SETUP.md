# RKLlama Integration Setup

This guide shows you how to run the ROS2 rover with rkllama server for VLM-based control.

## Overview

The system now uses **rkllama** (REST API server) instead of the direct RKLLM library. This provides:
- Better separation of concerns
- Easier debugging and monitoring
- Ability to use the VLM from multiple clients
- Ollama-compatible API

## Architecture

```
RealSense Camera → ROS2 VLM Controller → HTTP Request (base64 image)
                                              ↓
                                       rkllama Server (localhost:8080)
                                              ↓
                                       Qwen2.5-VL Model (RKLLM)
                                              ↓
                                       JSON Response (navigation commands)
                                              ↓
                                       Rover Motors
```

## Prerequisites

1. **Install Docker** (if not already installed):
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   # Log out and back in for group changes to take effect
   ```

2. **Verify model file exists**:
   ```bash
   ls -lh /home/ubuntu/models/Qwen2.5-VL-7B-Instruct-rk3588-1.2.1.rkllm
   ```

3. **Pull rkllama Docker image** (optional, will auto-pull on first run):
   ```bash
   docker pull ghcr.io/notpunchnox/rkllama:main
   ```

## Quick Start

### Simple Mode (Auto-Managed Docker Container)

The easiest way - just run one command:

```bash
cd /home/ubuntu/ros2-rover
./start_vlm_control.sh
```

This will:
1. **Automatically start** the rkllama Docker container if it's not running
2. **Launch** all rover systems (camera, motors, VLM, safety)
3. **Automatically stop** the rkllama container when you exit (Ctrl+C)

That's it! The script handles everything for you.

**Common Options:**
```bash
# Add teleop backup control
./start_vlm_control.sh --teleop

# Run without motors (dry run/testing)
./start_vlm_control.sh --no-motor

# Keep rkllama running after exit (for multiple tests)
./start_vlm_control.sh --no-auto-stop-rkllama

# Use existing rkllama container (don't auto-start)
./start_vlm_control.sh --no-auto-start-rkllama

# Custom models directory
./start_vlm_control.sh --models-dir=/path/to/models
```

### Manual Mode (Separate Docker Container)

If you prefer to manage the rkllama container separately:

**Terminal 1** - Start rkllama interactively:
```bash
cd /home/ubuntu/ros2-rover
./start_rkllama_server.sh
```

Or in background mode:
```bash
./start_rkllama_server_detached.sh
```

**Terminal 2** - Start ROS2 (don't auto-manage Docker):
```bash
./start_vlm_control.sh --no-auto-start-rkllama --no-auto-stop-rkllama
```

### Docker Management Commands

```bash
# View logs
docker logs -f rkllama-server

# Stop server
docker stop rkllama-server

# Remove container
docker rm rkllama-server

# Restart server
docker restart rkllama-server

# Check status
docker ps | grep rkllama
```

## Testing the Setup

### Test rkllama Server

```bash
# Check if server is running
curl http://localhost:8080/api/tags

# List available models (to get exact model name)
docker exec rkllama-server rkllama list

# Test with a simple prompt (no image)
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-VL-7B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-0.5",
    "prompt": "Hello, how are you?",
    "stream": false
  }'
```

### Monitor ROS2 Topics

```bash
# Watch VLM commands
ros2 topic echo /cmd_vel_vlm

# Monitor VLM responses
ros2 topic echo /vlm_response

# Check camera feed
ros2 topic hz /camera/camera/color/image_raw
```

## Configuration

### VLM Controller Parameters

Edit `src/tractor_bringup/launch/vlm_control.launch.py`:

```python
# rkllama server URL
"rkllama_url": "http://localhost:8080"

# Model name (must match model from 'rkllama list')
"model_name": "Qwen2.5-VL-7B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-0.5"

# Camera topic
"camera_topic": "/camera/camera/color/image_raw"

# Speed limits
"max_linear_speed": 0.15  # m/s
"max_angular_speed": 0.4  # rad/s

# Inference timing
"inference_interval": 1.5  # seconds between VLM calls
"request_timeout": 10.0    # HTTP request timeout
"command_timeout": 3.0     # stop if no new commands
```

## API Details

The VLM controller sends images to rkllama using the Ollama-compatible `/api/chat` endpoint:

```python
POST http://localhost:8080/api/chat
{
  "model": "Qwen2.5-VL-7B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-0.5",
  "messages": [
    {
      "role": "user",
      "content": "Analyze this image and provide navigation commands...",
      "images": ["<base64-encoded-jpeg>"]
    }
  ],
  "stream": false,
  "options": {
    "temperature": 0.3,
    "top_p": 0.9
  }
}
```

**Finding the model name:**
```bash
# Inside the container
docker exec rkllama-server rkllama list

# Output:
# Available models:
# - Qwen2.5-VL-7B-Instruct-rk3588-w8a8-opt-1-hybrid-ratio-0.5
```

Response format expected:
```json
{
  "message": {
    "content": "{\"analysis\": \"...\", \"linear_speed\": 0.1, \"angular_speed\": 0.0, \"reasoning\": \"...\"}"
  }
}
```

## Troubleshooting

### rkllama server won't start
- Check if Docker is installed: `docker --version`
- Verify Docker daemon is running: `docker ps`
- Check if models directory exists: `ls /home/ubuntu/models/`
- Check port 8080 is not in use: `sudo netstat -tlnp | grep 8080`
- View Docker logs: `docker logs rkllama-server`
- Try pulling image manually: `docker pull ghcr.io/notpunchnox/rkllama:main`

### VLM controller can't connect to rkllama
- Verify server is running: `curl http://localhost:8080/api/tags`
- Check logs: The VLM controller will print connection errors
- If connection fails, it falls back to simulation mode (random commands)

### Poor navigation performance
- Reduce inference interval for faster reactions (but higher load)
- Adjust temperature in the API request (lower = more consistent)
- Check camera feed quality: `ros2 run rqt_image_view rqt_image_view`
- Monitor inference time in rkllama debug logs

### Images not being sent
- Check camera topic: `ros2 topic hz /camera/camera/color/image_raw`
- Verify image encoding: Should be 640×480 and encode as JPEG base64
- Check rkllama logs for base64 decode errors
- View a sample: `ros2 run rqt_image_view rqt_image_view`

## Performance Tips

1. **Image size**:
   - Camera outputs 640×480 (307K pixels) - optimal for Qwen2.5-VL!
   - No resizing needed (Qwen2.5-VL range: 200K-1M pixels)
   - Change to 848×480 for wider FOV if needed
2. **Inference interval**: Default 1.5s is good for real-time navigation
3. **JPEG quality**: Set to 85% (good compression, minimal quality loss)
4. **Temperature**: 0.3 for consistent navigation (lower = more deterministic)
5. **FPS**: Camera runs at 15fps, but VLM only samples every 1.5s (efficient!)

## Files Modified

- `src/tractor_bringup/tractor_bringup/vlm_rover_controller.py` - Updated to use rkllama REST API
- `src/tractor_bringup/launch/vlm_control.launch.py` - New rkllama parameters
- `start_vlm_control.sh` - Updated arguments
- `start_rkllama_server.sh` - New script to start rkllama server
- `RKLLAMA_SETUP.md` - This file

## Next Steps

1. Start both servers and verify they communicate
2. Test with `--no-motor` first to verify VLM output
3. Add teleop backup for manual override: `./start_vlm_control.sh --teleop`
4. Monitor and tune parameters for your environment
5. Run rkllama in background mode for production:
   ```bash
   # Start detached
   docker run -d --privileged --restart=unless-stopped \
     --name rkllama-server \
     -p 8080:8080 \
     -v /home/ubuntu/models:/opt/rkllama/models \
     ghcr.io/notpunchnox/rkllama:main
   ```

## Running on Boot (Optional)

To start rkllama automatically on boot, the Docker container can use the `--restart=unless-stopped` flag (shown above), which will:
- Restart the container if it crashes
- Start automatically on system boot
- Not restart if you manually stop it

To stop auto-restart:
```bash
docker update --restart=no rkllama-server
```

## Reference

- rkllama repository: https://github.com/NotPunchnox/rkllama
- Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
