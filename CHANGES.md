# rkllama Integration Changes

## Summary

Updated the ROS2 rover VLM control system to use rkllama REST API server (Docker) instead of direct RKLLM library integration.

## What Changed

### 1. VLM Controller (`vlm_rover_controller.py`)
- **Before**: Used direct RKLLM Python library (`from rkllama.api import RKLLM`)
- **After**: Uses HTTP requests to rkllama server REST API
- **Benefits**:
  - No need to install rkllma Python package
  - Better separation of concerns
  - Can use remote rkllama servers
  - Easier debugging (view logs separately)
  - Ollama-compatible API

**Key Changes**:
- Removed RKLLM library dependency
- Added `requests` library for HTTP communication
- Converts images to base64-encoded JPEG
- Sends to `/api/chat` endpoint (Ollama format)
- Parses JSON responses with markdown code block handling
- Auto-fallback to simulation mode if server unavailable

### 2. Launch File (`vlm_control.launch.py`)
- **Before**: `vlm_model_path` parameter pointing to .rkllm file
- **After**: `rkllama_url` and `model_name` parameters

**New Parameters**:
- `rkllama_url`: Server URL (default: http://localhost:8080)
- `model_name`: Model name in rkllama (default: qwen2.5-vl-7b)
- `request_timeout`: HTTP timeout (default: 10.0s)

### 3. Startup Script (`start_vlm_control.sh`)
- **Major Feature**: Auto-manages Docker container lifecycle
- **Auto-starts** rkllama container if not running
- **Auto-stops** container on exit (configurable)
- Checks Docker availability
- Shows container status in output

**New Options**:
- `--models-dir=PATH`: Specify models directory
- `--no-auto-start-rkllama`: Don't auto-start container
- `--no-auto-stop-rkllama`: Keep container running after exit

### 4. New Scripts

**`start_rkllama_server.sh`**: Interactive Docker container start
- Manages Docker container lifecycle
- Interactive mode (see logs in terminal)
- Auto-cleanup of old containers

**`start_rkllama_server_detached.sh`**: Background Docker container start
- Detached mode (runs in background)
- Auto-restart on crash/reboot
- Shows management commands

### 5. Documentation (`RKLLAMA_SETUP.md`)
- Complete setup guide
- Docker-focused approach
- Testing procedures
- Troubleshooting tips
- API details

## Usage

### Simplest Way (One Command)
```bash
./start_vlm_control.sh
```
This automatically:
1. Starts rkllama Docker container
2. Launches rover with VLM control
3. Stops container on exit

### Manual Control
```bash
# Terminal 1: Start rkllama
./start_rkllama_server_detached.sh

# Terminal 2: Start rover
./start_vlm_control.sh --no-auto-start-rkllama --no-auto-stop-rkllama
```

## Benefits

1. **Simpler Setup**: No need to install rkllama Python package
2. **Docker Isolation**: Server runs in container with all dependencies
3. **Easier Testing**: Can test server independently
4. **Better Debugging**: Separate logs for server and ROS
5. **Flexibility**: Can run rkllama on different machine
6. **Auto-Management**: Script handles container lifecycle
7. **Production Ready**: Can run detached with auto-restart

## Migration Notes

If you were using the old direct RKLLM integration:

1. **No code changes needed** in your ROS workspace (already updated)
2. **Install Docker** if not already installed
3. **Use new startup script**: `./start_vlm_control.sh`
4. **Models location**: Still `/home/ubuntu/models/` (mounted into container)

## Files Modified

- `src/tractor_bringup/tractor_bringup/vlm_rover_controller.py`
- `src/tractor_bringup/launch/vlm_control.launch.py`
- `start_vlm_control.sh`

## Files Created

- `start_rkllama_server.sh`
- `start_rkllama_server_detached.sh`
- `RKLLAMA_SETUP.md`
- `CHANGES.md` (this file)

## Docker Image

Uses: `ghcr.io/notpunchnox/rkllama:main`

Container configuration:
- Privileged mode (for NPU access)
- Port 8080 exposed
- Models directory mounted from host
- Auto-restart policy (optional)

## Testing Checklist

- [ ] Docker installed and running
- [ ] Model file exists: `/home/ubuntu/models/Qwen2.5-VL-7B-Instruct-rk3588-1.2.1.rkllm`
- [ ] ROS workspace built: `colcon build --packages-select tractor_bringup`
- [ ] Test rkllama: `curl http://localhost:8080/api/tags`
- [ ] Test rover: `./start_vlm_control.sh --no-motor` (dry run)
- [ ] Full test: `./start_vlm_control.sh`

## Rollback

To revert to old direct RKLLM integration, restore from git:
```bash
git checkout HEAD -- src/tractor_bringup/tractor_bringup/vlm_rover_controller.py
git checkout HEAD -- src/tractor_bringup/launch/vlm_control.launch.py
git checkout HEAD -- start_vlm_control.sh
```

## Support

- rkllama repo: https://github.com/NotPunchnox/rkllama
- Docker docs: https://docs.docker.com/
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
