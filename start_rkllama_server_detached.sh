#!/usr/bin/env bash
set -e

# rkllama Server Startup Script (Docker Detached Mode)
# This script starts the rkllama server in background mode

MODELS_DIR="/home/ubuntu/models"
PORT=8080
CONTAINER_NAME="rkllama-server"

# Parse command line arguments
for arg in "$@"; do
  case "$arg" in
    --models-dir=*)
      MODELS_DIR="${arg#*=}" ;;
    --port=*)
      PORT="${arg#*=}" ;;
    --name=*)
      CONTAINER_NAME="${arg#*=}" ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Starts rkllama server in detached (background) mode"
      echo ""
      echo "Options:"
      echo "  --models-dir=PATH    Directory containing RKLLM model files (default: /home/ubuntu/models)"
      echo "  --port=PORT          Server port (default: 8080)"
      echo "  --name=NAME          Docker container name (default: rkllama-server)"
      echo "  --help               Show this help message"
      echo ""
      echo "Management commands:"
      echo "  docker logs -f $CONTAINER_NAME    # View logs"
      echo "  docker stop $CONTAINER_NAME       # Stop server"
      echo "  docker restart $CONTAINER_NAME    # Restart server"
      exit 0 ;;
  esac
done

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
  echo "ERROR: Models directory not found: $MODELS_DIR"
  echo "Please specify the correct directory with --models-dir=PATH"
  exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  echo "ERROR: Docker is not installed"
  echo "Please install Docker first: https://docs.docker.com/engine/install/"
  exit 1
fi

# Stop and remove existing container if it exists
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
  echo "Stopping existing container: $CONTAINER_NAME"
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

echo "Starting rkllama server in detached mode..."
echo "  Container name: $CONTAINER_NAME"
echo "  Models directory: $MODELS_DIR"
echo "  Port: $PORT"
echo "  Docker image: ghcr.io/notpunchnox/rkllama:main"
echo ""

# Start the Docker container in detached mode with auto-restart
docker run -d --privileged \
  --restart=unless-stopped \
  --name "$CONTAINER_NAME" \
  -p ${PORT}:8080 \
  -v "${MODELS_DIR}:/opt/rkllama/models" \
  ghcr.io/notpunchnox/rkllama:main

echo ""
echo "✓ Server started successfully!"
echo ""
echo "Management commands:"
echo "  docker logs -f $CONTAINER_NAME    # View logs"
echo "  docker stop $CONTAINER_NAME       # Stop server"
echo "  docker restart $CONTAINER_NAME    # Restart server"
echo "  docker rm $CONTAINER_NAME         # Remove container"
echo ""
echo "Test the server:"
echo "  curl http://localhost:${PORT}/api/tags"
