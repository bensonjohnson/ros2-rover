#!/usr/bin/env bash
set -e

# Unified RTAB-Map + Nav2 launcher with map management
# Usage:
#   ./start_mapping.sh [mode] [options]
# Modes:
#   mapping        # interactive map selection for mapping (default)
#   localization   # interactive map selection for localization
# Options:
#   --teleop       # Enable Xbox teleop
#   --no-auto      # Disable autonomous mapper
#   --no-motor     # Disable motor driver
#   --no-safety    # Disable safety monitor

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure ROS 2 + workspace environment is sourced
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
fi

if [ -f "install/setup.bash" ]; then
  source install/setup.bash
else
  echo -e "${YELLOW}Workspace not built yet; building tractor_bringup (fast)...${NC}"
  colcon build --packages-select tractor_bringup --cmake-args -DCMAKE_BUILD_TYPE=Release
  source install/setup.bash
fi

# Create maps directory if it doesn't exist
mkdir -p /home/ubuntu/maps

# Default values
MODE="mapping"
WITH_TELEOP=false
WITH_AUTO=true
WITH_MOTOR=true
WITH_SAFETY=true

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    mapping|localization)
      MODE="$1"
      shift
      ;;
    --teleop)
      WITH_TELEOP=true
      shift
      ;;
    --no-auto)
      WITH_AUTO=false
      shift
      ;;
    --no-motor)
      WITH_MOTOR=false
      shift
      ;;
    --no-safety)
      WITH_SAFETY=false
      shift
      ;;
    *)
      echo -e "${RED}Unknown argument: $1${NC}"
      exit 1
      ;;
  esac
done

# Function to show map selection menu
show_map_menu() {
  local mode=$1
  echo -e "${BLUE}=== RTAB-Map $mode Mode ===${NC}"
  echo -e "${BLUE}Available Maps:${NC}"
  
  # List available maps
  local maps=($(ls /home/ubuntu/maps/*.db 2>/dev/null | xargs -n 1 basename 2>/dev/null | sed 's/.db$//' || true))
  
  if [ ${#maps[@]} -eq 0 ]; then
    echo -e "${YELLOW}No existing maps found.${NC}"
  else
    for i in "${!maps[@]}"; do
      local idx=$((i + 1))
      local map_name="${maps[$i]}"
      local map_file="/home/ubuntu/maps/$map_name.db"
      local size=$(du -h "$map_file" 2>/dev/null | cut -f1 || echo "Unknown")
      local modified=$(stat -c %y "$map_file" 2>/dev/null | cut -d' ' -f1,2 || echo "Unknown")
      echo -e "${GREEN}${idx}.${NC} $map_name (${size}, $modified)"
    done
  fi
  
  echo ""
  echo -e "${BLUE}Options:${NC}"
  if [ "$mode" = "mapping" ]; then
    echo -e "${GREEN}n.${NC} Create new map"
  fi
  echo -e "${GREEN}q.${NC} Quit"
  echo ""
}

# Function to get map name input
get_map_name() {
  local prompt=$1
  local map_name
  
  while true; do
    echo -n -e "${YELLOW}${prompt}: ${NC}"
    read -r map_name
    
    # Validate map name
    if [ -z "$map_name" ]; then
      echo -e "${RED}Map name cannot be empty.${NC}"
      continue
    fi
    
    if [ ${#map_name} -gt 50 ]; then
      echo -e "${RED}Map name too long (max 50 characters).${NC}"
      continue
    fi
    
    # Check if map already exists (for new maps)
    if [ "$2" = "new" ] && [ -f "/home/ubuntu/maps/$map_name.db" ]; then
      echo -e "${RED}Map '$map_name' already exists.${NC}"
      continue
    fi
    
    echo "$map_name"
    return 0
  done
}

# Main logic
if [[ "$MODE" == "mapping" ]]; then
  while true; do
    show_map_menu "mapping"
    echo -n -e "${YELLOW}Select option: ${NC}"
    read -r choice
    
    case "$choice" in
      [0-9]*)
        # Resume existing map for mapping
        maps=($(ls /home/ubuntu/maps/*.db 2>/dev/null | xargs -n 1 basename 2>/dev/null | sed 's/.db$//' || true))
        if [ "$choice" -le "${#maps[@]}" ] && [ "$choice" -gt 0 ]; then
          map_name="${maps[$((choice-1))]}"
          echo -e "${GREEN}Resuming map: $map_name${NC}"
          
          DATABASE_PATH="/home/ubuntu/maps/$map_name.db"
          
          echo -e "${BLUE}Launching RTAB-Map mapping mode (resuming '$map_name')...${NC}"
          ros2 launch tractor_bringup rtabmap_nav2.launch.py \
            nav2_params_file:=`ros2 pkg prefix tractor_bringup`/share/tractor_bringup/config/nav2_params.yaml \
            with_teleop:=${WITH_TELEOP} with_autonomous_mapper:=${WITH_AUTO} with_motor:=${WITH_MOTOR} with_safety:=${WITH_SAFETY} \
            database_path:="${DATABASE_PATH}"
          break
        else
          echo -e "${RED}Invalid selection.${NC}"
        fi
        ;;
      n|N)
        # Create new map
        map_name=$(get_map_name "Enter new map name" "new")
        if [ $? -eq 0 ]; then
          echo -e "${GREEN}Creating new map: $map_name${NC}"
          
          # Create metadata file
          python3 -c "
import json
from datetime import datetime
metadata = {
    'name': '$map_name',
    'description': '',
    'created_at': datetime.now().isoformat(),
    'status': 'new',
    'environment': 'indoor'
}
with open('/home/ubuntu/maps/$map_name.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"
          
          DATABASE_PATH="/home/ubuntu/maps/$map_name.db"
          
          echo -e "${BLUE}Launching RTAB-Map mapping mode (new map '$map_name')...${NC}"
          ros2 launch tractor_bringup rtabmap_nav2.launch.py \
            nav2_params_file:=`ros2 pkg prefix tractor_bringup`/share/tractor_bringup/config/nav2_params.yaml \
            with_teleop:=${WITH_TELEOP} with_autonomous_mapper:=${WITH_AUTO} with_motor:=${WITH_MOTOR} with_safety:=${WITH_SAFETY} \
            database_path:="${DATABASE_PATH}"
          break
        fi
        ;;
      q|Q)
        echo -e "${YELLOW}Exiting.${NC}"
        exit 0
        ;;
      *)
        echo -e "${RED}Invalid option. Please try again.${NC}"
        ;;
    esac
  done
  
elif [[ "$MODE" == "localization" ]]; then
  while true; do
    show_map_menu "localization"
    echo -n -e "${YELLOW}Select option: ${NC}"
    read -r choice
    
    case "$choice" in
      [0-9]*)
        # Use existing map for localization
        maps=($(ls /home/ubuntu/maps/*.db 2>/dev/null | xargs -n 1 basename 2>/dev/null | sed 's/.db$//' || true))
        if [ "$choice" -le "${#maps[@]}" ] && [ "$choice" -gt 0 ]; then
          map_name="${maps[$((choice-1))]}"
          echo -e "${GREEN}Using map for localization: $map_name${NC}"
          
          DATABASE_PATH="/home/ubuntu/maps/$map_name.db"
          
          echo -e "${BLUE}Launching RTAB-Map localization mode (using '$map_name')...${NC}"
          ros2 launch tractor_bringup rtabmap_nav2.launch.py \
            nav2_params_file:=`ros2 pkg prefix tractor_bringup`/share/tractor_bringup/config/nav2_params_localization.yaml \
            with_teleop:=${WITH_TELEOP} with_autonomous_mapper:=${WITH_AUTO} with_motor:=${WITH_MOTOR} with_safety:=${WITH_SAFETY} \
            database_path:="${DATABASE_PATH}"
          break
        else
          echo -e "${RED}Invalid selection.${NC}"
        fi
        ;;
      q|Q)
        echo -e "${YELLOW}Exiting.${NC}"
        exit 0
        ;;
      *)
        echo -e "${RED}Invalid option. Please try again.${NC}"
        ;;
    esac
  done
  
else
  echo -e "${RED}Unknown mode: $MODE (use 'mapping' or 'localization')${NC}" >&2
  exit 1
fi
