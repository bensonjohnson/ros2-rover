#!/usr/bin/env bash
set -e

# Enhanced RTAB-Map + Nav2 launcher with map management
# Usage:
#   ./start_rtabmap_nav2_enhanced.sh mapping        # interactive map selection for mapping
#   ./start_rtabmap_nav2_enhanced.sh localization   # interactive map selection for localization

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure ROS 2 + workspace environment is sourced so pluginlib can find image_transport plugins
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

MODE=${1:-mapping}
WITH_TELEOP=false
WITH_AUTO=true
WITH_MOTOR=true
WITH_SAFETY=true

# Parse additional arguments
for arg in "$@"; do
  case "$arg" in
    --teleop)
      WITH_TELEOP=true ;;
    --no-autonomous)
      WITH_AUTO=false ;;
    --no-motor)
      WITH_MOTOR=false ;;
    --no-safety)
      WITH_SAFETY=false ;;
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
      local map_name="${maps[$i]}"
      local map_file="/home/ubuntu/maps/$map_name.db"
      local size=$(du -h "$map_file" 2>/dev/null | cut -f1 || echo "Unknown")
      local modified=$(stat -c %y "$map_file" 2>/dev/null | cut -d' ' -f1,2 || echo "Unknown")
      echo -e "${GREEN}$((i+1)).${NC} $map_name (${size}, $modified)"
    done
  fi
  
  echo ""
  echo -e "${BLUE}Options:${NC}"
  if [ "$mode" = "mapping" ]; then
    echo -e "${GREEN}n.${NC} Create new map"
    echo -e "${GREEN}l.${NC} List maps with details"
  elif [ "$mode" = "localization" ]; then
    echo -e "${GREEN}m.${NC} Manage maps (create/delete/assess)"
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
    
    # Check for invalid characters
    if [[ "$map_name" =~ [/:*?\"<>|] ]]; then
      echo -e "${RED}Map name contains invalid characters.${NC}"
      continue
    fi
    
    # Check if map already exists (for new maps)
    if [ "$2" = "new" ] && [ -f "/home/ubuntu/maps/$map_name.db" ]; then
      echo -e "${RED}Map '$map_name' already exists.${NC}"
      continue
    fi
    
    # Check if map exists (for existing maps)
    if [ "$2" = "existing" ] && [ ! -f "/home/ubuntu/maps/$map_name.db" ]; then
      echo -e "${RED}Map '$map_name' not found.${NC}"
      continue
    fi
    
    echo "$map_name"
    return 0
  done
}

# Function to launch map manager
launch_map_manager() {
  echo -e "${BLUE}Launching map manager...${NC}"
  python3 src/tractor_bringup/tractor_bringup/map_manager.py
}

# Function to assess map quality
assess_map() {
  local map_name=$1
  echo -e "${BLUE}Assessing map quality for '$map_name'...${NC}"
  
  # Use python to assess map
  python3 -c "
import sys
sys.path.append('src/tractor_bringup/tractor_bringup')
import rclpy
from map_manager import MapManager

rclpy.init()
manager = MapManager()
quality = manager.assess_map_quality('$map_name')

if 'error' in quality:
    print(f'Error: {quality[\"error\"]}')
else:
    print(f'Total Nodes: {quality[\"total_nodes\"]}')
    print(f'Bad Nodes: {quality[\"bad_nodes\"]}')
    print(f'Loop Closures: {quality[\"loop_closures\"]}')
    print(f'Quality Score: {quality[\"quality_score\"]}/100')
    print(f'Recommendation: {quality[\"recommendation\"]}')
    
    if quality['quality_issues']:
        print('Issues:')
        for issue in quality['quality_issues']:
            print(f'  - {issue}')

manager.destroy_node()
rclpy.shutdown()
"
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
          
          # Set environment variable for RTAB-Map database
          export RTABMAP_DATABASE_PATH="/home/ubuntu/maps/$map_name.db"
          
          echo -e "${BLUE}Launching RTAB-Map mapping mode (resuming '$map_name')...${NC}"
          ros2 launch tractor_bringup rtabmap_nav2.launch.py \
            nav2_params_file:=`ros2 pkg prefix tractor_bringup`/share/tractor_bringup/config/nav2_params.yaml \
            with_teleop:=${WITH_TELEOP} with_autonomous_mapper:=${WITH_AUTO} with_motor:=${WITH_MOTOR} with_safety:=${WITH_SAFETY}
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
          
          # Set environment variable for RTAB-Map database
          export RTABMAP_DATABASE_PATH="/home/ubuntu/maps/$map_name.db"
          
          echo -e "${BLUE}Launching RTAB-Map mapping mode (new map '$map_name')...${NC}"
          ros2 launch tractor_bringup rtabmap_nav2.launch.py \
            nav2_params_file:=`ros2 pkg prefix tractor_bringup`/share/tractor_bringup/config/nav2_params.yaml \
            with_teleop:=${WITH_TELEOP} with_autonomous_mapper:=${WITH_AUTO} with_motor:=${WITH_MOTOR} with_safety:=${WITH_SAFETY}
          break
        fi
        ;;
      l|L)
        # List maps with details
        echo -e "${BLUE}Detailed Map Information:${NC}"
        for db_file in /home/ubuntu/maps/*.db; do
          if [ -f "$db_file" ]; then
            map_name=$(basename "$db_file" .db)
            size=$(du -h "$db_file" 2>/dev/null | cut -f1 || echo "Unknown")
            modified=$(stat -c %y "$db_file" 2>/dev/null | cut -d' ' -f1,2 || echo "Unknown")
            echo -e "${GREEN}$map_name:${NC}"
            echo "  Size: $size"
            echo "  Modified: $modified"
            assess_map "$map_name"
            echo ""
          fi
        done
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
          
          # Assess map quality first
          echo -e "${BLUE}Checking map quality...${NC}"
          assess_map "$map_name"
          
          echo ""
          echo -n -e "${YELLOW}Continue with this map? (y/n): ${NC}"
          read -r confirm
          if [[ "$confirm" =~ ^[Yy]$ ]]; then
            # Set environment variable for RTAB-Map database
            export RTABMAP_DATABASE_PATH="/home/ubuntu/maps/$map_name.db"
            
            echo -e "${BLUE}Launching RTAB-Map localization mode (using '$map_name')...${NC}"
            ros2 launch tractor_bringup rtabmap_nav2.launch.py \
              nav2_params_file:=`ros2 pkg prefix tractor_bringup`/share/tractor_bringup/config/nav2_params_localization.yaml \
              with_teleop:=${WITH_TELEOP} with_autonomous_mapper:=${WITH_AUTO} with_motor:=${WITH_MOTOR} with_safety:=${WITH_SAFETY}
            break
          fi
        else
          echo -e "${RED}Invalid selection.${NC}"
        fi
        ;;
      m|M)
        # Map management
        launch_map_manager
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
