#!/bin/bash

get_map_name() {
  local prompt=$1
  echo -n -e "PROMPT: ${prompt}: "
  read -r map_name
  echo "$map_name"
}

echo "Select option: n"
# This should hang because the prompt is captured
map_name=$(get_map_name "Enter new map name")
echo "Map name: $map_name"
