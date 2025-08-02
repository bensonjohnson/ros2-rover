#!/bin/bash
# Script to manually activate Nav2 lifecycle nodes

echo "Activating Nav2 Lifecycle Nodes..."

nodes=("slam_toolbox" "controller_server" "planner_server" "behavior_server" "bt_navigator" "velocity_smoother" "collision_monitor")

# First configure all nodes (transition id=1)
echo "Configuring nodes..."
for node in "${nodes[@]}"; do
    echo "  Configuring $node..."
    ros2 service call /$node/change_state lifecycle_msgs/srv/ChangeState "{transition: {id: 1}}" --timeout 5 >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "    ✓ $node configured"
    else
        echo "    ✗ $node configuration failed"
    fi
done

sleep 2

# Then activate all nodes (transition id=3)
echo "Activating nodes..."
for node in "${nodes[@]}"; do
    echo "  Activating $node..."
    ros2 service call /$node/change_state lifecycle_msgs/srv/ChangeState "{transition: {id: 3}}" --timeout 5 >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "    ✓ $node activated"
    else
        echo "    ✗ $node activation failed"
    fi
done

echo ""
echo "Checking final states..."
for node in "${nodes[@]}"; do
    state=$(ros2 service call /$node/get_state lifecycle_msgs/srv/GetState --timeout 2 2>/dev/null | grep -o "label='[^']*'" | cut -d"'" -f2)
    if [ "$state" = "active" ]; then
        echo "  ✓ $node: $state"
    else
        echo "  ⚠ $node: $state"
    fi
done

echo ""
echo "Nav2 activation complete!"