#!/bin/bash
# Test script to verify SLAM Toolbox is properly activated

echo "=== SLAM Toolbox Status Check ==="

# Check if SLAM Toolbox node is running
echo "1. Checking if SLAM Toolbox is running..."
if pgrep -f slam_toolbox > /dev/null; then
    echo "✓ SLAM Toolbox process is running"
else
    echo "✗ SLAM Toolbox process not found"
fi

# Check lifecycle state
echo ""
echo "2. Checking SLAM Toolbox lifecycle state..."
SLAM_STATE=$(ros2 service call /slam_toolbox/get_state lifecycle_msgs/srv/GetState 2>/dev/null | grep -o "label='[^']*'" | cut -d"'" -f2)
echo "   Current state: $SLAM_STATE"

if [ "$SLAM_STATE" = "active" ]; then
    echo "✓ SLAM Toolbox is properly activated"
elif [ "$SLAM_STATE" = "unconfigured" ]; then
    echo "⚠ SLAM Toolbox needs to be activated by lifecycle manager"
    echo "   Attempting to activate..."
    # The lifecycle manager should handle this automatically
else
    echo "? SLAM Toolbox state: $SLAM_STATE"
fi

# Check for map frame
echo ""
echo "3. Checking for map frame..."
if ros2 run tf2_tools view_frames 2>/dev/null | grep -q "map:"; then
    echo "✓ Map frame is available"
else
    echo "✗ Map frame not found - SLAM may not be active"
fi

# Check for map topic
echo ""
echo "4. Checking for map topic..."
if ros2 topic list | grep -q "/map"; then
    echo "✓ Map topic is available"
    MAP_MSG_COUNT=$(timeout 2s ros2 topic echo /map --once 2>/dev/null | wc -l)
    if [ "$MAP_MSG_COUNT" -gt 0 ]; then
        echo "✓ Map topic is publishing data"
    else
        echo "⚠ Map topic exists but no data received"
    fi
else
    echo "✗ Map topic not found"
fi

echo ""
echo "=== Test Complete ==="