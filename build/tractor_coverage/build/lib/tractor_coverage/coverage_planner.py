#!/usr/bin/env python3

import math
import numpy as np
from typing import List, Tuple, Optional
from geometry_msgs.msg import Point, Polygon, Point32
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, LineString
from shapely.ops import unary_union
from shapely.affinity import rotate, translate


class CoveragePlanner:
    """
    Agricultural coverage path planner using boustrophedon (ox-turning) pattern.
    Optimized for lawn mowing and spraying operations.
    """
    
    def __init__(self):
        self.tool_width = 1.0  # meters
        self.overlap_ratio = 0.1  # 10% overlap
        self.turn_radius = 0.5  # minimum turning radius
        self.border_offset = 0.5  # offset from boundary
        
    def set_tool_parameters(self, width: float, overlap: float = 0.1):
        """Set tool width and overlap percentage"""
        self.tool_width = width
        self.overlap_ratio = overlap
        
    def set_vehicle_parameters(self, turn_radius: float, border_offset: float = 0.5):
        """Set vehicle turning radius and boundary offset"""
        self.turn_radius = turn_radius
        self.border_offset = border_offset
    
    def polygon_from_points(self, points: List[Point]) -> ShapelyPolygon:
        """Convert ROS Point list to Shapely Polygon"""
        coords = [(p.x, p.y) for p in points]
        return ShapelyPolygon(coords)
    
    def plan_coverage_path(self, boundary_points: List[Point], 
                          obstacles: List[List[Point]] = None) -> List[Point]:
        """
        Plan complete coverage path with perimeter + boustrophedon fill
        
        Args:
            boundary_points: List of boundary points (should form closed polygon)
            obstacles: List of obstacle polygons (each as list of points)
            
        Returns:
            List of waypoints for complete coverage
        """
        # Convert to Shapely polygon
        boundary = self.polygon_from_points(boundary_points)
        
        # Handle obstacles
        work_area = boundary
        if obstacles:
            obstacle_polygons = [self.polygon_from_points(obs) for obs in obstacles]
            obstacle_union = unary_union(obstacle_polygons)
            work_area = boundary.difference(obstacle_union)
        
        # Plan perimeter path (optional - for edge trimming)
        perimeter_path = self.plan_perimeter_path(work_area)
        
        # Plan fill pattern
        fill_path = self.plan_boustrophedon_path(work_area)
        
        # Combine paths
        complete_path = perimeter_path + fill_path
        
        # Convert back to ROS Points
        return [Point(x=p[0], y=p[1], z=0.0) for p in complete_path]
    
    def plan_perimeter_path(self, polygon: ShapelyPolygon) -> List[Tuple[float, float]]:
        """Plan perimeter path around the boundary"""
        # Create inset polygon for perimeter
        inset_polygon = polygon.buffer(-self.border_offset)
        
        if inset_polygon.is_empty:
            return []
        
        # Extract coordinates (handle MultiPolygon case)
        if hasattr(inset_polygon, 'geoms'):
            # MultiPolygon - take largest
            largest = max(inset_polygon.geoms, key=lambda p: p.area)
            coords = list(largest.exterior.coords)
        else:
            coords = list(inset_polygon.exterior.coords)
        
        return coords[:-1]  # Remove duplicate last point
    
    def plan_boustrophedon_path(self, polygon: ShapelyPolygon) -> List[Tuple[float, float]]:
        """
        Plan boustrophedon (back-and-forth) coverage pattern
        """
        # Get polygon bounds
        minx, miny, maxx, maxy = polygon.bounds
        
        # Calculate effective tool width with overlap
        effective_width = self.tool_width * (1 - self.overlap_ratio)
        
        # Create work area with border offset
        work_poly = polygon.buffer(-self.border_offset)
        
        if work_poly.is_empty:
            return []
        
        # Handle MultiPolygon
        if hasattr(work_poly, 'geoms'):
            work_poly = max(work_poly.geoms, key=lambda p: p.area)
        
        # Determine optimal sweep direction (minimize turns)
        sweep_angle = self.find_optimal_sweep_direction(work_poly)
        
        # Generate sweep lines
        path_points = []
        
        # Rotate polygon to align with sweep direction
        rotated_poly = rotate(work_poly, -math.degrees(sweep_angle), origin='centroid')
        rot_minx, rot_miny, rot_maxx, rot_maxy = rotated_poly.bounds
        
        # Generate parallel lines
        y = rot_miny + effective_width / 2
        line_index = 0
        
        while y <= rot_maxy:
            # Create sweep line
            line = LineString([(rot_minx - 1, y), (rot_maxx + 1, y)])
            
            # Find intersections with work area
            intersections = rotated_poly.intersection(line)
            
            if intersections.is_empty:
                y += effective_width
                continue
            
            # Extract line segments
            segments = []
            if hasattr(intersections, 'geoms'):
                for geom in intersections.geoms:
                    if geom.geom_type == 'LineString':
                        segments.append(geom)
            elif intersections.geom_type == 'LineString':
                segments.append(intersections)
            
            # Process segments for this sweep line
            for segment in segments:
                coords = list(segment.coords)
                
                # Alternate direction for boustrophedon pattern
                if line_index % 2 == 1:
                    coords.reverse()
                
                # Add turn at end of each line (except first)
                if path_points and coords:
                    # Add turn waypoints
                    turn_points = self.generate_turn_waypoints(
                        path_points[-1], coords[0], self.turn_radius
                    )
                    path_points.extend(turn_points)
                
                path_points.extend(coords)
            
            y += effective_width
            line_index += 1
        
        # Rotate points back to original orientation
        final_points = []
        for point in path_points:
            # Create point at origin
            p = ShapelyPoint(point[0], point[1])
            # Rotate back
            rotated_back = rotate(p, math.degrees(sweep_angle), origin=work_poly.centroid)
            final_points.append((rotated_back.x, rotated_back.y))
        
        return final_points
    
    def find_optimal_sweep_direction(self, polygon: ShapelyPolygon) -> float:
        """
        Find optimal sweep direction to minimize number of turns.
        Returns angle in radians.
        """
        # Test multiple angles and choose the one with minimum bounding box width
        best_angle = 0
        min_width = float('inf')
        
        for angle_deg in range(0, 180, 15):  # Test every 15 degrees
            angle_rad = math.radians(angle_deg)
            rotated = rotate(polygon, -angle_deg, origin='centroid')
            _, _, width, _ = rotated.bounds
            
            if width < min_width:
                min_width = width
                best_angle = angle_rad
        
        return best_angle
    
    def generate_turn_waypoints(self, start: Tuple[float, float], 
                               end: Tuple[float, float], 
                               radius: float) -> List[Tuple[float, float]]:
        """
        Generate smooth turn waypoints between two points.
        Uses circular arc approximation.
        """
        if not start or not end:
            return []
        
        # Calculate turn direction and arc
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 2 * radius:
            # Turn too tight, use simple intermediate point
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            return [(mid_x, mid_y)]
        
        # Generate arc waypoints
        num_points = max(3, int(distance / (radius * 0.5)))
        waypoints = []
        
        for i in range(1, num_points):
            t = i / num_points
            # Simple bezier curve approximation
            x = start[0] + t * dx
            y = start[1] + t * dy
            # Add slight curve
            curve_offset = radius * math.sin(t * math.pi) * 0.5
            perp_x = -dy / distance if distance > 0 else 0
            perp_y = dx / distance if distance > 0 else 0
            x += curve_offset * perp_x
            y += curve_offset * perp_y
            waypoints.append((x, y))
        
        return waypoints
    
    def optimize_path(self, waypoints: List[Point]) -> List[Point]:
        """
        Optimize path by removing redundant points and smoothing corners
        """
        if len(waypoints) < 3:
            return waypoints
        
        optimized = [waypoints[0]]  # Always keep first point
        
        for i in range(1, len(waypoints) - 1):
            prev_point = optimized[-1]
            curr_point = waypoints[i]
            next_point = waypoints[i + 1]
            
            # Check if current point is necessary (collinearity test)
            if not self.is_collinear(prev_point, curr_point, next_point, tolerance=0.1):
                optimized.append(curr_point)
        
        optimized.append(waypoints[-1])  # Always keep last point
        return optimized
    
    def is_collinear(self, p1: Point, p2: Point, p3: Point, tolerance: float = 0.1) -> bool:
        """Check if three points are approximately collinear"""
        # Calculate cross product to check collinearity
        cross_product = abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x))
        return cross_product < tolerance
    
    def calculate_coverage_area(self, boundary_points: List[Point], 
                               obstacles: List[List[Point]] = None) -> float:
        """Calculate total area to be covered"""
        boundary = self.polygon_from_points(boundary_points)
        work_area = boundary
        
        if obstacles:
            obstacle_polygons = [self.polygon_from_points(obs) for obs in obstacles]
            obstacle_union = unary_union(obstacle_polygons)
            work_area = boundary.difference(obstacle_union)
        
        # Account for border offset
        effective_area = work_area.buffer(-self.border_offset)
        return effective_area.area if not effective_area.is_empty else 0.0
    
    def estimate_coverage_time(self, path_length: float, work_speed: float = 1.0) -> float:
        """Estimate time to complete coverage in seconds"""
        return path_length / work_speed
    
    def validate_boundary(self, points: List[Point]) -> bool:
        """Validate that boundary forms a valid simple polygon"""
        if len(points) < 3:
            return False
        
        try:
            polygon = self.polygon_from_points(points)
            return polygon.is_valid and not polygon.is_empty
        except:
            return False