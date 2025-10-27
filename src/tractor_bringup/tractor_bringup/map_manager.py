#!/usr/bin/env python3
"""
RTAB-Map Map Management System
Handles creating, listing, resuming, and deleting maps with manual naming.
Stores maps in /home/ubuntu/maps with validation and quality assessment.
"""

import os
import sys
import sqlite3
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class MapManager(Node):
    def __init__(self):
        super().__init__('map_manager')
        
        # Configuration
        self.maps_dir = Path("/home/ubuntu/maps")
        self.maps_dir.mkdir(exist_ok=True)
        
        # Publishers
        self.status_pub = self.create_publisher(String, 'map_manager_status', 10)
        
        self.get_logger().info(f"Map Manager initialized. Maps directory: {self.maps_dir}")

    def list_maps(self) -> List[Dict[str, str]]:
        """List all available maps with metadata"""
        maps = []
        
        for map_file in self.maps_dir.glob("*.db"):
            map_name = map_file.stem
            metadata = self.get_map_metadata(map_file)
            
            # Get database statistics
            try:
                conn = sqlite3.connect(str(map_file))
                cursor = conn.cursor()
                
                # Get node count
                cursor.execute("SELECT COUNT(*) FROM Node")
                node_count = cursor.fetchone()[0]
                
                # Get map statistics
                cursor.execute("SELECT MAX(value) FROM Parameter WHERE name='Grid/RangeMax'")
                max_range = cursor.fetchone()[0] or 0
                
                # Get latest timestamp
                cursor.execute("SELECT MAX(stamp) FROM Node")
                latest_time = cursor.fetchone()[0] or 0
                
                conn.close()
                
                maps.append({
                    'name': map_name,
                    'path': str(map_file),
                    'size_mb': round(map_file.stat().st_size / (1024*1024), 2),
                    'node_count': node_count,
                    'max_range': max_range,
                    'latest_time': latest_time,
                    'metadata': metadata
                })
                
            except Exception as e:
                self.get_logger().warn(f"Could not read map {map_name}: {e}")
                maps.append({
                    'name': map_name,
                    'path': str(map_file),
                    'size_mb': round(map_file.stat().st_size / (1024*1024), 2),
                    'node_count': 0,
                    'error': str(e)
                })
        
        return sorted(maps, key=lambda x: x['name'])

    def get_map_metadata(self, map_file: Path) -> Dict[str, str]:
        """Get metadata from JSON file if it exists"""
        metadata_file = map_file.with_suffix('.json')
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_map_metadata(self, map_name: str, metadata: Dict[str, str]):
        """Save metadata for a map"""
        metadata_file = self.maps_dir / f"{map_name}.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Could not save metadata for {map_name}: {e}")

    def validate_map_name(self, name: str) -> Tuple[bool, str]:
        """Validate map name for safety and uniqueness"""
        if not name:
            return False, "Map name cannot be empty"
        
        if len(name) > 50:
            return False, "Map name too long (max 50 characters)"
        
        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in name for char in invalid_chars):
            return False, f"Map name contains invalid characters: {invalid_chars}"
        
        # Check if already exists
        existing_map = self.maps_dir / f"{name}.db"
        if existing_map.exists():
            return False, f"Map '{name}' already exists"
        
        return True, "Valid map name"

    def create_new_map(self, name: str, description: str = "") -> bool:
        """Create a new map entry"""
        is_valid, message = self.validate_map_name(name)
        if not is_valid:
            self.get_logger().error(f"Invalid map name: {message}")
            return False
        
        # Create metadata
        metadata = {
            'name': name,
            'description': description,
            'created_at': str(self.get_clock().now().to_msg()),
            'status': 'new',
            'environment': 'indoor'
        }
        
        self.save_map_metadata(name, metadata)
        
        status_msg = String()
        status_msg.data = f"Created new map: {name}"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(f"Created new map: {name}")
        return True

    def assess_map_quality(self, map_name: str) -> Dict[str, any]:
        """Assess the quality of a map based on various metrics"""
        map_file = self.maps_dir / f"{map_name}.db"
        if not map_file.exists():
            return {'error': 'Map not found'}
        
        try:
            conn = sqlite3.connect(str(map_file))
            cursor = conn.cursor()
            
            # Basic statistics
            cursor.execute("SELECT COUNT(*) FROM Node")
            total_nodes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM Node WHERE pose = '0'")
            bad_nodes = cursor.fetchone()[0]
            
            # Loop closures
            cursor.execute("SELECT COUNT(*) FROM Link WHERE type = 3")
            loop_closures = cursor.fetchone()[0]
            
            # Map coverage (simplified)
            cursor.execute("SELECT MIN(value), MAX(value) FROM Parameter WHERE name='Grid/RangeMax'")
            range_info = cursor.fetchone()
            
            conn.close()
            
            # Quality assessment
            quality_score = 0
            quality_issues = []
            
            if total_nodes < 50:
                quality_issues.append("Too few nodes for good map")
                quality_score += 20
            elif total_nodes < 200:
                quality_score += 40
            else:
                quality_score += 60
            
            if loop_closures > 0:
                quality_score += 20
            else:
                quality_issues.append("No loop closures detected")
            
            if bad_nodes / total_nodes < 0.1:
                quality_score += 20
            else:
                quality_issues.append("High percentage of bad nodes")
            
            return {
                'total_nodes': total_nodes,
                'bad_nodes': bad_nodes,
                'loop_closures': loop_closures,
                'quality_score': min(100, quality_score),
                'quality_issues': quality_issues,
                'recommendation': self._get_recommendation(quality_score, total_nodes, loop_closures)
            }
            
        except Exception as e:
            return {'error': f'Could not assess map: {e}'}

    def _get_recommendation(self, score: int, nodes: int, loops: int) -> str:
        """Get recommendation based on map quality"""
        if score >= 80:
            return "Excellent map, ready for navigation"
        elif score >= 60:
            return "Good map, consider adding more loop closures"
        elif score >= 40:
            return "Fair map, needs more coverage and loop closures"
        else:
            return "Poor map, continue mapping or restart"

    def delete_map(self, name: str) -> bool:
        """Delete a map and its metadata"""
        map_file = self.maps_dir / f"{name}.db"
        metadata_file = self.maps_dir / f"{name}.json"
        
        if not map_file.exists():
            self.get_logger().error(f"Map '{name}' does not exist")
            return False
        
        try:
            if map_file.exists():
                map_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            
            status_msg = String()
            status_msg.data = f"Deleted map: {name}"
            self.status_pub.publish(status_msg)
            
            self.get_logger().info(f"Deleted map: {name}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to delete map {name}: {e}")
            return False

    def setup_rtabmap_database(self, map_name: str) -> bool:
        """Setup RTAB-Map to use the specified map database"""
        map_file = self.maps_dir / f"{map_name}.db"
        
        if not map_file.exists():
            self.get_logger().error(f"Map database '{map_name}' does not exist")
            return False
        
        # Set environment variable for RTAB-Map database path
        os.environ['RTABMAP_DATABASE_PATH'] = str(map_file)
        
        status_msg = String()
        status_msg.data = f"Setup RTAB-Map with database: {map_name}"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(f"RTAB-Map will use database: {map_file}")
        return True


def interactive_map_manager():
    """Interactive command-line interface for map management"""
    rclpy.init()
    manager = MapManager()
    
    try:
        while True:
            print("\n" + "="*50)
            print("RTAB-Map Map Manager")
            print("="*50)
            print("1. List available maps")
            print("2. Create new map")
            print("3. Assess map quality")
            print("4. Delete map")
            print("5. Exit")
            print("="*50)
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\nAvailable Maps:")
                print("-" * 80)
                maps = manager.list_maps()
                if not maps:
                    print("No maps found.")
                else:
                    for i, map_info in enumerate(maps, 1):
                        print(f"{i}. {map_info['name']}")
                        print(f"   Size: {map_info['size_mb']} MB")
                        print(f"   Nodes: {map_info.get('node_count', 'N/A')}")
                        if 'error' in map_info:
                            print(f"   Error: {map_info['error']}")
                        print()
            
            elif choice == '2':
                while True:
                    name = input("Enter map name (or 'back' to return): ").strip()
                    if name.lower() == 'back':
                        break
                    
                    is_valid, message = manager.validate_map_name(name)
                    if not is_valid:
                        print(f"Invalid name: {message}")
                        continue
                    
                    description = input("Enter description (optional): ").strip()
                    
                    if manager.create_new_map(name, description):
                        print(f"Map '{name}' created successfully!")
                        print(f"Database will be: {manager.maps_dir / f'{name}.db'}")
                        break
                    else:
                        print("Failed to create map.")
            
            elif choice == '3':
                maps = manager.list_maps()
                if not maps:
                    print("No maps found.")
                    continue
                
                print("\nSelect map to assess:")
                for i, map_info in enumerate(maps, 1):
                    print(f"{i}. {map_info['name']}")
                
                try:
                    choice_idx = int(input("Enter map number: ").strip()) - 1
                    if 0 <= choice_idx < len(maps):
                        map_name = maps[choice_idx]['name']
                        quality = manager.assess_map_quality(map_name)
                        
                        print(f"\nQuality Assessment for '{map_name}':")
                        print("-" * 40)
                        if 'error' in quality:
                            print(f"Error: {quality['error']}")
                        else:
                            print(f"Total Nodes: {quality['total_nodes']}")
                            print(f"Bad Nodes: {quality['bad_nodes']}")
                            print(f"Loop Closures: {quality['loop_closures']}")
                            print(f"Quality Score: {quality['quality_score']}/100")
                            print(f"Recommendation: {quality['recommendation']}")
                            
                            if quality['quality_issues']:
                                print("Issues:")
                                for issue in quality['quality_issues']:
                                    print(f"  - {issue}")
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            
            elif choice == '4':
                maps = manager.list_maps()
                if not maps:
                    print("No maps found.")
                    continue
                
                print("\nSelect map to delete:")
                for i, map_info in enumerate(maps, 1):
                    print(f"{i}. {map_info['name']}")
                
                try:
                    choice_idx = int(input("Enter map number: ").strip()) - 1
                    if 0 <= choice_idx < len(maps):
                        map_name = maps[choice_idx]['name']
                        confirm = input(f"Are you sure you want to delete '{map_name}'? (yes/no): ").strip().lower()
                        if confirm == 'yes':
                            if manager.delete_map(map_name):
                                print(f"Map '{map_name}' deleted successfully.")
                            else:
                                print("Failed to delete map.")
                        else:
                            print("Deletion cancelled.")
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            
            elif choice == '5':
                print("Exiting Map Manager.")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\nExiting Map Manager.")
    
    finally:
        manager.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    interactive_map_manager()
