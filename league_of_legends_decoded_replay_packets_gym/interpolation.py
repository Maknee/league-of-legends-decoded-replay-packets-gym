#!/usr/bin/env python3
"""
Position Interpolation System

This module provides functionality to interpolate positions of champions and objects
between waypoints and time steps for smooth position tracking and analysis.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from .types import Position
from .packets import PacketType, WaypointGroup, WaypointGroupWithSpeed


@dataclass
class InterpolatedPosition:
    """Position with interpolation metadata"""
    position: Position
    timestamp: float
    confidence: float  # 0.0 to 1.0, higher means more accurate
    is_interpolated: bool  # True if position was calculated, False if from actual waypoint
    entity_id: int


@dataclass
class MovementVector:
    """Movement vector between two positions"""
    start_pos: Position
    end_pos: Position
    start_time: float
    end_time: float
    speed: float  # units per second
    
    def get_position_at_time(self, target_time: float) -> Position:
        """Get interpolated position at specific time"""
        if target_time <= self.start_time:
            return self.start_pos
        elif target_time >= self.end_time:
            return self.end_pos
        
        # Linear interpolation
        duration = self.end_time - self.start_time
        progress = (target_time - self.start_time) / duration
        
        x = self.start_pos.x + (self.end_pos.x - self.start_pos.x) * progress
        z = self.start_pos.z + (self.end_pos.z - self.start_pos.z) * progress
        
        return Position(x=x, z=z)


class PositionInterpolator:
    """Main interpolation system for entity positions"""
    
    def __init__(self):
        self.entity_waypoints: Dict[int, List[Tuple[float, Position]]] = {}  # entity_id -> [(time, position)]
        self.movement_vectors: Dict[int, List[MovementVector]] = {}  # entity_id -> movement vectors
        self.last_known_positions: Dict[int, InterpolatedPosition] = {}  # entity_id -> last position
        
    def add_waypoint_group(self, waypoint_packet: Union[WaypointGroup, WaypointGroupWithSpeed]) -> None:
        """Add waypoints from a WaypointGroup packet"""
        timestamp = waypoint_packet.time
        
        for entity_id, positions in waypoint_packet.waypoints.items():
            if entity_id not in self.entity_waypoints:
                self.entity_waypoints[entity_id] = []
                self.movement_vectors[entity_id] = []
            
            # Add all waypoints with the same timestamp
            for pos in positions:
                self.entity_waypoints[entity_id].append((timestamp, pos))
            
            # Update movement vectors
            self._update_movement_vectors(entity_id)
    
    def _update_movement_vectors(self, entity_id: int) -> None:
        """Update movement vectors for an entity"""
        waypoints = self.entity_waypoints[entity_id]
        if len(waypoints) < 2:
            return
        
        # Clear existing vectors and recalculate
        self.movement_vectors[entity_id] = []
        
        # Sort waypoints by time
        waypoints.sort(key=lambda x: x[0])
        
        # Create movement vectors between consecutive waypoints
        for i in range(len(waypoints) - 1):
            start_time, start_pos = waypoints[i]
            end_time, end_pos = waypoints[i + 1]
            
            # Calculate speed
            distance = start_pos.distance_to(end_pos)
            duration = end_time - start_time
            speed = distance / duration if duration > 0 else 0
            
            vector = MovementVector(
                start_pos=start_pos,
                end_pos=end_pos,
                start_time=start_time,
                end_time=end_time,
                speed=speed
            )
            
            self.movement_vectors[entity_id].append(vector)
    
    def get_position_at_time(self, entity_id: int, target_time: float) -> Optional[InterpolatedPosition]:
        """Get interpolated position for entity at specific time"""
        if entity_id not in self.entity_waypoints:
            return None
        
        waypoints = self.entity_waypoints[entity_id]
        if not waypoints:
            return None
        
        # Sort waypoints by time
        waypoints.sort(key=lambda x: x[0])
        
        # Check if we have exact waypoint at target time
        for timestamp, pos in waypoints:
            if abs(timestamp - target_time) < 0.001:  # Very close to exact time
                return InterpolatedPosition(
                    position=pos,
                    timestamp=target_time,
                    confidence=1.0,
                    is_interpolated=False,
                    entity_id=entity_id
                )
        
        # Find appropriate movement vector for interpolation
        vectors = self.movement_vectors[entity_id]
        for vector in vectors:
            if vector.start_time <= target_time <= vector.end_time:
                interpolated_pos = vector.get_position_at_time(target_time)
                
                # Calculate confidence based on how close we are to actual waypoints
                time_to_start = abs(target_time - vector.start_time)
                time_to_end = abs(target_time - vector.end_time)
                duration = vector.end_time - vector.start_time
                
                # Confidence is higher when closer to actual waypoints
                min_distance_to_waypoint = min(time_to_start, time_to_end)
                confidence = max(0.1, 1.0 - (min_distance_to_waypoint / (duration / 2)))
                
                return InterpolatedPosition(
                    position=interpolated_pos,
                    timestamp=target_time,
                    confidence=confidence,
                    is_interpolated=True,
                    entity_id=entity_id
                )
        
        # If no vector found, try to extrapolate from closest waypoint
        if waypoints:
            # Find closest waypoint
            closest_waypoint = min(waypoints, key=lambda x: abs(x[0] - target_time))
            closest_time, closest_pos = closest_waypoint
            
            # Low confidence for extrapolation
            time_diff = abs(target_time - closest_time)
            confidence = max(0.05, 0.5 * math.exp(-time_diff / 10.0))  # Exponential decay
            
            return InterpolatedPosition(
                position=closest_pos,
                timestamp=target_time,
                confidence=confidence,
                is_interpolated=True,
                entity_id=entity_id
            )
        
        return None
    
    def get_all_positions_at_time(self, target_time: float) -> Dict[int, InterpolatedPosition]:
        """Get interpolated positions for all entities at specific time"""
        positions = {}
        for entity_id in self.entity_waypoints:
            pos = self.get_position_at_time(entity_id, target_time)
            if pos:
                positions[entity_id] = pos
        return positions
    
    def get_entities_in_radius(self, center: Position, radius: float, target_time: float) -> List[InterpolatedPosition]:
        """Get all entities within radius of center at specific time"""
        all_positions = self.get_all_positions_at_time(target_time)
        entities_in_radius = []
        
        for pos_data in all_positions.values():
            distance = center.distance_to(pos_data.position)
            if distance <= radius:
                entities_in_radius.append(pos_data)
        
        return entities_in_radius
    
    def get_movement_speed(self, entity_id: int, target_time: float) -> Optional[float]:
        """Get estimated movement speed for entity at specific time"""
        if entity_id not in self.movement_vectors:
            return None
        
        vectors = self.movement_vectors[entity_id]
        for vector in vectors:
            if vector.start_time <= target_time <= vector.end_time:
                return vector.speed
        
        return None
    
    def get_entity_trajectory(self, entity_id: int, start_time: float, end_time: float, 
                            time_step: float = 0.1) -> List[InterpolatedPosition]:
        """Get trajectory of entity over time period"""
        trajectory = []
        current_time = start_time
        
        while current_time <= end_time:
            pos = self.get_position_at_time(entity_id, current_time)
            if pos:
                trajectory.append(pos)
            current_time += time_step
        
        return trajectory
    
    def clear_old_data(self, cutoff_time: float) -> None:
        """Clear waypoint data older than cutoff_time to save memory"""
        for entity_id in list(self.entity_waypoints.keys()):
            # Keep waypoints after cutoff time
            new_waypoints = [(t, p) for t, p in self.entity_waypoints[entity_id] if t >= cutoff_time]
            
            if new_waypoints:
                self.entity_waypoints[entity_id] = new_waypoints
                self._update_movement_vectors(entity_id)
            else:
                # Remove entity if no recent waypoints
                del self.entity_waypoints[entity_id]
                if entity_id in self.movement_vectors:
                    del self.movement_vectors[entity_id]
                if entity_id in self.last_known_positions:
                    del self.last_known_positions[entity_id]
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get interpolation system statistics"""
        total_entities = len(self.entity_waypoints)
        total_waypoints = sum(len(waypoints) for waypoints in self.entity_waypoints.values())
        total_vectors = sum(len(vectors) for vectors in self.movement_vectors.values())
        
        avg_waypoints_per_entity = total_waypoints / total_entities if total_entities > 0 else 0
        avg_vectors_per_entity = total_vectors / total_entities if total_entities > 0 else 0
        
        return {
            'total_entities': total_entities,
            'total_waypoints': total_waypoints,
            'total_movement_vectors': total_vectors,
            'average_waypoints_per_entity': avg_waypoints_per_entity,
            'average_vectors_per_entity': avg_vectors_per_entity
        }


class GameStateInterpolator:
    """Higher-level interpolator that works with complete game state"""
    
    def __init__(self):
        self.position_interpolator = PositionInterpolator()
        self.game_packets: List[PacketType] = []
        
    def add_packets(self, packets: List[PacketType]) -> None:
        """Add packets to the interpolator"""
        for packet in packets:
            self.game_packets.append(packet)
            
            # Process position-related packets
            if isinstance(packet, (WaypointGroup, WaypointGroupWithSpeed)):
                self.position_interpolator.add_waypoint_group(packet)
    
    def get_game_state_at_time(self, target_time: float) -> Dict[str, Union[Dict[int, InterpolatedPosition], List[PacketType]]]:
        """Get complete game state at specific time including interpolated positions"""
        # Get all interpolated positions
        positions = self.position_interpolator.get_all_positions_at_time(target_time)
        
        # Get relevant packets around this time (within 1 second window)
        time_window = 1.0
        relevant_packets = [
            packet for packet in self.game_packets
            if hasattr(packet, 'time') and abs(packet.time - target_time) <= time_window
        ]
        
        return {
            'positions': positions,
            'nearby_packets': relevant_packets,
            'timestamp': target_time
        }
    
    def analyze_movement_patterns(self, entity_id: int, time_window: float = 30.0) -> Dict[str, float]:
        """Analyze movement patterns for an entity over a time window"""
        if entity_id not in self.position_interpolator.entity_waypoints:
            return {}
        
        waypoints = self.position_interpolator.entity_waypoints[entity_id]
        if len(waypoints) < 2:
            return {}
        
        # Calculate basic movement statistics
        total_distance = 0.0
        max_speed = 0.0
        speeds = []
        
        waypoints.sort(key=lambda x: x[0])
        
        for i in range(len(waypoints) - 1):
            time1, pos1 = waypoints[i]
            time2, pos2 = waypoints[i + 1]
            
            distance = pos1.distance_to(pos2)
            duration = time2 - time1
            speed = distance / duration if duration > 0 else 0
            
            total_distance += distance
            speeds.append(speed)
            max_speed = max(max_speed, speed)
        
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        return {
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'max_speed': max_speed,
            'waypoint_count': len(waypoints),
            'time_span': waypoints[-1][0] - waypoints[0][0] if len(waypoints) > 1 else 0
        }


