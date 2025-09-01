#!/usr/bin/env python3
"""
Packet Diff System

This module provides functionality to compute differences between packet states
at different time steps, enabling analysis of game state changes over time.
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from .types import Position
from .packets import PacketType
from .interpolation import InterpolatedPosition


class ChangeType(Enum):
    """Types of changes that can occur between steps"""
    ADDED = "added"           # New packet/entity appeared
    REMOVED = "removed"       # Packet/entity disappeared  
    MODIFIED = "modified"     # Packet/entity changed
    MOVED = "moved"          # Position changed
    UNCHANGED = "unchanged"   # No change


@dataclass
class PacketDiff:
    """Represents a difference in a single packet between two steps"""
    packet_type: str
    entity_id: Optional[int]
    change_type: ChangeType
    old_data: Optional[Any] = None
    new_data: Optional[Any] = None
    field_changes: Optional[Dict[str, Tuple[Any, Any]]] = None  # field -> (old_value, new_value)
    timestamp_old: Optional[float] = None
    timestamp_new: Optional[float] = None


@dataclass
class PositionDiff:
    """Represents a position change for an entity"""
    entity_id: int
    old_position: Optional[InterpolatedPosition]
    new_position: Optional[InterpolatedPosition]
    distance_moved: float
    speed: float  # units per second
    time_delta: float


@dataclass
class StepDiff:
    """Complete diff between two game steps"""
    step1_time: float
    step2_time: float
    time_delta: float
    packet_diffs: List[PacketDiff]
    position_diffs: List[PositionDiff]
    summary: Dict[str, int]  # Summary statistics


class PacketDiffEngine:
    """Engine for computing packet differences between game steps"""
    
    def __init__(self):
        self.position_sensitive_packets = {
            'WaypointGroup', 'WaypointGroupWithSpeed', 'BasicAttackPos', 
            'CastSpellAns', 'SpawnMinion', 'CreateNeutral'
        }
    
    def compute_packet_diff(self, packets1: List[PacketType], packets2: List[PacketType],
                          time1: float, time2: float) -> StepDiff:
        """Compute complete diff between two packet lists"""
        
        # Group packets by type and entity for comparison
        grouped1 = self._group_packets(packets1)
        grouped2 = self._group_packets(packets2)
        
        packet_diffs = []
        
        # Find added, removed, and modified packets
        all_keys = set(grouped1.keys()) | set(grouped2.keys())
        
        for key in all_keys:
            packet_type, entity_id = key
            old_packets = grouped1.get(key, [])
            new_packets = grouped2.get(key, [])
            
            if not old_packets and new_packets:
                # Added packets
                for packet in new_packets:
                    packet_diffs.append(PacketDiff(
                        packet_type=packet_type,
                        entity_id=entity_id,
                        change_type=ChangeType.ADDED,
                        new_data=self._packet_to_dict(packet),
                        timestamp_new=time2
                    ))
            
            elif old_packets and not new_packets:
                # Removed packets
                for packet in old_packets:
                    packet_diffs.append(PacketDiff(
                        packet_type=packet_type,
                        entity_id=entity_id,
                        change_type=ChangeType.REMOVED,
                        old_data=self._packet_to_dict(packet),
                        timestamp_old=time1
                    ))
            
            elif old_packets and new_packets:
                # Compare packets for modifications
                for old_packet, new_packet in zip(old_packets, new_packets):
                    diff = self._compare_packets(old_packet, new_packet)
                    if diff:
                        packet_diffs.append(PacketDiff(
                            packet_type=packet_type,
                            entity_id=entity_id,
                            change_type=ChangeType.MODIFIED,
                            old_data=self._packet_to_dict(old_packet),
                            new_data=self._packet_to_dict(new_packet),
                            field_changes=diff,
                            timestamp_old=time1,
                            timestamp_new=time2
                        ))
                    else:
                        packet_diffs.append(PacketDiff(
                            packet_type=packet_type,
                            entity_id=entity_id,
                            change_type=ChangeType.UNCHANGED,
                            old_data=self._packet_to_dict(old_packet),
                            new_data=self._packet_to_dict(new_packet),
                            timestamp_old=time1,
                            timestamp_new=time2
                        ))
        
        # Generate summary statistics
        summary = self._generate_summary(packet_diffs)
        
        return StepDiff(
            step1_time=time1,
            step2_time=time2,
            time_delta=time2 - time1,
            packet_diffs=packet_diffs,
            position_diffs=[],  # Will be filled by position diff system
            summary=summary
        )
    
    def compute_position_diff(self, positions1: Dict[int, InterpolatedPosition], 
                            positions2: Dict[int, InterpolatedPosition],
                            time1: float, time2: float) -> List[PositionDiff]:
        """Compute position differences between two position dictionaries"""
        position_diffs = []
        time_delta = time2 - time1
        
        all_entities = set(positions1.keys()) | set(positions2.keys())
        
        for entity_id in all_entities:
            old_pos = positions1.get(entity_id)
            new_pos = positions2.get(entity_id)
            
            if old_pos and new_pos:
                # Calculate movement
                distance = old_pos.position.distance_to(new_pos.position)
                speed = distance / time_delta if time_delta > 0 else 0
                
                position_diffs.append(PositionDiff(
                    entity_id=entity_id,
                    old_position=old_pos,
                    new_position=new_pos,
                    distance_moved=distance,
                    speed=speed,
                    time_delta=time_delta
                ))
            
            elif not old_pos and new_pos:
                # Entity appeared
                position_diffs.append(PositionDiff(
                    entity_id=entity_id,
                    old_position=None,
                    new_position=new_pos,
                    distance_moved=0.0,
                    speed=0.0,
                    time_delta=time_delta
                ))
            
            elif old_pos and not new_pos:
                # Entity disappeared
                position_diffs.append(PositionDiff(
                    entity_id=entity_id,
                    old_position=old_pos,
                    new_position=None,
                    distance_moved=0.0,
                    speed=0.0,
                    time_delta=time_delta
                ))
        
        return position_diffs
    
    def _group_packets(self, packets: List[PacketType]) -> Dict[Tuple[str, Optional[int]], List[PacketType]]:
        """Group packets by type and entity ID"""
        grouped = {}
        
        for packet in packets:
            packet_type = type(packet).__name__
            entity_id = self._extract_entity_id(packet)
            key = (packet_type, entity_id)
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(packet)
        
        return grouped
    
    def _extract_entity_id(self, packet: PacketType) -> Optional[int]:
        """Extract entity ID from packet if available"""
        # Try common entity ID field names
        for field_name in ['net_id', 'source_net_id', 'caster_net_id', 'entity_id']:
            if hasattr(packet, field_name):
                return getattr(packet, field_name)
        return None
    
    def _packet_to_dict(self, packet: PacketType) -> Dict[str, Any]:
        """Convert packet to dictionary for comparison"""
        if hasattr(packet, '__dict__'):
            result = {}
            for key, value in packet.__dict__.items():
                if isinstance(value, Position):
                    result[key] = {"x": value.x, "z": value.z}
                elif isinstance(value, list) and value and isinstance(value[0], Position):
                    result[key] = [{"x": pos.x, "z": pos.z} for pos in value]
                else:
                    result[key] = value
            return result
        return {}
    
    def _compare_packets(self, packet1: PacketType, packet2: PacketType) -> Optional[Dict[str, Tuple[Any, Any]]]:
        """Compare two packets and return field differences"""
        if type(packet1) != type(packet2):
            return {"type": (type(packet1).__name__, type(packet2).__name__)}
        
        dict1 = self._packet_to_dict(packet1)
        dict2 = self._packet_to_dict(packet2)
        
        changes = {}
        all_fields = set(dict1.keys()) | set(dict2.keys())
        
        for field in all_fields:
            val1 = dict1.get(field)
            val2 = dict2.get(field)
            
            if val1 != val2:
                changes[field] = (val1, val2)
        
        return changes if changes else None
    
    def _generate_summary(self, packet_diffs: List[PacketDiff]) -> Dict[str, int]:
        """Generate summary statistics from packet diffs"""
        summary = {
            'total_changes': len(packet_diffs),
            'added': 0,
            'removed': 0,
            'modified': 0,
            'unchanged': 0,
            'moved': 0
        }
        
        for diff in packet_diffs:
            if diff.change_type == ChangeType.ADDED:
                summary['added'] += 1
            elif diff.change_type == ChangeType.REMOVED:
                summary['removed'] += 1
            elif diff.change_type == ChangeType.MODIFIED:
                summary['modified'] += 1
            elif diff.change_type == ChangeType.UNCHANGED:
                summary['unchanged'] += 1
            elif diff.change_type == ChangeType.MOVED:
                summary['moved'] += 1
        
        return summary


class GameStateDiffAnalyzer:
    """High-level analyzer for game state differences"""
    
    def __init__(self):
        self.diff_engine = PacketDiffEngine()
        self.step_history: List[Tuple[float, List[PacketType], Dict[int, InterpolatedPosition]]] = []
    
    def add_step(self, timestamp: float, packets: List[PacketType], 
                 positions: Dict[int, InterpolatedPosition]) -> None:
        """Add a game step to the analyzer"""
        self.step_history.append((timestamp, packets, positions))
    
    def get_step_diff(self, step1_idx: int, step2_idx: int) -> Optional[StepDiff]:
        """Get diff between two specific steps"""
        if (step1_idx < 0 or step1_idx >= len(self.step_history) or 
            step2_idx < 0 or step2_idx >= len(self.step_history)):
            return None
        
        time1, packets1, positions1 = self.step_history[step1_idx]
        time2, packets2, positions2 = self.step_history[step2_idx]
        
        # Compute packet diff
        step_diff = self.diff_engine.compute_packet_diff(packets1, packets2, time1, time2)
        
        # Add position diffs
        step_diff.position_diffs = self.diff_engine.compute_position_diff(
            positions1, positions2, time1, time2
        )
        
        return step_diff
    
    def get_consecutive_diffs(self, start_idx: int = 0, count: Optional[int] = None) -> List[StepDiff]:
        """Get differences between consecutive steps"""
        if count is None:
            count = len(self.step_history) - 1
        
        diffs = []
        end_idx = min(start_idx + count, len(self.step_history) - 1)
        
        for i in range(start_idx, end_idx):
            diff = self.get_step_diff(i, i + 1)
            if diff:
                diffs.append(diff)
        
        return diffs
    
    def analyze_entity_activity(self, entity_id: int, window_size: int = 10) -> Dict[str, Any]:
        """Analyze activity patterns for a specific entity"""
        recent_steps = self.step_history[-window_size:] if len(self.step_history) >= window_size else self.step_history
        
        packet_appearances = 0
        position_changes = 0
        total_distance = 0.0
        speeds = []
        
        for i in range(len(recent_steps) - 1):
            time1, packets1, positions1 = recent_steps[i]
            time2, packets2, positions2 = recent_steps[i + 1]
            
            # Check packet appearances
            for packet in packets1 + packets2:
                if self.diff_engine._extract_entity_id(packet) == entity_id:
                    packet_appearances += 1
                    break
            
            # Check position changes
            pos1 = positions1.get(entity_id)
            pos2 = positions2.get(entity_id)
            
            if pos1 and pos2:
                distance = pos1.position.distance_to(pos2.position)
                if distance > 1.0:  # Threshold for significant movement
                    position_changes += 1
                    total_distance += distance
                    
                    time_delta = time2 - time1
                    if time_delta > 0:
                        speeds.append(distance / time_delta)
        
        return {
            'entity_id': entity_id,
            'packet_appearances': packet_appearances,
            'position_changes': position_changes,
            'total_distance_moved': total_distance,
            'average_speed': sum(speeds) / len(speeds) if speeds else 0,
            'max_speed': max(speeds) if speeds else 0,
            'steps_analyzed': len(recent_steps)
        }
    
    def get_most_active_entities(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the most active entities based on recent activity"""
        # Get all unique entity IDs from recent history
        all_entities = set()
        for _, _, positions in self.step_history[-20:]:  # Look at last 20 steps
            all_entities.update(positions.keys())
        
        # Analyze each entity
        activities = []
        for entity_id in all_entities:
            activity = self.analyze_entity_activity(entity_id)
            activity['activity_score'] = (
                activity['packet_appearances'] + 
                activity['position_changes'] * 2 + 
                activity['total_distance_moved'] / 100.0
            )
            activities.append(activity)
        
        # Sort by activity score and return top N
        activities.sort(key=lambda x: x['activity_score'], reverse=True)
        return activities[:top_n]
    
    def export_diff_history(self, filename: str, format: str = 'json') -> None:
        """Export diff history to file"""
        consecutive_diffs = self.get_consecutive_diffs()
        
        if format == 'json':
            # Convert diffs to JSON-serializable format
            serializable_diffs = []
            for diff in consecutive_diffs:
                diff_dict = {
                    'step1_time': diff.step1_time,
                    'step2_time': diff.step2_time,
                    'time_delta': diff.time_delta,
                    'summary': diff.summary,
                    'packet_changes': len(diff.packet_diffs),
                    'position_changes': len(diff.position_diffs)
                }
                serializable_diffs.append(diff_dict)
            
            with open(filename, 'w') as f:
                json.dump(serializable_diffs, f, indent=2)
    
    def clear_old_history(self, keep_steps: int = 100) -> None:
        """Clear old step history to manage memory"""
        if len(self.step_history) > keep_steps:
            self.step_history = self.step_history[-keep_steps:]


