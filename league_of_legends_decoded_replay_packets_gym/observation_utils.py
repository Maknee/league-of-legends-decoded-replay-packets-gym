#!/usr/bin/env python3
"""
Observation Utilities and Helpers

This module provides utility functions and helpers for working with observations
in the League of Legends replay environment.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from gymnasium import spaces

from .types import Position

# Use TYPE_CHECKING to avoid circular imports  
if TYPE_CHECKING:
    from .league_replays_gym import GameState
    from .observations import ObservationCallback


def normalize_position(x: float, z: float, map_size: float = 15000.0) -> Tuple[float, float]:
    """
    Normalize world coordinates to [0, 1] range.
    
    Args:
        x: X coordinate in world space
        z: Z coordinate in world space (note: League uses x,z not x,y)
        map_size: Size of the map (default 15000 for League)
        
    Returns:
        Tuple of normalized (x, z) coordinates
    """
    # Assuming map center is at origin, shift to positive coordinates
    norm_x = (x + map_size/2) / map_size
    norm_z = (z + map_size/2) / map_size
    
    # Clip to [0, 1] range
    norm_x = np.clip(norm_x, 0.0, 1.0)
    norm_z = np.clip(norm_z, 0.0, 1.0)
    
    return norm_x, norm_z


def denormalize_position(norm_x: float, norm_z: float, map_size: float = 15000.0) -> Tuple[float, float]:
    """
    Convert normalized coordinates back to world coordinates.
    
    Args:
        norm_x: Normalized X coordinate [0, 1]
        norm_z: Normalized Z coordinate [0, 1]
        map_size: Size of the map (default 15000 for League)
        
    Returns:
        Tuple of world (x, z) coordinates
    """
    x = norm_x * map_size - map_size/2
    z = norm_z * map_size - map_size/2
    return x, z


def normalize_time(game_time: float, max_time: float = 1800.0) -> float:
    """
    Normalize game time to [0, 1] range.
    
    Args:
        game_time: Game time in seconds
        max_time: Maximum expected game time (default 30 minutes = 1800s)
        
    Returns:
        Normalized time [0, 1]
    """
    return np.clip(game_time / max_time, 0.0, 1.0)


def extract_hero_features(game_state: 'GameState', max_heroes: int = 10) -> Dict[str, np.ndarray]:
    """
    Extract basic hero features from game state.
    
    Args:
        game_state: Current game state
        max_heroes: Maximum number of heroes to extract
        
    Returns:
        Dictionary with hero feature arrays
    """
    positions = np.zeros((max_heroes, 2), dtype=np.float32)
    hero_ids = np.zeros(max_heroes, dtype=np.int32)
    hero_mask = np.zeros(max_heroes, dtype=np.bool_)
    
    heroes = list(game_state.heroes.items())[:max_heroes]
    for i, (net_id, hero_info) in enumerate(heroes):
        hero_ids[i] = net_id
        hero_mask[i] = True
        
        pos = game_state.get_position(net_id)
        if pos:
            norm_x, norm_z = normalize_position(pos.x, pos.z)
            positions[i] = [norm_x, norm_z]
    
    return {
        'positions': positions,
        'ids': hero_ids,
        'mask': hero_mask
    }


def create_minimap_grid(game_state: 'GameState', 
                       resolution: int = 64, 
                       map_size: float = 15000.0,
                       channels: List[str] = None) -> np.ndarray:
    """
    Create a minimap-style grid representation of the game state.
    
    Args:
        game_state: Current game state
        resolution: Grid resolution (resolution x resolution)
        map_size: Size of the game map
        channels: List of channels to include ['heroes', 'minions', 'structures']
        
    Returns:
        Grid array with shape (channels, resolution, resolution)
    """
    if channels is None:
        channels = ['heroes']
    
    grid = np.zeros((len(channels), resolution, resolution), dtype=np.float32)
    
    if 'heroes' in channels:
        hero_channel = channels.index('heroes')
        for net_id, hero_info in game_state.heroes.items():
            pos = game_state.get_position(net_id)
            if pos:
                # Convert to grid coordinates
                norm_x, norm_z = normalize_position(pos.x, pos.z, map_size)
                grid_x = int(norm_x * (resolution - 1))
                grid_z = int(norm_z * (resolution - 1))
                
                # Set hero presence (could differentiate by team/champion)
                grid[hero_channel, grid_z, grid_x] = 1.0
    
    # Additional channels can be added here for minions, structures, etc.
    
    return grid


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between all positions using vectorized operations.
    
    Args:
        positions: Array of positions with shape (n_entities, 2)
        
    Returns:
        Distance matrix with shape (n_entities, n_entities)
    """
    if positions.size == 0:
        return np.array([], dtype=np.float32).reshape(0, 0)
    
    # Use broadcasting to compute all pairwise distances efficiently
    # positions[:, None, :] has shape (n, 1, 2)
    # positions[None, :, :] has shape (1, n, 2)
    # Subtracting gives shape (n, n, 2)
    diff = positions[:, None, :] - positions[None, :, :]
    
    # Compute squared distances and then square root
    distances = np.sqrt(np.sum(diff**2, axis=2)).astype(np.float32)
    
    return distances


def find_nearby_entities(center_pos: Position, 
                        game_state: 'GameState', 
                        radius: float,
                        normalize: bool = True) -> List[Tuple[int, Position, float]]:
    """
    Find all entities within a radius of a center position.
    
    Args:
        center_pos: Center position to search from
        game_state: Current game state
        radius: Search radius (in normalized units if normalize=True)
        normalize: Whether positions are normalized
        
    Returns:
        List of (entity_id, position, distance) tuples
    """
    nearby = []
    
    for net_id, hero_info in game_state.heroes.items():
        pos = game_state.get_position(net_id)
        if pos:
            if normalize:
                # Convert to normalized coordinates for distance calculation
                center_x, center_z = normalize_position(center_pos.x, center_pos.z)
                pos_x, pos_z = normalize_position(pos.x, pos.z)
                
                dx = pos_x - center_x
                dz = pos_z - center_z
                distance = np.sqrt(dx * dx + dz * dz)
            else:
                distance = center_pos.distance_to(pos)
            
            if distance <= radius:
                nearby.append((net_id, pos, distance))
    
    # Sort by distance
    nearby.sort(key=lambda x: x[2])
    return nearby


def extract_event_features(events: List[Any], 
                          event_types: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Extract features from a list of game events.
    
    Args:
        events: List of game events
        event_types: List of event types to include
        
    Returns:
        Dictionary with event features
    """
    if event_types is None:
        event_types = ['CreateHero', 'UnitApplyDamage', 'CastSpellAns', 'BuyItem']
    
    # Count events by type
    event_counts = {event_type: 0 for event_type in event_types}
    event_times = []
    
    for event in events:
        if hasattr(event, 'event_type') and event.event_type in event_counts:
            event_counts[event.event_type] += 1
            event_times.append(event.time)
    
    # Convert to arrays
    features = {
        'event_counts': np.array([event_counts[et] for et in event_types], dtype=np.int32),
        'total_events': np.array([len(events)], dtype=np.int32),
        'recent_event_rate': np.array([len(event_times)], dtype=np.float32)
    }
    
    if event_times:
        features['last_event_time'] = np.array([max(event_times)], dtype=np.float32)
    else:
        features['last_event_time'] = np.array([0.0], dtype=np.float32)
    
    return features


def create_team_features(game_state: 'GameState') -> Dict[str, np.ndarray]:
    """
    Extract team-level features from game state.
    
    Args:
        game_state: Current game state
        
    Returns:
        Dictionary with team features
    """
    # Group heroes by team (this is simplified - real implementation would need team info)
    team_positions = {'team1': [], 'team2': []}
    
    # Simple team assignment based on net_id (this is a placeholder)
    for net_id, hero_info in game_state.heroes.items():
        pos = game_state.get_position(net_id)
        if pos:
            norm_x, norm_z = normalize_position(pos.x, pos.z)
            
            # Simple heuristic: even net_ids = team1, odd = team2
            team = 'team1' if net_id % 2 == 0 else 'team2'
            team_positions[team].append([norm_x, norm_z])
    
    features = {}
    
    for team, positions in team_positions.items():
        if positions:
            positions_array = np.array(positions, dtype=np.float32)
            # Team centroid using vectorized mean
            centroid = np.mean(positions_array, axis=0)
            
            # Team spread (average distance from centroid) using vectorized operations
            diff_from_centroid = positions_array - centroid
            distances = np.sqrt(np.sum(diff_from_centroid ** 2, axis=1))
            spread = np.mean(distances)
            
            features[f'{team}_centroid'] = centroid
            features[f'{team}_spread'] = np.array([spread], dtype=np.float32)
            features[f'{team}_count'] = np.array([len(positions)], dtype=np.int32)
        else:
            features[f'{team}_centroid'] = np.zeros(2, dtype=np.float32)
            features[f'{team}_spread'] = np.zeros(1, dtype=np.float32)
            features[f'{team}_count'] = np.zeros(1, dtype=np.int32)
    
    return features


def validate_observation_space(obs_space: spaces.Space, observation: Any) -> bool:
    """
    Validate that an observation matches its declared space.
    
    Args:
        obs_space: The observation space
        observation: The observation to validate
        
    Returns:
        True if observation is valid for the space
    """
    try:
        return obs_space.contains(observation)
    except Exception:
        return False


def debug_observation(observation: Any, name: str = "observation") -> None:
    """
    Print debug information about an observation.
    
    Args:
        observation: The observation to debug
        name: Name to use in debug output
    """
    print(f"Debug {name}:")
    
    if isinstance(observation, dict):
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                      f"min={value.min():.3f}, max={value.max():.3f}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
    
    elif isinstance(observation, np.ndarray):
        print(f"  Array: shape={observation.shape}, dtype={observation.dtype}, "
              f"min={observation.min():.3f}, max={observation.max():.3f}")
    
    else:
        print(f"  Type: {type(observation).__name__}")
        print(f"  Value: {observation}")


def create_observation_summary(observation: Any) -> Dict[str, Any]:
    """
    Create a summary of an observation for logging/monitoring.
    
    Args:
        observation: The observation to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    if isinstance(observation, dict):
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                summary[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'mean': float(value.mean()),
                    'std': float(value.std()),
                    'min': float(value.min()),
                    'max': float(value.max()),
                    'zeros': int(np.sum(value == 0)),
                    'nonzeros': int(np.sum(value != 0))
                }
            else:
                summary[key] = {'type': type(value).__name__, 'value': str(value)}
    
    elif isinstance(observation, np.ndarray):
        summary = {
            'shape': observation.shape,
            'dtype': str(observation.dtype),
            'mean': float(observation.mean()),
            'std': float(observation.std()),
            'min': float(observation.min()),
            'max': float(observation.max())
        }
    
    else:
        summary = {'type': type(observation).__name__, 'value': str(observation)}
    
    return summary