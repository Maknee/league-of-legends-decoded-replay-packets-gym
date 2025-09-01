#!/usr/bin/env python3
"""
Customizable Observation System for League of Legends Replays

This module provides a flexible observation callback system that allows users
to define custom observations for their specific machine learning tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from .types import Position
from .interpolation import InterpolatedPosition

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .league_replays_gym import GameState


class ObservationCallback(ABC):
    """
    Abstract base class for custom observation callbacks.
    
    Users can inherit from this class to create custom observations
    that extract specific features from the game state.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._observation_space = None
    
    @abstractmethod
    def compute_observation(self, game_state: 'GameState', info: Dict[str, Any]) -> Any:
        """
        Compute observation from current game state.
        
        Args:
            game_state: Current game state with heroes, positions, events
            info: Additional information from environment
            
        Returns:
            Observation data (format depends on specific callback)
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """
        Get the observation space for this callback.
        
        Returns:
            Gymnasium space describing the observation format
        """
        pass
    
    def reset(self) -> None:
        """Called when environment resets. Override for stateful callbacks."""
        pass


class PositionObservation(ObservationCallback):
    """
    Default observation providing normalized hero positions and game time.
    
    This is the recommended starting point for most users.
    """
    
    def __init__(self, 
                 max_heroes: int = 10,
                 normalize_positions: bool = True,
                 map_size: float = 15000.0,
                 include_game_time: bool = True):
        """
        Initialize position observation.
        
        Args:
            max_heroes: Maximum number of heroes to track
            normalize_positions: Whether to normalize positions to [0, 1]
            map_size: Size of the map for normalization (League map is ~15000 units)
            include_game_time: Whether to include normalized game time
        """
        super().__init__("position")
        self.max_heroes = max_heroes
        self.normalize_positions = normalize_positions
        self.map_size = map_size
        self.include_game_time = include_game_time
    
    def compute_observation(self, game_state: 'GameState', info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute normalized position observation."""
        # Initialize arrays
        positions = np.zeros((self.max_heroes, 2), dtype=np.float32)
        hero_ids = np.zeros(self.max_heroes, dtype=np.int32)
        hero_mask = np.zeros(self.max_heroes, dtype=np.bool_)
        
        # Fill hero data (up to max_heroes)
        heroes = list(game_state.heroes.items())[:self.max_heroes]
        for i, (net_id, hero_info) in enumerate(heroes):
            hero_ids[i] = net_id
            hero_mask[i] = True
            
            pos = game_state.get_position(net_id)
            if pos:
                x, z = pos.x, pos.z
                if self.normalize_positions:
                    # Normalize to [0, 1] assuming map center at origin
                    x = (x + self.map_size/2) / self.map_size
                    z = (z + self.map_size/2) / self.map_size
                    x = np.clip(x, 0, 1)
                    z = np.clip(z, 0, 1)
                positions[i] = [x, z]
        
        obs = {
            'hero_positions': positions,
            'hero_ids': hero_ids,
            'hero_mask': hero_mask,
        }
        
        if self.include_game_time:
            # Normalize game time to [0, 1] assuming 30 minute games
            normalized_time = np.clip(game_state.current_time / 1800.0, 0, 1)
            obs['game_time'] = np.array([normalized_time], dtype=np.float32)
        
        return obs
    
    def get_observation_space(self) -> spaces.Dict:
        """Get observation space for position data."""
        space_dict = {
            'hero_positions': spaces.Box(
                low=0.0 if self.normalize_positions else -self.map_size,
                high=1.0 if self.normalize_positions else self.map_size,
                shape=(self.max_heroes, 2),
                dtype=np.float32
            ),
            'hero_ids': spaces.Box(
                low=0, high=999999, shape=(self.max_heroes,), dtype=np.int32
            ),
            'hero_mask': spaces.Box(
                low=0, high=1, shape=(self.max_heroes,), dtype=np.bool_
            ),
        }
        
        if self.include_game_time:
            space_dict['game_time'] = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        
        return spaces.Dict(space_dict)


class MinimapObservation(ObservationCallback):
    """
    2D minimap-style observation showing hero positions on a grid.
    
    Useful for convolutional neural networks and spatial reasoning.
    """
    
    def __init__(self, 
                 resolution: int = 64,
                 map_size: float = 15000.0,
                 channels: List[str] = None):
        """
        Initialize minimap observation.
        
        Args:
            resolution: Size of the minimap grid (resolution x resolution)
            map_size: Size of the game map in units
            channels: List of channels to include ['heroes', 'minions', 'structures']
        """
        super().__init__("minimap")
        self.resolution = resolution
        self.map_size = map_size
        self.channels = channels or ['heroes']
        self.channel_map = {name: i for i, name in enumerate(self.channels)}
    
    def compute_observation(self, game_state: 'GameState', info: Dict[str, Any]) -> np.ndarray:
        """Compute minimap observation."""
        minimap = np.zeros((len(self.channels), self.resolution, self.resolution), dtype=np.float32)
        
        # Add heroes to minimap
        if 'heroes' in self.channel_map:
            channel_idx = self.channel_map['heroes']
            for net_id, hero_info in game_state.heroes.items():
                pos = game_state.get_position(net_id)
                if pos:
                    # Convert world position to grid coordinates
                    x = int((pos.x + self.map_size/2) / self.map_size * self.resolution)
                    z = int((pos.z + self.map_size/2) / self.map_size * self.resolution)
                    
                    # Clip to valid range
                    x = np.clip(x, 0, self.resolution - 1)
                    z = np.clip(z, 0, self.resolution - 1)
                    
                    # Set hero presence (could use different values for different teams)
                    minimap[channel_idx, z, x] = 1.0  # Note: z, x for image convention
        
        return minimap
    
    def get_observation_space(self) -> spaces.Box:
        """Get observation space for minimap."""
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.channels), self.resolution, self.resolution),
            dtype=np.float32
        )


class EventHistoryObservation(ObservationCallback):
    """
    Sliding window of recent game events.
    
    Useful for analyzing event sequences and temporal patterns.
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 event_types: Optional[List[str]] = None):
        """
        Initialize event history observation.
        
        Args:
            window_size: Number of recent events to keep
            event_types: List of event types to track (None = all types)
        """
        super().__init__("event_history")
        self.window_size = window_size
        self.event_types = set(event_types) if event_types else None
        self.event_history: List[Dict[str, Any]] = []
    
    def compute_observation(self, game_state: 'GameState', info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute event history observation."""
        # Add new events to history
        for event in game_state.events:
            if self.event_types is None or event.event_type in self.event_types:
                event_data = {
                    'type': event.event_type,
                    'time': event.time,
                    'data': event.data
                }
                self.event_history.append(event_data)
        
        # Keep only recent events
        self.event_history = self.event_history[-self.window_size:]
        
        # Convert to arrays (simplified representation)
        event_count = len(self.event_history)
        event_times = np.zeros(self.window_size, dtype=np.float32)
        event_type_ids = np.zeros(self.window_size, dtype=np.int32)
        
        # Simple event type to ID mapping (could be more sophisticated)
        type_to_id = {
            'CreateHero': 1, 'WaypointGroup': 2, 'UnitApplyDamage': 3,
            'CastSpellAns': 4, 'BuyItem': 5, 'HeroDie': 6
        }
        
        for i, event in enumerate(self.event_history):
            event_times[i] = event['time']
            event_type_ids[i] = type_to_id.get(event['type'], 0)
        
        return {
            'event_times': event_times,
            'event_type_ids': event_type_ids,
            'event_count': np.array([event_count], dtype=np.int32)
        }
    
    def get_observation_space(self) -> spaces.Dict:
        """Get observation space for event history."""
        return spaces.Dict({
            'event_times': spaces.Box(
                low=0.0, high=np.inf, shape=(self.window_size,), dtype=np.float32
            ),
            'event_type_ids': spaces.Box(
                low=0, high=10, shape=(self.window_size,), dtype=np.int32
            ),
            'event_count': spaces.Box(
                low=0, high=self.window_size, shape=(1,), dtype=np.int32
            )
        })
    
    def reset(self) -> None:
        """Clear event history on reset."""
        self.event_history.clear()


class CustomObservation(ObservationCallback):
    """
    Wrapper for user-defined observation functions.
    
    This allows users to define observations using simple functions
    without needing to inherit from ObservationCallback.
    """
    
    def __init__(self, 
                 name: str,
                 compute_fn: callable,
                 observation_space: spaces.Space,
                 reset_fn: Optional[callable] = None):
        """
        Initialize custom observation.
        
        Args:
            name: Name of the observation
            compute_fn: Function(game_state, info) -> observation
            observation_space: Gymnasium space for the observation
            reset_fn: Optional function to call on reset
        """
        super().__init__(name)
        self.compute_fn = compute_fn
        self._observation_space = observation_space
        self.reset_fn = reset_fn
    
    def compute_observation(self, game_state: 'GameState', info: Dict[str, Any]) -> Any:
        """Compute observation using user function."""
        return self.compute_fn(game_state, info)
    
    def get_observation_space(self) -> spaces.Space:
        """Get observation space."""
        return self._observation_space
    
    def reset(self) -> None:
        """Call user reset function if provided."""
        if self.reset_fn:
            self.reset_fn()


class CompositeObservation(ObservationCallback):
    """
    Combines multiple observation callbacks into a single observation.
    
    Useful for creating rich, multi-modal observations.
    """
    
    def __init__(self, callbacks: List[ObservationCallback]):
        """
        Initialize composite observation.
        
        Args:
            callbacks: List of observation callbacks to combine
        """
        super().__init__("composite")
        self.callbacks = callbacks
        self.callback_map = {cb.name: cb for cb in callbacks}
    
    def compute_observation(self, game_state: 'GameState', info: Dict[str, Any]) -> Dict[str, Any]:
        """Compute all sub-observations."""
        observation = {}
        for callback in self.callbacks:
            obs = callback.compute_observation(game_state, info)
            if isinstance(obs, dict):
                # Prefix keys with callback name to avoid conflicts
                for key, value in obs.items():
                    observation[f"{callback.name}_{key}"] = value
            else:
                observation[callback.name] = obs
        return observation
    
    def get_observation_space(self) -> spaces.Dict:
        """Get combined observation space."""
        space_dict = {}
        for callback in self.callbacks:
            cb_space = callback.get_observation_space()
            if isinstance(cb_space, spaces.Dict):
                # Add prefixed spaces
                for key, space in cb_space.spaces.items():
                    space_dict[f"{callback.name}_{key}"] = space
            else:
                space_dict[callback.name] = cb_space
        return spaces.Dict(space_dict)
    
    def reset(self) -> None:
        """Reset all sub-callbacks."""
        for callback in self.callbacks:
            callback.reset()


# Utility functions for creating common observations

def create_position_observation(**kwargs) -> PositionObservation:
    """Create a position observation with common defaults."""
    return PositionObservation(**kwargs)


def create_minimap_observation(**kwargs) -> MinimapObservation:
    """Create a minimap observation with common defaults."""
    return MinimapObservation(**kwargs)


def create_event_observation(**kwargs) -> EventHistoryObservation:
    """Create an event history observation with common defaults."""
    return EventHistoryObservation(**kwargs)


def create_custom_observation(compute_fn: callable, 
                            observation_space: spaces.Space,
                            name: str = "custom",
                            reset_fn: Optional[callable] = None) -> CustomObservation:
    """
    Create a custom observation from a simple function.
    
    Example:
        >>> def my_obs_fn(game_state, info):
        ...     return np.array([game_state.current_time, len(game_state.heroes)])
        >>> 
        >>> obs_space = spaces.Box(low=0, high=np.inf, shape=(2,))
        >>> callback = create_custom_observation(my_obs_fn, obs_space)
    """
    return CustomObservation(name, compute_fn, observation_space, reset_fn)


def combine_observations(*callbacks: ObservationCallback) -> CompositeObservation:
    """
    Combine multiple observation callbacks.
    
    Example:
        >>> pos_obs = create_position_observation()
        >>> minimap_obs = create_minimap_observation(resolution=32)
        >>> combined = combine_observations(pos_obs, minimap_obs)
    """
    return CompositeObservation(list(callbacks))