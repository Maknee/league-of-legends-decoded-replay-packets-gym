#!/usr/bin/env python3
"""
League of Legends Replays Gymnasium Environment

A Gymnasium-compliant interface for easily accessing and iterating through League of Legends replay data
with support for multiple parallel instances, filtering, and state management.
"""

import gzip
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Tuple, Callable, Union, TYPE_CHECKING
from enum import Enum
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType

from .types import Position, GameEvent

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .observations import ObservationCallback

@dataclass
class GameState:
    """
    Represents the complete state of a League of Legends game at a specific moment in time.
    
    This class provides a user-friendly interface to access game information including:
    - Hero positions and stats
    - Recent events that occurred
    - Spatial queries (finding nearby heroes)
    - Game timing information
    
    Attributes:
        game_id (int): Unique identifier for this game
        current_time (float): Current game time in seconds
        events (List[GameEvent]): Recent events that occurred at this time step
        heroes (Dict[int, Dict[str, Any]]): Map of net_id -> hero information
        positions (Dict[int, Position]): Map of net_id -> current position
    
    Example:
        >>> state = env.step()  # Get current game state
        >>> print(f"Game time: {state.current_time}s")
        >>> for net_id, hero in state.get_all_heroes():
        ...     pos = state.get_position(net_id)
        ...     print(f"Hero {hero.get('name', 'Unknown')} at ({pos.x:.1f}, {pos.z:.1f})")
    """
    game_id: int
    current_time: float
    events: List[GameEvent] = field(default_factory=list)
    heroes: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # net_id -> hero_info
    positions: Dict[int, Position] = field(default_factory=dict)     # net_id -> position
    
    def get_hero_by_net_id(self, net_id: int) -> Optional[Dict[str, Any]]:
        """
        Get hero information by network ID.
        
        Args:
            net_id (int): The network ID of the hero
            
        Returns:
            Optional[Dict[str, Any]]: Hero information dict or None if not found
            
        Example:
            >>> hero = state.get_hero_by_net_id(123)
            >>> if hero:
            ...     print(f"Hero name: {hero.get('name', 'Unknown')}")
        """
        return self.heroes.get(net_id)
    
    def get_position(self, net_id: int) -> Optional[Position]:
        """
        Get current position of any entity by network ID.
        
        Args:
            net_id (int): The network ID of the entity
            
        Returns:
            Optional[Position]: Position object or None if not found
            
        Example:
            >>> pos = state.get_position(123)
            >>> if pos:
            ...     print(f"Entity at ({pos.x:.1f}, {pos.z:.1f})")
        """
        return self.positions.get(net_id)
    
    def get_all_heroes(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Get all heroes in the game as a list of (net_id, hero_info) tuples.
        
        Returns:
            List[Tuple[int, Dict[str, Any]]]: List of (net_id, hero_info) pairs
            
        Example:
            >>> for net_id, hero in state.get_all_heroes():
            ...     team = hero.get('team', 'unknown')
            ...     name = hero.get('name', 'Unknown')
            ...     print(f"Team {team}: {name} (ID: {net_id})")
        """
        return [(net_id, hero) for net_id, hero in self.heroes.items()]
    
    def get_heroes_by_team(self, team: str) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Get all heroes belonging to a specific team.
        
        Args:
            team (str): Team identifier (e.g., "CHAOS", "ORDER")
            
        Returns:
            List[Tuple[int, Dict[str, Any]]]: List of (net_id, hero_info) pairs
            
        Example:
            >>> chaos_heroes = state.get_heroes_by_team("CHAOS")
            >>> print(f"CHAOS team has {len(chaos_heroes)} heroes")
        """
        return [(net_id, hero) for net_id, hero in self.heroes.items() 
                if hero.get('team') == team]
    
    def get_heroes_in_radius(self, center: Position, radius: float) -> List[Tuple[int, Dict[str, Any], Position]]:
        """
        Find all heroes within a specified radius of a center position.
        
        Args:
            center (Position): Center point for the search
            radius (float): Search radius in game units
            
        Returns:
            List[Tuple[int, Dict[str, Any], Position]]: List of (net_id, hero_info, position) tuples
            
        Example:
            >>> center = Position(x=1000, z=2000)  # Mid lane
            >>> nearby = state.get_heroes_in_radius(center, radius=500)
            >>> print(f"Found {len(nearby)} heroes near mid lane")
            >>> for net_id, hero, pos in nearby:
            ...     dist = center.distance_to(pos)
            ...     print(f"  {hero.get('name', 'Unknown')} at distance {dist:.1f}")
        """
        heroes_in_radius = []
        for net_id, hero in self.heroes.items():
            pos = self.get_position(net_id)
            if pos and pos.distance_to(center) <= radius:
                heroes_in_radius.append((net_id, hero, pos))
        return heroes_in_radius
    
    def get_game_time_formatted(self) -> str:
        """
        Get game time in a human-readable format (MM:SS).
        
        Returns:
            str: Formatted time string
            
        Example:
            >>> print(f"Game time: {state.get_game_time_formatted()}")
            Game time: 15:30
        """
        minutes = int(self.current_time // 60)
        seconds = int(self.current_time % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_events_by_type(self, event_type: str) -> List[GameEvent]:
        """
        Filter events by type for this time step.
        
        Args:
            event_type (str): The event type to filter for
            
        Returns:
            List[GameEvent]: List of events matching the type
            
        Example:
            >>> damage_events = state.get_events_by_type("UnitApplyDamage")
            >>> print(f"Found {len(damage_events)} damage events this step")
        """
        return [event for event in self.events if event.event_type == event_type]
    
    def summary(self) -> str:
        """
        Get a human-readable summary of the current game state.
        
        Returns:
            str: Summary string with key game state information
            
        Example:
            >>> print(state.summary())
            Game 12345 at 15:30 - 10 heroes, 5 events this step
        """
        hero_count = len(self.heroes)
        event_count = len(self.events)
        time_str = self.get_game_time_formatted()
        return f"Game {self.game_id} at {time_str} - {hero_count} heroes, {event_count} events this step"

class ReplayDataset:
    """Dataset class for managing multiple replay files"""
    
    def __init__(self, data_sources: List[str], repo_id: str = "maknee/league-of-legends-replays"):
        """
        Initialize dataset with data sources
        
        Args:
            data_sources: List of file paths (local) or filenames (HuggingFace)
            repo_id: HuggingFace repository ID
        """
        self.data_sources = data_sources
        self.repo_id = repo_id
        self.games: List[List[GameEvent]] = []
        self.loaded = False
    
    def _expand_directories(self, sources: List[str]) -> List[str]:
        """Expand directory patterns to individual files"""
        expanded = []
        
        for source in sources:
            if self._is_directory_pattern(source):
                try:
                    # List all files in the repository
                    all_files = list_repo_files(
                        repo_id=self.repo_id,
                        repo_type="dataset"
                    )
                    
                    # Filter files that match the directory pattern
                    directory_path = source.rstrip('/*').rstrip('/')
                    matching_files = [
                        f for f in all_files 
                        if f.startswith(directory_path + '/') and f.endswith('.jsonl.gz')
                    ]
                    
                    if not matching_files:
                        raise ValueError(f"No .jsonl.gz files found in directory: {source}")
                        
                    expanded.extend(matching_files)
                except Exception as e:
                    raise ValueError(f"Failed to expand directory {source}: {e}")
            else:
                expanded.append(source)
        
        return expanded
    
    def _is_directory_pattern(self, source: str) -> bool:
        """Check if a source is a directory pattern"""
        # Skip local files
        if source.startswith('/') or source.startswith('./') or source.startswith('test_data/'):
            return False
            
        # Directory if ends with / or /* or contains / but no file extension
        return (source.endswith('/') or 
                source.endswith('/*') or
                ('/' in source and not source.endswith('.gz') and not source.endswith('.jsonl')))

    def load(self, max_games: Optional[int] = None) -> None:
        """Load games from all data sources"""
        if not self.data_sources:
            raise ValueError("No data sources provided")
        
        # Expand directories to individual files
        try:
            expanded_sources = self._expand_directories(self.data_sources)
        except ValueError as e:
            raise ValueError(f"Failed to expand data sources: {e}")
        
        self.games.clear()
        load_errors = []
        
        for source in expanded_sources:
            if not isinstance(source, str) or not source.strip():
                load_errors.append(f"Invalid data source: {source}")
                continue
                
            try:
                lines = []
                if source.startswith('/') or source.startswith('./') or source.startswith('test_data/'):
                    # Local file
                    if not source.endswith('.gz'):
                        load_errors.append(f"Local file must be gzipped (.gz): {source}")
                        continue
                        
                    try:
                        with gzip.open(source, 'rt', encoding='utf-8') as f:
                            lines = f.readlines()
                    except FileNotFoundError:
                        load_errors.append(f"Local file not found: {source}")
                        continue
                    except gzip.BadGzipFile:
                        load_errors.append(f"Invalid gzip file: {source}")
                        continue
                else:
                    # HuggingFace file
                    try:
                        file_path = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=source,
                            repo_type="dataset"
                        )
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            lines = f.readlines()
                    except Exception as e:
                        load_errors.append(f"Failed to download from HuggingFace {source}: {e}")
                        continue
                
                games_parsed = 0
                for line_num, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                        
                    try:
                        game_data = json.loads(line)
                        if not isinstance(game_data, dict):
                            continue
                            
                        events_data = game_data.get('events', [])
                        if not isinstance(events_data, list):
                            continue
                            
                        events = []
                        for packet in events_data:
                            try:
                                event = GameEvent.from_packet(packet)
                                events.append(event)
                            except Exception:
                                # Skip invalid packets
                                continue
                                
                        if events:  # Only add games with valid events
                            self.games.append(events)
                            games_parsed += 1
                            
                            if max_games and len(self.games) >= max_games:
                                break
                                
                    except json.JSONDecodeError as e:
                        load_errors.append(f"JSON error in {source} line {line_num}: {e}")
                        continue
                    except Exception as e:
                        load_errors.append(f"Error processing {source} line {line_num}: {e}")
                        continue
                
                if games_parsed == 0:
                    load_errors.append(f"No valid games found in {source}")
                    
                if max_games and len(self.games) >= max_games:
                    break
                    
            except Exception as e:
                load_errors.append(f"Unexpected error loading {source}: {e}")
                continue
        
        self.loaded = True
        
        if not self.games:
            error_msg = "No games could be loaded. Errors:\n" + "\n".join(load_errors)
            raise RuntimeError(error_msg)
            
        if load_errors:
            import warnings
            warnings.warn(f"Some issues occurred during loading:\n" + "\n".join(load_errors[:5]))
            
        print(f"Successfully loaded {len(self.games)} games from {len(self.data_sources)} sources")
    
    def __len__(self) -> int:
        return len(self.games)
    
    def __getitem__(self, idx: int) -> List[GameEvent]:
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.games[idx]

class LeagueReplaysEnv(gym.Env):
    """Gymnasium environment for League of Legends replays"""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, 
                 dataset: ReplayDataset,
                 max_time: Optional[float] = None,
                 time_step: float = 1.0,
                 event_filter: Optional[Callable[[GameEvent], bool]] = None,
                 render_mode: Optional[str] = None,
                 observation_callback: Optional[Union[Any, List[Any]]] = None):
        """
        Initialize environment
        
        Args:
            dataset: ReplayDataset containing the games
            max_time: Maximum time to simulate (None for full game)
            time_step: Time increment per step
            event_filter: Optional filter function for events
            render_mode: Rendering mode
            observation_callback: Custom observation callback(s) or None for default
        """
        super().__init__()
        
        self.dataset = dataset
        self.max_time = max_time
        self.time_step = time_step
        self.event_filter = event_filter
        self.render_mode = render_mode
        
        # Setup observation callback
        self._setup_observation_callback(observation_callback)
        
        # Current episode state
        self.current_game_idx: int = 0
        self.current_time: float = 0.0
        self.current_events: List[GameEvent] = []
        self.game_state: Optional[GameState] = None
        self._terminated = False
        self._truncated = False
        
        if not dataset.loaded:
            dataset.load()
        
        # Define observation and action spaces
        self._setup_spaces()
    
    def _setup_observation_callback(self, observation_callback) -> None:
        """Setup observation callback system."""
        # Import here to avoid circular imports
        from .observations import ObservationCallback, PositionObservation, CompositeObservation
        
        if observation_callback is None:
            # Use default position observation
            self.observation_callback = PositionObservation()
        elif isinstance(observation_callback, list):
            # Multiple callbacks - combine them
            self.observation_callback = CompositeObservation(observation_callback)
        elif hasattr(observation_callback, 'compute_observation'):
            # Single callback
            self.observation_callback = observation_callback
        else:
            raise ValueError(f"Invalid observation_callback type: {type(observation_callback)}")
    
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces"""
        # Get observation space from callback
        self.observation_space = self.observation_callback.get_observation_space()
        
        # Action space: discrete actions for controlling the simulation
        # 0: continue simulation, 1: skip ahead, 2: reset to start
        self.action_space = spaces.Discrete(3)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset environment to start of a game"""
        super().reset(seed=seed)
        
        if not self.dataset.loaded:
            raise RuntimeError("Dataset not loaded. Call dataset.load() first.")
            
        if len(self.dataset) == 0:
            raise RuntimeError("Dataset is empty. No games available.")
        
        # Get game index from options or random
        game_idx = None
        if options and 'game_idx' in options:
            game_idx = options['game_idx']
            if not isinstance(game_idx, int) or game_idx < 0 or game_idx >= len(self.dataset):
                raise ValueError(f"Invalid game_idx {game_idx}. Must be 0 <= game_idx < {len(self.dataset)}")
        
        if game_idx is None:
            game_idx = self.np_random.integers(0, len(self.dataset))
        
        try:
            self.current_game_idx = game_idx
            self.current_time = 0.0
            self.current_events = self.dataset[game_idx].copy()
            self._terminated = False
            self._truncated = False
            
            if not self.current_events:
                raise ValueError(f"Game {game_idx} has no events")
            
            # Sort events by time and validate
            try:
                self.current_events.sort(key=lambda e: e.time)
            except (AttributeError, TypeError) as e:
                raise ValueError(f"Invalid event data in game {game_idx}: {e}")
            
            # Initialize game state
            self.game_state = GameState(
                game_id=game_idx,
                current_time=self.current_time
            )
            
            # Reset observation callback
            try:
                self.observation_callback.reset()
            except Exception as e:
                raise RuntimeError(f"Failed to reset observation callback: {e}")
            
            # Return observation and info
            try:
                observation = self._get_observation()
                info = self._get_info()
            except Exception as e:
                raise RuntimeError(f"Failed to get initial observation: {e}")
            
            return observation, info
            
        except Exception as e:
            # Reset to safe state on any error
            self.game_state = None
            self._terminated = True
            self._truncated = True
            raise RuntimeError(f"Failed to reset environment: {e}")
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Advance simulation by one time step
        
        Args:
            action: Action to take (0=continue, 1=skip ahead, 2=reset to start)
        
        Returns:
            observation: Current game state observation
            reward: Reward for this step
            terminated: Whether episode terminated normally
            truncated: Whether episode was truncated
            info: Additional info
        """
        if self.game_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        if self._terminated or self._truncated:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in range [0, {self.action_space.n-1}]")
        
        # Process action
        try:
            if action == 1:  # Skip ahead
                self.current_time += self.time_step * 5  # Skip 5 time steps
            elif action == 2:  # Reset to start
                self.current_time = 0.0
                self.game_state.heroes.clear()
                self.game_state.positions.clear()
                self.game_state.events.clear()
            else:  # Continue normally (action == 0)
                self.current_time += self.time_step
        except Exception as e:
            raise RuntimeError(f"Failed to process action {action}: {e}")
        
        # Get events that occur in this time window
        step_events = []
        for event in self.current_events:
            if (event.time > self.current_time - self.time_step and 
                event.time <= self.current_time):
                
                if self.event_filter is None or self.event_filter(event):
                    step_events.append(event)
                    self._update_game_state(event)
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        if self.max_time and self.current_time >= self.max_time:
            truncated = True
        elif not any(e.time > self.current_time for e in self.current_events):
            terminated = True  # No more events
        
        self._terminated = terminated
        self._truncated = truncated
        
        self.game_state.current_time = self.current_time
        self.game_state.events = step_events  # Only current step events
        
        # Calculate reward (example: reward based on number of events)
        reward = float(len(step_events))
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'step_events': len(step_events),
            'action_taken': action
        })
        
        return observation, reward, terminated, truncated, info
    
    def _update_game_state(self, event: GameEvent) -> None:
        """Update game state based on event"""
        if event.event_type == 'CreateHero':
            net_id = event.data.get('net_id')
            if net_id:
                self.game_state.heroes[net_id] = {
                    'name': event.data.get('name'),
                    'champion': event.data.get('champion'),
                    'net_id': net_id
                }
        
        elif event.event_type == 'WaypointGroup':
            waypoints = event.data.get('waypoints', {})
            for net_id_str, positions in waypoints.items():
                net_id = int(net_id_str)
                if positions:  # Take the last position in the waypoint list
                    last_pos = positions[-1]
                    self.game_state.positions[net_id] = Position(
                        x=last_pos['x'], 
                        z=last_pos['z']
                    )
        
        # Add more event type handlers as needed
    
    def _get_observation(self) -> ObsType:
        """Get current observation using callback"""
        if self.game_state is None:
            # Create empty game state for callback
            empty_state = GameState(game_id=0, current_time=0.0)
            return self.observation_callback.compute_observation(empty_state, {})
        
        # Use callback to compute observation
        info = self._get_info()
        return self.observation_callback.compute_observation(self.game_state, info)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        return {
            'game_id': self.current_game_idx,
            'current_time': self.current_time,
            'game_state': self.game_state,
            'terminated': self._terminated,
            'truncated': self._truncated,
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment"""
        if self.render_mode == "human":
            print(f"Game {self.current_game_idx} at t={self.current_time:.1f}s")
            if self.game_state:
                print(f"  Heroes: {len(self.game_state.heroes)}")
                print(f"  Events this step: {len(self.game_state.events)}")
                for i, (net_id, hero) in enumerate(list(self.game_state.heroes.items())[:5]):
                    pos = self.game_state.get_position(net_id)
                    pos_str = f"({pos.x:.0f}, {pos.z:.0f})" if pos else "unknown"
                    print(f"    {hero.get('name', f'Hero{net_id}')}: {pos_str}")
                if len(self.game_state.heroes) > 5:
                    print(f"    ... and {len(self.game_state.heroes) - 5} more")
        return None
    
    def close(self) -> None:
        """Close the environment"""
        pass
    
    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space description"""
        return {
            'heroes': 'Dict[int, Dict[str, Any]]',
            'positions': 'Dict[int, Position]',
            'current_time': 'float',
            'events': 'List[GameEvent]'
        }
    
    def sample_random_game(self) -> int:
        """Sample a random game index"""
        return random.randint(0, len(self.dataset) - 1)

class MultiEnvManager:
    """Manager for multiple parallel environments"""
    
    def __init__(self, 
                 dataset: ReplayDataset, 
                 num_envs: int = 4,
                 **env_kwargs):
        """
        Initialize multiple environments
        
        Args:
            dataset: Shared dataset
            num_envs: Number of parallel environments
            **env_kwargs: Additional arguments for LeagueReplaysEnv
        """
        self.dataset = dataset
        self.num_envs = num_envs
        self.envs = [
            LeagueReplaysEnv(dataset, **env_kwargs) 
            for _ in range(num_envs)
        ]
    
    def reset(self, game_indices: Optional[List[int]] = None) -> List[GameState]:
        """Reset all environments"""
        if game_indices is None:
            game_indices = [None] * self.num_envs
        
        states = []
        for i, env in enumerate(self.envs):
            state = env.reset(game_indices[i])
            states.append(state)
        
        return states
    
    def step(self) -> List[Tuple[GameState, List[GameEvent], bool, Dict[str, Any]]]:
        """Step all environments"""
        results = []
        for env in self.envs:
            result = env.step()
            results.append(result)
        return results
    
    def get_active_envs(self) -> List[int]:
        """Get indices of environments that are not done"""
        active = []
        for i, env in enumerate(self.envs):
            if env.game_state is not None:
                active.append(i)
        return active

