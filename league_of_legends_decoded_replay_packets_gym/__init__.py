"""
League of Legends Decoded Replay Packets Gym

A Gymnasium environment for League of Legends decoded replay packets,
enabling esports research, AI development, and gameplay analysis.

This package provides:
- Gymnasium-compliant environment for RL/ML applications
- Access to professional replay data from HuggingFace datasets  
- High-performance parsing with Rust backend acceleration
- Flexible observation system for custom feature extraction
- AI examples including OpenLeague5 action prediction system
- Visualization tools for champion movement and analysis

Main components:
- LeagueReplaysEnv: Gymnasium environment for decoded replay packets
- ReplayDataset: Dataset management with HuggingFace integration
- GameState: Rich game state representation with decoded packets
- Observation system: Flexible feature extraction framework
- Examples: OpenLeague5 AI system and visualization tools

Example usage:
    >>> import league_of_legends_decoded_replay_packets_gym as lol_gym
    
    # Load professional data from HuggingFace
    >>> dataset = lol_gym.ReplayDataset([
    ...     "12_22/batch_001.jsonl.gz"
    ... ], repo_id="maknee/league-of-legends-decoded-replay-packets")
    >>> dataset.load()
    
    # Create Gymnasium environment
    >>> env = lol_gym.LeagueReplaysEnv(dataset)
    >>> obs, info = env.reset()
"""

__version__ = "0.1.1"
__author__ = "League Parser Team"
__email__ = "parser@league.com"

from .types import Position, GameEvent
from .league_replays_gym import (
    ReplayDataset, 
    LeagueReplaysEnv, 
    MultiEnvManager,
    GameState
)
from .packets import (
    PacketParser,
    PacketType,
    get_packet_types,
    get_packet_schema
)
from .interpolation import (
    PositionInterpolator,
    GameStateInterpolator,
    InterpolatedPosition,
    MovementVector
)
from .diff_system import (
    PacketDiffEngine,
    GameStateDiffAnalyzer,
    StepDiff,
    PacketDiff,
    PositionDiff,
    ChangeType
)
from .observations import (
    ObservationCallback,
    PositionObservation,
    MinimapObservation,
    EventHistoryObservation,
    CustomObservation,
    CompositeObservation,
    create_position_observation,
    create_minimap_observation,
    create_event_observation,
    create_custom_observation,
    combine_observations
)
from .observation_utils import (
    normalize_position,
    denormalize_position,
    normalize_time,
    extract_hero_features,
    create_minimap_grid,
    compute_distance_matrix,
    find_nearby_entities,
    extract_event_features,
    create_team_features,
    validate_observation_space,
    debug_observation,
    create_observation_summary
)

__all__ = [
    # Shared Types
    "Position",
    "GameEvent",
    
    # Unified Parser
    "UnifiedLeagueParser",
    "ParseMethod", 
    "ParseResult",
    
    # Gymnasium Environment
    "ReplayDataset",
    "LeagueReplaysEnv",
    "MultiEnvManager", 
    "GameState",
    
    # Packet System
    "PacketParser",
    "PacketType",
    "get_packet_types",
    "get_packet_schema",
    
    # Interpolation System
    "PositionInterpolator",
    "GameStateInterpolator",
    "InterpolatedPosition",
    "MovementVector",
    
    # Diff System
    "PacketDiffEngine",
    "GameStateDiffAnalyzer",
    "StepDiff",
    "PacketDiff",
    "PositionDiff",
    "ChangeType",
    
    # Observation System
    "ObservationCallback",
    "PositionObservation",
    "MinimapObservation", 
    "EventHistoryObservation",
    "CustomObservation",
    "CompositeObservation",
    "create_position_observation",
    "create_minimap_observation",
    "create_event_observation",
    "create_custom_observation",
    "combine_observations",
    
    # Observation Utilities
    "normalize_position",
    "denormalize_position", 
    "normalize_time",
    "extract_hero_features",
    "create_minimap_grid",
    "compute_distance_matrix",
    "find_nearby_entities",
    "extract_event_features",
    "create_team_features",
    "validate_observation_space",
    "debug_observation",
    "create_observation_summary"
]