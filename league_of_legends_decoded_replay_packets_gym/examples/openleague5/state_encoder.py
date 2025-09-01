"""
Game State Encoder for OpenLeague5

This module converts League of Legends game states from the replay parser
into multi-modal neural network inputs suitable for the OpenLeague5 model.

The encoder handles:
1. Spatial features: Minimap-style 2D representations
2. Unit features: Hero stats, positions, abilities  
3. Global features: Game time, gold, objectives
4. Temporal sequencing: Rolling window of past states

Based on OpenAI Five and AlphaStar state representation principles.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
import league_replays_parser as lrp
from league_replays_parser.types import Position, GameEvent
from league_replays_parser.league_replays_gym import GameState


@dataclass
class GameStateVector:
    """Structured representation of encoded game state"""
    spatial_features: torch.Tensor  # [channels, height, width]
    unit_features: torch.Tensor     # [max_units, unit_feature_dim] 
    unit_mask: torch.Tensor         # [max_units] - boolean mask for valid units
    global_features: torch.Tensor   # [global_feature_dim]
    timestamp: float                # Game time
    hero_actions: Optional[Dict[str, Any]] = None  # Ground truth actions (for training)


class StateEncoder:
    """
    Converts League of Legends GameState objects into neural network inputs
    
    This encoder follows the multi-modal approach used in OpenAI Five and AlphaStar,
    creating spatial, unit-based, and global feature representations.
    """
    
    def __init__(self, 
                 spatial_resolution: int = 64,
                 max_units: int = 50,
                 map_size: float = 15000.0,
                 max_game_time: float = 3600.0):  # 1 hour max
        """
        Initialize state encoder
        
        Args:
            spatial_resolution: Size of spatial feature maps (64x64)
            max_units: Maximum number of units to track
            map_size: Size of the League map in game units
            max_game_time: Maximum game duration for normalization
        """
        self.spatial_resolution = spatial_resolution
        self.max_units = max_units
        self.map_size = map_size
        self.max_game_time = max_game_time
        
        # Feature dimensions
        self.spatial_channels = 16  # Different feature types in spatial representation
        self.unit_feature_dim = 64  # Features per unit
        self.global_feature_dim = 32  # Global game features
        
        # Spatial feature channel assignments
        self.SPATIAL_CHANNELS = {
            'ally_heroes': 0,
            'enemy_heroes': 1, 
            'ally_minions': 2,
            'enemy_minions': 3,
            'neutral_minions': 4,
            'ally_structures': 5,
            'enemy_structures': 6,
            'terrain': 7,
            'river': 8,
            'jungle': 9,
            'vision': 10,
            'objectives': 11,
            'recent_combat': 12,
            'pathing': 13,
            'danger_zones': 14,
            'strategic_locations': 15
        }
        
        # Unit feature assignments
        self.UNIT_FEATURES = {
            'position': (0, 2),      # x, z coordinates (normalized)
            'health': (2, 4),        # current, max health (normalized)
            'mana': (4, 6),          # current, max mana (normalized)
            'level': (6, 7),         # level (normalized 1-18)
            'gold': (7, 8),          # current gold (normalized)
            'abilities': (8, 12),    # ability cooldowns (normalized)
            'items': (12, 18),       # item presence (6 slots, binary)
            'stats': (18, 26),       # AD, AP, armor, MR, AS, MS, CDR, lifesteal
            'team': (26, 27),        # team (0=ally, 1=enemy)
            'champion_id': (27, 28), # champion identifier (normalized)
            'role': (28, 32),        # role one-hot (top, jg, mid, adc, sup) 
            'recently_damaged': (32, 33),  # damage taken in last 5s
            'recently_killed': (33, 34),   # killed enemy in last 10s
            'vision_score': (34, 35),      # normalized vision score
            'cs_score': (35, 36),          # normalized CS
            'kda': (36, 39),               # kills, deaths, assists (normalized)
            'positioning': (39, 43),       # relative to team, enemies, objectives
            'threat_level': (43, 44),      # estimated threat (normalized)
            'utility_score': (44, 45),     # estimated utility (normalized)
            'last_action': (45, 51),       # one-hot of last action type
            'momentum': (51, 53),          # velocity vector (normalized)
            'strategic_value': (53, 56),   # proximity to objectives (dragon, baron, towers)
            'communication': (56, 60),     # recent pings/communication features
            'itemization_path': (60, 64)   # item build direction features
        }
        
        # Global feature assignments  
        self.GLOBAL_FEATURES = {
            'game_time': (0, 1),          # normalized game time
            'game_phase': (1, 4),         # early/mid/late game one-hot
            'team_gold_diff': (4, 5),     # normalized gold difference
            'team_exp_diff': (5, 6),      # normalized experience difference 
            'tower_differential': (6, 7), # tower advantage (normalized)
            'dragon_stacks': (7, 11),     # dragon types taken (4 types)
            'baron_buff': (11, 12),       # baron buff active
            'elder_dragon': (12, 13),     # elder dragon active
            'inhibitor_down': (13, 15),   # inhibitors down (ally, enemy)
            'team_fight_state': (15, 18), # no fight, skirmish, team fight
            'map_control': (18, 22),      # vision control quadrants
            'objective_pressure': (22, 26), # pressure on dragon/baron/towers
            'team_positioning': (26, 28),  # spread vs grouped
            'aggression_level': (28, 29),  # recent combat frequency
            'economy_state': (29, 30),     # farming vs fighting focus
            'win_probability': (30, 31),   # estimated win probability
            'strategic_timer': (31, 32)    # time to next major objective
        }
        
    def encode_game_state(self, game_state: GameState,
                         previous_states: Optional[List[GameState]] = None) -> GameStateVector:
        """
        Encode a complete game state into neural network inputs
        
        Args:
            game_state: Current game state from replay
            previous_states: List of recent previous states for temporal features
            
        Returns:
            GameStateVector with all encoded features
        """
        # Encode spatial features (minimap-style representation)
        spatial_features = self._encode_spatial_features(game_state, previous_states)
        
        # Encode unit features (all heroes, minions, etc.)
        unit_features, unit_mask = self._encode_unit_features(game_state)
        
        # Encode global features (game-wide state)
        global_features = self._encode_global_features(game_state, previous_states)
        
        return GameStateVector(
            spatial_features=spatial_features,
            unit_features=unit_features,
            unit_mask=unit_mask,
            global_features=global_features,
            timestamp=game_state.current_time
        )
    
    def _encode_spatial_features(self, game_state: GameState, 
                               previous_states: Optional[List[GameState]] = None) -> torch.Tensor:
        """
        Create minimap-style spatial feature representation
        
        Returns:
            Tensor of shape [spatial_channels, height, width]
        """
        spatial_map = torch.zeros(self.spatial_channels, 
                                self.spatial_resolution, 
                                self.spatial_resolution)
        
        # Helper function to convert world coordinates to grid coordinates
        def world_to_grid(pos: Position) -> Tuple[int, int]:
            x = int((pos.x + self.map_size/2) / self.map_size * self.spatial_resolution)
            z = int((pos.z + self.map_size/2) / self.map_size * self.spatial_resolution)
            x = np.clip(x, 0, self.spatial_resolution - 1)
            z = np.clip(z, 0, self.spatial_resolution - 1)
            return x, z
        
        # Encode hero positions
        ally_heroes = []
        enemy_heroes = []
        
        for net_id, hero_info in game_state.heroes.items():
            pos = game_state.get_position(net_id)
            if pos is None:
                continue
                
            x, z = world_to_grid(pos)
            team = hero_info.get('team', 'unknown')
            
            if team in ['ORDER', 'BLUE']:  # Assuming ORDER/BLUE is ally
                spatial_map[self.SPATIAL_CHANNELS['ally_heroes'], z, x] = 1.0
                ally_heroes.append((net_id, pos))
            elif team in ['CHAOS', 'RED']:  # Assuming CHAOS/RED is enemy
                spatial_map[self.SPATIAL_CHANNELS['enemy_heroes'], z, x] = 1.0
                enemy_heroes.append((net_id, pos))
        
        # Add strategic terrain features (simplified)
        self._add_terrain_features(spatial_map)
        
        # Add temporal features if previous states available
        if previous_states:
            self._add_temporal_spatial_features(spatial_map, game_state, previous_states)
        
        return spatial_map
    
    def _add_terrain_features(self, spatial_map: torch.Tensor):
        """Add basic terrain features to spatial map"""
        # River (diagonal through middle of map)
        river_channel = self.SPATIAL_CHANNELS['river']
        for i in range(self.spatial_resolution):
            for j in range(self.spatial_resolution):
                # Diagonal river approximation
                if abs(i - j) < 3:  # River width
                    dist_to_center = abs((i + j) / 2 - self.spatial_resolution / 2)
                    if dist_to_center < self.spatial_resolution * 0.3:
                        spatial_map[river_channel, i, j] = 1.0
        
        # Jungle areas (corners and sides)
        jungle_channel = self.SPATIAL_CHANNELS['jungle']
        for i in range(self.spatial_resolution):
            for j in range(self.spatial_resolution):
                # Distance from edges
                edge_dist = min(i, j, self.spatial_resolution - 1 - i, self.spatial_resolution - 1 - j)
                if 5 < edge_dist < 15:  # Jungle ring
                    spatial_map[jungle_channel, i, j] = 0.5
    
    def _add_temporal_spatial_features(self, spatial_map: torch.Tensor, 
                                     current_state: GameState,
                                     previous_states: List[GameState]):
        """Add temporal features like recent movement paths and combat"""
        if not previous_states:
            return
            
        # Track recent hero positions for pathing
        pathing_channel = self.SPATIAL_CHANNELS['pathing']
        combat_channel = self.SPATIAL_CHANNELS['recent_combat']
        
        def world_to_grid(pos: Position) -> Tuple[int, int]:
            x = int((pos.x + self.map_size/2) / self.map_size * self.spatial_resolution)
            z = int((pos.z + self.map_size/2) / self.map_size * self.spatial_resolution)
            x = np.clip(x, 0, self.spatial_resolution - 1)
            z = np.clip(z, 0, self.spatial_resolution - 1)
            return x, z
        
        # Add movement trails from previous states
        for i, prev_state in enumerate(previous_states[-5:]):  # Last 5 states
            decay_factor = (i + 1) / len(previous_states[-5:])  # Newer = stronger
            
            for net_id in current_state.heroes.keys():
                prev_pos = prev_state.get_position(net_id)
                curr_pos = current_state.get_position(net_id)
                
                if prev_pos and curr_pos:
                    # Add to pathing if moved significantly
                    if prev_pos.distance_to(curr_pos) > 100:  # Moved more than 100 units
                        px, pz = world_to_grid(prev_pos)
                        spatial_map[pathing_channel, pz, px] = max(
                            spatial_map[pathing_channel, pz, px], 
                            0.5 * decay_factor
                        )
            
            # Add combat indicators from damage events
            for event in prev_state.events:
                if event.event_type == 'UnitApplyDamage':
                    # Find damage location (simplified)
                    target_id = event.data.get('target_net_id')
                    if target_id:
                        target_pos = prev_state.get_position(target_id)
                        if target_pos:
                            cx, cz = world_to_grid(target_pos)
                            spatial_map[combat_channel, cz, cx] = max(
                                spatial_map[combat_channel, cz, cx],
                                0.8 * decay_factor
                            )
    
    def _encode_unit_features(self, game_state: GameState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all units into feature vectors
        
        Returns:
            unit_features: [max_units, unit_feature_dim]
            unit_mask: [max_units] - boolean mask for valid units
        """
        unit_features = torch.zeros(self.max_units, self.unit_feature_dim)
        unit_mask = torch.zeros(self.max_units, dtype=torch.bool)
        
        unit_idx = 0
        
        # Process heroes first (most important)
        for net_id, hero_info in game_state.heroes.items():
            if unit_idx >= self.max_units:
                break
                
            features = self._encode_single_unit(net_id, hero_info, game_state, is_hero=True)
            if features is not None:
                unit_features[unit_idx] = features
                unit_mask[unit_idx] = True
                unit_idx += 1
        
        # Could add minions, jungle monsters, etc. here
        # For now, focus on heroes only
        
        return unit_features, unit_mask
    
    def _encode_single_unit(self, net_id: int, unit_info: Dict[str, Any], 
                          game_state: GameState, is_hero: bool = False) -> Optional[torch.Tensor]:
        """
        Encode a single unit into feature vector
        
        Returns:
            Feature tensor of size [unit_feature_dim] or None if invalid
        """
        features = torch.zeros(self.unit_feature_dim)
        
        # Get position
        pos = game_state.get_position(net_id)
        if pos is None:
            return None
        
        # Position features (normalized to [0, 1])
        pos_start, pos_end = self.UNIT_FEATURES['position']
        features[pos_start] = (pos.x + self.map_size/2) / self.map_size
        features[pos_start + 1] = (pos.z + self.map_size/2) / self.map_size
        
        if is_hero:
            # Hero-specific features
            
            # Health (normalized - using placeholder values)
            health_start, health_end = self.UNIT_FEATURES['health']
            features[health_start] = 0.8  # Current health ratio (placeholder)
            features[health_start + 1] = 1.0  # Max health (normalized)
            
            # Mana (placeholder)
            mana_start, mana_end = self.UNIT_FEATURES['mana']
            features[mana_start] = 0.6  # Current mana ratio (placeholder)
            features[mana_start + 1] = 1.0  # Max mana (normalized)
            
            # Level (normalized 1-18)
            level_start, level_end = self.UNIT_FEATURES['level']
            features[level_start] = min(unit_info.get('level', 1), 18) / 18.0
            
            # Team encoding
            team_start, team_end = self.UNIT_FEATURES['team']
            team = unit_info.get('team', 'unknown')
            features[team_start] = 0.0 if team in ['ORDER', 'BLUE'] else 1.0
            
            # Champion ID (normalized placeholder)
            champ_start, champ_end = self.UNIT_FEATURES['champion_id']
            champion_name = unit_info.get('champion', 'Unknown')
            # Simple hash-based champion encoding (placeholder)
            champ_hash = hash(champion_name) % 150  # ~150 champions
            features[champ_start] = champ_hash / 150.0
            
            # Role estimation (placeholder - would need more sophisticated logic)
            role_start, role_end = self.UNIT_FEATURES['role']
            # For now, assign roles based on position or other heuristics
            if pos.z > self.map_size * 0.3:  # Upper part of map
                features[role_start] = 1.0  # Top lane
            elif pos.x > self.map_size * 0.3:  # Right part of map
                features[role_start + 3] = 1.0  # ADC
            else:
                features[role_start + 2] = 1.0  # Mid lane (default)
            
            # Add more placeholder features for items, abilities, stats, etc.
            # These would be filled with real data from the replay parser
            
            # Items (binary encoding - placeholder)
            items_start, items_end = self.UNIT_FEATURES['items']
            for i in range(items_end - items_start):
                features[items_start + i] = np.random.random() > 0.7  # Placeholder
            
            # Stats (placeholder values)
            stats_start, stats_end = self.UNIT_FEATURES['stats']
            features[stats_start:stats_end] = torch.rand(stats_end - stats_start) * 0.5 + 0.25
            
            # Recent activity indicators (placeholder)
            features[self.UNIT_FEATURES['recently_damaged'][0]] = 0.0
            features[self.UNIT_FEATURES['recently_killed'][0]] = 0.0
            
            # KDA (normalized placeholder)
            kda_start, kda_end = self.UNIT_FEATURES['kda']
            features[kda_start:kda_end] = torch.tensor([0.1, 0.05, 0.15])  # K, D, A rates
            
            # Momentum (velocity estimation from position)
            momentum_start, momentum_end = self.UNIT_FEATURES['momentum']
            # Would calculate from previous positions - placeholder
            features[momentum_start:momentum_end] = torch.randn(2) * 0.1
            
        return features
    
    def _encode_global_features(self, game_state: GameState,
                              previous_states: Optional[List[GameState]] = None) -> torch.Tensor:
        """
        Encode global game state features
        
        Returns:
            Feature tensor of size [global_feature_dim]
        """
        features = torch.zeros(self.global_feature_dim)
        
        # Game time (normalized)
        time_start, time_end = self.GLOBAL_FEATURES['game_time']
        features[time_start] = min(game_state.current_time / self.max_game_time, 1.0)
        
        # Game phase (early/mid/late)
        phase_start, phase_end = self.GLOBAL_FEATURES['game_phase']
        if game_state.current_time < 900:  # First 15 minutes
            features[phase_start] = 1.0  # Early game
        elif game_state.current_time < 1800:  # 15-30 minutes
            features[phase_start + 1] = 1.0  # Mid game
        else:
            features[phase_start + 2] = 1.0  # Late game
        
        # Team composition and balance features (placeholders)
        # Would analyze team compositions, gold differentials, objective states
        
        # Team gold difference (placeholder)
        gold_start, gold_end = self.GLOBAL_FEATURES['team_gold_diff']
        features[gold_start] = 0.1  # Slight gold lead (placeholder)
        
        # Tower differential (placeholder)
        tower_start, tower_end = self.GLOBAL_FEATURES['tower_differential']  
        features[tower_start] = 0.05  # Slight tower advantage (placeholder)
        
        # Dragon/Baron states (placeholder)
        dragon_start, dragon_end = self.GLOBAL_FEATURES['dragon_stacks']
        features[dragon_start:dragon_end] = torch.tensor([0.3, 0.0, 0.2, 0.1])  # Dragon types
        
        features[self.GLOBAL_FEATURES['baron_buff'][0]] = 0.0  # No baron
        features[self.GLOBAL_FEATURES['elder_dragon'][0]] = 0.0  # No elder
        
        # Map control and strategic state (placeholders)
        control_start, control_end = self.GLOBAL_FEATURES['map_control']
        features[control_start:control_end] = torch.rand(control_end - control_start)
        
        # Team positioning and aggression (placeholders)
        features[self.GLOBAL_FEATURES['aggression_level'][0]] = 0.3
        features[self.GLOBAL_FEATURES['economy_state'][0]] = 0.6
        
        # Estimated win probability (placeholder)
        features[self.GLOBAL_FEATURES['win_probability'][0]] = 0.52  # Slightly favored
        
        return features
    
    def encode_sequence(self, game_states: List[GameState],
                       sequence_length: int = 10) -> List[GameStateVector]:
        """
        Encode a sequence of game states with temporal context
        
        Args:
            game_states: List of GameState objects in chronological order
            sequence_length: Maximum length of sequence to encode
            
        Returns:
            List of GameStateVector objects with temporal context
        """
        encoded_states = []
        
        for i, state in enumerate(game_states[-sequence_length:]):
            # Get previous states for temporal context
            previous_states = game_states[max(0, i-5):i] if i > 0 else None
            
            encoded_state = self.encode_game_state(state, previous_states)
            encoded_states.append(encoded_state)
        
        return encoded_states
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the encoding scheme"""
        return {
            'spatial_channels': self.spatial_channels,
            'spatial_resolution': self.spatial_resolution,
            'unit_feature_dim': self.unit_feature_dim,
            'global_feature_dim': self.global_feature_dim,
            'max_units': self.max_units,
            'spatial_channel_mapping': self.SPATIAL_CHANNELS,
            'unit_feature_mapping': self.UNIT_FEATURES,
            'global_feature_mapping': self.GLOBAL_FEATURES
        }


def create_encoder(spatial_resolution: int = 64, max_units: int = 50) -> StateEncoder:
    """Factory function to create state encoder"""
    return StateEncoder(spatial_resolution=spatial_resolution, max_units=max_units)


if __name__ == "__main__":
    # Test the state encoder
    from league_replays_parser.league_replays_gym import GameState
    from league_replays_parser.types import Position
    
    # Create a dummy game state for testing
    test_state = GameState(
        game_id=12345,
        current_time=900.0,  # 15 minutes
        heroes={
            101: {'name': 'Player1', 'champion': 'Jinx', 'team': 'ORDER', 'level': 10},
            102: {'name': 'Player2', 'champion': 'Thresh', 'team': 'ORDER', 'level': 9},
            201: {'name': 'Enemy1', 'champion': 'Caitlyn', 'team': 'CHAOS', 'level': 11},
            202: {'name': 'Enemy2', 'champion': 'Blitzcrank', 'team': 'CHAOS', 'level': 8},
        },
        positions={
            101: Position(x=2000, z=-3000),  # Bot lane ADC position
            102: Position(x=1900, z=-3100),  # Support position
            201: Position(x=-2000, z=3000),  # Enemy ADC
            202: Position(x=-1900, z=3100),  # Enemy support
        }
    )
    
    # Test encoding
    encoder = create_encoder()
    encoded = encoder.encode_game_state(test_state)
    
    print("State encoding test successful!")
    print(f"Spatial features shape: {encoded.spatial_features.shape}")
    print(f"Unit features shape: {encoded.unit_features.shape}")
    print(f"Unit mask shape: {encoded.unit_mask.shape}")
    print(f"Global features shape: {encoded.global_features.shape}")
    print(f"Valid units: {encoded.unit_mask.sum().item()}")
    print(f"Timestamp: {encoded.timestamp}")
    
    # Print feature information
    info = encoder.get_feature_info()
    print(f"\\nEncoder configuration:")
    print(f"- Spatial channels: {info['spatial_channels']}")
    print(f"- Spatial resolution: {info['spatial_resolution']}")
    print(f"- Unit feature dim: {info['unit_feature_dim']}")
    print(f"- Global feature dim: {info['global_feature_dim']}")
    print(f"- Max units: {info['max_units']}")