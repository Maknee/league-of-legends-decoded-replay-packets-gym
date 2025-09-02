"""
Action Space Definition for OpenLeague5

This module defines the action space for League of Legends action prediction,
following the auto-regressive approach used in AlphaStar and OpenAI Five.

The action space is hierarchical and auto-regressive:
1. Action Type (move, attack, ability, item, recall)
2. Target Selection (coordinates, unit, item slot)
3. Additional Parameters (ability rank, movement modifiers)

This decomposition allows the model to predict complex actions step-by-step,
similar to how human players make decisions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import IntEnum
import sys
import os

# Import from the renamed package
from ...types import Position


class ActionType(IntEnum):
    """Primary action types in League of Legends"""
    NOOP = 0           # No action / continue current action
    MOVE = 1           # Move to location
    ATTACK = 2         # Attack target (unit or ground)
    ABILITY_Q = 3      # Use Q ability
    ABILITY_W = 4      # Use W ability  
    ABILITY_E = 5      # Use E ability
    ABILITY_R = 6      # Use R (ultimate) ability
    ITEM_ACTIVE = 7    # Use active item
    RECALL = 8         # Recall to base
    SHOP = 9           # Shop interaction
    LEVEL_UP = 10      # Level up ability
    PING = 11          # Communication ping


class TargetType(IntEnum):
    """Target selection types for actions"""
    NONE = 0           # No target required
    GROUND = 1         # Ground coordinate target
    UNIT = 2           # Unit target (enemy, ally, neutral)
    ITEM_SLOT = 3      # Inventory slot
    ABILITY_SLOT = 4   # Ability slot for level up
    SHOP_ITEM = 5      # Shop item purchase


@dataclass
class ActionPrediction:
    """Complete action prediction with all parameters"""
    action_type: int                    # Primary action type
    target_type: int                    # Target selection type  
    coordinates: Optional[Tuple[float, float]] = None  # World coordinates (normalized)
    unit_target: Optional[int] = None   # Target unit index
    ability_slot: Optional[int] = None  # Ability slot (Q=0, W=1, E=2, R=3)
    item_slot: Optional[int] = None     # Item slot (0-5)
    shop_item_id: Optional[int] = None  # Shop item identifier
    confidence: float = 0.0             # Model confidence in prediction
    value: float = 0.0                  # Estimated state value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'action_type': self.action_type,
            'target_type': self.target_type,
            'coordinates': self.coordinates,
            'unit_target': self.unit_target,
            'ability_slot': self.ability_slot,
            'item_slot': self.item_slot,
            'shop_item_id': self.shop_item_id,
            'confidence': self.confidence,
            'value': self.value
        }
    
    def get_action_description(self) -> str:
        """Get human-readable description of the action"""
        action_names = {
            ActionType.NOOP: "No Action",
            ActionType.MOVE: "Move",
            ActionType.ATTACK: "Attack",
            ActionType.ABILITY_Q: "Use Q Ability",
            ActionType.ABILITY_W: "Use W Ability", 
            ActionType.ABILITY_E: "Use E Ability",
            ActionType.ABILITY_R: "Use R Ability",
            ActionType.ITEM_ACTIVE: "Use Item",
            ActionType.RECALL: "Recall",
            ActionType.SHOP: "Shop",
            ActionType.LEVEL_UP: "Level Up",
            ActionType.PING: "Ping"
        }
        
        base_action = action_names.get(self.action_type, f"Unknown({self.action_type})")
        
        if self.coordinates:
            base_action += f" at ({self.coordinates[0]:.2f}, {self.coordinates[1]:.2f})"
        
        if self.unit_target is not None:
            base_action += f" targeting unit {self.unit_target}"
            
        if self.ability_slot is not None:
            ability_names = ['Q', 'W', 'E', 'R']
            if 0 <= self.ability_slot < len(ability_names):
                base_action += f" ({ability_names[self.ability_slot]})"
                
        if self.item_slot is not None:
            base_action += f" item slot {self.item_slot}"
            
        return base_action


class ActionSpace:
    """
    Defines the complete action space for League of Legends
    
    Uses auto-regressive prediction following AlphaStar's approach:
    1. Predict action type
    2. Predict target type (conditioned on action)
    3. Predict specific parameters (conditioned on action + target type)
    """
    
    def __init__(self, coordinate_bins: int = 64, max_units: int = 50):
        """
        Initialize action space
        
        Args:
            coordinate_bins: Number of discrete coordinate bins per axis
            max_units: Maximum number of units that can be targeted
        """
        self.coordinate_bins = coordinate_bins
        self.max_units = max_units
        
        # Action type constraints - which target types are valid for each action
        self.action_target_constraints = {
            ActionType.NOOP: [TargetType.NONE],
            ActionType.MOVE: [TargetType.GROUND],
            ActionType.ATTACK: [TargetType.GROUND, TargetType.UNIT],
            ActionType.ABILITY_Q: [TargetType.NONE, TargetType.GROUND, TargetType.UNIT],
            ActionType.ABILITY_W: [TargetType.NONE, TargetType.GROUND, TargetType.UNIT],
            ActionType.ABILITY_E: [TargetType.NONE, TargetType.GROUND, TargetType.UNIT],
            ActionType.ABILITY_R: [TargetType.NONE, TargetType.GROUND, TargetType.UNIT],
            ActionType.ITEM_ACTIVE: [TargetType.ITEM_SLOT, TargetType.GROUND, TargetType.UNIT],
            ActionType.RECALL: [TargetType.NONE],
            ActionType.SHOP: [TargetType.SHOP_ITEM],
            ActionType.LEVEL_UP: [TargetType.ABILITY_SLOT],
            ActionType.PING: [TargetType.GROUND]
        }
        
        # Ability slots for level up
        self.ability_slots = 4  # Q, W, E, R
        
        # Item slots
        self.item_slots = 7  # 6 inventory + 1 trinket
        
        # Shop items (simplified - would be much larger in practice)
        self.shop_items = 100  # Placeholder for item catalog size
    
    def get_action_type_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert action type logits to probabilities"""
        return torch.softmax(logits, dim=-1)
    
    def get_valid_target_types(self, action_type: int) -> List[int]:
        """Get valid target types for a given action type"""
        return self.action_target_constraints.get(action_type, [TargetType.NONE])
    
    def sample_action_type(self, logits: torch.Tensor, 
                          temperature: float = 1.0) -> Tuple[int, float]:
        """
        Sample action type from logits
        
        Returns:
            (action_type, confidence)
        """
        probs = torch.softmax(logits / temperature, dim=-1)
        action_type = torch.multinomial(probs, 1).item()
        confidence = probs[action_type].item()
        return action_type, confidence
    
    def sample_target_type(self, action_type: int, 
                          logits: Optional[torch.Tensor] = None) -> int:
        """
        Sample target type given action type
        
        If logits not provided, uses uniform sampling over valid targets
        """
        valid_targets = self.get_valid_target_types(action_type)
        
        if logits is not None:
            # Use model predictions, masked to valid targets
            masked_logits = logits.clone()
            mask = torch.ones_like(masked_logits) * float('-inf')
            for target in valid_targets:
                if target < len(masked_logits):
                    mask[target] = 0
            masked_logits += mask
            probs = torch.softmax(masked_logits, dim=-1)
            target_type = torch.multinomial(probs, 1).item()
        else:
            # Uniform sampling over valid targets
            target_type = np.random.choice(valid_targets)
        
        return target_type
    
    def sample_coordinates(self, coord_x_logits: torch.Tensor,
                          coord_y_logits: torch.Tensor,
                          temperature: float = 1.0) -> Tuple[float, float]:
        """
        Sample coordinates from discretized distributions
        
        Returns:
            (x, y) coordinates normalized to [0, 1]
        """
        x_probs = torch.softmax(coord_x_logits / temperature, dim=-1)
        y_probs = torch.softmax(coord_y_logits / temperature, dim=-1)
        
        x_bin = torch.multinomial(x_probs, 1).item()
        y_bin = torch.multinomial(y_probs, 1).item()
        
        # Convert bin indices to normalized coordinates
        x_coord = (x_bin + 0.5) / self.coordinate_bins  # Center of bin
        y_coord = (y_bin + 0.5) / self.coordinate_bins
        
        return x_coord, y_coord
    
    def sample_unit_target(self, unit_logits: torch.Tensor,
                          unit_mask: torch.Tensor,
                          temperature: float = 1.0) -> int:
        """
        Sample unit target using pointer network logits
        
        Args:
            unit_logits: [max_units] - logits for each unit
            unit_mask: [max_units] - mask for valid units
            temperature: Sampling temperature
            
        Returns:
            Unit index
        """
        # Mask invalid units
        masked_logits = unit_logits.clone()
        masked_logits[~unit_mask] = float('-inf')
        
        probs = torch.softmax(masked_logits / temperature, dim=-1)
        unit_idx = torch.multinomial(probs, 1).item()
        
        return unit_idx
    
    def predict_full_action(self, 
                           action_logits: torch.Tensor,
                           coord_x_logits: torch.Tensor,
                           coord_y_logits: torch.Tensor,
                           unit_logits: torch.Tensor,
                           unit_mask: torch.Tensor,
                           values: torch.Tensor,
                           temperature: float = 1.0) -> ActionPrediction:
        """
        Predict complete action using auto-regressive sampling
        
        Args:
            action_logits: [num_action_types] - logits for action types
            coord_x_logits: [coordinate_bins] - logits for x coordinates
            coord_y_logits: [coordinate_bins] - logits for y coordinates  
            unit_logits: [max_units] - logits for unit targets
            unit_mask: [max_units] - mask for valid units
            values: [1] - state value estimate
            temperature: Sampling temperature
            
        Returns:
            Complete ActionPrediction
        """
        # Step 1: Sample action type
        action_type, action_confidence = self.sample_action_type(action_logits, temperature)
        
        # Step 2: Determine target type (simplified - using first valid target)
        valid_targets = self.get_valid_target_types(action_type)
        target_type = valid_targets[0]
        
        # Step 3: Sample parameters based on target type
        coordinates = None
        unit_target = None
        ability_slot = None
        item_slot = None
        shop_item_id = None
        
        if target_type == TargetType.GROUND:
            coordinates = self.sample_coordinates(coord_x_logits, coord_y_logits, temperature)
        
        elif target_type == TargetType.UNIT:
            unit_target = self.sample_unit_target(unit_logits, unit_mask, temperature)
            # Also sample coordinates for unit targeting (for skillshots, etc.)
            coordinates = self.sample_coordinates(coord_x_logits, coord_y_logits, temperature)
        
        elif target_type == TargetType.ABILITY_SLOT:
            # For level up actions
            ability_slot = np.random.randint(0, self.ability_slots)
        
        elif target_type == TargetType.ITEM_SLOT:
            # For item usage
            item_slot = np.random.randint(0, self.item_slots)
        
        elif target_type == TargetType.SHOP_ITEM:
            # For shop purchases
            shop_item_id = np.random.randint(0, self.shop_items)
        
        return ActionPrediction(
            action_type=action_type,
            target_type=target_type,
            coordinates=coordinates,
            unit_target=unit_target,
            ability_slot=ability_slot,
            item_slot=item_slot,
            shop_item_id=shop_item_id,
            confidence=action_confidence,
            value=values.item() if values.numel() > 0 else 0.0
        )
    
    def coordinates_to_world(self, norm_coords: Tuple[float, float],
                            map_size: float = 15000.0) -> Position:
        """
        Convert normalized coordinates to world coordinates
        
        Args:
            norm_coords: (x, y) coordinates in [0, 1] range
            map_size: Size of the game map
            
        Returns:
            Position in world coordinates
        """
        x_world = (norm_coords[0] - 0.5) * map_size
        z_world = (norm_coords[1] - 0.5) * map_size
        
        return Position(x=x_world, z=z_world)
    
    def world_to_coordinates(self, position: Position, 
                            map_size: float = 15000.0) -> Tuple[float, float]:
        """
        Convert world coordinates to normalized coordinates
        
        Args:
            position: Position in world coordinates
            map_size: Size of the game map
            
        Returns:
            (x, y) coordinates in [0, 1] range
        """
        x_norm = (position.x / map_size) + 0.5
        z_norm = (position.z / map_size) + 0.5
        
        # Clamp to valid range
        x_norm = max(0.0, min(1.0, x_norm))
        z_norm = max(0.0, min(1.0, z_norm))
        
        return x_norm, z_norm
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space"""
        return {
            'num_action_types': len(ActionType),
            'coordinate_bins': self.coordinate_bins,
            'max_units': self.max_units,
            'ability_slots': self.ability_slots,
            'item_slots': self.item_slots,
            'shop_items': self.shop_items,
            'action_target_constraints': {
                int(k): [int(t) for t in v] 
                for k, v in self.action_target_constraints.items()
            }
        }


def create_action_space(coordinate_bins: int = 64, max_units: int = 50) -> ActionSpace:
    """Factory function to create action space"""
    return ActionSpace(coordinate_bins=coordinate_bins, max_units=max_units)


class ActionSequenceAnalyzer:
    """Utility class for analyzing sequences of actions from replays"""
    
    def __init__(self, action_space: ActionSpace):
        self.action_space = action_space
        self.action_sequences = []
    
    def extract_action_from_events(self, events: List[Any],
                                 hero_positions: Dict[int, Position]) -> List[ActionPrediction]:
        """
        Extract action sequences from replay events
        
        This is a simplified implementation - real action extraction would be
        much more sophisticated, analyzing movement patterns, ability usage,
        item purchases, etc.
        """
        actions = []
        
        for event in events:
            if event.event_type == 'WaypointGroup':
                # Extract movement actions from waypoint data
                waypoints = event.data.get('waypoints', {})
                for net_id_str, positions in waypoints.items():
                    if positions:
                        # Create move action
                        last_pos = positions[-1]
                        norm_coords = self.action_space.world_to_coordinates(
                            Position(x=last_pos['x'], z=last_pos['z'])
                        )
                        
                        action = ActionPrediction(
                            action_type=ActionType.MOVE,
                            target_type=TargetType.GROUND,
                            coordinates=norm_coords,
                            confidence=1.0  # Perfect confidence for ground truth
                        )
                        actions.append(action)
            
            elif event.event_type == 'CastSpellAns':
                # Extract ability usage
                spell_data = event.data
                spell_slot = spell_data.get('spell_slot', 0)
                
                # Map spell slot to action type
                ability_actions = {
                    0: ActionType.ABILITY_Q,
                    1: ActionType.ABILITY_W, 
                    2: ActionType.ABILITY_E,
                    3: ActionType.ABILITY_R
                }
                
                action_type = ability_actions.get(spell_slot, ActionType.ABILITY_Q)
                
                # Determine target type and parameters
                target_pos = spell_data.get('target_position')
                target_unit = spell_data.get('target_unit_net_id')
                
                if target_pos:
                    target_type = TargetType.GROUND
                    coordinates = self.action_space.world_to_coordinates(
                        Position(x=target_pos['x'], z=target_pos['z'])
                    )
                elif target_unit:
                    target_type = TargetType.UNIT
                    coordinates = None
                    # Would need to map target_unit to unit index
                else:
                    target_type = TargetType.NONE
                    coordinates = None
                
                action = ActionPrediction(
                    action_type=action_type,
                    target_type=target_type,
                    coordinates=coordinates,
                    unit_target=target_unit if target_unit else None,
                    ability_slot=spell_slot,
                    confidence=1.0
                )
                actions.append(action)
        
        return actions
    
    def analyze_action_patterns(self, action_sequences: List[List[ActionPrediction]]) -> Dict[str, Any]:
        """
        Analyze patterns in action sequences
        
        Returns:
            Dictionary with action statistics and patterns
        """
        all_actions = [action for sequence in action_sequences for action in sequence]
        
        if not all_actions:
            return {'total_actions': 0}
        
        # Action type distribution
        action_counts = {}
        for action in all_actions:
            action_type = action.action_type
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Target type distribution
        target_counts = {}
        for action in all_actions:
            target_type = action.target_type
            target_counts[target_type] = target_counts.get(target_type, 0) + 1
        
        # Coordinate distribution (for spatial analysis)
        coordinates = [action.coordinates for action in all_actions 
                      if action.coordinates is not None]
        
        coord_stats = {}
        if coordinates:
            x_coords = [c[0] for c in coordinates]
            y_coords = [c[1] for c in coordinates]
            
            coord_stats = {
                'mean_x': np.mean(x_coords),
                'mean_y': np.mean(y_coords),
                'std_x': np.std(x_coords),
                'std_y': np.std(y_coords)
            }
        
        return {
            'total_actions': len(all_actions),
            'action_type_distribution': action_counts,
            'target_type_distribution': target_counts,
            'coordinate_statistics': coord_stats,
            'sequences_analyzed': len(action_sequences)
        }


if __name__ == "__main__":
    # Test the action space
    action_space = create_action_space()
    
    # Test action prediction
    batch_size = 1
    num_actions = len(ActionType)
    coordinate_bins = action_space.coordinate_bins
    max_units = action_space.max_units
    
    # Mock model outputs
    action_logits = torch.randn(batch_size, num_actions)
    coord_x_logits = torch.randn(batch_size, coordinate_bins)
    coord_y_logits = torch.randn(batch_size, coordinate_bins) 
    unit_logits = torch.randn(batch_size, max_units)
    unit_mask = torch.ones(batch_size, max_units, dtype=torch.bool)
    unit_mask[0, 10:] = False  # Only first 10 units are valid
    values = torch.tensor([[0.75]])  # High confidence state value
    
    # Predict action
    predicted_action = action_space.predict_full_action(
        action_logits[0], coord_x_logits[0], coord_y_logits[0],
        unit_logits[0], unit_mask[0], values[0]
    )
    
    print("Action space test successful!")
    print(f"Predicted action: {predicted_action.get_action_description()}")
    print(f"Action type: {predicted_action.action_type}")
    print(f"Target type: {predicted_action.target_type}")
    print(f"Coordinates: {predicted_action.coordinates}")
    print(f"Unit target: {predicted_action.unit_target}")
    print(f"Confidence: {predicted_action.confidence:.3f}")
    print(f"Value: {predicted_action.value:.3f}")
    
    # Test action space info
    info = action_space.get_action_space_info()
    print(f"\\nAction space info:")
    print(f"- Action types: {info['num_action_types']}")
    print(f"- Coordinate bins: {info['coordinate_bins']}")
    print(f"- Max units: {info['max_units']}")
    print(f"- Ability slots: {info['ability_slots']}")
    print(f"- Item slots: {info['item_slots']}")
    
    # Test coordinate conversion
    world_pos = action_space.coordinates_to_world((0.3, 0.7))
    norm_coords = action_space.world_to_coordinates(world_pos)
    print(f"\\nCoordinate conversion test:")
    print(f"Normalized (0.3, 0.7) -> World {world_pos.x:.0f}, {world_pos.z:.0f}")
    print(f"World -> Normalized {norm_coords[0]:.3f}, {norm_coords[1]:.3f}")