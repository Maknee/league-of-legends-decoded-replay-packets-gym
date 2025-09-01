#!/usr/bin/env python3
"""
Shared types and data structures for League of Legends replay parsing.

This module contains common data structures used throughout the package
to avoid circular imports and ensure consistency.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class Position:
    """2D position in the game world"""
    x: float
    z: float
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return ((self.x - other.x) ** 2 + (self.z - other.z) ** 2) ** 0.5
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.z])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {"x": self.x, "z": self.z}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position':
        """Create Position from dictionary"""
        return cls(x=data["x"], z=data["z"])


@dataclass
class GameEvent:
    """Base class for all game events"""
    event_type: str
    time: float
    data: Dict[str, Any]
    
    @classmethod
    def from_packet(cls, packet: Dict[str, Any]) -> 'GameEvent':
        """Create GameEvent from packet dictionary"""
        event_type = list(packet.keys())[0]
        event_data = packet[event_type]
        return cls(
            event_type=event_type,
            time=event_data.get('time', 0.0),
            data=event_data
        )