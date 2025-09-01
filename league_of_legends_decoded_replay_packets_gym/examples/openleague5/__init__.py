"""
OpenLeague5: Action Prediction for League of Legends

An implementation inspired by OpenAI Five and AlphaStar approaches,
adapted for League of Legends replay analysis and action prediction.

This package provides:
- Neural network architectures for multi-modal game state processing
- Auto-regressive action prediction with pointer networks
- Behavior cloning from professional replay data
- Command-line interface for training and prediction

Main components:
- openleague5_model: Core neural network architectures
- state_encoder: Multi-modal state representation
- action_space: Auto-regressive action definitions
- trainer: Behavior cloning training pipeline
- cli: Command-line interface
"""

__version__ = "1.0.0"
__author__ = "OpenLeague5 Team"

from .openleague5_model import OpenLeague5Model
from .state_encoder import StateEncoder, GameStateVector
from .action_space import ActionSpace, ActionPrediction
from .trainer import Trainer, TrainingConfig
from .cli import main as cli_main

__all__ = [
    "OpenLeague5Model",
    "StateEncoder", 
    "GameStateVector",
    "ActionSpace",
    "ActionPrediction", 
    "Trainer",
    "TrainingConfig",
    "cli_main"
]