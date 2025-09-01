"""
OpenLeague5 Main Entry Point

This module provides the main entry point for the OpenLeague5 action prediction system.
It can be run as a module using: python -m examples.openleague5

The module supports direct execution and provides example usage for different components.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from .cli import main as cli_main
from .openleague5_model import OpenLeague5Model, ModelConfig
from .state_encoder import StateEncoder
from .action_space import ActionSpace
from .trainer import TrainingConfig


def print_banner():
    """Print OpenLeague5 banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                OpenLeague5                                    ║
║                    League of Legends Action Prediction                       ║
║                                                                              ║
║  Inspired by OpenAI Five and AlphaStar approaches                           ║
║  Multi-modal neural networks for professional gameplay analysis             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def example_model_creation():
    """Example: Create and inspect OpenLeague5 model"""
    print("🤖 OpenLeague5 Model Creation Example")
    print("=" * 50)
    
    # Create model configuration
    config = ModelConfig(
        spatial_channels=16,
        spatial_resolution=64,
        max_units=50,
        transformer_dim=256,
        lstm_hidden_size=1024,
        coordinate_bins=64
    )
    
    print(f"📋 Model Configuration:")
    print(f"   Spatial resolution: {config.spatial_resolution}x{config.spatial_resolution}")
    print(f"   Max units: {config.max_units}")
    print(f"   LSTM hidden size: {config.lstm_hidden_size}")
    print(f"   Transformer dimensions: {config.transformer_dim}")
    print(f"   Coordinate bins: {config.coordinate_bins}")
    
    # Create model
    model = OpenLeague5Model(config)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lstm_params = sum(p.numel() for p in model.temporal_lstm.parameters())
    
    print(f"\n🔢 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   LSTM parameters: {lstm_params:,} ({lstm_params/total_params*100:.1f}%)")
    
    # Test forward pass
    print(f"\n🔄 Testing forward pass...")
    
    batch_size = 2
    seq_len = 3
    
    # Create dummy inputs
    spatial = torch.randn(batch_size, seq_len, config.spatial_channels, 
                         config.spatial_resolution, config.spatial_resolution)
    units = torch.randn(batch_size, seq_len, config.max_units, config.unit_feature_dim)
    mask = torch.ones(batch_size, seq_len, config.max_units, dtype=torch.bool)
    mask[:, :, 30:] = False  # Only first 30 units are valid
    global_feat = torch.randn(batch_size, seq_len, config.global_feature_dim)
    
    try:
        with torch.no_grad():
            predictions, hidden = model(spatial, units, mask, global_feat)
        
        print(f"✅ Forward pass successful!")
        print(f"   Action predictions shape: {predictions['action_types'].shape}")
        print(f"   Coordinate predictions shape: {predictions['coordinates_x'].shape}")
        print(f"   Unit target predictions shape: {predictions['unit_targets'].shape}")
        print(f"   Value predictions shape: {predictions['values'].shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
    
    print()


def example_state_encoding():
    """Example: State encoding from game data"""
    print("🔄 State Encoding Example")
    print("=" * 50)
    
    # Import here to avoid circular imports in main module
    import league_replays_parser as lrp
    from league_replays_parser.types import Position
    from league_replays_parser.league_replays_gym import GameState
    
    # Create state encoder
    encoder = StateEncoder(spatial_resolution=64, max_units=50)
    
    print(f"📋 State Encoder Configuration:")
    info = encoder.get_feature_info()
    print(f"   Spatial channels: {info['spatial_channels']}")
    print(f"   Spatial resolution: {info['spatial_resolution']}x{info['spatial_resolution']}")
    print(f"   Unit feature dimension: {info['unit_feature_dim']}")
    print(f"   Global feature dimension: {info['global_feature_dim']}")
    print(f"   Max units: {info['max_units']}")
    
    # Create example game state
    test_state = GameState(
        game_id=12345,
        current_time=900.0,  # 15 minutes
        heroes={
            101: {'name': 'Faker', 'champion': 'Azir', 'team': 'ORDER', 'level': 12},
            102: {'name': 'Gumayusi', 'champion': 'Jinx', 'team': 'ORDER', 'level': 11},
            103: {'name': 'Keria', 'champion': 'Thresh', 'team': 'ORDER', 'level': 10},
            201: {'name': 'Chovy', 'champion': 'Orianna', 'team': 'CHAOS', 'level': 12},
            202: {'name': 'Viper', 'champion': 'Caitlyn', 'team': 'CHAOS', 'level': 11},
        },
        positions={
            101: Position(x=0, z=1000),      # Mid lane
            102: Position(x=2000, z=-3000),  # Bot lane ADC
            103: Position(x=1900, z=-3100),  # Bot lane Support
            201: Position(x=0, z=-1000),     # Enemy mid
            202: Position(x=-2000, z=3000),  # Enemy ADC
        }
    )
    
    print(f"\n🎮 Example Game State:")
    print(f"   Game ID: {test_state.game_id}")
    print(f"   Time: {test_state.current_time:.0f}s ({test_state.current_time/60:.1f} min)")
    print(f"   Heroes: {len(test_state.heroes)}")
    
    for net_id, hero in test_state.heroes.items():
        pos = test_state.get_position(net_id)
        pos_str = f"({pos.x:.0f}, {pos.z:.0f})" if pos else "Unknown"
        print(f"     {hero['name']} ({hero['champion']}): {pos_str}")
    
    # Encode the game state
    print(f"\n🔄 Encoding game state...")
    try:
        encoded = encoder.encode_game_state(test_state)
        
        print(f"✅ Encoding successful!")
        print(f"   Spatial features: {encoded.spatial_features.shape}")
        print(f"   Unit features: {encoded.unit_features.shape}")
        print(f"   Unit mask: {encoded.unit_mask.shape} ({encoded.unit_mask.sum().item()} valid)")
        print(f"   Global features: {encoded.global_features.shape}")
        print(f"   Timestamp: {encoded.timestamp:.0f}s")
        
        # Show some feature statistics
        print(f"\n📊 Feature Statistics:")
        print(f"   Spatial features - min: {encoded.spatial_features.min():.3f}, max: {encoded.spatial_features.max():.3f}")
        print(f"   Unit features - min: {encoded.unit_features.min():.3f}, max: {encoded.unit_features.max():.3f}")
        print(f"   Global features - min: {encoded.global_features.min():.3f}, max: {encoded.global_features.max():.3f}")
        
    except Exception as e:
        print(f"❌ Encoding failed: {e}")
    
    print()


def example_action_prediction():
    """Example: Action space and prediction"""
    print("🎯 Action Prediction Example")
    print("=" * 50)
    
    # Create action space
    action_space = ActionSpace(coordinate_bins=64, max_units=50)
    
    print(f"📋 Action Space Configuration:")
    info = action_space.get_action_space_info()
    print(f"   Action types: {info['num_action_types']}")
    print(f"   Coordinate bins: {info['coordinate_bins']}")
    print(f"   Max units: {info['max_units']}")
    print(f"   Ability slots: {info['ability_slots']}")
    print(f"   Item slots: {info['item_slots']}")
    
    # Simulate model predictions
    print(f"\n🤖 Simulating model predictions...")
    
    batch_size = 1
    num_actions = info['num_action_types']
    coordinate_bins = info['coordinate_bins']
    max_units = info['max_units']
    
    # Mock model outputs with realistic distributions
    action_logits = torch.tensor([[2.1, 3.2, 1.8, 2.5, 1.2, 1.0, 0.5, 0.8, 0.3, 0.2, 0.1, 0.1]])  # Favor move/attack
    coord_x_logits = torch.randn(batch_size, coordinate_bins) + torch.linspace(-1, 1, coordinate_bins)  # Center bias
    coord_y_logits = torch.randn(batch_size, coordinate_bins) + torch.linspace(-1, 1, coordinate_bins)
    unit_logits = torch.randn(batch_size, max_units)
    unit_mask = torch.ones(batch_size, max_units, dtype=torch.bool)
    unit_mask[0, 10:] = False  # Only first 10 units are valid
    values = torch.tensor([[0.65]])  # Slightly positive state value
    
    try:
        # Predict action
        predicted_action = action_space.predict_full_action(
            action_logits[0], coord_x_logits[0], coord_y_logits[0],
            unit_logits[0], unit_mask[0], values[0],
            temperature=1.0
        )
        
        print(f"✅ Action prediction successful!")
        print(f"\n🎯 Predicted Action:")
        print(f"   Description: {predicted_action.get_action_description()}")
        print(f"   Action type: {predicted_action.action_type}")
        print(f"   Target type: {predicted_action.target_type}")
        print(f"   Confidence: {predicted_action.confidence:.3f}")
        print(f"   State value: {predicted_action.value:.3f}")
        
        if predicted_action.coordinates:
            world_pos = action_space.coordinates_to_world(predicted_action.coordinates)
            print(f"   Coordinates: {predicted_action.coordinates} (normalized)")
            print(f"   World position: ({world_pos.x:.0f}, {world_pos.z:.0f})")
        
        if predicted_action.unit_target is not None:
            print(f"   Unit target: {predicted_action.unit_target}")
        
        # Test multiple predictions with different temperatures
        print(f"\n🌡️  Temperature Effects:")
        for temp in [0.5, 1.0, 2.0]:
            temp_prediction = action_space.predict_full_action(
                action_logits[0], coord_x_logits[0], coord_y_logits[0],
                unit_logits[0], unit_mask[0], values[0],
                temperature=temp
            )
            print(f"   T={temp}: {temp_prediction.get_action_description()[:30]}... (conf: {temp_prediction.confidence:.3f})")
        
    except Exception as e:
        print(f"❌ Action prediction failed: {e}")
    
    print()


def example_training_setup():
    """Example: Training configuration"""
    print("🏋️  Training Setup Example")
    print("=" * 50)
    
    # Create training configuration
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        sequence_length=10,
        num_epochs=100,
        warmup_steps=1000,
        checkpoint_dir="checkpoints/openleague5",
        log_dir="logs/openleague5"
    )
    
    print(f"📋 Training Configuration:")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Sequence length: {config.sequence_length}")
    print(f"   Number of epochs: {config.num_epochs}")
    print(f"   Warmup steps: {config.warmup_steps}")
    print(f"   Gradient clipping: {config.gradient_clip_norm}")
    
    print(f"\n💰 Loss Weights:")
    print(f"   Action loss: {config.action_loss_weight}")
    print(f"   Coordinate loss: {config.coordinate_loss_weight}")
    print(f"   Unit target loss: {config.unit_target_loss_weight}")
    print(f"   Value loss: {config.value_loss_weight}")
    
    print(f"\n💾 Storage:")
    print(f"   Checkpoint directory: {config.checkpoint_dir}")
    print(f"   Log directory: {config.log_dir}")
    print(f"   Save interval: {config.save_interval} steps")
    print(f"   Evaluation interval: {config.eval_interval} steps")
    
    print(f"\n🖥️  Hardware:")
    print(f"   Mixed precision: {config.use_amp}")
    print(f"   Number of workers: {config.num_workers}")
    print(f"   Pin memory: {config.pin_memory}")
    print(f"   Available device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print()


def show_usage_examples():
    """Show CLI usage examples"""
    print("💡 Usage Examples")
    print("=" * 50)
    
    examples = [
        ("Train model on replay data", 
         "python -m examples.openleague5 train --train-dir data/replays --epochs 50"),
        
        ("Predict action at specific time",
         "python -m examples.openleague5 predict --model model.pt --input replay.jsonl.gz --time 900"),
        
        ("Evaluate model performance",
         "python -m examples.openleague5 evaluate --model model.pt --eval-dir data/test"),
        
        ("Analyze replay patterns",
         "python -m examples.openleague5 analyze --input replay.jsonl.gz --output analysis.json"),
        
        ("Interactive demo",
         "python -m examples.openleague5 demo --model model.pt --input replay.jsonl.gz"),
        
        ("Train with custom configuration",
         "python -m examples.openleague5 train --config config.json --batch-size 64"),
        
        ("Predict with temperature control",
         "python -m examples.openleague5 predict --model model.pt --input replay.jsonl.gz --time 1200 --temperature 0.5"),
        
        ("Batch evaluation",
         "python -m examples.openleague5 evaluate --model model.pt --eval-files *.jsonl.gz --output results.json")
    ]
    
    for i, (description, command) in enumerate(examples, 1):
        print(f"{i:2d}. {description}")
        print(f"    {command}")
        print()


def show_system_info():
    """Show system and dependency information"""
    print("🔧 System Information")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"\nOpenLeague5 location: {Path(__file__).parent}")
    
    # Check for required dependencies
    missing_deps = []
    try:
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import huggingface_hub
        print(f"HuggingFace Hub version: {huggingface_hub.__version__}")
    except ImportError:
        missing_deps.append("huggingface_hub")
    
    try:
        import gymnasium
        print(f"Gymnasium version: {gymnasium.__version__}")
    except ImportError:
        missing_deps.append("gymnasium")
    
    if missing_deps:
        print(f"\n⚠️  Missing optional dependencies: {', '.join(missing_deps)}")
        print("   Some features may not work without these packages.")
    else:
        print(f"\n✅ All dependencies available!")
    
    print()


def main():
    """Main entry point"""
    print_banner()
    
    # If called with arguments, run CLI
    if len(sys.argv) > 1:
        return cli_main()
    
    # Otherwise, show examples and information
    print("Welcome to OpenLeague5! 🎮")
    print("This system provides state-of-the-art action prediction for League of Legends.")
    print("Inspired by OpenAI Five and AlphaStar approaches.\n")
    
    try:
        # Show system information
        show_system_info()
        
        # Run examples
        example_model_creation()
        example_state_encoding()
        example_action_prediction()
        example_training_setup()
        
        # Show usage examples
        show_usage_examples()
        
        print("🚀 Ready to predict the future of League of Legends!")
        print("Use --help to see all available commands and options.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())