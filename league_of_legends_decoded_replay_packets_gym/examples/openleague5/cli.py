"""
Command Line Interface for OpenLeague5

This module provides a comprehensive CLI for the OpenLeague5 action prediction system,
supporting training, evaluation, and prediction tasks.

Commands:
- train: Train model on replay data using behavior cloning
- predict: Predict next action given current game state
- evaluate: Evaluate trained model performance
- analyze: Analyze replay data and action patterns
- demo: Interactive demonstration of predictions

The CLI follows the design patterns from the main League replays parser CLI,
providing a consistent user experience.
"""

import argparse
import sys
import os
import json
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

# Import from the renamed package
import league_of_legends_decoded_replay_packets_gym as lrp

from .openleague5_model import OpenLeague5Model, ModelConfig
from .state_encoder import StateEncoder, GameStateVector
from .action_space import ActionSpace, ActionPrediction, ActionSequenceAnalyzer
from .trainer import Trainer, TrainingConfig


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def train_command(args: argparse.Namespace) -> int:
    """Execute training command"""
    try:
        print("🚀 OpenLeague5 Training")
        print("=" * 50)
        
        # Setup logging
        setup_logging(args.log_level)
        
        # Load training configuration
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = TrainingConfig(**config_dict)
        else:
            config = TrainingConfig()
        
        # Override config with command line arguments
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        if args.epochs:
            config.num_epochs = args.epochs
        if args.checkpoint_dir:
            config.checkpoint_dir = args.checkpoint_dir
        
        # HuggingFace configuration
        if hasattr(args, 'huggingface_repo') and args.huggingface_repo:
            config.huggingface_repo_id = args.huggingface_repo
        if hasattr(args, 'huggingface_filter') and args.huggingface_filter:
            config.huggingface_file_filter = args.huggingface_filter
        if hasattr(args, 'huggingface_max_files') and args.huggingface_max_files:
            config.huggingface_max_files = args.huggingface_max_files
        if hasattr(args, 'cache_dir') and args.cache_dir:
            config.cache_dir = args.cache_dir
        
        print(f"📋 Training Configuration:")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Sequence length: {config.sequence_length}")
        print(f"   Checkpoint dir: {config.checkpoint_dir}")
        
        if config.huggingface_repo_id:
            print(f"🤗 HuggingFace Configuration:")
            print(f"   Repository: {config.huggingface_repo_id}")
            if config.huggingface_file_filter:
                print(f"   File filter: {config.huggingface_file_filter}")
            if config.huggingface_max_files:
                print(f"   Max files: {config.huggingface_max_files}")
            if config.cache_dir:
                print(f"   Cache directory: {config.cache_dir}")
        
        print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Create trainer
        trainer = Trainer(config)
        
        # Load checkpoint if provided
        if args.resume:
            print(f"📂 Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Prepare file lists
        train_files = []
        val_files = []
        
        if args.train_files:
            train_files = args.train_files
        elif args.train_dir:
            train_dir = Path(args.train_dir)
            train_files = list(train_dir.glob("*.jsonl.gz"))
            train_files = [str(f) for f in train_files]
        
        if args.val_files:
            val_files = args.val_files
        elif args.val_dir:
            val_dir = Path(args.val_dir)
            val_files = list(val_dir.glob("*.jsonl.gz"))
            val_files = [str(f) for f in val_files]
        
        if not train_files and not config.huggingface_repo_id:
            print("❌ No training files provided! Use --train-files, --train-dir, or --huggingface-repo")
            return 1
        
        if train_files:
            print(f"📚 Local training files: {len(train_files)}")
        if val_files:
            print(f"📊 Validation files: {len(val_files)}")
        if config.huggingface_repo_id and not train_files:
            print("📁 Training files will be downloaded from HuggingFace")
        
        # Start training
        trainer.train(train_files, val_files)
        
        print("✅ Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def predict_command(args: argparse.Namespace) -> int:
    """Execute prediction command"""
    try:
        print("🔮 OpenLeague5 Action Prediction")
        print("=" * 50)
        
        # Load model
        if not os.path.exists(args.model):
            print(f"❌ Model file not found: {args.model}")
            return 1
        
        print(f"📂 Loading model from: {args.model}")
        checkpoint = torch.load(args.model, map_location='cpu')
        
        # Create model
        if 'model_config' in checkpoint:
            model_config = ModelConfig(**checkpoint['model_config'])
        else:
            model_config = ModelConfig()
        
        model = OpenLeague5Model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded on {device}")
        
        # Create encoders
        state_encoder = StateEncoder(
            spatial_resolution=model_config.spatial_resolution,
            max_units=model_config.max_units
        )
        
        action_space = ActionSpace(
            coordinate_bins=model_config.coordinate_bins,
            max_units=model_config.max_units
        )
        
        # Load replay data
        print(f"📚 Loading replay data from: {args.input}")
        dataset = lrp.ReplayDataset([args.input])
        dataset.load(max_games=1)
        
        if len(dataset) == 0:
            print("❌ No games found in replay data!")
            return 1
        
        # Create environment
        env = lrp.LeagueReplaysEnv(dataset, time_step=1.0)
        obs, info = env.reset()
        
        print(f"🎮 Loaded game {info['game_id']}")
        
        # Find the target timestamp
        target_time = args.time
        current_time = 0
        
        print(f"⏰ Seeking to timestamp: {target_time}s")
        
        # Step to target time
        while current_time < target_time and not (info.get('terminated', False) or info.get('truncated', False)):
            obs, reward, terminated, truncated, info = env.step(0)
            current_time = info['current_time']
            
            if terminated or truncated:
                print(f"⚠️  Game ended before target time (reached {current_time:.1f}s)")
                break
        
        # Get current game state
        game_state = info['game_state']
        if not game_state or not game_state.heroes:
            print("❌ No valid game state at target time!")
            env.close()
            return 1
        
        print(f"📊 Game state at {current_time:.1f}s:")
        print(f"   Heroes: {len(game_state.heroes)}")
        print(f"   Events this step: {len(game_state.events)}")
        
        # Encode game state
        print("🔄 Encoding game state...")
        encoded_state = state_encoder.encode_game_state(game_state)
        
        # Prepare inputs for model
        spatial = encoded_state.spatial_features.unsqueeze(0).to(device)  # [1, C, H, W]
        units = encoded_state.unit_features.unsqueeze(0).to(device)      # [1, max_units, feat_dim]
        mask = encoded_state.unit_mask.unsqueeze(0).to(device)          # [1, max_units]
        global_feat = encoded_state.global_features.unsqueeze(0).to(device)  # [1, global_dim]
        
        # Make prediction
        print("🤖 Predicting next action...")
        with torch.no_grad():
            predicted_action = model.predict_next_action(
                spatial, units, mask, global_feat, 
                temperature=args.temperature
            )
        
        # Display prediction results
        print("\n🎯 Prediction Results:")
        print("=" * 30)
        
        # Map action type to readable description
        action_names = {
            0: "No Action",
            1: "Move", 
            2: "Attack",
            3: "Use Q Ability",
            4: "Use W Ability",
            5: "Use E Ability", 
            6: "Use R Ability",
            7: "Use Item",
            8: "Recall",
            9: "Shop",
            10: "Level Up",
            11: "Ping"
        }
        
        action_type = predicted_action['action_type']
        action_description = action_names.get(action_type, f"Action {action_type}")
        
        print(f"Action: {action_description}")
        print(f"Confidence: {predicted_action['action_confidence']:.3f}")
        print(f"State Value: {predicted_action['value']:.3f}")
        
        if predicted_action['coordinates']:
            world_pos = action_space.coordinates_to_world(predicted_action['coordinates'])
            print(f"Target Position: ({world_pos.x:.0f}, {world_pos.z:.0f}) world coords")
            print(f"Coordinate Confidence: X={predicted_action['coordinate_confidence'][0]:.3f}, "
                  f"Y={predicted_action['coordinate_confidence'][1]:.3f}")
        
        if predicted_action['unit_target'] is not None:
            print(f"Unit Target: {predicted_action['unit_target']}")
            print(f"Unit Confidence: {predicted_action['unit_confidence']:.3f}")
        
        # Save prediction if requested
        if args.output:
            prediction_data = {
                'timestamp': current_time,
                'game_id': game_state.game_id,
                'prediction': predicted_action,
                'model_file': args.model,
                'temperature': args.temperature
            }
            
            with open(args.output, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            print(f"💾 Prediction saved to: {args.output}")
        
        env.close()
        print("✅ Prediction completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def evaluate_command(args: argparse.Namespace) -> int:
    """Execute evaluation command"""
    try:
        print("📊 OpenLeague5 Model Evaluation")
        print("=" * 50)
        
        # Load model
        print(f"📂 Loading model from: {args.model}")
        checkpoint = torch.load(args.model, map_location='cpu')
        
        if 'model_config' in checkpoint:
            model_config = ModelConfig(**checkpoint['model_config'])
        else:
            model_config = ModelConfig()
        
        model = OpenLeague5Model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        model.to(device)
        
        # Create trainer for evaluation
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = TrainingConfig(**config_dict)
        else:
            config = TrainingConfig()
        
        trainer = Trainer(config, model, device)
        
        # Prepare evaluation files
        eval_files = []
        if args.eval_files:
            eval_files = args.eval_files
        elif args.eval_dir:
            eval_dir = Path(args.eval_dir)
            eval_files = list(eval_dir.glob("*.jsonl.gz"))
            eval_files = [str(f) for f in eval_files]
        
        if not eval_files:
            print("❌ No evaluation files provided!")
            return 1
        
        print(f"📚 Evaluation files: {len(eval_files)}")
        
        # Create evaluation dataloader
        eval_loader = trainer.create_dataloader(eval_files, shuffle=False)
        print(f"📊 Evaluation batches: {len(eval_loader)}")
        
        # Run evaluation
        print("🔄 Running evaluation...")
        start_time = time.time()
        
        metrics = trainer.evaluate(eval_loader)
        
        eval_time = time.time() - start_time
        
        # Display results
        print("\n📈 Evaluation Results:")
        print("=" * 30)
        print(f"Total Loss: {metrics['total']:.4f}")
        print(f"Action Loss: {metrics['action']:.4f}")
        print(f"Coordinate Loss: {metrics['coordinate']:.4f}")
        print(f"Unit Target Loss: {metrics['unit_target']:.4f}")
        print(f"Value Loss: {metrics['value']:.4f}")
        print(f"Evaluation Time: {eval_time:.1f}s")
        
        # Save results if requested
        if args.output:
            results = {
                'model_file': args.model,
                'eval_files': eval_files,
                'metrics': metrics,
                'evaluation_time': eval_time,
                'timestamp': time.time()
            }
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {args.output}")
        
        print("✅ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def analyze_command(args: argparse.Namespace) -> int:
    """Execute analysis command"""
    try:
        print("🔍 OpenLeague5 Replay Analysis")
        print("=" * 50)
        
        # Create action space and analyzer
        action_space = ActionSpace()
        analyzer = ActionSequenceAnalyzer(action_space)
        
        # Load replay data
        print(f"📚 Loading replay data from: {args.input}")
        dataset = lrp.ReplayDataset([args.input])
        dataset.load(max_games=args.max_games)
        
        if len(dataset) == 0:
            print("❌ No games found in replay data!")
            return 1
        
        print(f"🎮 Analyzing {len(dataset)} games...")
        
        # Analyze each game
        all_action_sequences = []
        game_summaries = []
        
        for game_idx in range(len(dataset)):
            print(f"   Processing game {game_idx + 1}/{len(dataset)}...")
            
            try:
                # Create environment for this game
                env = lrp.LeagueReplaysEnv(dataset, time_step=5.0)  # 5-second steps
                obs, info = env.reset(seed=None, options={'game_idx': game_idx})
                
                game_actions = []
                step_count = 0
                max_steps = args.max_time // 5 if args.max_time else 1000
                
                while step_count < max_steps:
                    game_state = info['game_state']
                    
                    if game_state and game_state.events:
                        # Extract actions from events
                        hero_positions = {net_id: game_state.get_position(net_id) 
                                        for net_id in game_state.heroes.keys()}
                        actions = analyzer.extract_action_from_events(
                            game_state.events, hero_positions
                        )
                        game_actions.extend(actions)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(0)
                    step_count += 1
                    
                    if terminated or truncated:
                        break
                
                env.close()
                
                # Game summary
                game_summary = {
                    'game_idx': game_idx,
                    'total_actions': len(game_actions),
                    'duration_steps': step_count,
                    'duration_seconds': step_count * 5
                }
                game_summaries.append(game_summary)
                all_action_sequences.append(game_actions)
                
                print(f"     {len(game_actions)} actions extracted")
                
            except Exception as e:
                print(f"     Error processing game {game_idx}: {e}")
                continue
        
        # Analyze action patterns
        print("\n🔄 Analyzing action patterns...")
        pattern_analysis = analyzer.analyze_action_patterns(all_action_sequences)
        
        # Display results
        print("\n📈 Analysis Results:")
        print("=" * 30)
        print(f"Games Analyzed: {len(game_summaries)}")
        print(f"Total Actions: {pattern_analysis['total_actions']}")
        print(f"Average Actions per Game: {pattern_analysis['total_actions'] / len(game_summaries):.1f}")
        
        print("\n📊 Action Type Distribution:")
        action_dist = pattern_analysis['action_type_distribution']
        total_actions = sum(action_dist.values())
        for action_type, count in sorted(action_dist.items()):
            percentage = (count / total_actions) * 100
            print(f"   Action {action_type}: {count} ({percentage:.1f}%)")
        
        print("\n🎯 Target Type Distribution:")
        target_dist = pattern_analysis['target_type_distribution']
        total_targets = sum(target_dist.values())
        for target_type, count in sorted(target_dist.items()):
            percentage = (count / total_targets) * 100
            print(f"   Target {target_type}: {count} ({percentage:.1f}%)")
        
        coord_stats = pattern_analysis['coordinate_statistics']
        if coord_stats:
            print(f"\n📍 Coordinate Statistics:")
            print(f"   Mean Position: ({coord_stats['mean_x']:.3f}, {coord_stats['mean_y']:.3f})")
            print(f"   Std Position: ({coord_stats['std_x']:.3f}, {coord_stats['std_y']:.3f})")
        
        # Save results if requested
        if args.output:
            results = {
                'input_file': args.input,
                'games_analyzed': len(game_summaries),
                'game_summaries': game_summaries,
                'pattern_analysis': pattern_analysis,
                'analysis_timestamp': time.time()
            }
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Analysis results saved to: {args.output}")
        
        print("✅ Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def demo_command(args: argparse.Namespace) -> int:
    """Execute demo command"""
    try:
        print("🎮 OpenLeague5 Interactive Demo")
        print("=" * 50)
        
        # Load model
        print(f"📂 Loading model from: {args.model}")
        checkpoint = torch.load(args.model, map_location='cpu')
        
        if 'model_config' in checkpoint:
            model_config = ModelConfig(**checkpoint['model_config'])
        else:
            model_config = ModelConfig()
        
        model = OpenLeague5Model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        model.to(device)
        model.eval()
        
        # Create encoders
        state_encoder = StateEncoder(
            spatial_resolution=model_config.spatial_resolution,
            max_units=model_config.max_units
        )
        
        action_space = ActionSpace(
            coordinate_bins=model_config.coordinate_bins,
            max_units=model_config.max_units
        )
        
        # Load replay data
        print(f"📚 Loading replay data from: {args.input}")
        dataset = lrp.ReplayDataset([args.input])
        dataset.load(max_games=1)
        
        if len(dataset) == 0:
            print("❌ No games found in replay data!")
            return 1
        
        # Create environment
        env = lrp.LeagueReplaysEnv(dataset, time_step=args.step_size)
        obs, info = env.reset()
        
        print(f"🎮 Demo: Game {info['game_id']}")
        print("\nControls:")
        print("  ENTER: Step forward and predict next action")
        print("  'q' + ENTER: Quit demo")
        print("  'j <time>' + ENTER: Jump to specific time")
        print("  's <temp>' + ENTER: Set prediction temperature")
        print()
        
        temperature = args.temperature
        step_count = 0
        
        while True:
            current_time = info['current_time']
            game_state = info['game_state']
            
            # Display current state
            print(f"⏰ Time: {current_time:.1f}s | Step: {step_count}")
            if game_state:
                print(f"   Heroes: {len(game_state.heroes)} | Events: {len(game_state.events)}")
                
                # Show hero positions
                for net_id, hero_info in list(game_state.heroes.items())[:3]:  # Show first 3
                    pos = game_state.get_position(net_id)
                    if pos:
                        name = hero_info.get('name', f'Hero{net_id}')[:10]
                        print(f"   {name}: ({pos.x:.0f}, {pos.z:.0f})")
            
            # Get user input
            try:
                user_input = input("\n> ").strip().lower()
                
                if user_input == 'q':
                    break
                elif user_input.startswith('j '):
                    # Jump to time
                    try:
                        target_time = float(user_input.split()[1])
                        while current_time < target_time:
                            obs, reward, terminated, truncated, info = env.step(0)
                            current_time = info['current_time']
                            if terminated or truncated:
                                print("⚠️  Game ended!")
                                break
                        continue
                    except (ValueError, IndexError):
                        print("❌ Invalid time format. Use: j <seconds>")
                        continue
                elif user_input.startswith('s '):
                    # Set temperature
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"🌡️  Temperature set to: {temperature}")
                        continue
                    except (ValueError, IndexError):
                        print("❌ Invalid temperature format. Use: s <temperature>")
                        continue
                elif user_input != '':
                    print("❌ Unknown command")
                    continue
                
                # Make prediction if we have a valid state
                if game_state and game_state.heroes:
                    encoded_state = state_encoder.encode_game_state(game_state)
                    
                    spatial = encoded_state.spatial_features.unsqueeze(0).to(device)
                    units = encoded_state.unit_features.unsqueeze(0).to(device)
                    mask = encoded_state.unit_mask.unsqueeze(0).to(device)
                    global_feat = encoded_state.global_features.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        predicted_action = model.predict_next_action(
                            spatial, units, mask, global_feat, temperature=temperature
                        )
                    
                    print(f"🤖 Predicted: {predicted_action.get_action_description()}")
                    print(f"   Confidence: {predicted_action['action_confidence']:.3f} | Value: {predicted_action['value']:.3f}")
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(0)
                step_count += 1
                
                if terminated or truncated:
                    print("🏁 Game ended!")
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted!")
                break
            except EOFError:
                break
        
        env.close()
        print("✅ Demo completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="OpenLeague5 - League of Legends Action Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model on replay data
  python -m openleague5.cli train --train-dir data/train --val-dir data/val --epochs 50
  
  # Predict action at specific timestamp
  python -m openleague5.cli predict --model model.pt --input replay.jsonl.gz --time 900
  
  # Evaluate trained model
  python -m openleague5.cli evaluate --model model.pt --eval-dir data/test
  
  # Analyze replay patterns
  python -m openleague5.cli analyze --input replay.jsonl.gz --output analysis.json
  
  # Interactive demo
  python -m openleague5.cli demo --model model.pt --input replay.jsonl.gz
        """
    )
    
    # Global options
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train OpenLeague5 model')
    train_parser.add_argument('--train-files', nargs='+', help='Training replay files')
    train_parser.add_argument('--train-dir', help='Directory containing training files')
    train_parser.add_argument('--val-files', nargs='+', help='Validation replay files')
    train_parser.add_argument('--val-dir', help='Directory containing validation files')
    train_parser.add_argument('--config', help='Training configuration file')
    train_parser.add_argument('--resume', help='Resume from checkpoint')
    train_parser.add_argument('--batch-size', type=int, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--checkpoint-dir', help='Checkpoint directory')
    
    # HuggingFace integration options
    train_parser.add_argument('--huggingface-repo', help='HuggingFace dataset repository (e.g. "maknee/league-of-legends-decoded-replay-packets")')
    train_parser.add_argument('--huggingface-filter', help='File filter pattern (e.g. "12_22/*.jsonl.gz" or "*/batch_00*.jsonl.gz")')
    train_parser.add_argument('--huggingface-max-files', type=int, help='Maximum number of files to download from HuggingFace')
    train_parser.add_argument('--cache-dir', help='Local cache directory for downloaded files')
    
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict next action')
    predict_parser.add_argument('--model', required=True, help='Trained model file')
    predict_parser.add_argument('--input', required=True, help='Input replay file')
    predict_parser.add_argument('--time', type=float, required=True, help='Timestamp (seconds)')
    predict_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    predict_parser.add_argument('--output', help='Output prediction file')
    predict_parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    predict_parser.set_defaults(func=predict_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', required=True, help='Trained model file')
    eval_parser.add_argument('--eval-files', nargs='+', help='Evaluation replay files')
    eval_parser.add_argument('--eval-dir', help='Directory containing evaluation files')
    eval_parser.add_argument('--config', help='Training configuration file')
    eval_parser.add_argument('--output', help='Output results file')
    eval_parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    eval_parser.set_defaults(func=evaluate_command)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze replay data')
    analyze_parser.add_argument('--input', required=True, help='Input replay file')
    analyze_parser.add_argument('--output', help='Output analysis file')
    analyze_parser.add_argument('--max-games', type=int, default=10, help='Maximum games to analyze')
    analyze_parser.add_argument('--max-time', type=int, help='Maximum time per game (seconds)')
    analyze_parser.set_defaults(func=analyze_command)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Interactive demo')
    demo_parser.add_argument('--model', required=True, help='Trained model file')
    demo_parser.add_argument('--input', required=True, help='Input replay file')
    demo_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    demo_parser.add_argument('--step-size', type=float, default=5.0, help='Time step size (seconds)')
    demo_parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    demo_parser.set_defaults(func=demo_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up basic error handling
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())