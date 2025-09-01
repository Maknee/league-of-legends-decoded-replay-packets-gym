#!/usr/bin/env python3
"""
League Replays Parser CLI

Command-line interface for parsing League of Legends replay data
and managing datasets.

This module provides a comprehensive CLI for:
- Parsing individual replay files
- Managing replay datasets 
- Running gymnasium environments
- Checking parser capabilities

The CLI supports multiple parsing backends (Python/Rust) and various
observation types for machine learning applications.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Union
import json

from .unified_parser import UnifiedLeagueParser, ParseMethod
from .league_replays_gym import ReplayDataset, LeagueReplaysEnv
from .observations import PositionObservation, MinimapObservation, EventHistoryObservation


def parse_command(args: argparse.Namespace) -> int:
    """Handle parse command"""
    try:
        parser = UnifiedLeagueParser(
            preferred_method=ParseMethod(args.method) if args.method else ParseMethod.AUTO
        )
        
        if args.input.endswith('.gz'):
            # Parse single file
            result = parser.parse_file(args.input)
            print(f"Parsed {result.games_parsed} games with {result.total_events} events")
            print(f"Method used: {result.method_used}")
            print(f"Time: {result.elapsed_time:.2f}s ({result.games_per_sec:.1f} games/sec)")
            
            if result.errors:
                print("Errors:")
                for error in result.errors:
                    print(f"  - {error}")
            
        else:
            print(f"Error: Unsupported file format. Expected .gz file, got {args.input}")
            return 1
            
    except Exception as e:
        print(f"Error parsing file: {e}")
        return 1
    
    return 0


def dataset_command(args: argparse.Namespace) -> int:
    """Handle dataset command"""
    try:
        dataset = ReplayDataset(
            data_sources=[args.input] if isinstance(args.input, str) else args.input,
            repo_id=args.repo_id
        )
        
        print(f"Loading dataset from {len(dataset.data_sources)} sources...")
        dataset.load(max_games=args.max_games)
        
        print(f"Successfully loaded {len(dataset)} games")
        
        if args.output:
            # Export dataset info
            info = {
                'num_games': len(dataset),
                'data_sources': dataset.data_sources,
                'repo_id': dataset.repo_id
            }
            
            with open(args.output, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"Dataset info exported to {args.output}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    return 0


def gym_command(args: argparse.Namespace) -> int:
    """Handle gym environment command"""
    try:
        # Create dataset
        dataset = ReplayDataset([args.input])
        dataset.load(max_games=args.max_games or 1)
        
        if len(dataset) == 0:
            print("No games loaded from dataset")
            return 1
        
        # Create observation callback based on type
        if args.observation == 'position':
            obs_callback = PositionObservation(max_heroes=args.max_heroes)
        elif args.observation == 'minimap':
            obs_callback = MinimapObservation(resolution=args.minimap_resolution)
        elif args.observation == 'events':
            obs_callback = EventHistoryObservation(window_size=args.event_window)
        else:
            obs_callback = PositionObservation(max_heroes=args.max_heroes)
        
        # Create environment
        env = LeagueReplaysEnv(
            dataset=dataset,
            max_time=args.max_time,
            time_step=args.time_step,
            render_mode="human" if args.render else None,
            observation_callback=obs_callback
        )
        
        print(f"Created environment with {len(dataset)} games")
        print(f"Observation space: {type(env.observation_space).__name__}")
        print(f"Action space: {env.action_space}")
        
        # Run demo
        obs, info = env.reset()
        print(f"Reset to game {info['game_id']}")
        
        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if args.render:
                env.render()
            
            print(f"Step {step + 1}: reward={reward:.1f}, time={info['current_time']:.1f}s")
            
            if terminated or truncated:
                print(f"Episode ended: {'terminated' if terminated else 'truncated'}")
                break
        
        env.close()
        
    except Exception as e:
        print(f"Error running gym environment: {e}")
        return 1
    
    return 0


def capabilities_command(args: argparse.Namespace) -> int:
    """Handle capabilities command"""
    try:
        parser = UnifiedLeagueParser()
        caps = parser.get_capabilities()
        
        print("League Replays Parser Capabilities:")
        print("=" * 40)
        
        for key, value in caps.items():
            if isinstance(value, list):
                value = [v for v in value if v is not None]
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
        
        # Show packet types
        from .packets import get_packet_types
        packet_types = get_packet_types()
        print(f"\nSupported packet types ({len(packet_types)}):")
        for i, packet_type in enumerate(packet_types, 1):
            print(f"  {i:2d}. {packet_type}")
            
    except Exception as e:
        print(f"Error getting capabilities: {e}")
        return 1
    
    return 0


def main() -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="League of Legends Replay Parser CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse a single replay file
  league-parser parse data/replay.jsonl.gz
  
  # Load dataset and show info
  league-parser dataset data/replay.jsonl.gz --output dataset_info.json
  
  # Run gym environment demo
  league-parser gym data/replay.jsonl.gz --steps 10 --render
  
  # Show parser capabilities
  league-parser capabilities
        """
    )
    
    parser.add_argument('--version', action='version', version='0.1.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse replay files')
    parse_parser.add_argument('input', help='Input replay file (.jsonl.gz)')
    parse_parser.add_argument('--method', choices=['auto', 'rust', 'rust_parallel', 'python'],
                            help='Parsing method (default: auto)')
    parse_parser.set_defaults(func=parse_command)
    
    # Dataset command
    dataset_parser = subparsers.add_parser('dataset', help='Manage datasets')
    dataset_parser.add_argument('input', help='Input data source')
    dataset_parser.add_argument('--repo-id', default='maknee/league-of-legends-replays',
                               help='HuggingFace repository ID')
    dataset_parser.add_argument('--max-games', type=int, help='Maximum games to load')
    dataset_parser.add_argument('--output', help='Output file for dataset info')
    dataset_parser.set_defaults(func=dataset_command)
    
    # Gym command
    gym_parser = subparsers.add_parser('gym', help='Run gym environment')
    gym_parser.add_argument('input', help='Input data source')
    gym_parser.add_argument('--max-games', type=int, default=1, help='Maximum games to load')
    gym_parser.add_argument('--max-time', type=float, help='Maximum simulation time')
    gym_parser.add_argument('--time-step', type=float, default=1.0, help='Time step for simulation')
    gym_parser.add_argument('--steps', type=int, default=10, help='Number of steps to run')
    gym_parser.add_argument('--render', action='store_true', help='Enable rendering')
    gym_parser.add_argument('--observation', choices=['position', 'minimap', 'events'], 
                          default='position', help='Observation type')
    gym_parser.add_argument('--max-heroes', type=int, default=10, help='Max heroes for position observation')
    gym_parser.add_argument('--minimap-resolution', type=int, default=64, help='Minimap resolution')
    gym_parser.add_argument('--event-window', type=int, default=50, help='Event history window size')
    gym_parser.set_defaults(func=gym_command)
    
    # Capabilities command
    cap_parser = subparsers.add_parser('capabilities', help='Show parser capabilities')
    cap_parser.set_defaults(func=capabilities_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())