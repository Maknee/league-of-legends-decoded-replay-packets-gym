#!/usr/bin/env python3
"""
Simple prediction script for OpenLeague5
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="OpenLeague5 Simple Prediction")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input', required=True, help='Input replay file')
    parser.add_argument('--time', type=float, required=True, help='Time in seconds')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Get the root directory and change to it
    current_file = Path(__file__)
    openleague5_dir = current_file.parent
    examples_dir = openleague5_dir.parent  
    gym_dir = examples_dir.parent
    root_dir = gym_dir.parent
    
    # Change to root directory
    os.chdir(root_dir)
    
    # Build the command
    cmd = [
        sys.executable, '-m', 
        'league_of_legends_decoded_replay_packets_gym.examples.openleague5',
        'predict',
        '--model', args.model,
        '--input', args.input,
        '--time', str(args.time),
        '--temperature', str(args.temperature)
    ]
    
    print(f"🔮 Running OpenLeague5 prediction")
    print(f"📝 Command: {' '.join(cmd)}")
    print("=" * 50)
    
    # Execute the command
    try:
        import subprocess
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\n👋 Prediction interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())