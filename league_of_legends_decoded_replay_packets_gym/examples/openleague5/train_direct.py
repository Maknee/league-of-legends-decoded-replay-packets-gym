#!/usr/bin/env python3
"""
Simple wrapper to run OpenLeague5 trainer directly
Usage: python train_direct.py [arguments...]
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the root directory (3 levels up from this file)
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
        'train'
    ]
    
    # Add all arguments passed to this script
    cmd.extend(sys.argv[1:])
    
    print(f"🚀 Running OpenLeague5 trainer from {root_dir}")
    print(f"📝 Command: {' '.join(cmd)}")
    print("=" * 50)
    
    # Execute the command
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\n👋 Training interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())