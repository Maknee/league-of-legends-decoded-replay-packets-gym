#!/usr/bin/env python3
"""
Direct trainer execution script for OpenLeague5
Run this with: python run_trainer.py
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add the gym package to path
current_dir = Path(__file__).parent
gym_dir = current_dir.parent.parent
sys.path.insert(0, str(gym_dir))

# Import the gym package
import league_of_legends_decoded_replay_packets_gym as lrp

def main():
    """Main entry point for direct trainer execution"""
    parser = argparse.ArgumentParser(description="OpenLeague5 Trainer - Direct Execution")
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    
    # Data sources
    parser.add_argument('--train-files', nargs='+', help='Local training files')
    parser.add_argument('--train-dir', help='Directory with training files')
    
    # HuggingFace options
    parser.add_argument('--huggingface-repo', default='maknee/league-of-legends-decoded-replay-packets',
                       help='HuggingFace repository')
    parser.add_argument('--huggingface-filter', default='12_22/batch_00*.jsonl.gz',
                       help='File filter pattern')
    parser.add_argument('--huggingface-max-files', type=int, default=3,
                       help='Max files to download')
    parser.add_argument('--cache-dir', default='data/hf_cache',
                       help='Cache directory')
    
    # Other options
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--config', help='JSON config file')
    
    args = parser.parse_args()
    
    try:
        print("🚀 OpenLeague5 Direct Trainer")
        print("=" * 40)
        
        # Now import the trainer components after setting up paths
        from examples.openleague5.trainer import Trainer, TrainingConfig
        
        # Load config
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = TrainingConfig(**config_dict)
        else:
            # Create config from args
            config = TrainingConfig(
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
                checkpoint_dir=args.checkpoint_dir,
                huggingface_repo_id=args.huggingface_repo,
                huggingface_file_filter=args.huggingface_filter,
                huggingface_max_files=args.huggingface_max_files,
                cache_dir=args.cache_dir
            )
        
        print(f"📋 Configuration:")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   HuggingFace repo: {config.huggingface_repo_id}")
        print(f"   File filter: {config.huggingface_file_filter}")
        print(f"   Max files: {config.huggingface_max_files}")
        
        # Prepare training files
        train_files = []
        if args.train_files:
            train_files = args.train_files
        elif args.train_dir:
            train_dir = Path(args.train_dir)
            train_files = [str(f) for f in train_dir.glob("*.jsonl.gz")]
        
        # Create trainer
        trainer = Trainer(config)
        
        # Resume if needed
        if args.resume:
            print(f"📂 Resuming from: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        print("🎯 Starting training...")
        trainer.train(train_files)
        
        print("✅ Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())