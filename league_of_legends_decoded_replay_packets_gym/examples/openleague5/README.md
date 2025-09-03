# OpenLeague5: Action Prediction for League of Legends

OpenLeague5 is a state-of-the-art neural network system for predicting player actions in League of Legends, inspired by the approaches used in OpenAI Five (DOTA 2) and AlphaStar (StarCraft II).

## 🎯 Overview

This system implements multi-modal neural networks that can analyze League of Legends replay data and predict what actions professional players are likely to take next, given the current game state.

### Key Features

- **Multi-modal Architecture**: Combines spatial CNN, unit-based Transformer, and temporal LSTM
- **Auto-regressive Action Prediction**: Predicts complex action sequences step-by-step
- **Professional Replay Analysis**: Learns from high-level gameplay patterns
- **Real-time Prediction**: Fast inference for live game analysis
- **Behavior Cloning**: Imitates expert player decision-making

### Architecture Highlights

- **Spatial CNN**: Processes minimap-style spatial features (64x64 grid)
- **Unit Transformer**: Handles variable-length unit sequences with attention
- **Temporal LSTM**: 1024-unit LSTM for sequence modeling (matching OpenAI Five)
- **Pointer Networks**: For unit targeting (inspired by AlphaStar)
- **Auto-regressive Head**: Step-by-step action prediction

## 📁 Project Structure

```
openleague5/
├── __init__.py              # Package initialization
├── __main__.py              # Main entry point with examples
├── cli.py                   # Command-line interface
├── openleague5_model.py     # Neural network architecture
├── state_encoder.py         # Game state encoding
├── action_space.py          # Action space definition
├── trainer.py               # Training pipeline
└── README.md               # This file
```

## 🚀 Quick Start

### Training

Train OpenLeague5 on League of Legends replay data from HuggingFace:

```bash
cd league_of_legends_decoded_replay_packets_gym/examples/openleague5

python train_direct.py \
    --huggingface-repo maknee/league-of-legends-decoded-replay-packets \
    --huggingface-filter "12_22/*.jsonl.gz" \
    --huggingface-max-files 1 \
    --learning-rate 1e-6 \
    --epochs 5 \
    --batch-size 8
```

**Expected Training Results:**
- ✅ **Data Loading**: `Successfully loaded 23 games from 1 sources` (755 sequences)
- ✅ **Model Creation**: `Model parameters: 13852007` (13.8M parameters)
- ✅ **Training Progress**: `Epoch X completed in Y.Zs`
- ✅ **Checkpoint Saving**: `checkpoint_step_final.pt` saved in checkpoints directory
- ✅ **GPU Utilization**: Training on CUDA device

### Prediction

Make action predictions using the trained model:

#### Early Game Prediction (5 minutes)
```bash
python predict_simple.py \
    --model checkpoints/checkpoint_step_final.pt \
    --input data/huggingface_cache/12_22/batch_001.jsonl.gz \
    --time 300
```

#### Mid Game Prediction (10 minutes)
```bash
python predict_simple.py \
    --model ../../../checkpoints/checkpoint_step_final.pt \
    --input ../../../data/huggingface_cache/12_22/batch_001.jsonl.gz \
    --time 600 \
    --temperature 0.8
```

#### Different Action Diversity (7.5 minutes)
```bash
python predict_simple.py \
    --model ../../../checkpoints/checkpoint_step_final.pt \
    --input ../../../data/huggingface_cache/12_22/batch_001.jsonl.gz \
    --time 450 \
    --temperature 1.2
```

**Expected Prediction Results:**
- ✅ **Model Loading**: `Model loaded on cuda`
- ✅ **Game Data Loading**: `Successfully loaded 1 games from 1 sources`
- ✅ **Action Prediction**: Actions like "No Action", "Attack", "Use W Ability" with confidence scores
- ✅ **Coordinate Mapping**: World coordinates for target positions
- ✅ **Unit Targeting**: Unit selection with confidence values

### Action Types

The model can predict the following actions:
- **0**: No Action
- **1**: Move
- **2**: Attack  
- **3**: Use Q Ability
- **4**: Use W Ability
- **5**: Use E Ability
- **6**: Use R Ability
- **7**: Use Item
- **8**: Recall
- **9**: Shop
- **10**: Level Up
- **11**: Ping

### Temperature Effects

- **temperature < 1.0** (e.g., 0.8): More conservative, higher confidence predictions
- **temperature = 1.0**: Balanced predictions
- **temperature > 1.0** (e.g., 1.2): More diverse, exploratory predictions

