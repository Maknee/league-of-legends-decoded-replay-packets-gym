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

### Installation

```bash
# Install PyTorch (choose version for your system)
pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy matplotlib gymnasium huggingface_hub

# Navigate to the League replays parser directory
cd league-of-legends-replays-parser
```

### Basic Usage

```bash
# Run examples and system info
python -m examples.openleague5

# Train a model (requires replay data)
python -m examples.openleague5 train --train-dir data/replays --epochs 50

# Predict action at a specific timestamp
python -m examples.openleague5 predict --model model.pt --input replay.jsonl.gz --time 900

# Interactive demo
python -m examples.openleague5 demo --model model.pt --input replay.jsonl.gz

# Analyze replay patterns
python -m examples.openleague5 analyze --input replay.jsonl.gz --output analysis.json
```

## 🏗️ Architecture Details

### Neural Network Components

1. **Spatial CNN**
   - Processes 16-channel spatial features (heroes, minions, terrain, vision, etc.)
   - 64x64 resolution minimap representation
   - Convolutional feature extraction with batch normalization

2. **Unit Transformer**
   - Handles up to 50 units with attention mechanisms
   - 256-dimensional transformer with 8 attention heads
   - Variable-length sequences with positional encoding

3. **Temporal LSTM**
   - 1024-unit LSTM (matching OpenAI Five architecture)
   - Processes fused spatial + unit + global features
   - Maintains temporal context across game states

4. **Action Prediction Head**
   - Auto-regressive action type prediction (12 action types)
   - Discretized coordinate prediction (64x64 bins)
   - Pointer network for unit targeting
   - Value function estimation

### State Representation

**Spatial Features (16 channels)**:
- Ally/enemy heroes and minions
- Structures and terrain
- Vision and jungle areas
- Recent combat and movement trails

**Unit Features (64 dimensions per unit)**:
- Position, health, mana, level
- Champion type and team affiliation
- Items, abilities, and stats
- Recent actions and momentum

**Global Features (32 dimensions)**:
- Game time and phase
- Team gold/experience differences
- Objective states (dragons, baron)
- Map control and strategic timers

### Action Space

**Primary Actions**:
- Move, Attack, Use Abilities (Q/W/E/R)
- Use Items, Recall, Shop, Level Up, Ping

**Target Types**:
- Ground coordinates (discretized)
- Unit targets (via pointer network)
- Item/ability slots

## 🎓 Training

### Behavior Cloning Pipeline

The system uses behavior cloning to learn from professional replay data:

1. **Data Preprocessing**: Convert replay events to state-action sequences
2. **Multi-modal Encoding**: Transform game states to neural network inputs
3. **Sequence Learning**: Train on temporal sequences with LSTM
4. **Multi-objective Loss**: Combine action, coordinate, unit, and value losses

### Training Configuration

```python
config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    sequence_length=10,
    num_epochs=100,
    gradient_clip_norm=1.0,
    use_amp=True  # Mixed precision training
)
```

### Loss Components

- **Action Loss**: CrossEntropy for action type prediction
- **Coordinate Loss**: CrossEntropy for discretized coordinates
- **Unit Target Loss**: CrossEntropy with pointer network attention
- **Value Loss**: MSE for state value estimation

## 📊 Evaluation

### Metrics

- **Action Accuracy**: Percentage of correct action type predictions
- **Coordinate Error**: Mean distance error for spatial predictions
- **Unit Target Accuracy**: Correct unit selection percentage
- **Value Error**: State value estimation accuracy

### Example Results

```bash
# Evaluate trained model
python -m examples.openleague5 evaluate --model model.pt --eval-dir data/test

# Sample output:
# Total Loss: 1.245
# Action Loss: 0.823 (68.4% accuracy)
# Coordinate Loss: 1.156 (avg error: 234 units)
# Unit Target Loss: 0.745 (71.2% accuracy)
# Value Loss: 0.089 (±0.298 error)
```

## 🔧 Advanced Usage

### Custom Model Configuration

```python
from openleague5 import OpenLeague5Model, ModelConfig

config = ModelConfig(
    spatial_resolution=128,     # Higher resolution
    lstm_hidden_size=2048,      # Larger LSTM
    transformer_layers=6,       # Deeper transformer
    coordinate_bins=128         # More precise coordinates
)

model = OpenLeague5Model(config)
```

### Custom Training

```python
from openleague5 import Trainer, TrainingConfig

config = TrainingConfig(
    batch_size=64,
    learning_rate=2e-4,
    sequence_length=20,
    action_loss_weight=1.5,
    coordinate_loss_weight=1.0
)

trainer = Trainer(config)
trainer.train(train_files, val_files)
```

### Prediction with Temperature Control

```python
# More conservative predictions
prediction = model.predict_next_action(
    spatial, units, mask, global_features,
    temperature=0.5
)

# More diverse predictions
prediction = model.predict_next_action(
    spatial, units, mask, global_features,
    temperature=2.0
)
```

## 🎮 CLI Commands

### Training
```bash
# Basic training
python -m examples.openleague5 train --train-dir data/train --val-dir data/val

# Advanced training
python -m examples.openleague5 train \
    --config training_config.json \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --epochs 100 \
    --checkpoint-dir checkpoints/
```

### Prediction
```bash
# Predict at specific time
python -m examples.openleague5 predict \
    --model model.pt \
    --input replay.jsonl.gz \
    --time 900 \
    --temperature 1.0

# Save prediction results
python -m examples.openleague5 predict \
    --model model.pt \
    --input replay.jsonl.gz \
    --time 1200 \
    --output prediction.json
```

### Analysis
```bash
# Analyze replay patterns
python -m examples.openleague5 analyze \
    --input replay.jsonl.gz \
    --max-games 10 \
    --output analysis.json
```

### Interactive Demo
```bash
# Live prediction demo
python -m examples.openleague5 demo \
    --model model.pt \
    --input replay.jsonl.gz \
    --step-size 5.0
```

## 🔬 Research Applications

### Potential Use Cases

1. **Gameplay Analysis**: Understand professional player decision-making
2. **Coaching Tools**: Identify strategic patterns and mistakes
3. **Game Balance**: Analyze champion and item effectiveness
4. **AI Development**: Train stronger League of Legends bots
5. **Esports Analytics**: Predict game outcomes and player performance

### Extensions

- **Multi-agent Prediction**: Predict all 10 players simultaneously
- **Strategic Planning**: Long-term objective and macro strategy prediction
- **Champion-specific Models**: Specialized models for different champions
- **Real-time Integration**: Live game analysis and coaching

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 1.12+
- NumPy 1.20+
- league-replays-parser (parent package)

### Optional Dependencies
- Matplotlib (for visualizations)
- HuggingFace Hub (for dataset loading)
- Gymnasium (for environment interface)

### Hardware Recommendations
- **Training**: NVIDIA GPU with 8GB+ VRAM
- **Inference**: CPU or GPU with 4GB+ VRAM
- **Memory**: 16GB+ RAM for large replay datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the league-of-legends-replays-parser package.

## 🙏 Acknowledgments

- **OpenAI Five Team**: For pioneering work in MOBA game AI
- **DeepMind AlphaStar Team**: For transformer-based game AI architectures
- **Riot Games**: For League of Legends and replay data access
- **Community Contributors**: For replay data collection and parsing tools

## 📚 References

1. OpenAI Five: [Dota 2 with Large Scale Deep Reinforcement Learning](https://arxiv.org/abs/1912.06680)
2. AlphaStar: [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z)
3. Attention Mechanisms: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
4. Pointer Networks: [Pointer Networks](https://arxiv.org/abs/1506.03134)

---

**OpenLeague5** - Bringing the future of AI to League of Legends 🚀