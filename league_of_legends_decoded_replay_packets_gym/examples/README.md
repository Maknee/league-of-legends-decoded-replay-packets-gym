# Examples

This directory contains example applications and tools for the League of Legends Decoded Replay Packets Gym.

## 🤖 [OpenLeague5 AI System](openleague5/)

**Action Prediction Neural Network inspired by OpenAI Five and AlphaStar**

The OpenLeague5 system demonstrates how to use decoded replay packets for advanced AI development. It includes:

- **Multi-modal Neural Network**: CNN + Transformer + LSTM architecture
- **Action Prediction**: Predicts what professional players will do next
- **Behavior Cloning**: Learn from professional gameplay patterns
- **Interactive Demo**: Real-time prediction visualization

```bash
cd openleague5/
python -m openleague5 demo --model models/pro.pt --input worlds_final.jsonl.gz
```

See [openleague5/README.md](openleague5/README.md) for detailed documentation.

## 📊 [Champion Movement Visualizer](champion_gif_generator.py)

**Generate animated GIFs showing champion positioning over time**

Creates beautiful visualizations of champion movements during professional matches:

- **Animated GIFs**: Champion trails and positioning
- **Customizable**: Time range, resolution, speed
- **Professional Data**: Works with HuggingFace datasets

```bash
python champion_gif_generator.py --dataset "worlds_2022/finals.jsonl.gz" --max-time 10 --fps 8
```

**Key Features:**
- Summoner's Rift map visualization
- Champion movement trails  
- Team positioning analysis
- Export as high-quality GIF

## 🚀 Getting Started

All examples work with the HuggingFace dataset by default:

```python
import league_of_legends_decoded_replay_packets_gym as lol_gym

# Load professional data
dataset = lol_gym.ReplayDataset([
    "12_22/batch_001.jsonl.gz"
], repo_id="maknee/league-of-legends-decoded-replay-packets")

dataset.load(max_games=1)
```

## 📚 Example Use Cases

### Research Applications
- **Esports Analytics**: Analyze professional gameplay patterns
- **AI Development**: Train League of Legends playing agents
- **Behavioral Studies**: Study decision-making in competitive gaming

### Educational Applications  
- **RL Learning**: Practical reinforcement learning examples
- **Data Science**: Real-world dataset for analysis projects
- **Game AI**: Neural network architectures for games

### Industry Applications
- **Coaching Tools**: Automated analysis for teams
- **Content Creation**: Generate insights for streaming/broadcast
- **Performance Analysis**: Player and team evaluation

## 🛠️ Requirements

Most examples require additional dependencies:

```bash
# For AI examples (OpenLeague5)
pip install torch torchvision matplotlib

# For visualizations  
pip install matplotlib pillow

# Or install everything
pip install league-of-legends-decoded-replay-packets-gym[ai]
```

## 💡 Contributing Examples

Want to add your own example? Great! Here's what makes a good example:

1. **Clear Purpose**: Solves a specific problem or demonstrates a concept
2. **Professional Data**: Works with the HuggingFace dataset
3. **Documentation**: Include README.md explaining usage
4. **Self-contained**: Minimal external dependencies
5. **Educational Value**: Helps others learn or research

Example structure:
```
my_example/
├── README.md           # What it does and how to use it
├── my_example.py      # Main implementation
├── requirements.txt   # Additional dependencies
└── demo.py           # Simple usage demonstration
```

## 🔗 Related Resources

- **Main Package**: [league-of-legends-decoded-replay-packets-gym](../)
- **HuggingFace Dataset**: [maknee/league-of-legends-decoded-replay-packets](https://huggingface.co/datasets/maknee/league-of-legends-decoded-replay-packets)
- **Gymnasium Documentation**: [gymnasium.farama.org](https://gymnasium.farama.org/)

---

**Ready to build something cool with League of Legends data?** 🚀

Start with the basic gym environment, then explore the AI and visualization examples!