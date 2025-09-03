# League of Legends Decoded Replay Packets Gym 🏋️‍♀️

# Disclaimer

This work isn’t endorsed by Riot Games and doesn’t reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends. League of Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc.

**A Gymnasium Environment for League of Legends Decoded Replay Packets**

[![PyPI version](https://badge.fury.io/py/league-of-legends-decoded-replay-packets-gym.svg)](https://badge.fury.io/py/league-of-legends-decoded-replay-packets-gym)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance **Gymnasium environment** for League of Legends replay analysis, AI development, and esports research. Access decoded replay packets from professional matches with a simple, standardized interface.

## 🚀 Quick Start

```bash
pip install league-of-legends-decoded-replay-packets-gym
```

```python
import league_of_legends_decoded_replay_packets_gym as lol_gym

# Load professional replay data from HuggingFace
dataset = lol_gym.ReplayDataset([
    "12_22/"  # Download entire patch directory
], repo_id="maknee/league-of-legends-decoded-replay-packets")

# Or specific files
dataset = lol_gym.ReplayDataset([
    "12_22/batch_001.jsonl.gz",  # Professional matches from patch 12.22
    "12_22/batch_002.jsonl.gz"
], repo_id="maknee/league-of-legends-decoded-replay-packets")

dataset.load(max_games=10)  # Load first 10 games

# Create Gymnasium environment
env = lol_gym.LeagueReplaysEnv(dataset, time_step=1.0)
obs, info = env.reset()

print(f"🎮 Loaded game {info['game_id']}")
print(f"⏰ Starting at time: {info['current_time']:.1f}s")

# Step through decoded replay packets
for step in range(100):
    obs, reward, terminated, truncated, info = env.step(0)  # Continue action
    
    game_state = info['game_state']
    
    if game_state.heroes:
        print(f"Step {step}: t={game_state.current_time:.1f}s, "
              f"heroes={len(game_state.heroes)}, "
              f"events={len(game_state.events)}")
        
        # Access decoded packet data
        for net_id, hero in list(game_state.heroes.items())[:3]:
            pos = game_state.get_position(net_id)
            if pos:
                print(f"  {hero.get('name', 'Hero')}: ({pos.x:.0f}, {pos.z:.0f})")
    
    if terminated or truncated:
        print("🏁 Game ended, resetting...")
        obs, info = env.reset()

env.close()
```

## 🎯 Features

- **🏃‍♂️ Gymnasium Interface**: Standard RL environment for easy integration
- **⚡ High Performance**: Rust-accelerated replay parsing with Python fallback
- **📊 Professional Data**: Access to decoded packets from real esports matches
- **🧠 AI Ready**: Includes neural network examples (OpenLeague5)
- **🔧 Flexible Observations**: Minimap, positional, event-based, and custom observations
- **🎮 Real Game Data**: Professional tournament replays from HuggingFace

## 📚 Data Sources

### HuggingFace Dataset (Primary)
The main data source is [maknee/league-of-legends-decoded-replay-packets](https://huggingface.co/datasets/maknee/league-of-legends-decoded-replay-packets):

```python
# Available datasets
dataset = lol_gym.ReplayDataset([
    "12_22/",                        # Entire patch directory
    "worlds_2022/",                  # Entire tournament directory  
    "13_1/batch_001.jsonl.gz"        # Specific file
], repo_id="maknee/league-of-legends-decoded-replay-packets")

# Individual files also supported
dataset = lol_gym.ReplayDataset([
    "12_22/batch_001.jsonl.gz",      # Pro matches, patch 12.22
    "12_22/batch_002.jsonl.gz",      # More pro matches
    "worlds_2022/semifinals.jsonl.gz", # Championship matches
    "worlds_2022/finals.jsonl.gz"      # Grand finals
], repo_id="maknee/league-of-legends-decoded-replay-packets")
```

### Local Files
```python
# Use your own replay files
dataset = lol_gym.ReplayDataset(["local_replay.jsonl.gz"])
```

## 🤖 AI Examples

### Champion Movement Visualization

![movement](champion_movement.gif)

```python
from league_of_legends_decoded_replay_packets_gym.examples.champion_gif_generator import ChampionGifGenerator

# Create animated GIF of champion movements
dataset = lol_gym.ReplayDataset(
    ["worlds_2022/finals.jsonl.gz"],
    repo_id="maknee/league-of-legends-decoded-replay-packets"
)
dataset.load(max_games=1)

generator = ChampionGifGenerator()
generator.create_gif(
    dataset=dataset,
    output_path="worlds_final_movements.gif",
    max_time_minutes=5,
    fps=6
)
```

### Action Prediction with OpenLeague5
```python
from league_of_legends_decoded_replay_packets_gym.examples.openleague5 import OpenLeague5Model

# Load professional data  
dataset = lol_gym.ReplayDataset(
    ["12_22/"],  # Download entire patch directory
    repo_id="maknee/league-of-legends-decoded-replay-packets"
)
dataset.load(max_games=1)

# Create environment and jump to 15 minutes
env = lol_gym.LeagueReplaysEnv(dataset)
obs, info = env.reset()

# Step to 15 minutes (900 seconds)
while info['current_time'] < 900:
    obs, reward, terminated, truncated, info = env.step(0)
    if terminated or truncated:
        break

# AI predicts what players will do next
model = OpenLeague5Model()
game_state = info['game_state']

prediction = model.predict_next_action(game_state, temperature=1.0)
print(f"🔮 AI Prediction: {prediction.get_action_description()}")
print(f"   Confidence: {prediction.confidence:.3f}")
```

```bash
🎯 Prediction Results:
==============================
Action: Use W Ability
Confidence: 0.354
State Value: -0.681
Target Position: (7266, 3750) world coords
Coordinate Confidence: X=0.158, Y=0.080
Unit Target: 0
Unit Confidence: 1.000
✅ Prediction completed successfully!
```

## 🔧 Advanced Usage

### Custom Observations
```python
from league_of_legends_decoded_replay_packets_gym.observations import MinimapObservation

# Create 128x128 minimap observation
minimap_obs = MinimapObservation(
    resolution=128, 
    channels=['heroes', 'minions', 'structures']
)

env = lol_gym.LeagueReplaysEnv(dataset, observation_callback=minimap_obs)
obs, info = env.reset()

print(f"Minimap shape: {obs['minimap'].shape}")  # [3, 128, 128]
```

### Raw Parser Access
```python
# Direct access to replay parsing
parser = lol_gym.UnifiedLeagueParser()
result = parser.parse_file("replay.jsonl.gz")

print(f"Parsed {result.games_parsed} games")
print(f"Total events: {result.total_events}")
print(f"Method used: {result.method_used}")
```

### Multi-Environment Training
```python
# Multiple parallel environments for RL training
manager = lol_gym.MultiEnvManager(dataset, num_envs=4)
states = manager.reset()

for epoch in range(100):
    # Step all environments in parallel
    results = manager.step()
    
    for i, (obs, reward, terminated, truncated, info) in enumerate(results):
        if terminated or truncated:
            print(f"Environment {i} finished game")
```

## 🛠️ Installation Options

```bash
# Basic installation (core gym environment)
pip install league-of-legends-decoded-replay-packets-gym

# With AI examples (includes PyTorch, matplotlib)
pip install league-of-legends-decoded-replay-packets-gym[ai]

# Development installation
pip install league-of-legends-decoded-replay-packets-gym[dev]

# Everything
pip install league-of-legends-decoded-replay-packets-gym[all]
```

## 🎮 Command Line Interface

```bash
# Basic gym environment demo
league-gym env --data "12_22/batch_001.jsonl.gz" --steps 100

# Parse replay files directly
league-gym parse local_replay.jsonl.gz

# AI prediction demo
league-gym ai predict --model openleague5 --time 900 --data "worlds_2022/finals.jsonl.gz"

# Generate champion movement GIF
league-gym viz movement --data "12_22/batch_001.jsonl.gz" --output movements.gif
```

## 📁 Examples

All examples are included in the package and have their own documentation:

### 🎯 [OpenLeague5 AI System](league_of_legends_decoded_replay_packets_gym/examples/openleague5/)
Neural network system for action prediction, inspired by OpenAI Five and AlphaStar.

### 📊 [Champion Movement Visualizer](league_of_legends_decoded_replay_packets_gym/examples/champion_gif_generator/)
Generate animated GIFs showing champion positioning over time.

See [`examples/README.md`](league_of_legends_decoded_replay_packets_gym/examples/README.md) for a complete overview.

## 🏗️ Architecture

### Gymnasium Environment
- **Observation Space**: Configurable (positions, minimap, events, custom)
- **Action Space**: Discrete actions (continue, skip, jump to time)
- **Reward Function**: Customizable based on research needs
- **Info Dict**: Rich game state with decoded packet access

### Game State Access
```python
game_state = info['game_state']

# Core information
game_state.current_time          # Game time in seconds
game_state.heroes               # All heroes {net_id: hero_info}
game_state.positions            # All positions {net_id: Position}
game_state.events              # Recent events [GameEvent]

# Convenience methods
game_state.get_heroes_by_team('ORDER')           # Team filtering
game_state.get_heroes_in_radius(pos, 1000)      # Spatial queries  
game_state.get_events_by_type('CastSpellAns')   # Event filtering
```

### Packet Types
The environment provides access to decoded packet data including:
- `CreateHero`: Hero spawn events
- `WaypointGroup`: Movement and positioning
- `CastSpellAns`: Ability usage
- `UnitApplyDamage`: Combat events
- `BuyItem`: Item purchases
- `HeroDie`: Elimination events

## 🎓 Research Applications

- **Esports Analytics**: Analyze professional gameplay patterns
- **AI Development**: Train League of Legends playing agents
- **Reinforcement Learning**: Standard gym environment for RL research  
- **Behavioral Analysis**: Study decision-making in competitive gaming
- **Meta-game Research**: Track strategic evolution across patches

## 🤝 Contributing

Contributions are welcome! Please see the examples for adding new analysis tools or AI models.

```bash
# Development setup
git clone https://github.com/your-org/league-of-legends-decoded-replay-packets-gym.git
cd league-of-legends-decoded-replay-packets-gym
pip install -e .[dev]

# Run tests
python -m pytest

# Format code
black league_of_legends_decoded_replay_packets_gym/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Riot Games** for League of Legends
- **Professional Players** for the gameplay data
- **maknee** for decoded replay packet dataset
- **Gymnasium Project** for the RL environment standard
- **OpenAI & DeepMind** for AI research inspiration

---

**Ready to analyze professional League of Legends gameplay?** 🚀

```bash
pip install league-of-legends-decoded-replay-packets-gym
```