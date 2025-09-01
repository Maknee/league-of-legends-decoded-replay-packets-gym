#!/usr/bin/env python3
"""
League of Legends Champion Movement GIF Generator

Command-line tool for generating champion movement GIFs with customizable parameters.

Usage:
    python examples/champion_gif_generator.py [options]

Examples:
    # Basic usage (5-minute GIF with 5-second timesteps)
    python examples/champion_gif_generator.py
    
    # Custom duration and timesteps
    python examples/champion_gif_generator.py --max-time 10 --timestep 2.0
    
    # High-resolution, longer GIF
    python examples/champion_gif_generator.py --max-time 15 --timestep 1.0 --fps 10
    
    # Quick preview
    python examples/champion_gif_generator.py --max-time 2 --timestep 10.0 --fps 5
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import defaultdict, deque

# Add the python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import league_replays_parser as lrp

class ChampionGifGenerator:
    def __init__(self, map_width=14800, map_height=14800):
        """Initialize the champion GIF generator"""
        self.map_width = map_width
        self.map_height = map_height
        
        # Enhanced color palette
        self.champion_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#FF8B94', '#6BCF7F'
        ]
        
        self.position_history = defaultdict(lambda: deque(maxlen=25))
        
    def get_entity_to_champion_mapping(self, dataset):
        """Get the mapping from position entities to champion info"""
        print("📋 Building entity-to-champion mapping...")
        
        # Load environment to get hero data
        env = lrp.LeagueReplaysEnv(dataset, time_step=5.0)
        obs, info = env.reset()
        
        # Get heroes (wait for them to spawn)
        heroes = {}
        step = 0
        while step < 20 and len(heroes) < 10:
            obs, reward, terminated, truncated, info = env.step(0)
            heroes = info['game_state'].heroes.copy()
            step += 1
        
        # Get position entities order
        entity_first_seen = {}
        step = 0
        while step < 20:
            obs, reward, terminated, truncated, info = env.step(0)
            
            for entity_id, pos in info['game_state'].positions.items():
                if entity_id not in entity_first_seen:
                    entity_first_seen[entity_id] = step
            
            step += 1
            if terminated or truncated:
                break
        
        env.close()
        
        # Create mapping based on chronological order
        hero_creation_order = []
        for net_id, hero_info in heroes.items():
            hero_creation_order.append((net_id, hero_info))
        
        entity_ids_ordered = sorted(entity_first_seen.keys(), key=lambda x: entity_first_seen[x])
        
        mapping = {}
        for i, entity_id in enumerate(entity_ids_ordered):
            if i < len(hero_creation_order):
                hero_net_id, hero_info = hero_creation_order[i]
                username = hero_info.get('name', f'Player_{i+1}')
                champion = hero_info.get('champion', f'Champion_{i+1}')
                
                # Clean up names for display
                username = username[:15] + '...' if len(username) > 15 else username
                champion = champion.title()
                
                mapping[entity_id] = {
                    'username': username,
                    'champion': champion,
                    'hero_id': hero_net_id,
                    'color': self.champion_colors[i % len(self.champion_colors)]
                }
                
                print(f"   Entity {entity_id} → {username} ({champion})")
        
        return mapping
    
    def normalize_position(self, x, z):
        """Normalize game coordinates to [0,1] range"""
        norm_x = (x + self.map_width/2) / self.map_width
        norm_z = (z + self.map_height/2) / self.map_width
        return np.clip(norm_x, 0, 1), np.clip(norm_z, 0, 1)
    
    def setup_plot(self, entity_mapping, show_names=True):
        """Setup the matplotlib plot"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_facecolor('#0F2027')
        
        # Add Summoner's Rift features
        river = patches.Polygon([(0.3, 0.3), (0.7, 0.7), (0.75, 0.65), (0.35, 0.25)], 
                              facecolor='#1a4a5c', alpha=0.7)
        ax.add_patch(river)
        
        # Lanes
        top_lane = patches.Rectangle((0.1, 0.7), 0.8, 0.05, facecolor='#2d2d2d', alpha=0.5)
        ax.add_patch(top_lane)
        
        mid_lane = patches.Polygon([(0.1, 0.1), (0.9, 0.9), (0.85, 0.95), (0.05, 0.15)], 
                                 facecolor='#2d2d2d', alpha=0.5)
        ax.add_patch(mid_lane)
        
        bot_lane = patches.Rectangle((0.7, 0.1), 0.05, 0.8, facecolor='#2d2d2d', alpha=0.5)
        ax.add_patch(bot_lane)
        
        # Jungle camps
        camps = [(0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)]
        for camp_x, camp_y in camps:
            camp = patches.Circle((camp_x, camp_y), 0.03, facecolor='#4a4a4a', alpha=0.6)
            ax.add_patch(camp)
        
        ax.set_title('League of Legends - Champion Movement', 
                    fontsize=18, color='white', weight='bold', pad=20)
        ax.set_xlabel('Map X Position', color='white', fontsize=12)
        ax.set_ylabel('Map Z Position', color='white', fontsize=12)
        ax.tick_params(colors='white')
        
        # Create legend
        if show_names and entity_mapping:
            legend_elements = []
            for entity_id in sorted(entity_mapping.keys())[:8]:
                info = entity_mapping[entity_id]
                color = info['color']
                username = info['username']
                champion = info['champion']
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                              markersize=8, label=f"{username}\n({champion})",
                              markeredgecolor='white', markeredgewidth=1)
                )
            
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                     fancybox=True, shadow=True, fontsize=10)
        
        fig.patch.set_facecolor('#1a1a1a')
        plt.tight_layout()
        
        return fig, ax
    
    def extract_movement_data(self, dataset, max_time_minutes, timestep_seconds):
        """Extract movement data with configurable parameters"""
        print(f"📊 Extracting movement data...")
        print(f"   Max time: {max_time_minutes} minutes")
        print(f"   Time step: {timestep_seconds} seconds")
        
        # Get entity mapping
        entity_mapping = self.get_entity_to_champion_mapping(dataset)
        
        # Extract movement data
        movement_data = defaultdict(lambda: defaultdict(tuple))
        
        env = lrp.LeagueReplaysEnv(
            dataset, 
            max_time=max_time_minutes * 60.0, 
            time_step=timestep_seconds
        )
        
        obs, info = env.reset()
        print(f"🎮 Processing game {info['game_id']}")
        
        step_count = 0
        entities_found = False
        
        while True:
            current_time = step_count * timestep_seconds
            
            game_state = info['game_state']
            
            # Extract position data
            active_entities = 0
            for entity_id, pos in game_state.positions.items():
                if entity_id in entity_mapping:
                    active_entities += 1
                    movement_data[current_time][entity_id] = (pos.x, pos.z)
            
            if active_entities > 0 and not entities_found:
                entities_found = True
                print(f"   🎯 Found {active_entities} champions with movement data")
            
            obs, reward, terminated, truncated, info = env.step(0)
            step_count += 1
            
            if terminated or truncated:
                break
                
            # Progress reporting based on timestep
            progress_interval = max(1, int(60 / timestep_seconds))  # Every minute
            if step_count % progress_interval == 0:
                elapsed_minutes = (step_count * timestep_seconds) // 60
                print(f"   Processed {elapsed_minutes} minutes...")
        
        env.close()
        
        total_time = step_count * timestep_seconds
        print(f"✅ Extracted {step_count} time steps ({total_time:.0f}s total) with {len(entity_mapping)} champions")
        return movement_data, entity_mapping
    
    def create_gif(self, movement_data, entity_mapping, output_path, fps, timestep_seconds, show_names=True):
        """Create GIF with configurable parameters"""
        print(f"🎬 Creating GIF animation...")
        print(f"   Output: {output_path}")
        print(f"   FPS: {fps}")
        print(f"   Show names: {show_names}")
        
        fig, ax = self.setup_plot(entity_mapping, show_names)
        
        # Initialize plots
        entity_plots = {}
        trail_plots = {}
        name_texts = {}
        
        for entity_id, info in entity_mapping.items():
            color = info['color']
            
            entity_plots[entity_id] = ax.scatter([], [], s=120, c=color, 
                                               marker='o', edgecolors='white', linewidth=2,
                                               zorder=10)
            trail_plots[entity_id] = ax.scatter([], [], s=20, c=color, 
                                              alpha=0.5, marker='o', zorder=5)
            
            if show_names:
                name_texts[entity_id] = ax.text(0, 0, '', fontsize=8, color='white', 
                                              weight='bold', ha='center', va='bottom',
                                              bbox=dict(boxstyle="round,pad=0.2", 
                                                      facecolor=color, alpha=0.7),
                                              zorder=15)
        
        # Time display
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=14, color='white', weight='bold',
                           verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
        
        def animate(frame):
            current_time = frame * timestep_seconds
            active_champions = 0
            
            for entity_id, info in entity_mapping.items():
                if current_time in movement_data and entity_id in movement_data[current_time]:
                    world_x, world_z = movement_data[current_time][entity_id]
                    norm_x, norm_z = self.normalize_position(world_x, world_z)
                    
                    self.position_history[entity_id].append((norm_x, norm_z))
                    
                    entity_plots[entity_id].set_offsets([[norm_x, norm_z]])
                    
                    if len(self.position_history[entity_id]) > 1:
                        trail_positions = list(self.position_history[entity_id])
                        trail_x = [pos[0] for pos in trail_positions[:-1]]
                        trail_z = [pos[1] for pos in trail_positions[:-1]]
                        trail_plots[entity_id].set_offsets(list(zip(trail_x, trail_z)))
                    
                    if show_names:
                        username = info['username']
                        name_texts[entity_id].set_position((norm_x, norm_z + 0.03))
                        name_texts[entity_id].set_text(username)
                    
                    active_champions += 1
                else:
                    entity_plots[entity_id].set_offsets(np.empty((0, 2)))
                    trail_plots[entity_id].set_offsets(np.empty((0, 2)))
                    if show_names:
                        name_texts[entity_id].set_text('')
            
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            time_text.set_text(f'Time: {minutes:02d}:{seconds:02d}\nActive: {active_champions}/10')
            
            artists = (list(entity_plots.values()) + list(trail_plots.values()) + 
                      list(name_texts.values()) + [time_text])
            return artists
        
        # Calculate frames
        max_time = max(movement_data.keys()) if movement_data else 0
        num_frames = int(max_time / timestep_seconds) + 1
        
        print(f"   Creating animation with {num_frames} frames")
        
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000//fps, 
                           blit=False, repeat=True)
        
        # Save GIF
        print(f"💾 Saving GIF...")
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=120)
        plt.close()
        
        print(f"✅ GIF saved successfully!")
        return output_path

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate League of Legends champion movement GIFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Default: 5 minutes, 5.0s timestep
  %(prog)s --max-time 10                # 10 minutes with default timestep  
  %(prog)s --timestep 2.0               # 5 minutes with 2-second timestep
  %(prog)s --max-time 3 --timestep 1.0  # 3 minutes with 1-second timestep
  %(prog)s --fps 12 --no-names          # High FPS without player names
        """
    )
    
    parser.add_argument('--max-time', type=float, default=5.0,
                       help='Maximum time to extract in minutes (default: 5.0)')
    
    parser.add_argument('--timestep', type=float, default=5.0,
                       help='Time step interval in seconds (default: 5.0)')
    
    parser.add_argument('--fps', type=int, default=6,
                       help='Frames per second for GIF (default: 6)')
    
    parser.add_argument('--output', type=str, default='champion_movement.gif',
                       help='Output filename (default: champion_movement.gif)')
    
    parser.add_argument('--dataset', type=str, default='12_22/batch_001.jsonl.gz',
                       help='Dataset file to use (default: 12_22/batch_001.jsonl.gz)')
    
    parser.add_argument('--no-names', action='store_true',
                       help='Hide player names in the visualization')
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick preview mode (2 min, 10s timestep, 5 fps)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.max_time = 2.0
        args.timestep = 10.0
        args.fps = 5
        args.output = 'champion_movement_quick.gif'
        print("🚀 Quick preview mode activated")
    
    print("🎮 League of Legends Champion Movement GIF Generator")
    print("=" * 60)
    print(f"⚙️  Configuration:")
    print(f"   Max time: {args.max_time} minutes")
    print(f"   Time step: {args.timestep} seconds")
    print(f"   FPS: {args.fps}")
    print(f"   Output: {args.output}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Show names: {not args.no_names}")
    print("=" * 60)
    
    try:
        # Load dataset
        print("📦 Loading League of Legends replay data...")
        dataset = lrp.ReplayDataset([args.dataset])
        dataset.load(max_games=1)
        
        print(f"✅ Loaded {len(dataset)} games")
        
        # Create generator
        generator = ChampionGifGenerator()
        
        # Extract movement data
        movement_data, entity_mapping = generator.extract_movement_data(
            dataset, args.max_time, args.timestep
        )
        
        if not movement_data:
            print("❌ No movement data found!")
            return 1
        
        # Create GIF
        generator.create_gif(
            movement_data, entity_mapping, args.output, 
            args.fps, args.timestep, show_names=not args.no_names
        )
        
        # Summary
        file_size = os.path.getsize(args.output) / (1024*1024)
        print(f"\n🎉 Successfully created champion movement GIF!")
        print(f"📁 File: {args.output} ({file_size:.1f} MB)")
        print(f"📊 Champions tracked: {len(entity_mapping)}")
        print(f"⏱️  Duration: {args.max_time} minutes at {args.timestep}s intervals")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())