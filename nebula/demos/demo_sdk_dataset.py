#!/usr/bin/env python3
"""
NEBULA Dataset Loader Script using SDK
Follows GROOT dataset loading logic adapted for NEBULA dataset using the SDK wrapper.
"""

import time
from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import tyro

# Use SDK imports instead of direct imports
from nebula.dataset.nebula_sdk import load_nebula_dataset, list_available_robots, NEBULADatasetSDK
from nebula.dataset.embodiment import Embodiment


def print_yellow(text: str) -> None:
    """Print text in yellow color"""
    print(f"\033[93m{text}\033[0m")


def print_blue(text: str) -> None:
    """Print text in blue color"""
    print(f"\033[94m{text}\033[0m")


@dataclass
class NEBULAArgsConfig:
    """Configuration for loading the NEBULA dataset."""

    dataset_root: str = "/path/to/nebula/dataset"
    """Root path to the NEBULA dataset."""

    robot_config_path: str = "./nebula/dataset/embodiment_configs.py"
    """Path to robot configuration file."""

    robot_name: str = "franka_panda_single_arm_2gripper"
    """Robot configuration name to use."""

    task_filter: Optional[List[str]] = field(default_factory=lambda: None)
    """Optional list of task names to filter (None = all tasks)."""

    load_statistics: bool = True
    """Whether to compute episode statistics."""

    plot_state_action: bool = True
    """Whether to plot the state and action trajectories."""

    plot_images: bool = True
    """Whether to plot camera images."""

    steps: int = 100
    """Number of episodes to process for visualization."""

    camera_view: str = "base_camera"
    """Camera view to use for image visualization."""


#####################################################################################


def get_dataset_info(dataset: NEBULADatasetSDK) -> dict:
    """
    Get comprehensive information about the NEBULA dataset using SDK.
    """
    # Use SDK methods instead of direct access
    stats = dataset.get_statistics()
    robot_info = dataset.get_robot_info()
    
    return {
        "dataset_stats": stats,
        "robot_config": robot_info
    }


def plot_state_action_trajectories(
    state_trajectories: dict[str, np.ndarray],
    action_trajectories: dict[str, np.ndarray],
    limb_names: List[str]
):
    """
    Plot state and action trajectories for each limb.
    
    Args:
        state_trajectories: dict with limb_name -> [Time, Dimension] arrays
        action_trajectories: dict with limb_name -> [Time, Dimension] arrays  
        limb_names: list of limb names to plot
    """
    fig, axes = plt.subplots(len(limb_names), 1, figsize=(12, 4 * len(limb_names)))
    if len(limb_names) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10.colors
    
    for i, limb_name in enumerate(limb_names):
        ax = axes[i]
        
        if limb_name not in state_trajectories or limb_name not in action_trajectories:
            ax.text(0.5, 0.5, f"No data for {limb_name}", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{limb_name} (No Data)")
            continue
            
        state_data = state_trajectories[limb_name]
        action_data = action_trajectories[limb_name]
        
        print(f"{limb_name} - State shape: {state_data.shape}, Action shape: {action_data.shape}")
        
        # Plot each dimension
        min_dims = min(state_data.shape[1], action_data.shape[1])
        
        for dim in range(min_dims):
            state_time = np.arange(len(state_data))
            action_time = np.arange(len(action_data))
            
            # State with dashed line
            ax.plot(state_time, state_data[:, dim], '--', 
                   color=colors[dim % len(colors)], linewidth=1.5,
                   label=f'state dim {dim}')
            
            # Action with solid line
            ax.plot(action_time, action_data[:, dim], '-',
                   color=colors[dim % len(colors)], linewidth=2,
                   label=f'action dim {dim}')
        
        ax.set_title(f"{limb_name} Trajectories")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_camera_images(images: List[np.ndarray], camera_name: str, skip_frames: int):
    """
    Plot a grid of camera images.
    """
    num_images = len(images)
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Step {i * skip_frames}")
    
    # Hide extra subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"{camera_name} Images", fontsize=16)
    plt.tight_layout()
    plt.show()


def analyze_episode_data(episode, episode_idx: int, robot_info: dict):
    """
    Analyze and print detailed information about an episode.
    """
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx} Analysis")
    print(f"{'='*60}")
    
    print(f"Task ID: {episode.task_id}")
    print(f"Robot: {episode.robot_name}")
    print(f"Instruction: '{episode.instruction.raw_text}'")
    print(f"Steps: {len(episode.steps)}")
    print(f"Success: {episode.meta.success}")
    print(f"Source: {episode.meta.source}")
    print(f"Environment: {episode.meta.environment}")
    
    if episode.meta.additional:
        print(f"Control Mode: {episode.meta.additional.get('control_mode')}")
        print(f"Elapsed Steps: {episode.meta.additional.get('elapsed_steps')}")
        print(f"Episode Seed: {episode.meta.additional.get('episode_seed')}")
    
    # Analyze first step
    first_step = episode.steps[0]
    obs = first_step.observation
    
    print(f"\nObservation Data:")
    if obs.image:
        print(f"  Camera Views: {list(obs.image.keys())}")
        for camera, img in obs.image.items():
            print(f"    {camera}: {img.shape}")
    
    if obs.depth:
        print(f"  Depth Views: {list(obs.depth.keys())}")
    
    if obs.robot_state:
        print(f"  Robot Limbs: {list(obs.robot_state.limbs.keys())}")
        for limb_name, limb_state in obs.robot_state.limbs.items():
            print(f"    {limb_name}: {limb_state.joint_positions.shape[0]} joints")
    
    if first_step.action.robot_action:
        print(f"  Action Limbs: {list(first_step.action.robot_action.limbs.keys())}")
        for limb_name, action in first_step.action.robot_action.limbs.items():
            print(f"    {limb_name}: {action.shape[0]} actions")
    
    if episode.statistics:
        print(f"\nEpisode Statistics:")
        print(f"  State dims: {episode.statistics.state_mean.shape[0]}")
        print(f"  Action dims: {episode.statistics.action_min.shape[0]}")
        print(f"  State range: [{episode.statistics.state_min.min():.3f}, {episode.statistics.state_max.max():.3f}]")
        print(f"  Action range: [{episode.statistics.action_min.min():.3f}, {episode.statistics.action_max.max():.3f}]")


def extract_trajectories(dataset: NEBULADatasetSDK, num_episodes: int, robot_info: dict):
    """
    Extract state and action trajectories from multiple episodes using SDK.
    """
    # Get limb names from robot info instead of config object
    state_limbs = robot_info['state_limbs']
    action_limbs = robot_info['action_limbs']
    
    state_trajectories = {limb: [] for limb in state_limbs}
    action_trajectories = {limb: [] for limb in action_limbs}
    camera_images = []
    instructions = []
    
    for i in range(min(num_episodes, len(dataset))):
        print(f"Processing episode {i}/{min(num_episodes, len(dataset))}")
        
        episode = dataset[i]
        instructions.append(episode.instruction.raw_text)
        
        # Extract trajectories for this episode
        episode_states = {limb: [] for limb in state_limbs}
        episode_actions = {limb: [] for limb in action_limbs}
        episode_images = []
        
        for step in episode.steps:
            # Extract robot state
            for limb_name, limb_state in step.observation.robot_state.limbs.items():
                if limb_name in episode_states:
                    episode_states[limb_name].append(limb_state.joint_positions)
            
            # Extract actions
            if step.action.robot_action:
                for limb_name, action in step.action.robot_action.limbs.items():
                    if limb_name in episode_actions:
                        episode_actions[limb_name].append(action)
            
            # Extract images (only from first few steps)
            if len(episode_images) < 5 and step.observation.image:
                camera_views = list(step.observation.image.keys())
                if camera_views:
                    episode_images.append(step.observation.image[camera_views[0]])
        
        # Convert to numpy arrays and add to overall trajectories
        for limb_name in state_limbs:
            if episode_states[limb_name]:
                state_traj = np.array(episode_states[limb_name])
                state_trajectories[limb_name].append(state_traj)
        
        for limb_name in action_limbs:
            if episode_actions[limb_name]:
                action_traj = np.array(episode_actions[limb_name])
                action_trajectories[limb_name].append(action_traj)
        
        camera_images.extend(episode_images)
    
    # Concatenate all episode trajectories
    for limb_name in state_limbs:
        if state_trajectories[limb_name]:
            state_trajectories[limb_name] = np.concatenate(state_trajectories[limb_name])
    
    for limb_name in action_limbs:
        if action_trajectories[limb_name]:
            action_trajectories[limb_name] = np.concatenate(action_trajectories[limb_name])
    
    return state_trajectories, action_trajectories, camera_images, instructions


def load_nebula_dataset_demo(
    dataset_root: str,
    robot_config_path: str,
    robot_name: str,
    task_filter: Optional[List[str]] = None,
    load_statistics: bool = True,
    plot_state_action: bool = True,
    plot_images: bool = True,
    steps: int = 100,
    camera_view: str = "base_camera"
):
    """
    Load and visualize NEBULA dataset using SDK following GROOT loading pattern.
    """
    
    print("="*100)
    print(f"{'NEBULA Dataset Loading via SDK':=^100}")
    print("="*100)
    
    # 1. Show available robot configurations
    print_blue("Available robot configurations:")
    try:
        available_robots = list_available_robots(robot_config_path)
        print(f"  Available robots: {available_robots}")
        if robot_name not in available_robots:
            print_yellow(f"Warning: {robot_name} not in available configurations")
    except Exception as e:
        print_yellow(f"Could not list available robots: {e}")
    
    # 2. Initialize dataset using SDK
    print_blue("Initializing NEBULA dataset via SDK...")
    dataset = load_nebula_dataset(
        dataset_root=dataset_root,
        robot_config_path=robot_config_path,
        robot_name=robot_name,
        task_filter=task_filter,
        load_statistics=load_statistics
    )
    
    # 3. Get dataset information using SDK methods
    print_blue("Analyzing dataset structure via SDK...")
    dataset_info = get_dataset_info(dataset)
    
    print("\nDataset Statistics:")
    pprint(dataset_info["dataset_stats"])
    
    print("\nRobot Configuration:")
    pprint(dataset_info["robot_config"])
    
    # 4. Demonstrate SDK query capabilities
    print_blue("Demonstrating SDK query capabilities...")
    
    # Show total episodes
    print(f"Total episodes: {len(dataset)}")
    
    # Show tasks
    tasks = dataset.get_tasks()
    print(f"Available tasks: {tasks}")
    
    
    # 5. Analyze first episode in detail
    if len(dataset) > 0:
        print_blue("Analyzing first episode...")
        first_episode = dataset.get_episode(0)
        analyze_episode_data(first_episode, 0, dataset_info["robot_config"])
    
    # 6. Extract and visualize trajectories
    num_episodes = min(steps // 20, len(dataset))  # Use fewer episodes for visualization
    print_blue(f"Extracting trajectories from {num_episodes} episodes...")
    
    state_trajectories, action_trajectories, camera_images, instructions = extract_trajectories(
        dataset, num_episodes, dataset_info["robot_config"]
    )
    
    print(f"\nExtracted data:")
    print(f"  Instructions: {len(instructions)} episodes")
    print(f"  Camera images: {len(camera_images)} frames")
    
    for limb, traj in state_trajectories.items():
        if traj.size > 0:
            print(f"  State {limb}: {traj.shape}")
    
    for limb, traj in action_trajectories.items():
        if traj.size > 0:
            print(f"  Action {limb}: {traj.shape}")
    
    # 7. Plot state and action trajectories
    if plot_state_action and any(traj.size > 0 for traj in state_trajectories.values()):
        print_blue("Plotting state/action trajectories...")
        
        # Filter out empty trajectories
        valid_limbs = [limb for limb in dataset_info["robot_config"]["state_limbs"]
                      if limb in state_trajectories and state_trajectories[limb].size > 0]
        
        if valid_limbs:
            plot_state_action_trajectories(state_trajectories, action_trajectories, valid_limbs)
        else:
            print_yellow("No valid state/action data to plot")
    
    # 8. Plot camera images
    if plot_images and camera_images:
        print_blue("Plotting camera images...")
        skip_frames = max(1, len(camera_images) // 20)
        sampled_images = camera_images[::skip_frames][:20]  # Show max 20 images
        plot_camera_images(sampled_images, camera_view, skip_frames)
    
    # 9. Show sample instructions
    print_blue("Sample task instructions:")
    unique_instructions = list(set(instructions[:10]))  # Show unique instructions
    for i, instruction in enumerate(unique_instructions[:5]):
        print(f"  {i+1}. '{instruction}'")
    
    # 10. SDK usage examples
    print_blue("SDK usage examples:")
    print("""
# Basic SDK usage:
from nebula_sdk import load_nebula_dataset

# Load dataset
dataset = load_nebula_dataset("/path/to/data", robot_name="franka_panda_single_arm_2gripper")

# Query successful episodes
successful_episodes = dataset.episodes().success(True).execute()

# Query specific task
pick_episodes = dataset.episodes().task("Control-PlaceSphere-Easy").execute()

# Complex queries
episodes = (dataset.episodes()
           .task("pick_and_place") 
           .success(True)
           .min_steps(50)
           .limit(10)
           .execute())

# Get specific episode
episode = dataset.get_episode(0)

# Access data
arm_state = episode.steps[0].observation.robot_state.limbs['arm'].joint_positions
base_camera = episode.steps[0].observation.image['base_camera']

# Train/test split
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2, stratify_by_task=True)

# Random sampling
random_episodes = dataset.sample(n=5, random_state=42)
    """)
    
    # 11. Demonstrate advanced SDK features
    print_blue("Demonstrating advanced SDK features...")
    
    if len(dataset) >= 10:
        # Train/test split demo
        train_dataset, test_dataset = dataset.train_test_split(
            test_size=0.2, 
            random_state=42,
            stratify_by_task=True
        )
        print(f"Train/test split: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Sampling demo
        random_episodes = dataset.sample(n=3, random_state=42)
        print(f"Random sampling: {len(random_episodes)} episodes")
        for i, ep in enumerate(random_episodes):
            print(f"  Sample {i+1}: {ep.task_id} - Success: {ep.meta.success}")
    
    print("="*100)
    print(f"{'SDK Dataset Loading Complete!':=^100}")
    print("="*100)
    
    return dataset


if __name__ == "__main__":
    config = tyro.cli(NEBULAArgsConfig)
    
    try:
        dataset = load_nebula_dataset_demo(
            dataset_root=config.dataset_root,
            robot_config_path=config.robot_config_path,
            robot_name=config.robot_name,
            task_filter=config.task_filter,
            load_statistics=config.load_statistics,
            plot_state_action=config.plot_state_action,
            plot_images=config.plot_images,
            steps=config.steps,
            camera_view=config.camera_view
        )
        
        print_yellow(f"Successfully loaded dataset with {len(dataset)} episodes using SDK!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the dataset_root path to point to your NEBULA dataset")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()