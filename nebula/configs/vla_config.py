# File: configs/vla_config.py

"""
Configuration file for VLA Pick and Place Environment
"""

VLA_CONFIG = {
    # Environment settings
    "env_id": "VLAPickPlace-v1",
    "max_episode_steps": 200,
    "control_timestep": 0.02,
    
    # Task parameters
    "cube_half_size": 0.02,
    "target_radius": 0.05,
    "time_limit": 15.0,
    "success_threshold": 0.025,
    
    # Visual settings
    "camera_configs": {
        "base_camera": {
            "width": 128,
            "height": 128,
            "fov": 1.5708,  # pi/2
        },
        "hand_camera": {
            "width": 128,
            "height": 128,
            "fov": 1.5708,
        }
    },
    
    # Language settings
    "instruction_complexity": "medium",  # simple, medium, complex
    "max_instruction_length": 100,
    
    # Evaluation settings
    "evaluation": {
        "num_episodes": 100,
        "num_parallel_envs": 4,
        "success_threshold": 0.025,
        "efficiency_weight": 0.3,
    },
    
    # Data collection settings
    "data_collection": {
        "episodes_per_task": 1000,
        "randomization_level": "medium",
        "save_rgb": True,
        "save_depth": False,
        "save_trajectory": True,
    }
}


# File: examples/vla_data_collection.py

import gymnasium as gym
import nebula.envs
import torch
import numpy as np
import h5py
import os
from pathlib import Path
import time
from typing import Dict, List
import argparse
import json


class VLADataCollector:
    """
    Data collector for VLA training datasets
    Collects multi-modal data: vision + language + actions
    """
    
    def __init__(self, env_id: str, output_dir: str, num_envs: int = 4):
        self.env_id = env_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_envs = num_envs
        
        # Create environment
        self.env = gym.make(
            env_id,
            num_envs=num_envs,
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            render_mode="rgb_array",
            sim_backend="gpu"
        )
        
        # Data storage
        self.episode_data = []
        self.current_episodes = [[] for _ in range(num_envs)]
        
    def collect_demonstration_data(self, num_episodes: int, policy_type: str = "random"):
        """
        Collect demonstration data using specified policy
        """
        print(f"Collecting {num_episodes} episodes with {policy_type} policy...")
        
        episodes_collected = 0
        obs, info = self.env.reset()
        step_count = 0
        
        while episodes_collected < num_episodes:
            # Store current step data
            for env_idx in range(self.num_envs):
                step_data = self._extract_step_data(obs, info, env_idx, step_count)
                self.current_episodes[env_idx].append(step_data)
            
            # Get action based on policy type
            if policy_type == "random":
                action = self.env.action_space.sample()
            elif policy_type == "heuristic":
                action = self._get_heuristic_action(obs)
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Add action to step data
            for env_idx in range(self.num_envs):
                if self.current_episodes[env_idx]:
                    self.current_episodes[env_idx][-1]["action"] = action[env_idx].cpu().numpy()
                    self.current_episodes[env_idx][-1]["reward"] = reward[env_idx].cpu().numpy()
            
            done = terminated | truncated
            step_count += 1
            
            # Handle episode completion
            if torch.any(done):
                for env_idx in range(self.num_envs):
                    if done[env_idx] and episodes_collected < num_episodes:
                        # Save completed episode
                        episode_data = {
                            "episode_id": episodes_collected,
                            "env_idx": env_idx,
                            "success": bool(obs.get("is_success", [False])[env_idx]),
                            "steps": self.current_episodes[env_idx].copy(),
                            "total_steps": len(self.current_episodes[env_idx]),
                            "instruction": obs.get("instruction", [""])[0] if isinstance(obs.get("instruction", [""]), list) else obs.get("instruction", "")
                        }
                        
                        self.episode_data.append(episode_data)
                        self.current_episodes[env_idx] = []
                        episodes_collected += 1
                        
                        if episodes_collected % 50 == 0:
                            print(f"Collected {episodes_collected}/{num_episodes} episodes")
                            self._save_intermediate_data(episodes_collected)
                
                # Reset if needed
                if episodes_collected < num_episodes:
                    obs, info = self.env.reset()
                    step_count = 0
        
        # Final save
        self._save_final_data()
        print(f"Data collection complete! Saved to {self.output_dir}")
        
    def _extract_step_data(self, obs: Dict, info: Dict, env_idx: int, step: int) -> Dict:
        """Extract data for a single step"""
        step_data = {
            "step": step,
            "timestamp": time.time(),
            
            # Visual observations
            "base_camera_rgb": obs.get("sensor_data", {}).get("base_camera", {}).get("rgb", [None])[env_idx],
            "hand_camera_rgb": obs.get("sensor_data", {}).get("hand_camera", {}).get("rgb", [None])[env_idx],
            
            # State information
            "cube_pos": obs.get("cube_pos", [None])[env_idx],
            "target_pos": obs.get("target_pos", [None])[env_idx],
            "distance_to_target": obs.get("distance_to_target", [None])[env_idx],
            "remaining_time": obs.get("remaining_time", [None])[env_idx],
            
            # Language
            "instruction": obs.get("instruction", ""),
            
            # Will be filled in next step
            "action": None,
            "reward": None
        }
        
        # Convert tensors to numpy
        for key, value in step_data.items():
            if isinstance(value, torch.Tensor):
                step_data[key] = value.cpu().numpy()
        
        return step_data
    
    def _get_heuristic_action(self, obs: Dict) -> torch.Tensor:
        """Simple heuristic policy for demonstration"""
        batch_size = self.num_envs
        actions = torch.zeros((batch_size, 4), device=self.env.device)
        
        for env_idx in range(batch_size):
            cube_pos = obs.get("cube_pos", torch.zeros((batch_size, 3)))[env_idx]
            target_pos = obs.get("target_pos", torch.zeros((batch_size, 3)))[env_idx]
            distance = obs.get("distance_to_target", torch.ones(batch_size))[env_idx]
            
            if distance > 0.1:
                # Move towards target
                direction = target_pos - cube_pos
                direction = direction / (torch.norm(direction) + 1e-6)
                actions[env_idx, :3] = direction * 0.05
                actions[env_idx, 3] = 1.0  # Close gripper
            else:
                # At target
                actions[env_idx, 3] = -1.0  # Open gripper
        
        return actions
    
    def _save_intermediate_data(self, episodes_so_far: int):
        """Save data periodically"""
        filepath = self.output_dir / f"intermediate_data_{episodes_so_far}.json"
        
        # Save metadata only for intermediate saves
        metadata = {
            "episodes_collected": episodes_so_far,
            "total_episodes": len(self.episode_data),
            "env_id": self.env_id,
            "timestamp": time.time()
        }
        
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _save_final_data(self):
        """Save complete dataset"""
        # Save as HDF5 for efficient storage
        hdf5_path = self.output_dir / "vla_dataset.h5"
        
        with h5py.File(hdf5_path, "w") as f:
            # Dataset metadata
            f.attrs["env_id"] = self.env_id
            f.attrs["num_episodes"] = len(self.episode_data)
            f.attrs["collection_time"] = time.time()
            
            # Create groups for each episode
            for i, episode in enumerate(self.episode_data):
                ep_group = f.create_group(f"episode_{i}")
                ep_group.attrs["episode_id"] = episode["episode_id"]
                ep_group.attrs["success"] = episode["success"]
                ep_group.attrs["total_steps"] = episode["total_steps"]
                ep_group.attrs["instruction"] = episode["instruction"]
                
                # Store step data
                steps_group = ep_group.create_group("steps")
                for j, step in enumerate(episode["steps"]):
                    step_group = steps_group.create_group(f"step_{j}")
                    
                    for key, value in step.items():
                        if value is not None and key != "instruction":
                            try:
                                step_group.create_dataset(key, data=value)
                            except (TypeError, ValueError):
                                # Handle non-numeric data
                                step_group.attrs[key] = str(value)
        
        # Also save as JSON for easy inspection
        json_path = self.output_dir / "vla_dataset_summary.json"
        summary = {
            "dataset_info": {
                "env_id": self.env_id,
                "num_episodes": len(self.episode_data),
                "success_rate": np.mean([ep["success"] for ep in self.episode_data]),
                "avg_episode_length": np.mean([ep["total_steps"] for ep in self.episode_data]),
            },
            "sample_episodes": self.episode_data[:5]  # First 5 episodes as examples
        }
        
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Collect VLA training data")
    parser.add_argument("--env-id", default="VLAPickPlace-v1", help="Environment ID")
    parser.add_argument("--output-dir", default="vla_data", help="Output directory")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--policy", default="random", choices=["random", "heuristic"], help="Data collection policy")
    
    args = parser.parse_args()
    
    collector = VLADataCollector(
        env_id=args.env_id,
        output_dir=args.output_dir,
        num_envs=args.num_envs
    )
    
    collector.collect_demonstration_data(
        num_episodes=args.num_episodes,
        policy_type=args.policy
    )


if __name__ == "__main__":
    main()