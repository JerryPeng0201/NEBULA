import os
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from dataclasses import dataclass

from nebula.dataset.embodiment import Embodiment
from nebula.dataset.structures import (
    Episode, Step, Observation, Action, LanguageInstruction,
    RobotState, RobotLimbState, RobotAction, MetaInfo, EpisodeStatistics
)


class NEBULADataset:
    """
    Dataset class for loading NEBULA data in Unified Data Format with transform support.
    
    NEBULA Dataset Structure:
    dataset_root/
    ├── task_name_1/
    │   └── motionplanning/
    │       ├── s0/
    │       │   ├── data.h5     (contains traj_0, traj_1, ...)
    │       │   └── meta.json   (episodes[0], episodes[1], ...)
    │       └── 1/
    │           ├── data.h5
    │           └── meta.json
    └── task_name_2/
        └── ...
    
    For more details, please refer to README.md in the NEBULA dataset.
    """
    
    def __init__(self, 
                 dataset_root: Union[str, Path],
                 robot_config_path: Union[str, Path],
                 robot_name: str,
                 task_filter: Optional[List[str]] = None,
                 load_statistics: bool = True,
                 transform: Optional[Callable[[Episode], Episode]] = None):
        """
        Initialize NEBULA dataset.
        
        Args:
            dataset_root: Root directory of NEBULA dataset
            robot_config_path: Path to robot configuration file
            robot_name: Name of robot configuration to use
            task_filter: Optional list of task names to include (None = all tasks)
            load_statistics: Whether to compute episode statistics
            transform: Optional transform function to apply to episodes.
                      Should take an Episode object and return a modified Episode object.
        """
        self.dataset_root = Path(dataset_root)
        self.robot_config = Embodiment(robot_config_path, robot_name)
        self.task_filter = task_filter
        self.load_statistics = load_statistics
        self.transform = transform
        
        # Discover all episodes
        self.episodes_info = self._discover_episodes()
        print(f"Discovered {len(self.episodes_info)} episodes across {len(set(ep['task_name'] for ep in self.episodes_info))} tasks")
    
    def set_transform(self, transform: Optional[Callable[[Episode], Episode]]):
        """
        Set or update the transform function.
        
        Args:
            transform: Transform function to apply to episodes, or None to disable transforms
        """
        self.transform = transform
    
    def _discover_episodes(self) -> List[Dict]:
        """Discover all episodes in the dataset."""
        episodes_info = []

        for task_dir in self.dataset_root.iterdir():
            if not task_dir.is_dir():
                print(f"Warning: Skipping {task_dir} - not a directory")
                continue
                
            task_name = task_dir.name
            if self.task_filter and task_name not in self.task_filter:
                print(f"Warning: Skipping {task_name} - not in task filter")
                continue
            
            motionplanning_dir = task_dir / "motionplanning"
            if not motionplanning_dir.exists() or not motionplanning_dir.is_dir():
                print(f"Warning: Skipping {motionplanning_dir} - not a directory")
                continue
            
            for subtask_dir in motionplanning_dir.iterdir():
                if not subtask_dir.is_dir():
                    print(f"Warning: Skipping {subtask_dir} - not a directory")
                    continue
                
                # Find h5 and json files (guaranteed to have exactly 1 of each)
                h5_files = list(subtask_dir.glob("*.h5"))
                json_files = list(subtask_dir.glob("*.json"))
                
                if len(h5_files) == 1 and len(json_files) == 1:
                    h5_path = h5_files[0]
                    json_path = json_files[0]
                    
                    # Load JSON to get all episodes in this subtask
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add each episode
                    for episode_meta in metadata.get('episodes', []):
                        episode_id = episode_meta['episode_id']
                        episodes_info.append({
                            'task_name': task_name,
                            'subtask_idx': subtask_dir.name,
                            'episode_id': episode_id,
                            'h5_path': h5_path,
                            'json_path': json_path,
                            'episode_meta': episode_meta,
                            'env_info': metadata.get('env_info', {})
                        })
                else:
                    print(f"Warning: Skipping {subtask_dir} - found {len(h5_files)} h5 files and {len(json_files)} json files")
                    print(f"Warning: There should be exactly 1 h5 file and 1 json file in each subtask directory")
        
        return episodes_info
    
    def __len__(self) -> int:
        """Return total number of episodes."""
        return len(self.episodes_info)
    
    def __getitem__(self, idx: int) -> Episode:
        """Load and convert a single episode to EVA format with transforms applied."""
        return self.load_episode(idx)
    
    def load_episode(self, idx: int, apply_transform: bool = True) -> Episode:
        """
        Load a specific episode and convert to EVA format.
        
        Args:
            idx: Episode index
            apply_transform: Whether to apply the transform (if set)
            
        Returns:
            Episode object, potentially transformed
        """
        if idx >= len(self.episodes_info):
            raise IndexError(f"Episode index {idx} out of range (max: {len(self.episodes_info)-1})")
        
        episode_info = self.episodes_info[idx]
        
        # Load HDF5 trajectory data
        with h5py.File(episode_info['h5_path'], 'r') as f:
            # Get the specific trajectory for this episode
            traj_key = f"traj_{episode_info['episode_id']}"
            
            if traj_key not in f:
                raise ValueError(f"Trajectory {traj_key} not found in {episode_info['h5_path']}")
            
            eva_episode = self._convert_trajectory_to_eva(
                f[traj_key], 
                episode_info['episode_meta'], 
                episode_info['env_info'],
                episode_info
            )
        
        # Apply transform if provided and requested
        if apply_transform and self.transform is not None:
            try:
                eva_episode = self.transform(eva_episode)
            except Exception as e:
                print(f"Warning: Transform failed for episode {idx}: {e}")
                # Return original episode if transform fails
        
        return eva_episode
    
    def load_episode_raw(self, idx: int) -> Episode:
        """
        Load an episode without applying any transforms.
        
        Args:
            idx: Episode index
            
        Returns:
            Raw Episode object without transforms
        """
        return self.load_episode(idx, apply_transform=False)
    
    def _convert_trajectory_to_eva(self, 
                                   traj_group: h5py.Group, 
                                   episode_meta: Dict,
                                   env_info: Dict,
                                   episode_info: Dict) -> Episode:
        """Convert a single HDF5 trajectory to EVA Episode format."""
        
        # Create language instruction from the episode-specific task_instruction
        instruction = LanguageInstruction(
            raw_text=episode_meta.get('task_instruction', ''),
            tokens=None,
            token_ids=None,
            history=None
        )
        
        # Load trajectory data
        actions = np.array(traj_group['actions'])  # [T, action_dim]
        obs_group = traj_group['obs']
        
        # Optional fields
        rewards = np.array(traj_group['rewards']) if 'rewards' in traj_group else None
        success = np.array(traj_group['success']) if 'success' in traj_group else None
        terminated = np.array(traj_group['terminated']) if 'terminated' in traj_group else None
        
        T = actions.shape[0]  # Number of timesteps
        
        # Create steps
        steps = []
        for t in range(T):
            # Create observation (using t+1 for most obs, since obs is [T+1])
            obs_idx = min(t + 1, obs_group['agent']['qpos'].shape[0] - 1)
            observation = self._create_observation(obs_group, obs_idx)
            
            # Create action
            action = self._create_action(actions[t])
            
            # Create step
            step = Step(
                observation=observation,
                action=action,
                reward=rewards[t] if rewards is not None else None,
                metadata={
                    'frame_index': t,
                    'timestamp': float(t),
                    'success': bool(success[t]) if success is not None else None,
                    'terminated': bool(terminated[t]) if terminated is not None else None
                }
            )
            steps.append(step)
        
        # Create metadata
        meta = MetaInfo(
            success=bool(episode_meta.get('success', False)),
            scene_id=f"{episode_info['task_name']}_{episode_info['subtask_idx']}_ep{episode_info['episode_id']}",
            source="NEBULA",
            environment=env_info.get('env_id', 'unknown'),
            robot_init_pose=None,  # Could extract from first observation if needed
            additional={
                'control_mode': episode_meta.get('control_mode'),
                'elapsed_steps': episode_meta.get('elapsed_steps'),
                'episode_seed': episode_meta.get('episode_seed'),
                'reset_kwargs': episode_meta.get('reset_kwargs'),
                'env_kwargs': env_info.get('env_kwargs'),
                'max_episode_steps': env_info.get('max_episode_steps'),
                'videos': episode_meta.get('videos', {}),
                'dataset_idx': episode_info.get('dataset_idx')  # Add dataset index for tracking
            }
        )
        
        # Create episode statistics if requested
        statistics = None
        if self.load_statistics:
            statistics = self._compute_episode_statistics(obs_group, actions)
        
        # Create episode
        episode = Episode(
            task_id=f"{episode_info['task_name']}_{episode_info['subtask_idx']}_ep{episode_info['episode_id']}",
            robot_name=self.robot_config.robot_name,
            instruction=instruction,
            steps=steps,
            meta=meta,
            statistics=statistics
        )
        
        return episode
    
    def _create_observation(self, obs_group: h5py.Group, obs_idx: int) -> Observation:
        """Create EVA Observation from NEBULA obs data at specific timestep."""
        
        # Create robot state
        robot_state = self._create_robot_state(obs_group, obs_idx)
        
        # Extract camera data
        images = {}
        depth_maps = {}
        segmentation_maps = {}
        camera_poses = {}
        
        if 'sensor_data' in obs_group:
            sensor_data = obs_group['sensor_data']
            for camera_name in sensor_data.keys():
                camera_group = sensor_data[camera_name]
                
                # RGB images
                if 'rgb' in camera_group:
                    images[camera_name] = np.array(camera_group['rgb'][obs_idx])
                
                # Depth maps
                if 'depth' in camera_group:
                    depth_maps[camera_name] = np.array(camera_group['depth'][obs_idx])
                
                # Segmentation
                if 'segmentation' in camera_group:
                    segmentation_maps[camera_name] = np.array(camera_group['segmentation'][obs_idx])
        
        # Extract camera poses if available
        if 'sensor_param' in obs_group:
            sensor_param = obs_group['sensor_param']
            for camera_name in sensor_param.keys():
                if 'extrinsic_cv' in sensor_param[camera_name]:
                    camera_poses[camera_name] = np.array(sensor_param[camera_name]['extrinsic_cv'][obs_idx])
        
        return Observation(
            image=images if images else None,
            depth=depth_maps if depth_maps else None,
            segmentation=segmentation_maps if segmentation_maps else None,
            pointcloud=None,  # Could add if available in NEBULA
            robot_state=robot_state,
            camera_pose=camera_poses if camera_poses else None,
            timestamp=float(obs_idx)
        )
    
    def _create_robot_state(self, obs_group: h5py.Group, obs_idx: int) -> RobotState:
        """Create EVA RobotState from NEBULA obs data."""
        limbs = {}
        
        # Get joint positions and velocities
        qpos = np.array(obs_group['agent']['qpos'][obs_idx])
        qvel = np.array(obs_group['agent']['qvel'][obs_idx]) if 'qvel' in obs_group['agent'] else None
        
        # Map to limbs using robot config
        for limb_name in self.robot_config.get_all_limb_names():
            # Get slice information from robot config
            state_slice = self.robot_config.get_state_slice(limb_name)
            if state_slice:
                start, end = state_slice
                limb_qpos = qpos[start:end]
                limb_qvel = qvel[start:end] if qvel is not None else None
            else:
                # Fallback: use DOF information
                limb_info = self.robot_config.get_limb_info(limb_name)
                if limb_info:
                    dof = limb_info.get('dof', 0)
                    # Simple assumption: limbs are in order
                    offset = sum(self.robot_config.get_action_dim(l) 
                               for l in self.robot_config.get_all_limb_names() 
                               if self.robot_config.get_all_limb_names().index(l) < 
                                  self.robot_config.get_all_limb_names().index(limb_name))
                    limb_qpos = qpos[offset:offset+dof]
                    limb_qvel = qvel[offset:offset+dof] if qvel is not None else None
                else:
                    continue
            
            limbs[limb_name] = RobotLimbState(
                joint_positions=limb_qpos,
                joint_velocities=limb_qvel,
                joint_torques=None  # Not available in NEBULA
            )
        
        return RobotState(
            robot_name=self.robot_config.robot_name,
            limbs=limbs
        )
    
    def _create_action(self, action_array: np.ndarray) -> Action:
        """Create EVA Action from NEBULA action array."""
        limb_actions = {}
        
        # Map action array to limbs using robot config
        for limb_name in self.robot_config.get_all_action_limbs():
            action_slice = self.robot_config.get_action_slice(limb_name)
            if action_slice:
                start, end = action_slice
                limb_actions[limb_name] = action_array[start:end]
            else:
                # Fallback: use DOF information
                limb_info = self.robot_config.get_limb_info(limb_name)
                if limb_info:
                    dof = limb_info.get('dof', 0)
                    offset = sum(self.robot_config.get_action_dim(l) 
                               for l in self.robot_config.get_all_limb_names() 
                               if self.robot_config.get_all_limb_names().index(l) < 
                                  self.robot_config.get_all_limb_names().index(limb_name))
                    limb_actions[limb_name] = action_array[offset:offset+dof]
        
        robot_action = RobotAction(
            control_type="joint",  # Could be extracted from metadata
            limbs=limb_actions
        )
        
        return Action(
            robot_action=robot_action,
            symbolic=None
        )
    
    def _compute_episode_statistics(self, obs_group: h5py.Group, actions: np.ndarray) -> EpisodeStatistics:
        """Compute episode-level statistics for normalization."""
        # Get all state data
        qpos = np.array(obs_group['agent']['qpos'])  # [T+1, state_dim]
        qvel = np.array(obs_group['agent']['qvel']) if 'qvel' in obs_group['agent'] else None
        
        # Combine state data
        if qvel is not None:
            state_data = np.concatenate([qpos, qvel], axis=-1)
        else:
            state_data = qpos
        
        return EpisodeStatistics(
            state_min=np.min(state_data, axis=0),
            state_max=np.max(state_data, axis=0),
            state_mean=np.mean(state_data, axis=0),
            state_std=np.std(state_data, axis=0),
            action_min=np.min(actions, axis=0),
            action_max=np.max(actions, axis=0)
        )
    
    def get_task_names(self) -> List[str]:
        """Get list of all unique task names in the dataset."""
        return list(set(ep['task_name'] for ep in self.episodes_info))
    
    def get_episodes_by_task(self, task_name: str) -> List[int]:
        """Get episode indices for a specific task."""
        return [i for i, ep in enumerate(self.episodes_info) if ep['task_name'] == task_name]
    
    def get_episodes_by_subtask(self, task_name: str, subtask_idx: str) -> List[int]:
        """Get episode indices for a specific subtask."""
        return [i for i, ep in enumerate(self.episodes_info) 
                if ep['task_name'] == task_name and ep['subtask_idx'] == subtask_idx]
    
    def get_dataset_statistics(self) -> Dict:
        """Get overall dataset statistics."""
        task_counts = {}
        subtask_counts = {}
        
        for ep in self.episodes_info:
            task_name = ep['task_name']
            subtask_key = f"{task_name}/{ep['subtask_idx']}"
            
            task_counts[task_name] = task_counts.get(task_name, 0) + 1
            subtask_counts[subtask_key] = subtask_counts.get(subtask_key, 0) + 1
        
        return {
            'total_episodes': len(self.episodes_info),
            'total_tasks': len(set(ep['task_name'] for ep in self.episodes_info)),
            'total_subtasks': len(set(f"{ep['task_name']}/{ep['subtask_idx']}" for ep in self.episodes_info)),
            'task_distribution': task_counts,
            'subtask_distribution': subtask_counts,
            'robot_name': self.robot_config.robot_name,
            'robot_type': self.robot_config.robot_type
        }