"""
nebula_dataset.py

Custom dataset loader for Nebula HDF5 format, designed to output the same format as RLDS pipeline.
"""

import os
import h5py
import json
import glob
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import tensorflow as tf
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

def load_nebula_trajectory(h5_path: str, json_path: str, traj_idx: int, task_instruction: str) -> Dict:
    """
    Load a single trajectory from Nebula HDF5 format and convert to RLDS-compatible format.
    
    Args:
        h5_path: Path to HDF5 file
        json_path: Path to JSON metadata file  
        traj_idx: Trajectory index within the file
        task_instruction: Task instruction string
        
    Returns:
        Dictionary in RLDS-compatible format
    """
    with h5py.File(h5_path, 'r') as f:
        traj_key = f"traj_{traj_idx}"
        if traj_key not in f:
            raise ValueError(f"Trajectory {traj_key} not found in {h5_path}")
            
        traj_data = f[traj_key]
        
        # Extract trajectory length
        #actions = np.array(traj_data['actions'])
        #traj_len = len(actions)
        
        # Extract observations
        obs_data = traj_data['obs']

        # Extract TCP pose for computing EEF actions
        tcp_pose = np.array(obs_data['extra']['tcp_pose'])  # [T, 7] (pos + quat)

        # Extract gripper action from original actions
        original_actions = np.array(traj_data['actions'])  # [T, 8]
        gripper_action = original_actions[:, 7:8]  # [T, 1] - gripper from action 7

        # Compute EEF actions from TCP pose differences
        eef_pos = tcp_pose[:, :3]  # [T, 3] positions
        eef_quat = tcp_pose[:, 3:7]  # [T, 4] quaternions

        # Compute position deltas
        pos_deltas = np.diff(eef_pos, axis=0, prepend=eef_pos[:1])  # [T, 3]

        # Compute rotation deltas (simplified - convert quat to euler then diff)
        from scipy.spatial.transform import Rotation as R
        eef_euler = R.from_quat(eef_quat).as_euler('xyz')  # [T, 3]
        rot_deltas = np.diff(eef_euler, axis=0, prepend=eef_euler[:1])  # [T, 3]

        # Combine into 7D EEF actions
        # Truncate deltas to match gripper action length
        action_len = len(gripper_action)
        pos_deltas = pos_deltas[:action_len]  # [T-1, 3]
        rot_deltas = rot_deltas[:action_len]  # [T-1, 3]

        # Combine into 7D EEF actions
        actions = np.concatenate([
            pos_deltas,      # [T-1, 3] translation
            rot_deltas,      # [T-1, 3] rotation  
            gripper_action   # [T-1, 1] gripper
        ], axis=1).astype(np.float32)  # [T-1, 7]
        traj_len = len(actions)
        
        # Extract agent state (qpos, qvel)
        agent_data = obs_data['agent']
        qpos = np.array(agent_data['qpos'])  # [T, joint_dim]
        qvel = np.array(agent_data['qvel'])  # [T, joint_dim]
        
        # Extract TCP pose
        tcp_pose = np.array(obs_data['extra']['tcp_pose'])  # [T, 7] (pos + quat)
        
        # Extract camera data - using base_camera as primary
        sensor_data = obs_data['sensor_data']
        
        # Get camera intrinsics for base_camera
        sensor_params = obs_data['sensor_param']
        base_camera_params = sensor_params['base_camera']
        intrinsic = np.array(base_camera_params['intrinsic_cv'])  # [3, 3]
        
        # Extract images - base_camera as primary, hand_camera as wrist
        base_camera_data = sensor_data['base_camera']
        hand_camera_data = sensor_data['hand_camera']
        
        # RGB images [T, H, W, 3]
        image_primary = np.array(base_camera_data['rgb'])
        image_wrist = np.array(hand_camera_data['rgb'])
        
        # Depth images [T, H, W] -> [T, H, W, 1]  
        depth_primary = np.array(base_camera_data['depth'])
        if len(depth_primary.shape) == 3:
            depth_primary = depth_primary[..., None]
            
        # Extract other trajectory info
        terminated = np.array(traj_data['terminated'])
        truncated = np.array(traj_data['truncated'])
        success = np.array(traj_data['success'])
        
        # Create RLDS-compatible format
        trajectory = {
            'observation': {
                # Images - convert to bytes for consistency with RLDS
                'image_primary': _encode_images(image_primary),
                'image_wrist': _encode_images(image_wrist), 
                'depth_primary': _encode_depth(depth_primary),
                
                # Robot state - concatenate qpos, qvel, tcp_pose
                'state': np.concatenate([
                    tcp_pose[:, :3],  # TCP position [3]
                    tcp_pose[:, 3:],  # TCP orientation (quat) [4] 
                    qpos[:, :1] if qpos.shape[1] > 0 else np.zeros((traj_len, 1)),  # Gripper state [1]
                ], axis=1).astype(np.float32),
                
                # Additional state info
                'cartesian_position': tcp_pose.astype(np.float32),
                'joint_positions': qpos.astype(np.float32),
                'joint_velocities': qvel.astype(np.float32),
            },
            
            'action': actions.astype(np.float32),
            
            # Task information
            'language_instruction': np.array([task_instruction.encode('utf-8')] * traj_len),
            
            # Episode metadata  
            'episode_metadata': {
                'file_path': h5_path,
                'traj_index': traj_idx,
            },
            
            # Trajectory metadata
            '_traj_index': traj_idx,
            'terminated': terminated.astype(bool),
            'truncated': truncated.astype(bool), 
            'success': success.astype(bool),
        }
        
        # Add intrinsic parameters
        trajectory['intrinsic'] = intrinsic.astype(np.float32)
        
    return trajectory

def _encode_images(images: np.ndarray) -> np.ndarray:
    """Encode images as bytes strings for consistency with RLDS format."""
    import cv2
    encoded_images = []
    for img in images:
        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # Encode as JPEG bytes
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        encoded_images.append(buffer.tobytes())
    
    return np.array(encoded_images, dtype=object)

def _encode_depth(depth: np.ndarray) -> np.ndarray:
    """Encode depth as bytes strings for consistency with RLDS format."""
    import cv2
    encoded_depth = []
    for d in depth:
        # Normalize depth to 0-255 range
        d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)
        
        # Encode as PNG bytes (better for depth)
        _, buffer = cv2.imencode('.png', d_norm)
        encoded_depth.append(buffer.tobytes())
    
    return np.array(encoded_depth, dtype=object)

def make_nebula_dataset(
    data_root_dir: str,
    task_names: List[str],
    task_instruction_map: Dict[str, str],
    train: bool = True,
    shuffle_seed: int = 42,
    max_trajectories_per_task: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Create a dataset from Nebula HDF5 files.
    
    Args:
        data_root_dir: Root directory containing task folders
        task_names: List of task names to include
        task_instruction_map: Mapping from task name to instruction
        train: Whether to load train split (affects file selection)
        shuffle_seed: Random seed for shuffling
        max_trajectories_per_task: Limit trajectories per task (for debugging)
        
    Returns:
        Tuple of (trajectories_list, dataset_statistics)
    """
    np.random.seed(shuffle_seed)
    
    all_trajectories = []
    actions_list = []
    
    for task_name in task_names:
        
        """if task_name not in task_instruction_map:
            print(f"Warning: No instruction found for task {task_name}, skipping...")
            continue"""
        
        if task_name in task_instruction_map.keys():
            task_instruction = task_instruction_map[task_name]
        
        task_dir = os.path.join(data_root_dir, task_name, "motionplanning")
        
        if not os.path.exists(task_dir):
            print(f"Warning: Task directory {task_dir} not found, skipping...")
            continue
            
        # Find all subtask directories
        subtask_dirs = glob.glob(os.path.join(task_dir, "[0-9]*"))
        subtask_dirs.sort(key=lambda x: int(os.path.basename(x)))
        
        task_trajectories = 0
        for subtask_dir in subtask_dirs:
            # Find all h5 files in subtask
            h5_files = glob.glob(os.path.join(subtask_dir, "*.h5"))
            h5_files.sort()
            
            for h5_file in h5_files:
                # Find corresponding metadata.json
                json_file = os.path.join(subtask_dir, "metadata.json")
                if not os.path.exists(json_file):
                    print(f"Warning: metadata.json not found for {h5_file}, skipping...")
                    continue
                
                # Load trajectories from this file (100 per file)
                try:
                    with h5py.File(h5_file, 'r') as f:
                        available_trajs = [k for k in f.keys() if k.startswith('traj_')]

                    for traj_key in available_trajs:
                        if max_trajectories_per_task and task_trajectories >= max_trajectories_per_task:
                            break
                            
                        traj_idx = int(traj_key.split('_')[1])
                        
                        try:
                            task_instruction = json_file["episodes"][traj_idx]["task_instruction"]
                        except:
                            task_instruction = task_instruction_map.get(task_name, "No instruction available.")

                        try:
                            trajectory = load_nebula_trajectory(
                                h5_file, json_file, traj_idx, task_instruction
                            )
                            all_trajectories.append(trajectory)
                            actions_list.append(trajectory['action'])
                            task_trajectories += 1
                            
                        except Exception as e:
                            print(f"Error loading {h5_file}:traj_{traj_idx}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error processing {h5_file}: {e}")
                    continue
                    
                if max_trajectories_per_task and task_trajectories >= max_trajectories_per_task:
                    break
                    
            if max_trajectories_per_task and task_trajectories >= max_trajectories_per_task:
                break
                
        print(f"Loaded {task_trajectories} trajectories for task {task_name}")
    
    # Compute dataset statistics
    if actions_list:
        all_actions = np.concatenate(actions_list, axis=0)
        
        # Collect proprio data from trajectories
        proprio_list = []
        for traj in all_trajectories:
            if 'state' in traj['observation']:
                proprio_list.append(traj['observation']['state'])
        
        if proprio_list:
            all_proprio = np.concatenate(proprio_list, axis=0)
            proprio_stats = {
                "mean": all_proprio.mean(0),
                "std": all_proprio.std(0),
                "max": all_proprio.max(0),
                "min": all_proprio.min(0),
                "q01": np.quantile(all_proprio, 0.01, axis=0),
                "q99": np.quantile(all_proprio, 0.99, axis=0),
            }
        else:
            # Fallback if no proprio data
            proprio_dim = 8  # Default size
            proprio_stats = {
                "mean": np.zeros(proprio_dim),
                "std": np.ones(proprio_dim),
                "max": np.ones(proprio_dim),
                "min": np.ones(proprio_dim) * -1,
                "q01": np.ones(proprio_dim) * -1,
                "q99": np.ones(proprio_dim),
            }
        
        dataset_statistics = {
            "action": {
                "mean": all_actions.mean(0),
                "std": all_actions.std(0),
                "max": all_actions.max(0),
                "min": all_actions.min(0),
                "q01": np.quantile(all_actions, 0.01, axis=0),
                "q99": np.quantile(all_actions, 0.99, axis=0),
            },
            "proprio": proprio_stats,
            "num_transitions": np.array(sum(len(traj['action']) for traj in all_trajectories)),
            "num_trajectories": np.array(len(all_trajectories)),
        }
    else:
        # Empty dataset - use numpy arrays with 0 elements
        dataset_statistics = {
            "action": {
                "mean": np.array([]), "std": np.array([]), 
                "max": np.array([]), "min": np.array([]), 
                "q01": np.array([]), "q99": np.array([])
            },
            "proprio": {
                "mean": np.array([]), "std": np.array([]), 
                "max": np.array([]), "min": np.array([]), 
                "q01": np.array([]), "q99": np.array([])
            },
            "num_transitions": [0],
            "num_trajectories": [0],
        }
    
    print(f"Total loaded: {len(all_trajectories)} trajectories")
    return all_trajectories, dataset_statistics

def nebula_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform nebula trajectory to match SpatialVLA expected format.
    Similar to bridge_orig_dataset_transform but for nebula data.
    """
    # Actions are already in the right format [T, 7] 
    # Assuming: [dx, dy, dz, drx, dry, drz, gripper]
    
    # Extract EEF state and gripper state from observation
    state = trajectory["observation"]["state"]  # [T, 8] 
    trajectory["observation"]["EEF_state"] = state[:, :6]  # [T, 6] - position + orientation
    trajectory["observation"]["gripper_state"] = state[:, -1:]  # [T, 1] - gripper
    
    return trajectory