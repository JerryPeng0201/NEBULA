import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import torch

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.schema import (
    DatasetMetadata, 
    DatasetStatistics, 
    DatasetModalities,
    VideoMetadata,
    StateActionMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
    LeRobotModalityField,
    LeRobotActionMetadata,
    LeRobotStateMetadata,
    RotationType
)
from gr00t.data.embodiment_tags import EmbodimentTag


class HDF5LeRobotDataset(LeRobotSingleDataset):
    """
    Custom dataset adapter for HDF5 format robotics data.
    Loads data directly from HDF5 files without conversion.
    """
    
    def __init__(
        self,
        dataset_path: Path | str,
        task_name: str,
        subtask_name: str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        camera_names: Optional[List[str]] = None,
        state_keys: Optional[List[str]] = None,
        action_dim: int = 7,  # Default for 7-DOF robot
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms = None,
    ):
        """
        Initialize the HDF5 dataset adapter.
        
        Args:
            dataset_path: Root path to dataset
            task_name: Name of the task folder
            subtask_name: Name of the subtask folder  
            modality_configs: Modality configurations for GR00T
            embodiment_tag: Robot embodiment tag
            camera_names: List of camera names to use
            state_keys: List of state keys to extract
            action_dim: Dimension of action space
        """
        self.task_name = task_name
        self.subtask_name = subtask_name
        self.action_dim = action_dim
        
        # Default camera and state configurations
        self.camera_names = camera_names or [
            "base_camera", "hand_camera", "back_right_camera", 
            "back_left_camera", "front_right_camera", "front_left_camera"
        ]
        self.state_keys = state_keys or ["qpos", "qvel", "tcp_pose"]
        
        # Build full dataset path
        self.full_dataset_path = Path(dataset_path) / task_name / "motionplanning" / subtask_name
        
        # Cache for loaded data
        self._h5_cache = {}
        self._json_cache = {}
        
        # Initialize parent class
        super().__init__(
            dataset_path=self.full_dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
        )

    def _get_h5_files(self) -> List[Path]:
        """Get all HDF5 files in the dataset directory."""
        return sorted(list(self.full_dataset_path.glob("*.h5")))
    
    def _get_json_files(self) -> List[Path]:
        """Get all JSON files in the dataset directory."""
        return sorted(list(self.full_dataset_path.glob("*.json")))
    
    def _load_h5_file(self, file_path: Path) -> h5py.File:
        """Load and cache HDF5 file."""
        if file_path not in self._h5_cache:
            self._h5_cache[file_path] = h5py.File(file_path, 'r')
        return self._h5_cache[file_path]
    
    def _load_json_file(self, file_path: Path) -> dict:
        """Load and cache JSON file."""
        if file_path not in self._json_cache:
            with open(file_path, 'r') as f:
                self._json_cache[file_path] = json.load(f)
        return self._json_cache[file_path]

    def _get_modality_mapping(self) -> dict:
        """Load modality mapping from JSON file or return default."""
        modality_json_path = self.full_dataset_path / "modality.json"
        if modality_json_path.exists():
            with open(modality_json_path, 'r') as f:
                return json.load(f)
        else:
            return self._create_default_panda_modality_mapping()

    def _compute_dataset_statistics(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """Compute dataset statistics for all state and action modalities based on modality mapping."""
        print("Computing dataset statistics...")
        
        # Load modality mapping
        modality_mapping = self._get_modality_mapping()
        
        state_data = defaultdict(list)
        action_data = defaultdict(list)
        
        h5_files = self._get_h5_files()
        for h5_file in h5_files:
            h5_data = self._load_h5_file(h5_file)
            
            # Process each trajectory in the file
            for traj_key in h5_data.keys():
                if not traj_key.startswith('traj_'):
                    continue
                    
                traj = h5_data[traj_key]
                
                # Extract state data according to modality mapping
                if 'obs/agent/qpos' in traj:
                    qpos_data = np.array(traj['obs/agent/qpos'])  # Shape: [T, 8]
                    
                    for state_key, config in modality_mapping["state"].items():
                        start_idx = config["start"]
                        end_idx = config["end"]
                        state_slice = qpos_data[:, start_idx:end_idx]
                        state_data[state_key].append(state_slice)
                
                # Extract action data according to modality mapping
                if 'actions' in traj:
                    actions_data = np.array(traj['actions'])  # Shape: [T, 8]
                    
                    for action_key, config in modality_mapping["action"].items():
                        start_idx = config["start"]
                        end_idx = config["end"]
                        action_slice = actions_data[:, start_idx:end_idx]
                        action_data[action_key].append(action_slice)
        
        # Compute statistics
        statistics = {"state": {}, "action": {}}
        
        # State statistics
        for key, data_list in state_data.items():
            if data_list:
                all_data = np.concatenate(data_list, axis=0)
                statistics["state"][key] = self._compute_stats(all_data)
        
        # Action statistics  
        for key, data_list in action_data.items():
            if data_list:
                all_data = np.concatenate(data_list, axis=0)
                statistics["action"][key] = self._compute_stats(all_data)
                
        return statistics
    
    def _compute_stats(self, data: np.ndarray) -> Dict[str, List[float]]:
        """Compute statistical values for data array."""
        return {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(), 
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Create metadata compatible with GR00T format."""
        print("Creating dataset metadata...")
        
        # Load modality mapping to understand structure
        modality_mapping = self._get_modality_mapping()
        
        # Get video metadata from first trajectory
        h5_files = self._get_h5_files()
        first_h5 = self._load_h5_file(h5_files[0])
        first_traj = first_h5[list(first_h5.keys())[0]]
        
        # Video metadata - use camera names from modality mapping
        video_metadata = {}
        for video_key, config in modality_mapping["video"].items():
            # Extract camera name from original_key or use video_key
            original_key = config.get("original_key", f"obs/sensor_data/{video_key}/rgb")
            if original_key.startswith("obs/sensor_data/"):
                # Extract camera name from path like "obs/sensor_data/base_camera/rgb"
                cam_name = original_key.split("/")[2]
                if f'obs/sensor_data/{cam_name}/rgb' in first_traj:
                    rgb_data = first_traj[f'obs/sensor_data/{cam_name}/rgb']
                    height, width = rgb_data.shape[1:3]
                    video_metadata[video_key] = VideoMetadata(
                        resolution=(width, height),
                        channels=3,
                        fps=30.0  # Default FPS
                    )
        
        # State metadata - use modality mapping to determine dimensions
        state_metadata = {}
        for state_key, config in modality_mapping["state"].items():
            start_idx = config["start"]
            end_idx = config["end"]
            shape = (end_idx - start_idx,)
            
            # Check for rotation type
            rotation_type = None
            if "rotation_type" in config:
                rotation_type = RotationType(config["rotation_type"])
            
            state_metadata[state_key] = StateActionMetadata(
                absolute=config.get("absolute", True),
                rotation_type=rotation_type,
                shape=shape,
                continuous=True
            )
        
        # Action metadata - use modality mapping to determine dimensions
        action_metadata = {}
        for action_key, config in modality_mapping["action"].items():
            start_idx = config["start"]
            end_idx = config["end"]
            shape = (end_idx - start_idx,)
            
            # Check for rotation type
            rotation_type = None
            if "rotation_type" in config:
                rotation_type = RotationType(config["rotation_type"])
            
            action_metadata[action_key] = StateActionMetadata(
                absolute=config.get("absolute", False),
                rotation_type=rotation_type,
                shape=shape,
                continuous=True
            )
        
        # Compute dataset statistics
        statistics_dict = self._compute_dataset_statistics()
        statistics = DatasetStatistics(
            state={k: DatasetStatisticalValues(**v) for k, v in statistics_dict["state"].items()},
            action={k: DatasetStatisticalValues(**v) for k, v in statistics_dict["action"].items()}
        )
        
        # Create full metadata
        modalities = DatasetModalities(
            video=video_metadata,
            state=state_metadata, 
            action=action_metadata
        )
        
        return DatasetMetadata(
            statistics=statistics,
            modalities=modalities,
            embodiment_tag=embodiment_tag
        )

    def _get_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get trajectory IDs and lengths."""
        trajectory_ids = []
        trajectory_lengths = []
        
        h5_files = self._get_h5_files()
        
        for file_idx, h5_file in enumerate(h5_files):
            h5_data = self._load_h5_file(h5_file)
            
            for traj_key in sorted(h5_data.keys()):
                if not traj_key.startswith('traj_'):
                    continue
                    
                # Create global trajectory ID: file_index * 100 + traj_index
                traj_idx = int(traj_key.split('_')[1])
                global_traj_id = file_idx * 100 + traj_idx
                
                # Get trajectory length from actions
                traj = h5_data[traj_key]
                if 'actions' in traj:
                    length = len(traj['actions'])
                else:
                    length = 1  # Default
                    
                trajectory_ids.append(global_traj_id)
                trajectory_lengths.append(length)
        
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Create LeRobot modality metadata."""
        # Load modality mapping from JSON file if it exists
        modality_json_path = self.full_dataset_path / "modality.json"
        if modality_json_path.exists():
            with open(modality_json_path, 'r') as f:
                modality_mapping = json.load(f)
        else:
            # Create default mapping for Panda robot
            modality_mapping = self._create_default_panda_modality_mapping()
        
        # State metadata from mapping
        state_meta = {}
        for key, config in modality_mapping["state"].items():
            if "rotation_type" in config:
                rotation_type = RotationType(config["rotation_type"])
            else:
                rotation_type = None
                
            state_meta[key] = LeRobotStateMetadata(
                start=config["start"],
                end=config["end"],
                rotation_type=rotation_type,
                absolute=config.get("absolute", True),
                dtype=config.get("dtype", "float32"),
                original_key=config.get("original_key", f"state.{key}")
            )
        
        # Action metadata from mapping
        action_meta = {}
        for key, config in modality_mapping["action"].items():
            if "rotation_type" in config:
                rotation_type = RotationType(config["rotation_type"])
            else:
                rotation_type = None
                
            action_meta[key] = LeRobotActionMetadata(
                start=config["start"],
                end=config["end"],
                rotation_type=rotation_type,
                absolute=config.get("absolute", False),
                dtype=config.get("dtype", "float32"),
                original_key=config.get("original_key", f"action.{key}")
            )
        
        # Video metadata from mapping
        video_meta = {}
        for key, config in modality_mapping["video"].items():
            video_meta[key] = LeRobotModalityField(
                original_key=config.get("original_key", f"video.{key}")
            )
        
        # Annotation metadata if present
        annotation_meta = None
        if "annotation" in modality_mapping:
            annotation_meta = {}
            for key, config in modality_mapping["annotation"].items():
                annotation_meta[key] = LeRobotModalityField(
                    original_key=config.get("original_key", key)
                )
        
        return LeRobotModalityMetadata(
            state=state_meta,
            action=action_meta,
            video=video_meta,
            annotation=annotation_meta
        )
    
    def _create_default_panda_modality_mapping(self) -> dict:
        """Create default modality mapping for Panda robot."""
        return {
            "state": {
                "end_effector_position_relative": {
                    "start": 0,
                    "end": 3,
                    "absolute": True
                },
                "end_effector_rotation_relative": {
                    "start": 3,
                    "end": 7,
                    "rotation_type": "quaternion",
                    "absolute": True
                },
                "gripper_qpos": {
                    "start": 7,
                    "end": 8,
                    "absolute": True
                },
                "base_position": {
                    "start": 8,
                    "end": 11,
                    "absolute": True
                },
                "base_rotation": {
                    "start": 11,
                    "end": 15,
                    "rotation_type": "quaternion",
                    "absolute": True
                }
            },
            "action": {
                "end_effector_position": {
                    "start": 0,
                    "end": 3,
                    "absolute": False
                },
                "end_effector_rotation": {
                    "start": 3,
                    "end": 6,
                    "rotation_type": "axis_angle",
                    "absolute": False
                },
                "gripper_close": {
                    "start": 6,
                    "end": 7,
                    "absolute": False
                },
                "base_motion": {
                    "start": 7,
                    "end": 10,
                    "absolute": False
                },
                "control_mode": {
                    "start": 10,
                    "end": 11,
                    "absolute": False
                }
            },
            "video": {
                "left_view": {
                    "original_key": "observation.images.left_view"
                },
                "right_view": {
                    "original_key": "observation.images.right_view"
                },
                "wrist_view": {
                    "original_key": "observation.images.wrist_view"
                }
            },
            "annotation": {
                "human.action.task_description": {
                    "original_key": "task_index"
                }
            }
        }

    def _get_lerobot_info_meta(self) -> dict:
        """Create LeRobot info metadata."""
        return {
            "data_path": "data/{episode_chunk:04d}/{episode_index:06d}.parquet",
            "video_path": "videos/{episode_chunk:04d}/{episode_index:06d}_{video_key}.mp4",
            "chunks_size": 100,
            "features": {}  # Will be populated as needed
        }

    def _get_chunk_size(self) -> int:
        """Get chunk size (trajectories per file)."""
        return 100

    def _get_tasks(self) -> pd.DataFrame:
        """Create tasks dataframe from JSON metadata."""
        tasks = []
        task_index = 0
        
        json_files = self._get_json_files()
        for json_file in json_files:
            json_data = self._load_json_file(json_file)
            
            if "episodes" in json_data:
                for episode in json_data["episodes"]:
                    if "task_instruction" in episode:
                        tasks.append({
                            "task_index": task_index,
                            "task": episode["task_instruction"]
                        })
                        task_index += 1
        
        if not tasks:
            # Default task if no instructions found
            tasks = [{"task_index": 0, "task": "Robotic manipulation task"}]
            
        return pd.DataFrame(tasks).set_index("task_index")
    
    def get_language(self, trajectory_id: int, key: str, base_index: int) -> list[str]:
        """Get task instruction for the trajectory."""
        # Get step indices (though we'll return the same instruction for all steps)
        step_indices = self.delta_indices[key] + base_index
        
        # Create a simple task instruction based on task name
        task_instruction_map = {

            # ===== Control Tasks =====

            "Control-PlaceSphere-Easy": "Pick up the blue sphere and place it into the bin",
            "Control-PushCube-Easy": "Push the cube to the target goal position",
            "Control-StackCube-Easy": "Stack the cube on top of another object",

            "Control-PegInsertionSide-Medium": "Pick up the peg and insert the orange end into the box with a hole in it",
            "Control-PlaceSphere-Medium": "Pick up the blue sphere and place it into the purple bin, and then pick up the sphere again and place it into the blue bin",
            "Control-StackCube-Medium": "Pick up the red cube and place it by the green cube, and then pick up the blue cube and place it on top of the two cubes",

            "Control-PlaceSphere-Hard": "Place a sphere to the red bin, and move it to the blue bin, then move it to the green bin",
            "Control-StackCube-Hard": "Pick up the red cube and place it by the green cube, and then pick up the blue cube and place it on top of the two cubes, and then pick up the purple cube and place it by the green cube",
            "Control-PlugCharger-Hard": "Pick up the plug and insert it into the correct empty slot",

            # ===== Perception Tasks =====

            "Perception-PickBiggerSphere-Easy": "Place the bigger sphere into the bin",
            "Perception-PickRedSphere-Easy": "Place the red sphere into the bin",
            "Perception-PickSphere-Easy": "Place the sphere into the bin",

            "Perception-PickDiffCubes-Medium": "Place the cube that has different sizes into the bin",
            "Perception-PickRedT-Medium": "Place the red 'T' into the bin",
            "Perception-PickWhitePeg-Medium": "Place the peg that has white color into the bin",

            "Perception-PickPeg-Hard": "Place the peg that has red color at the middle into the bin",
            "Perception-PickRedT-Hard": "Place the red 'T' into the bin",
            "Perception-PickRightCubes-Hard": "Place the cube that can fit the bin into the bin",

            # ===== Language Tasks =====
            "LanguageEasy-GrabBlock": "Grab the red block",
            "LanguageEasy-PickCube": "Pick up the red cube",
            "LanguageEasy-PlaceCube": "Place the pink cube",

            # ===== Dynamic Adaptation Tasks =====
            "DynamicEasy-PressSwitch": "Only press the switch after the light turns red",
            "DynamicMedium-PickSlidingCube": "Pick up the cube that is sliding on the table",
            "DynamicHard-ColorSwitchPickCube": "Pick up the red cube",
            "DynamicHard-ShapeSwitchPickCube": "Pick up the cube",

            # ===== Spatial Reference Tasks =====
            "SpatialEasy-MoveCube": "Move the red cube to the right of the green cube",
            "SpatialEasy-PickCube": "Pick up the cube and place it at the middle",

            # Add more task mappings as needed
        }
        
        # Get task instruction or use default
        task_instruction = task_instruction_map.get(
            self.task_name, 
            f"Complete the {self.task_name} task"  # Default instruction
        )
        
        # Return the same instruction for all requested timesteps
        return [task_instruction] * len(step_indices)

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Load trajectory data from HDF5 file and split according to modality mapping."""
        # Determine which file and trajectory index
        file_idx = trajectory_id // 100
        traj_idx = trajectory_id % 100
        
        h5_files = self._get_h5_files()
        h5_file = h5_files[file_idx]
        h5_data = self._load_h5_file(h5_file)
        
        traj_key = f"traj_{traj_idx}"
        traj = h5_data[traj_key]
        
        # Extract data and create DataFrame
        data_dict = {}
        modality_mapping = self._get_modality_mapping()
        
        # Get the raw data first
        qpos_data = np.array(traj['obs/agent/qpos']) if 'obs/agent/qpos' in traj else None
        action_data = np.array(traj['actions']) if 'actions' in traj else None
        
        # Determine trajectory length based on actions (the constraining factor)
        if action_data is not None:
            length = len(action_data)
        elif qpos_data is not None:
            length = len(qpos_data) - 1  # Use states minus 1 if no actions
        else:
            length = 1
        
        data_dict["timestamp"] = np.arange(length, dtype=np.float32)
        
        # Handle state data - use states[0:length] to align with actions
        if qpos_data is not None:
            # Use first 'length' states to align with actions
            aligned_qpos = qpos_data[:length]
            
            for state_key, config in modality_mapping["state"].items():
                start_idx = config["start"]
                end_idx = config["end"]
                data_dict[f"state.{state_key}"] = [
                    aligned_qpos[i, start_idx:end_idx] for i in range(length)
                ]
        
        # Handle action data normally
        if action_data is not None:
            for action_key, config in modality_mapping["action"].items():
                start_idx = config["start"] 
                end_idx = config["end"]
                data_dict[f"action.{action_key}"] = [
                    action_data[i, start_idx:end_idx] for i in range(length)
                ]
        
        return pd.DataFrame(data_dict)

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        """This method won't be used since we load from HDF5."""
        return Path("dummy_path.mp4")

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        """Extract video frames from HDF5 file."""
        # Get step indices  
        step_indices = self.delta_indices[key] + base_index
        
        # Determine file and trajectory
        file_idx = trajectory_id // 100
        traj_idx = trajectory_id % 100
        
        h5_files = self._get_h5_files()
        h5_file = h5_files[file_idx]
        h5_data = self._load_h5_file(h5_file)
        
        traj_key = f"traj_{traj_idx}"
        traj = h5_data[traj_key]
        
        # Extract camera name from key
        camera_name = key.replace("video.", "")
        
        # Get video data path in HDF5
        video_path = f"obs/sensor_data/{camera_name}/rgb"
        
        if video_path not in traj:
            raise ValueError(f"Video path {video_path} not found in trajectory {traj_key}")
        
        video_data = np.array(traj[video_path])  # Shape: [T, H, W, C]
        
        # Handle out-of-bounds indices with padding
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        
        # Clip indices to valid range
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        
        return video_data[step_indices]  # Shape: [len(step_indices), H, W, C]

    def get_state_or_action(self, trajectory_id: int, modality: str, key: str, base_index: int) -> np.ndarray:
        """Extract state or action data from HDF5 file."""
        # Get step indices
        step_indices = self.delta_indices[f"{key}"] + base_index
        
        # Load trajectory data with split joints
        traj_data = self.get_trajectory_data(trajectory_id)
        
        # Get trajectory length for padding
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        
        # Get data array based on modality and key
        data_key = f"{key}" if modality != "observation" else f"observation.state.{key}"
        
        if data_key in traj_data.columns:
            # Data is already split according to modality mapping
            data_array = np.stack(traj_data[data_key])
        else:
            raise ValueError(f"Key {data_key} not found in trajectory data. Available keys: {traj_data.columns.tolist()}")
        
        # Use parent method for padding
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if modality == "state" else "zero"
        )

    def __del__(self):
        """Clean up HDF5 file handles."""
        for h5_file in self._h5_cache.values():
            h5_file.close()


def create_hdf5_dataset(
    dataset_path: str,
    task_name: str, 
    subtask_name: str,
    embodiment_tag: str = "custom_robot",
    camera_names: Optional[List[str]] = None,
    delta_indices: List[int] = [0],
    video_backend: str = "decord"
) -> HDF5LeRobotDataset:
    """
    Factory function to create HDF5 dataset with default configurations.
    
    Args:
        dataset_path: Root path to dataset
        task_name: Task folder name
        subtask_name: Subtask folder name
        embodiment_tag: Robot embodiment identifier
        camera_names: List of cameras to use
        delta_indices: Temporal sampling indices
        video_backend: Video loading backend
    
    Returns:
        Configured HDF5LeRobotDataset instance
    """
    
    # Default camera configuration
    if camera_names is None:
        camera_names = [
            "base_camera", "hand_camera", "back_right_camera",
            "back_left_camera", "front_right_camera", "front_left_camera"
        ]
    
    # Create modality configs
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=delta_indices,
            modality_keys=[f"video.{cam}" for cam in camera_names]
        ),
        "state": ModalityConfig(
            delta_indices=delta_indices,
            modality_keys=["state.qpos", "state.qvel", "state.tcp_pose"]
        ),
        "action": ModalityConfig(
            delta_indices=delta_indices,
            modality_keys=["action.robot_actions"]
        )
    }
    
    return HDF5LeRobotDataset(
        dataset_path=dataset_path,
        task_name=task_name,
        subtask_name=subtask_name,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        camera_names=camera_names,
        video_backend=video_backend
    )