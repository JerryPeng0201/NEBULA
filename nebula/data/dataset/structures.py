from dataclasses import dataclass
from typing import Optional, List, Dict, Literal
import numpy as np


# ====== Language Instruction ======
@dataclass
class LanguageInstruction:
    """
    Represents the natural language command and optionally its tokenized form.
    Useful for language-conditioned embodied tasks.
    """
    raw_text: str                                           # Original instruction text
    tokens: Optional[List[str]] = None                      # Tokenized words (if preprocessed)
    token_ids: Optional[np.ndarray] = None                  # Token IDs (e.g., for BERT/CLIP models)
    history: Optional[List[str]] = None                     # Dialogue or multi-step instruction history

# ====== Robot State Representation ======
@dataclass
class RobotLimbState:
    """
    Represents the state of a single robot limb (e.g., arm, gripper).
    """
    joint_positions: np.ndarray                             # Joint angles, shape: (DOF,)
    joint_velocities: Optional[np.ndarray] = None           # Optional joint velocities
    joint_torques: Optional[np.ndarray] = None              # Optional joint torques

@dataclass
class RobotState:
    """
    Describes the full robot state, organized by limb (e.g., left_arm, right_arm).
    """
    robot_name: str                                         # Identifier for robot model/type
    limbs: Dict[str, RobotLimbState]                        # Dictionary mapping limb names to their states

# ====== Perception & Observations ======
@dataclass
class Observation:
    """
    Captures all sensor and perception data at a single timestep.
    """
    image: Optional[Dict[str, np.ndarray]] = None           # RGB images from cameras: {"cam1": (H, W, 3)}
    depth: Optional[Dict[str, np.ndarray]] = None           # Depth maps: {"cam1": (H, W)}
    segmentation: Optional[Dict[str, np.ndarray]] = None    # Segmentation masks: {"cam1": (H, W)}
    pointcloud: Optional[Dict[str, np.ndarray]] = None      # 3D point clouds: {"cam1": (N, 3)}
    robot_state: Optional[RobotState] = None                # Full robot state at this timestep
    camera_pose: Optional[Dict[str, np.ndarray]] = None     # Camera extrinsics: {"cam1": (4, 4) matrix}
    timestamp: Optional[float] = None                       # Timestamp of the observation

# ====== Action (Continuous or Symbolic) ======
@dataclass
class RobotAction:
    """
    Low-level or continuous control signal to the robot.
    """
    control_type: Literal["joint", "torque", "symbolic"]    # Type of control (joint-level, torque, etc.)
    limbs: Dict[str, np.ndarray]                            # Limb-wise control vectors, e.g., {"left_arm": (DOF,)}

@dataclass
class Action:
    """
    Action issued at a single timestep, either low-level or symbolic.
    """
    robot_action: Optional[RobotAction] = None              # Continuous control action
    symbolic: Optional[str] = None                          # Optional high-level action (e.g., "pick_red_cube")

# ====== Timestep Data ======
@dataclass
class Step:
    """
    A single timestep in the trajectory, consisting of observation and action.
    """
    observation: Observation  # Perception and sensor data
    action: Action            # Executed action
    reward: Optional[float] = None                          # Optional reward (for RL or evaluation)
    metadata: Optional[Dict[str, any]] = None               # Extra per-step info (e.g., success flag, collisions)

# ====== Episode-Level Metadata ======
@dataclass
class MetaInfo:
    """
    Episode-level metadata and scenario information.
    """
    success: Optional[bool] = None                          # Whether the task was completed successfully
    scene_id: Optional[str] = None                          # Identifier for the scene/environment
    source: Optional[str] = None                            # Data source or collection pipeline (e.g., "GROOT")
    environment: Optional[str] = None                       # Simulation environment or real-world setting
    robot_init_pose: Optional[np.ndarray] = None            # Initial robot pose: (x, y, z, roll, pitch, yaw)
    additional: Optional[Dict[str, any]] = None             # Arbitrary extensible metadata

# ====== Episode-Level Statistics (Optional) ======
@dataclass
class EpisodeStatistics:
    """
    Optional summary statistics for the episode (e.g., for normalization).
    """
    state_min: Optional[np.ndarray] = None
    state_max: Optional[np.ndarray] = None
    state_mean: Optional[np.ndarray] = None
    state_std: Optional[np.ndarray] = None
    action_min: Optional[np.ndarray] = None
    action_max: Optional[np.ndarray] = None

# ====== Full Episode ======
@dataclass
class Episode:
    """
    A complete trajectory (episode) including instruction, observations, actions, and metadata.
    """
    task_id: str                                            # Task identifier (e.g., from NEBULA benchmark)
    robot_name: str                                         # Robot type used in the episode
    instruction: LanguageInstruction                        # Language command for the task
    steps: List[Step]                                       # Sequence of observations and actions
    meta: Optional[MetaInfo] = None                         # Metadata about the episode
    statistics: Optional[EpisodeStatistics] = None          # Optional summary statistics