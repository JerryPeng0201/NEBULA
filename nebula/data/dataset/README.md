# NEBULA Unified Data Platform

NEBULA provides a unified, modular, and extensible dataset platform for embodied AI and robotics research. It provides a clean and scalable interface for working with large-scale Vision-Language-Action (VLA) datasets — including multi-modal observations, robot control trajectories, and language instructions — in a structured and robot-agnostic way.

NEBULA Data Platform is able to

- Supports multi-robot datasets

- Fully compatible with single-arm or dual-arm setups

- Designed for both simulation and real-world data

- Plug-and-play for learning, benchmarking, and evaluation

## 🔧 What is this?

This section of the NEBULA codebase implements the data loading and representation layer. It defines how standard NEBULA dataset files (e.g., data.h5 + meta.json) are:

1. Discovered and parsed from disk

2. Translated into structured Python objects (Episode, Step, Observation, etc.)

3. Decoded into robot-specific joint/action/image data using a config system

4. Made accessible through a powerful, queryable Python SDK

## 🧠 How it works?

### Structured Episode Format

At the core is a standard representation for robot episodes:

```markdown
Episode
├── LanguageInstruction  (text command)
├── List[Step]
│   ├── Observation (images, depth, robot state)
│   └── Action (joint or symbolic control)
├── MetaInfo (task, environment, success)
├── Statistics (optional: min/max/mean/std)
```

An Episode represents one complete trajectory of an agent (robot) interacting with the environment under a specific task, usually paired with a language instruction, a sequence of multimodal observations, actions, and optional metadata/statistics.

#### instruction: LanguageInstruction

```markdown
LanguageInstruction
├── raw_text: str                       # e.g., "Pick up the red cube and place it on the green platform"
├── tokenized: List[str]                # e.g., ["Pick", "up", "the", "red", "cube", ...]
├── embedding: Optional[np.ndarray]     # e.g., BERT or CLIP embedding vector (optional)
```

#### steps: List[Step]

Each Step corresponds to one timestep in the trajectory.

```markdown
Step (for each timestep t)
├── observation: Observation
├── action: Action
├── timestamp: float                # Optional: real-time or simulation time
```

#### observation: Observation

This contains all the sensor data at time $t$, including images, depth maps, and robot states.

```markdown
Observation
├── views:
│   ├── base_camera: RGBImage
│   ├── hand_camera: RGBImage
│   └── ... (other views as defined in config)
│
├── state:
│   ├── arm: np.ndarray              # shape (DOF,) e.g., [q0, q1, ..., q6]
│   ├── gripper: np.ndarray          # e.g., [open_amount]
│   └── ... (other limbs if present)
```

Each view (e.g., base_camera) is typically a `RGBImage: np.ndarray of shape (H, W, 3), dtype=uint8`

#### action: Action

This contains the control command issued at time $t$.

```markdown
Action
├── arm: np.ndarray                 # shape (DoF,), e.g., desired joint velocity or position
├── gripper: np.ndarray             # e.g., single value for open/close
├── raw: Optional[np.ndarray]       # Flat concatenated array if needed
```

#### meta: MetaInfo

Contains metadata about the episode (task, environment, success, etc.).

```markdown
MetaInfo
├── task_name: str                 # e.g., "Control-Easy-PlaceSphere"
├── subtask_id: str                # e.g., "0034"
├── robot_name: str                # e.g., "franka_panda_single_arm_2gripper"
├── success: bool                  # Whether the task was completed successfully
├── env_id: Optional[str]          # Environment hash or ID
├── trajectory_id: Optional[str]   # Unique identifier (for cross-referencing)
├── episode_idx: int               # Index in dataset
```

#### stats: Optional[Stats]

Optional summary statistics, auto-computed or cached.

```markdown
Stats
├── step_count: int
├── state_min: np.ndarray
├── state_max: np.ndarray
├── state_mean: np.ndarray
├── action_min: np.ndarray
├── action_max: np.ndarray
├── action_mean: np.ndarray
├── task_difficulty: Optional[str]  # e.g., "easy", "medium", "hard"
```

This makes it easy to train models, evaluate behavior, or inspect data.

### Robot-Abstracted Embodiment Layer

Robot properties are not hardcoded. Instead, they’re defined in a centralized config (`embodiment_configs.py`) and loaded via the `Embodiment` class (`embodiment.py`).

```python
Embodiment("robot_configs.py", "franka_panda_single_arm_2gripper")
```

This enables NEBULA to handle:

1. Different numbers of joints
2. Different gripper types (1-finger, 2-finger)
3. Multi-arm configurations (e.g., left/right arms)
4. Different state/action slicing rules
5. Different sensor views (cameras)

### NEBULA Dataset Folder Structure

The NEBULADataset class loads raw .h5 files and .json metadata from:

```markdown
dataset_root/
└── <task_name>/
    └── motionplanning/
        └── <subtask_id>/
            ├── data.h5
            └── meta.json
```

### High-Level SDK for Querying & Sampling

The NEBULADatasetSDK exposes a clean Python API:

```python
from nebula.dataset.nebula_sdk import load_nebula_dataset

sdk = load_nebula_dataset("path/to/data")

# Get a successful episode for a given task
ep = sdk.episodes().task("stack_cube").success(True).first()

# Sample 10 episodes
batch = sdk.sample(10)

# Train-test split
train_ds, test_ds = sdk.train_test_split()
```

You can filter by task or subtask, success or failure, instruction text (natural language), step count, or robot type

## 🚀 How to use it?

### Load the dataset

```python
from nebula.dataset.nebula_sdk import load_nebula_dataset

dataset = load_nebula_dataset(
    dataset_root="/path/to/nebula",
    robot_config_path="robot_configs.py",
    robot_name="franka_panda_single_arm_2gripper"
)
```

### Query and Access Data

```python
# Get an episode
episode = dataset.get_episode(0)

# Filter for successful long-horizon episodes (This make take long time)
results = dataset.episodes().success(True).min_steps(100).execute()

# Iterate over episodes for a specific task
for ep in dataset.episodes().task("pick_cube"):
    print(ep.instruction.raw_text)
```

### Explore Metadata

```python
# Random sampling
batch = dataset.sample(5)

# Train/test split (stratified by task)
train_ds, test_ds = dataset.train_test_split(test_size=0.2)
```