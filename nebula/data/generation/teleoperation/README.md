# Running Nebula teleoperation (Simulation) with Dockerized `mplib`

## Overview: Teleoperation with Dockerized Motion Planning

![teleop_banner](/figures/teleop_banner.png)

Nebula teleoperation provides an intuitive click-and-drag interface, powered by the Sapien engine, for efficient data collection. The system leverages **gRPC** to expose the Ubuntu-exclusive **mplib** motion planner as a remote service. By running mplib inside a Ubuntu Docker container, users can connect and interact with the planner from their local machine (macOS or Ubuntu) via gRPC, enabling cross-platform teleoperation.

**Key Features:**

- Use Nebula teleoperation on both Ubuntu and macOS.
- The motion planner (mplib) runs remotely in Docker and integrates seamlessly with Nebula agents.
- Each robot can have its own remote planner service (currently implemented for Panda robots).

For practical integration examples, see [`remote_motionplanner.py`](panda/remote_motionplanner.py). The [protos](panda/protos) directory contains gRPC protocol definitions, and [server.py](panda/server.py) provides the server implementation.

---

## Instructions

### 1. Install Dependencies

1. Ensure **conda** (Anaconda or Miniconda) and **git** are installed on your system.
2. Create and activate a `nebula` conda environment. Refer to the [📦 Installation](/README.md#-installation) section for detailed steps.
3. With the `nebula` environment activated, install all required dependencies before proceeding with the following instructions.
4. If you are using macOS, refer to the [Vulkan macOS installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html) for platform-specific steps.
5. Install `pinocchio` (macOS):
  ```bash
  conda install pinocchio -c conda-forge
  ```
6. Install `gRPC`:
  ```bash
  pip install grpcio grpcio-tools
  ```

Use the **nebula** environment for all subsequent operations.

---

### 2. Build the Docker Image for **mplib** Motion Planning Service (macOS) 

1. Navigate to the Nebula directory:

  ```bash
  cd Nebula-ALPHA
  ```

2. For macOS (Apple Silicon or Intel):

  **Build the image (amd64 emulation on Apple Silicon):**

  ```bash
  docker build --platform=linux/amd64 -t mplib-grpc-panda:amd64 ./nebula/data/generation/teleoperation/panda
  ```

  **Run the container:**

  ```bash
  docker run --platform=linux/amd64 --rm -p 50051:50051 -v $(pwd)/nebula/assets:/app/assets mplib-grpc-panda:amd64
  ```

---

### 3. Start Data Collection

**For macOS, in another terminal**, set environment variables and start data collection:

- For macOS (Apple Silicon or Intel):

  ```bash
  export MPLIB_GRPC_ADDR=localhost:50051
  python -m nebula.data.generation.teleoperation.panda.interactive -e Control-PegInsertionSide-Medium --save-video --subtask-idx 3 --task_instruction="Pick up a orange-white peg and insert the orange end into the box with a hole in it." --use-remote
  ```

- For Ubuntu:

  ```bash
  python -m nebula.data.generation.teleoperation.panda.interactive -e Control-PegInsertionSide-Medium --save-video --subtask-idx 3 --task_instruction="Pick up a orange-white peg and insert the orange end into the box with a hole in it."
  ```

For more options and details, display the help message with:

```bash
python -m nebula.data.generation.teleoperation.panda.interactive -h
```

**Workflow:**

1. Review task instructions in `nebula/benchmarks/capabilities/*`.
2. Start the data collection script.
3. Use the click-and-drag interface and keyboard controls to operate the robot.
4. Monitor the terminal for `"success = True"` to confirm successful execution.
5. Press `"c"` to begin a new episode.
6. Press `"q"` to save collected data and exit.

**Note:** The `--subtask-idx` argument organizes collected data into separate subfolders for each session.

**Keyboard Commands:**

```bash
h: print help menu
g: toggle gripper open/close
t: print task instruction
u: move hand up
j: move hand down
k/l: rotate grasp pose in Yaw
i/o: rotate grasp pose in Pitch
arrow_keys: move hand in arrow direction
n: execute motion planning to target pose
c: end episode and record trajectory
q: quit and save data
```

---
### 4. Data formatter

A provided data-formatter script merges recorded demos into a single organized folder using the following CLI arguments (see dataclass Args):

- env_id / -e: environment id to reformat demos for (default: "ControlEasy-PlaceSphere")
- record_dir: directory containing recorded data and optional videos (default: "demos")
- num_episodes: maximum number of episodes to reformat (default: 100)
- target_id: name of the organized folder for all collections (default: "0")
- only_success: flag to include only successful episodes (default: False)

What it does
- Operates on copies only — original files and folders are preserved.
- Collect video files (e.g., .mp4) (in folders); merge HDF5 recordings (.h5) and metadata (.json).
- Reassigns and renumbers trajectory IDs so they are contiguous in the merged output.
- Produces a new merged folder under record_dir named by target_id.

Usage (example)
- Basic command (include -e alias for env_id):

```bash
python -m nebula.data.generation.teleoperation.panda.data_formatter \
  -e ControlEasy-PlaceSphere \
  --record-dir /path/to/raw_data \
  --num-episodes 100 \
  --target-id 0 \
  --only-success
```

Notes and expectations

- A new directory `record_dir/teleop/tmp_<target_id>` will contain the merged episodes and updated metadata. You must manually remove the `tmp_` prefix from the folder and file names (.h5, .json) for better security.

---

### 5. Customization

To adapt Nebula Teleoperation for a different robot, use the [Panda implementation](panda) as a reference. For each new robot, implement a local motion planner solver and/or a remote motion planner solver. The local solver can follow the approach in [Nebula Motion Planning data collection](../motionplanning/README.md). Update relevant files and configurations to ensure compatibility with your robot model and specific tasks.

---
