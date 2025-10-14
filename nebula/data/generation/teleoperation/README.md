# Running Nebula teleoperation (Simulation) with Dockerized `mplib`

## Overview: Teleoperation with Dockerized Motion Planning

Nebula teleoperation provides an intuitive click-and-drag interface, powered by the Sapien engine, for efficient data collection. The system leverages **gRPC** to expose the Ubuntu-exclusive **mplib** motion planner as a remote service. By running mplib inside a Docker container on Ubuntu, users can connect and interact with the planner from their local machine (macOS or Ubuntu) via gRPC, enabling cross-platform teleoperation.

**Key Features:**
- Use Nebula teleoperation on both Ubuntu and macOS.
- The motion planner (mplib) runs remotely in Docker and integrates seamlessly with Nebula agents.
- Each robot can have its own remote planner service (currently implemented for Panda robots).

For practical integration examples, see [`remote_motionplanner.py`](panda/remote_motionplanner.py). The [protos](panda/protos) directory contains gRPC protocol definitions, and [server.py](panda/server.py) provides the server implementation.

---
## Instructions
### 1. Install Dependencies

1. Ensure **conda** (Anaconda or Miniconda) and **git** are installed. Create and activate a conda environment:
  ```bash
  conda create -n nebula python=3.10
  conda activate nebula
  python -m pip install --upgrade pip
  ```

2. With the **nebula** environment activated, install the required dependencies. If you are using macOS, refer to the [Vulkan macOS installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html) for platform-specific steps.
  ```bash
  pip install --upgrade mani_skill torch
  ```

3. Install `pinocchio`:
  ```bash
  conda install pinocchio -c conda-forge
  ```

4. Install `gRPC`:
  ```bash
  pip install grpcio grpcio-tools
  ```

Use the **nebula** environment for all subsequent operations.

---

### 2. Build the Docker Image for **mplib** Motion Planning Service

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
```
h: print help menu
g: toggle gripper open/close
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

### 4. Customization

To adapt Nebula Teleoperation for a different robot, use the [Panda implementation](panda) as a reference. For each new robot, implement a local motion planner solver and/or a remote motion planner solver. The local solver can follow the approach in [Nebula Motion Planning data collection](../motionplanning/README.md). Update relevant files and configurations to ensure compatibility with your robot model and specific tasks.

---

**Acknowledgement:**  
This implementation is inspired by [ManiSkill3](https://github.com/haosulab/ManiSkill):

```
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao et al.},
  journal = {Robotics: Science and Systems},
  year={2025},
}
```
