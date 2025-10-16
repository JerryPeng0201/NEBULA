# Running Nebula Motion Planning (Simulation)

## Overview: Motion Planning in Nebula
![MP_banner](/figures/MP_banner.png)
Motion planning is a traditional control and data collection method in robotics, using algorithmic planners to generate feasible trajectories for robot manipulation. 

For the current implementation with the Panda robot, Nebula provides complete solutions for motion planning data collection across all Nebula benchmark tasks.  

> **Note:** For optimal consistency and efficiency, we recommend running Nebula Motion Planning on **Ubuntu**.

---

## Instructions

### 1. Install Dependencies

1. Ensure **conda** (Anaconda or Miniconda) and **git** are installed on your system.
2. Create and activate a `nebula` conda environment. Refer to the [📦 Installation](/README.md#-installation) section for detailed steps.
3. With the `nebula` environment activated, install all required dependencies before proceeding with the following instructions.

---

### 2. Start Motion Planning Data Collection

To begin collecting motion planning data with Nebula, use the following example command:

```bash
python -m nebula.data.generation.motionplanning.panda.run -e Control-PegInsertionSide-Medium -n 10 --save-video --subtask-idx 3
```

For more options and details, display the help message with:

```bash
python -m nebula.data.generation.motionplanning.panda.run -h
```
**Note:** The `--subtask-idx` argument organizes collected data into separate subfolders for each session.

**Workflow:**
1. Review the task instructions in `nebula/benchmarks/capabilities/*`.
2. Start the data collection script for the target task.
3. Wait for data collection to complete.

---

### 3. Customization

To customize Nebula motion planning data collection for a different robot, use the current [Panda](panda) implementation as a reference. For each new robot, define the robot in `nebula/core/embodiment/robots`, implement a motion planner solver, and provide solutions for the relevant tasks. Update any related files to ensure compatibility with your robot and tasks.

---

