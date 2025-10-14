# NEBULA Dual-Axis Evaluation Framework

NEBULA’s evaluation framework is designed to provide **fine-grained**, **interpretable**, and **comprehensive** assessment of embodied agents through a **dual-axis structure**:

- **Capability Tests**: Measure what the agent *can* do under controlled, isolated skill conditions.
- **Stress Tests**: Measure how well the agent *holds up* under increasing real-world constraints and perturbations.

Together, they move beyond binary success metrics to uncover how, when, and why agents succeed or fail in complex manipulation tasks.

## Axis 1: Capability Evaluation

![Alt text](./../../figures/task_example.png)

The Capability Axis diagnoses agent performance across six fundamental skill families. Each capability test isolates a particular competency to ensure targeted evaluation without confounding variables. Each test is categorized into Easy / Medium / Hard to support curriculum learning and multi-stage diagnosis.

| Capability                                                            | Description                                               |
|-----------------------------------------------------------------------|-----------------------------------------------------------|
| 👁️ [Perception](./capabilities/perception/README.md)                  | Color, shape, and size recognition                        |
| 🎮 [Control](./capabilities/control/README.md)                        | Low-level motor control, stability, and sequencing        |
| 🗣️ [Language](./capabilities/language/README.md)                      | Grounding, logic, and conditional instruction parsing     |
| 🔄 [Dynamic Adaptation](./capabilities/dynamic/README.md)             | Responding to object motion and time-sensitive changes    |
| 📍 [Spatial Reasoning](./capabilities/spatial/README.md)              | 3D spatial relationships, multi-object layout planning    |
| 🧪 [Robustness / Generalization](./capabilities/robustness/README.md) | Cross-domain and distribution shift handling              |


>**Note:**  Click the links above for detailed descriptions of each capability task.

## Axis 2: Stress-Based Evaluation

The Stress Axis measures how agents degrade under targeted constraints or resource limits. Each diagnostic test focuses on a single performance dimension to isolate failure causes. Each stress probe is divided into 3 pressure levels (v1, v2, v3) for controlled ablation.

| Stress Test               | Description                                                   |
|---------------------------|---------------------------------------------------------------|
| ⚡ Inference Frequency (Hz)| Max decision rate under real-time constraints                 |
| ⏱️ Latency                | Per-step response delay under compute/load pressure           |
| 📉 Stability Score        | Behavioral consistency across repeated runs                   |
| 🔁 Adaptation Speed       | Adjustment time when tasks/goals/scene shifts occur           |
| 🧠 Resource Usage         | Memory, compute, and bandwidth cost per episode               |

## Output Format & Automation
To ensure both usability and reproducibility, NEBULA's dual-axis evaluation framework supports **fully automated testing and result generation** through a single-line command. NEBULA aims to make evaluation as simple and fast as possible, while maintaining fairness, reproducibility, and comparability across agents and methods.

### Capability Tests

For each task on the Capability Axis, we provide:

- Radar plots visualizing multi-skill performance in a compact and interpretable form

- Structured .json outputs containing per-capability and per-difficulty scores for detailed logging and comparison

### Stress Tests

For each diagnostic probe on the Stress Axis, we generate:

- Bar charts highlighting system performance under increasing pressure levels (v1, v2, v3)

- Summary tables for recording metric breakdowns such as latency, adaptation time, inference rate, etc.

