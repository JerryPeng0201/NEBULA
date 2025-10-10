# NEBULA Capability Tests: Control

The Control capability family evaluates an agent’s ability to perform precise and reliable motor actions under varying levels of complexity. These tasks test the low-level and mid-level control proficiency of the system, including joint-space motion execution, trajectory following, sequential manipulation, and conditional action chaining.

Unlike Perception or Language tasks that rely heavily on understanding, Control tasks emphasize execution fidelity, action precision, and response consistency.

## Difficulty Level

| **Difficulty** | **Action Length**      | **Precision Requirement**                                 | **Example Task**                      |
|----------------|------------------------|------------------------------------------------------------|----------------------------------------|
| Easy           | 1–2 sequential actions | No fine precision required                                   | Pick and Place                         |
| Medium         | 3–4 sequential actions | Coarse precision (*e.g.,* insert magnet into a large socket) | Insert magnet into oversized hole      |
| Hard           | 5+ sequential actions  | Fine precision (*e.g.,* peg-in-hole alignment)               | Plug connector into electrical socket  |

## Task Preview

We will put a figure at here

## Task Details Table

The table below summarizes the key properties of each Control task included in the benchmark. The “Task Names” column lists the environment identifiers (env_id) corresponding to each specific task. “Dense Reward” indicates whether a dense reward signal is available during training. “Eval Conditions” specifies any predefined evaluation constraints or success criteria. The “Demos” column shows whether demonstration data is provided, such as motion planning trajectories or scripted rollouts. “Max Episode Steps” denotes the default upper limit on the number of steps per episode. Finally, “Teleoperation” indicates whether human-collected trajectories are available for the task, typically gathered through manual teleoperation.


|          Task Names               | Dense Reward | Eval Conditions | Demos | Max Episode Steps | Teleoperation |
|-----------------------------------|--------------|-----------------|-------|-------------------|---------------|
| Control-PlaceSphere-Easy          |      ✅      |        ✅        |  ✅   | 50                |       ❌      |
| Control-PushCube-Easy             |      ✅      |        ✅        |  ✅   | 50                |       ❌      |
| Control-StackCube-Easy            |      ✅      |        ✅        |  ✅   | 50                |       ❌      |
| Control-PegInsertionSide-Medium   |      ✅      |        ✅        |  ✅   | 100               |       ✅      |
| Control-PlaceSphere-Medium        |      ✅      |        ✅        |  ✅   | 50                |       ❌      |
| Control-StackCube-Medium          |      ✅      |        ✅        |  ✅   | 250               |       ❌      |
| Control-PlaceSphere-Hard          |      ✅      |        ✅        |  ✅   | 100               |       ❌      |
| Control-PlugCharger-Hard          |      ✅      |        ✅        |  ✅   | 200               |       ✅      |
| Control-StackCube-Hard            |      ✅      |        ✅        |  ✅   | 250               |       ❌      |

## Future Works
The current Control Capability Test suite focuses on basic single‑ and multi‑step manipulation behaviors. In upcoming updates, we plan to extend the benchmark with more fine‑grained control evaluation modules, targeting the robustness, precision, and adaptability of robotic motion. Specifically, the following tests are under development:

- [ ] **Action Stability Test**: Measures the temporal smoothness and consistency of control outputs over time, assessing the model’s ability to maintain stable motion without oscillations or overshooting.

- [ ] **Joint‑Level Control Test**: Evaluates the precision of individual joint control, including rotation accuracy, velocity regulation, and synchronization across multiple degrees of freedom.