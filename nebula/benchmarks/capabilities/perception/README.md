# NEBULA Capability Tests: Perception

The Perception capability test is designed to evaluate an agent’s ability to identify target objects based solely on their visual attributes, such as color, shape, or size. In order to isolate perception performance from language complexity and motor control, this test uses minimal language instructions and requires only a single, fixed behavior: grasp the identified object.

This setting enables precise assessment of the system’s ability to ground visual input to discrete object attributes, without interference from multi-step planning, ambiguous language, or complex interaction logic.

## Difficulty Level

All Perception tasks are categorized into three primary **attribute types**: **color**, **shape**, and **size**. Each task requires the model to identify and select an object based on a specific attribute. Difficulty is determined by the **number of candidate objects** and the **attribute complexity**, as outlined below:

| **Difficulty** | **Color Tasks**                                                | **Shape Tasks**                        | **Size Tasks**                         |
|----------------|----------------------------------------------------------------|----------------------------------------|----------------------------------------|
| **Easy**       | 2 distinct colors (e.g., red vs. blue)                         | 2 distinct shapes (e.g., cube vs. sphere) | 2 distinct sizes (e.g., small vs. large) |
| **Medium**     | 3–5 objects<br>2 mixed-color variants (e.g., red-green, blue-red) | 3–5 different shapes                   | 3–5 size levels (e.g., small, medium, large) |
| **Hard**       | >5 objects<br>3-way mixed colors (e.g., red-green-blue variants) | >5 similar and complex shapes          | >5 finely graded size variants         |

> Each episode tests only one attribute type at a time to isolate perception performance. Tasks with multiple overlapping attributes are not used in this benchmark to avoid confounding evaluation.

## Task Preview

We will put a figure at here

## Task Details Table

The table below summarizes the key properties of each Control task included in the benchmark. The “Task Names” column lists the environment identifiers (env_id) corresponding to each specific task. “Dense Reward” indicates whether a dense reward signal is available during training. “Eval Conditions” specifies any predefined evaluation constraints or success criteria. The “Demos” column shows whether demonstration data is provided, such as motion planning trajectories or scripted rollouts. “Max Episode Steps” denotes the default upper limit on the number of steps per episode. Finally, “Teleoperation” indicates whether human-collected trajectories are available for the task, typically gathered through manual teleoperation.


|          Task Names               | Dense Reward | Eval Conditions | Demos | Max Episode Steps | Teleoperation |
|-----------------------------------|--------------|-----------------|-------|-------------------|---------------|
| Perception-PickBiggerSphere-Easy  |      ✅      |        ✅       |  ✅   | 50                |       ❌      |
| Perception-PickRedSphere-Easy     |      ✅      |        ✅       |  ✅   | 50                |       ❌      |
| Perception-PickSphere-Easy        |      ✅      |        ✅       |  ✅   | 50                |       ❌      |
| Perception-PickDiffCubes-Medium   |      ✅      |        ✅       |  ✅   | 50                |       ❌      |
| Perception-PickRedT-Medium        |      ✅      |        ✅       |  ✅   | 50                |       ❌      |
| Perception-PickWhitePeg-Medium    |      ✅      |        ✅       |  ✅   | 50                |       ❌      |
| Perception-PickRedT-Hard          |      ✅      |        ✅       |  ✅   | 50                |       ❌      |
| Perception-PickRightCubes-Hard    |      ✅      |        ✅       |  ✅   | 50                |       ❌      |
| Perception-PickPeg-Hard           |      ✅      |        ✅       |  ✅   | 50                |       ❌      |

## Future Works

- [ ] **Multi-Object Discrimination**: Require the model to select multiple objects based on a shared attribute (e.g., "pick all red cubes") or count-based grounding (e.g., "pick the second tallest").