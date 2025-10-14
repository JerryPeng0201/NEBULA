# NEBULA Capability Tests: Dynamics

The Dynamic Adaptation capability tests in NEBULA are designed to evaluate an agent’s ability to perceive and respond to changes in the environment in real time. Unlike static scenes where all elements remain fixed, dynamic tasks introduce moving objects, time-sensitive events, or context shifts that require the agent to adjust its perception, planning, and control strategies on the fly.

## Difficulty Level

| Difficulty | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| Easy       | Static dynamic: object changes color or shape but remains stationary        |
| Medium     | Presence of distractor objects; target object does not move                 |
| Hard       | Target object moves during execution, requiring real-time adaptation        |


## Task Details Table

The table below summarizes the key properties of each Control task included in the benchmark. The “Task Names” column lists the environment identifiers (env_id) corresponding to each specific task. “Dense Reward” indicates whether a dense reward signal is available during training. “Eval Conditions” specifies any predefined evaluation constraints or success criteria. The “Demos” column shows whether demonstration data is provided, such as motion planning trajectories or scripted rollouts. “Max Episode Steps” denotes the default upper limit on the number of steps per episode. Finally, “Teleoperation” indicates whether human-collected trajectories are available for the task, typically gathered through manual teleoperation.


|          Task Names                   | Dense Reward | Eval Conditions | Demos | Max Episode Steps | Teleoperation |
|---------------------------------------|--------------|-----------------|-------|-------------------|---------------|
| Dynamic-ColorSwitchPick-Easy          |      ✅      |        ✅        |  ✅   | 100               |       ❌      |
| Dynamic-ShapeSwitchPick-Easy          |      ✅      |        ✅        |  ✅   | 250               |       ❌      |
| Dynamic-PressSwitch-Easy              |      ✅      |        ✅        |  ✅   | 200               |       ❌      |
| Dynamic-PickCubeWithCollision-Medium  |      ✅      |        ✅        |  ✅   | 200               |       ✅      |
| Dynamic-PickCubeWithSliding-Medium    |      ✅      |        ✅        |  ✅   | 300               |       ❌      |
| Dynamic-PlaceRollingSphere-Medium     |      ✅      |        ✅        |  ✅   | 300               |       ✅      |
| Dynamic-DistractorBallPickCube-Hard   |      ✅      |        ✅        |  ✅   | 100               |       ❌      |
| Dynamic-CatchRollingSphere-Hard       |      ✅      |        ✅        |  ✅   | 300               |       ✅      |
| Dynamic-RollBall-Hard                 |      ✅      |        ✅        |  ✅   | 200               |       ✅      |

> <u>**Note:**</u> Due to the high complexity of dynamic scenes, most of the data in this category is **manually collected** to ensure realistic and precise interaction patterns. We intentionally avoid using automated motion planning to generate dynamic behavior, as it often lacks the nuanced, reactive motion required for robust evaluation. For best results and fair benchmarking, we **do not recommend using motion planning tools** to create synthetic data for this capability test.