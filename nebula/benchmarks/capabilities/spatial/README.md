# NEBULA Capability Tests: Spatial Reasoning

The Spatial Reasoning capability family evaluates an agent's ability to interpret and execute spatial relationships and geometric constraints in 3D manipulation tasks. These tasks test spatial understanding by holding visual appearance and control difficulty fixed, ensuring that success depends solely on processing spatial concepts like positioning, orientation, containment, and relative placement.

Unlike Perception tasks that focus on attribute recognition or Control tasks that emphasize motor execution, Spatial Reasoning tasks isolate geometric reasoning proficiency across progressively complex spatial dimensions.

## Difficulty Level

| **Difficulty** | **Spatial Complexity**                          | **Geometric Reasoning**                                      | **Example Task**                           |
|----------------|-------------------------------------------------|--------------------------------------------------------------|--------------------------------------------|
| Easy           | 2D planar reasoning                             | Simple relative positioning (left, right, between)           | Move cube to the right of another cube     |
| Medium         | 3D spatial concepts                             | Horizontal and vertical relationships (closest, on top)      | Place the cube inside a container         |
| Hard           | Full 6-DoF motion planning                      | Complex 3D arrangement with rotational precision             | Stack objects in specific vertical order   |

## Task Preview
Examples of spatial reasoning tasks across three difficulty levels in the benchmark. Green marks objects, red marks targets, and blue indicates contextual cues. <u>**Bold underlined**</u> text shows actions; <u>*italic underlined*</u> text gives clarifications.

![Task Examples](../../../../figures/Spatial_README.png)

## Task Details Table

The table below summarizes the key properties of each Spatial Reasoning task included in the benchmark. The "Task Names" column lists the environment identifiers (env_id) corresponding to each specific task. "Dense Reward" indicates whether a dense reward signal is available during training. "Eval Conditions" specifies any predefined evaluation constraints or success criteria based on spatial relationships. The "Demos" column shows whether demonstration data is provided, such as motion planning trajectories or scripted rollouts. "Max Episode Steps" denotes the default upper limit on the number of steps per episode. Finally, "Teleoperation" indicates whether human-collected trajectories are available for the task.

|          Task Names                    | Dense Reward | Eval Conditions | Demos | Max Episode Steps | Teleoperation |
|----------------------------------------|--------------|-----------------|-------|-------------------|---------------|
| Spatial-MoveCube-Easy                  |      ✅      |        ✅        |  ✅   | 50                |       ❌      |
| Spatial-PlaceBetween-Easy              |      ✅      |        ✅        |  ✅   | 50                |       ❌      |
| Spatial-PickCube-Easy                  |      ✅      |        ✅        |  ✅   | 50                |       ❌      |
| Spatial-PickClosest-Medium             |      ✅      |        ✅        |  ✅   | 75                |       ❌      |
| Spatial-PlaceContainer-Medium          |      ✅      |        ✅        |  ✅   | 75                |       ❌      |
| Spatial-PickCube-Medium           |      ✅      |        ✅        |  ✅   | 75                |       ❌      |
| Spatial-KitchenAssembly-Hard           |      ✅      |        ✅        |  ✅   | 150               |       ✅      |
| Spatial-BuildBlock-Hard                |      ✅      |        ✅        |  ✅   | 150               |       ❌      |
| Spatial-PickCube-Hard         |      ✅      |        ✅        |  ✅   | 150               |       ❌      |

## Future Works
In upcoming updates, we plan to extend the benchmark with more advanced spatial evaluation modules, targeting geometric precision, occlusion handling, and multi-object spatial reasoning. Specifically, the following tests are under development:

- [ ] **Occlusion Reasoning Test**: Evaluates the ability to infer object positions and relationships when targets are partially or fully occluded by other objects, testing spatial memory and scene understanding.

- [ ] **Perspective and Viewpoint Reasoning Test**: Evaluates spatial reasoning across different camera viewpoints and the ability to mentally rotate or transform spatial relationships from one perspective to another.

- [ ] **Multi-Object Spatial Coordination Test**: Assesses the ability to reason about and maintain spatial relationships among multiple objects simultaneously, such as arranging items in specific patterns or maintaining relative distances during manipulation.