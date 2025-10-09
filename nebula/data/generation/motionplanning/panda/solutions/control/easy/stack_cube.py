import numpy as np
import sapien
from transforms3d.euler import euler2quat

from nebula.benchmarks.capabilities.control.easy.stack_cube import ControlStackCubeEasyEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def ControlStackCubeEasySolution(env: ControlStackCubeEasyEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Stack the red cube on top of the green cube"})
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    
    cubeA = env.cubeA
    cubeB = env.cubeB
    
    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(cubeA)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, cubeA.pose.sp.p)
    
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    
    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Reach to B with proper height adjustment
    # -------------------------------------------------------------------------- #

    reach_position = cubeB.pose.sp.p.copy()
    reach_position[2] += 0.04 # Adjust Z coordinate

    # Create reach pose maintaining grasp orientation
    reach_pose = sapien.Pose(reach_position, grasp_pose.q)
    res = planner.move_to_pose_with_screw(reach_pose)

    planner.open_gripper()

    planner.close()
    return res

