import numpy as np
import sapien
from transforms3d.euler import euler2quat
import random
from nebula.benchmarks.capabilities.perception.easy.pick_redsphere import PerceptionPickRedSphereEasyEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def PerceptionPickRedSphereEasySolution(env: PerceptionPickRedSphereEasyEnv, seed=None, debug=False, vis=False):
    """
    Perception task solution: Only grasp the red sphere.
    No placement is required for perception tasks.
    """
    env.reset(seed=seed, options={"task_instruction": "Grasp the red sphere."})
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

    # Target: the red sphere
    sphere = env.red_sphere

    # -------------------------------------------------------------------------- #
    # Compute grasp pose for the target sphere
    # -------------------------------------------------------------------------- #
    # Retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(sphere)

    approaching = np.array([0, 0, -1])  # Approach from above
    # Get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # Build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, sphere.pose.sp.p)
    
    # -------------------------------------------------------------------------- #
    # Phase 1: Reach the sphere
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if not res:
        print("Failed to reach the red sphere")
        planner.close()
        return res
    
    # -------------------------------------------------------------------------- #
    # Phase 2: Grasp the sphere
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if not res:
        print("Failed to move to grasp pose")
        planner.close()
        return res
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Phase 3: Lift the sphere (optional for perception, but helps verify grasp)
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.05]) * grasp_pose  # Small lift
    res = planner.move_to_pose_with_screw(lift_pose)
    if not res:
        print("Failed to lift the red sphere")
        planner.close()
        return res

    # Task complete: the robot has successfully grasped the red sphere
    planner.close()
    return res

