import numpy as np
import sapien
from transforms3d.euler import euler2quat
import random
import torch
from nebula.benchmarks.capabilities.perception.hard.pick_redT import PerceptionPickRedTHardEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from nebula.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

def PerceptionPickRedTHardSolution(env: PerceptionPickRedTHardEnv, seed=None, debug=False, vis=False):
    """
    Perception task solution: Only grasp the red T-shape among 6 objects.
    No placement is required for perception tasks.
    """
    env.reset(seed=seed, options={"task_instruction": "Grasp the red T-shape."})
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

    # Target: the red T-shape
    redT = env.red_tee

    # -------------------------------------------------------------------------- #
    # Compute grasp pose for the target T-shape
    # -------------------------------------------------------------------------- #
    # Retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(redT)

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
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, redT.pose.sp.p)
    
    # Rotate the grasp pose to align with the T-shape orientation
    xyz_angles = torch.tensor([0, 0, np.pi*0.5])  # Roll, Pitch, Yaw angles in radians
    newq = matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))
    
    # -------------------------------------------------------------------------- #
    # Phase 1: Reach the T-shape
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05], newq)
    res = planner.move_to_pose_with_screw(reach_pose)
    if not res:
        print("Failed to reach the red T-shape")
        planner.close()
        return res
    
    # -------------------------------------------------------------------------- #
    # Phase 2: Grasp the T-shape
    # -------------------------------------------------------------------------- #
    # Adjust grasp position slightly for better grip on T-shape
    grasp_pose = grasp_pose * sapien.Pose([0, -0.02, 0])
    grasp_pose = sapien.Pose(grasp_pose.p, reach_pose.q)
    res = planner.move_to_pose_with_screw(grasp_pose)
    if not res:
        print("Failed to move to grasp pose")
        planner.close()
        return res
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Phase 3: Lift the T-shape (optional for perception, but helps verify grasp)
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.05])  # Small lift
    res = planner.move_to_pose_with_screw(lift_pose)
    if not res:
        print("Failed to lift the red T-shape")
        planner.close()
        return res

    # Task complete: the robot has successfully grasped the red T-shape
    planner.close()
    return res

