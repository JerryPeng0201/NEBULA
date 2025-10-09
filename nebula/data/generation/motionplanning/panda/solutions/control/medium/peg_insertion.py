import numpy as np
import sapien
from transforms3d.euler import euler2quat
import random
import torch

from nebula.benchmarks.capabilities.control.medium.peg_insertion_side import  ControlPegInsertionSideMediumEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from nebula.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_euler_angles,
)
import random


def ControlPegInsertionSideMediumSolution(env: ControlPegInsertionSideMediumEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Pick up a orange-white peg and insert the orange end into the box with a hole in it."})
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

    
    peg = env.peg    
    box_hole_offset = env.box_hole_offsets

    # ========================================================================== #
    # STAGE 1: Pick up peg
    # ========================================================================== #
    
    # -------------------------------------------------------------------------- #
    # Phase 1.1: Grasp peg
    # -------------------------------------------------------------------------- #
    obb_peg = get_actor_obb(peg)
    approaching = np.array([0, 0, -1])  # Approach from above
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info_red = compute_grasp_info_by_obb(
        obb_peg,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info_red["closing"], grasp_info_red["center"]
    peg_grasp_pose = env.agent.build_grasp_pose(approaching, closing, peg.pose.sp.p)

    
    # Reach peg
    reach_pose_peg = peg_grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose_peg)
    if not res:
        print("Failed to reach peg")
        planner.close()
        return res
    grab_adjust = env.peg_half_sizes.tolist()[0][0]*0.45
    peg_grasp_peg  = peg_grasp_pose * sapien.Pose([-grab_adjust, 0, 0])  # Adjust for better grasping point
    # Grasp peg
    res = planner.move_to_pose_with_screw(peg_grasp_peg)
    if not res:
        print("Failed to grasp peg")
        planner.close()
        return res
    planner.close_gripper()

    # Lift peg
    lift_pose_peg = peg_grasp_peg *  sapien.Pose([0, 0, -0.08])   # Higher lift 
    res = planner.move_to_pose_with_screw(lift_pose_peg)
    if not res:
        print("Failed to lift peg")
        planner.close()
        return res
    

    # ========================================================================== #
    box = env.box_hole_pose  # box (stationary target)
    box_p = box.sp.p
    box_quat = box.sp.q 
    box_q_euler = matrix_to_euler_angles(quaternion_to_matrix(torch.tensor(box_quat)),convention="XYZ")

    grip_q_euler = matrix_to_euler_angles(quaternion_to_matrix(torch.tensor(lift_pose_peg.q)),convention="XYZ")

    target_q_euler =  [grip_q_euler[0].item(), grip_q_euler[1].item(), -box_q_euler[2]]  # Align the yaw angle with box

    target_q = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor(target_q_euler), convention="XYZ"))
    
    reach_pose = sapien.Pose(p=box_p,q=target_q) * sapien.Pose([-0.4, 0, 0])  # Approach from 
    
    # ===== do not change the order and value of these three reach poses ===== #


    res = planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(p=box_p,q=target_q) * sapien.Pose([-0.3, 0.0043, -0.004999])  

    

    res = planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(p=box_p,q=target_q) * sapien.Pose([-0.12, 0, -0.006])  

    

    res = planner.move_to_pose_with_screw(reach_pose)

    if not res:
        print("Failed to reach box")
        planner.close()
        return res
    planner.open_gripper()
    planner.close()
    return res