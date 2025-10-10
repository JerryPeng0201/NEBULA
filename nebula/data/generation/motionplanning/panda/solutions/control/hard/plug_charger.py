import numpy as np
import sapien
from transforms3d.euler import euler2quat
import random
import torch
from nebula.benchmarks.capabilities.control.hard.plug_charger import ControlPlugChargerHardEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
from nebula.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_euler_angles,
)
import random


def ControlPlugChargerHardSolution(env: ControlPlugChargerHardEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Pick up the charger and insert it into the socket."})
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

    # Get the charger objects
    charger = env.charger    # Charger (needs to be plugged in)
    
    # ========================================================================== #
    # STAGE 1: Pick up charger
    # ========================================================================== #
    
    # -------------------------------------------------------------------------- #
    # Phase 1.1: Grasp charger
    # -------------------------------------------------------------------------- #
    obb_red = get_actor_obb(charger)
    approaching = np.array([0, 0, -1])  # Approach from above
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info_red = compute_grasp_info_by_obb(
        obb_red,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info_red["closing"], grasp_info_red["center"]
    charger_grasp_pose = env.agent.build_grasp_pose(approaching, closing, charger.pose.sp.p)

    
    # Reach charger
    reach_pose_charger = charger_grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose_charger)
    if not res:
        print("Failed to reach charger")
        planner.close()
        return res
    
    charger_grasp_pose = charger_grasp_pose * sapien.Pose([-0.03, 0, 0])  # Adjust for better grasping point
    # Grasp charger
    res = planner.move_to_pose_with_screw(charger_grasp_pose)
    if not res:
        print("Failed to grasp charger")
        planner.close()
        return res
    planner.close_gripper()

    # Lift charger
    lift_pose_charger = charger_grasp_pose *  sapien.Pose([0, 0, -0.08])   # Higher lift for charger
    res = planner.move_to_pose_with_screw(lift_pose_charger)
    if not res:
        print("Failed to lift charger")
        planner.close()
        return res
    

    # ========================================================================== #
    socket = env.receptacle  # Socket (stationary target)
    socket_p = socket.pose.sp.p

    socket_quat = socket.pose.sp.q
    socket_q_euler = matrix_to_euler_angles(quaternion_to_matrix(torch.tensor(socket_quat)),convention="XYZ")

    grip_q_euler = matrix_to_euler_angles(quaternion_to_matrix(torch.tensor(lift_pose_charger.q)),convention="XYZ")
  
    target_q_euler =  [grip_q_euler[0].item(), socket_q_euler[1].item(), torch.pi - socket_q_euler[2].item()]  # Align the yaw angle with socket

    target_q = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor(target_q_euler), convention="XYZ"))

    reach_pose = sapien.Pose(p=socket_p,q=target_q) * sapien.Pose([-0.08, 0, 0])  

    
    # ===== do not change the order and value of these three reach poses ===== #


    res = planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(p=socket_p,q=target_q) * sapien.Pose([-0.05, 0, -0.0043])  

    

    res = planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(p=socket_p,q=target_q) * sapien.Pose([-0.005, 0, 0.0043])  

    

    res = planner.move_to_pose_with_screw(reach_pose)

    if not res:
        print("Failed to reach socket")
        planner.close()
        return res
    planner.open_gripper()
    planner.close()
    return res