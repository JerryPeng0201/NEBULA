import numpy as np
import sapien
from transforms3d.euler import euler2quat
import random

from nebula.benchmarks.capabilities.control.medium.place_sphere import ControlPlaceSphereMediumEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import (compute_grasp_info_by_obb, get_actor_obb)

def ControlPlaceSphereMediumSolution(env: ControlPlaceSphereMediumEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Place the sphere in the purple bin, and then move it to the blue bin"})
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

    # Get the sphere object
    sphere = env.obj
    
    # The red bin is actually purple colored, and blue bin is blue
    purple_bin = env.red_bin  # This is the purple bin (despite the variable name)
    blue_bin = env.blue_bin   # This is the blue bin

    # ========================================================================== #
    # STAGE 1: Pick up sphere and place in purple bin
    # ========================================================================== #
    
    # Get object oriented bounding box for initial grasp
    obb = get_actor_obb(sphere)
    approaching = np.array([0, 0, -1])  # Approach from above
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    # Compute initial grasp pose
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    initial_grasp_pose = env.agent.build_grasp_pose(approaching, closing, sphere.pose.sp.p)
    
    # -------------------------------------------------------------------------- #
    # Phase 1.1: Reach sphere
    # -------------------------------------------------------------------------- #
    reach_pose = initial_grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    
    # -------------------------------------------------------------------------- #
    # Phase 1.2: Grasp sphere
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(initial_grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Phase 1.3: Lift sphere
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * initial_grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Phase 1.4: Move to purple bin and place
    # -------------------------------------------------------------------------- #
    # Correct height: bin_base + bin_thickness + sphere_radius
    purple_target_position = purple_bin.pose.sp.p.copy()
    purple_target_position[2] = purple_bin.pose.sp.p[2] + env.block_half_size[0] + env.radius
    
    purple_place_pose = sapien.Pose(purple_target_position, initial_grasp_pose.q)
    planner.move_to_pose_with_screw(purple_place_pose)

    # Place in purple bin
    planner.open_gripper()

    # -------------------------------------------------------------------------- #
    # Phase 1.5: Move away from purple bin
    # -------------------------------------------------------------------------- #
    retreat_pose = purple_place_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(retreat_pose)

    # ========================================================================== #
    # STAGE 2: Pick up sphere from purple bin and move to blue bin
    # ========================================================================== #
    
    # Update sphere position after it settled in purple bin
    current_sphere_pos = sphere.pose.sp.p
    
    # -------------------------------------------------------------------------- #
    # Phase 2.1: Reach sphere in purple bin
    # -------------------------------------------------------------------------- #
    reach_pose_2 = sapien.Pose(current_sphere_pos, initial_grasp_pose.q)
    planner.move_to_pose_with_screw(reach_pose_2)
    
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Phase 2.2: Lift sphere from purple bin
    # -------------------------------------------------------------------------- #
    lift_pose_2 = sapien.Pose([0, 0, 0.1]) * reach_pose_2
    planner.move_to_pose_with_screw(lift_pose_2)

    # -------------------------------------------------------------------------- #
    # Phase 2.3: Move to blue bin and place
    # -------------------------------------------------------------------------- #
    # Correct height: bin_base + bin_thickness + sphere_radius
    blue_target_position = blue_bin.pose.sp.p.copy()
    blue_target_position[2] = blue_bin.pose.sp.p[2] + env.block_half_size[0] + env.radius
    
    blue_place_pose = sapien.Pose(blue_target_position, reach_pose_2.q)
    planner.move_to_pose_with_screw(blue_place_pose)

    # Place in blue bin
    # Capture the final state after completing both placements
    res = planner.open_gripper()

    planner.close()

    return res