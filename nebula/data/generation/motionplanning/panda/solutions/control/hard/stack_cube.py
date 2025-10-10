import numpy as np
import sapien
from transforms3d.euler import euler2quat
import random
import torch
from nebula.benchmarks.capabilities.control.hard.stack_cube import ControlStackCubeHardEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from nebula.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
import random


def ControlStackCubeHardSolution(env: ControlStackCubeHardEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Place the red cube next to the green cube, stack the blue cube on top of both red and green cubes, and place the purple cube next to the arrangement"})
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

    # Get the cube objects
    red_cube = env.cubeA    # Red cube (needs to be moved next to green)
    green_cube = env.cubeB  # Green cube (stationary reference)
    blue_cube = env.cubeC   # Blue cube (needs to be stacked on top)
    purple_cube = env.cubeD # Purple cube (needs to be placed next to the arrangement)
    # print("Task: Place the red cube next to the green cube, stack the blue cube on top of both red and green cubes, and place the purple cube next to the arrangement")

    # ========================================================================== #
    # STAGE 1: Pick up red cube and place next to green cube
    # ========================================================================== #
    
    # -------------------------------------------------------------------------- #
    # Phase 1.1: Grasp red cube
    # -------------------------------------------------------------------------- #
    obb_red = get_actor_obb(red_cube)
    approaching = np.array([0, 0, -1])  # Approach from above
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info_red = compute_grasp_info_by_obb(
        obb_red,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info_red["closing"], grasp_info_red["center"]
    red_grasp_pose = env.agent.build_grasp_pose(approaching, closing, red_cube.pose.sp.p)

    
    # Reach red cube
    reach_pose_red = red_grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose_red)
    if not res:
        print("Failed to reach red cube")
        planner.close()
        return res
    
    # Grasp red cube
    res = planner.move_to_pose_with_screw(red_grasp_pose)
    if not res:
        print("Failed to grasp red cube")
        planner.close()
        return res
    planner.close_gripper()

    # Lift red cube
    lift_pose_red = sapien.Pose([0, 0, 0.08]) * red_grasp_pose  # Higher lift for cubes
    res = planner.move_to_pose_with_screw(lift_pose_red)
    if not res:
        print("Failed to lift red cube")
        planner.close()
        return res

    # -------------------------------------------------------------------------- #
    # Phase 1.2: Place red cube next to green cube
    # -------------------------------------------------------------------------- #
    # Calculate position next to green cube
    green_pos = green_cube.pose.sp.p
    cube_size = 0.04  # 2 * half_size (0.02)
    
    # Place red cube adjacent to green cube (slightly offset to avoid collision)
    red_target_pos = green_pos.copy()
    red_target_pos[0] -= (cube_size + 0.005)  # Place to the left of green cube with small gap
    red_target_pos[2] = 0.02  # On the table surface (cube half height)
    
    red_place_pose = sapien.Pose(red_target_pos, red_grasp_pose.q)
    res = planner.move_to_pose_with_screw(red_place_pose)
    if not res:
        print("Failed to reach red cube placement position")
        planner.close()
        return res

    # Place red cube
    planner.open_gripper()


    # ========================================================================== #
    # STAGE 2: Pick up blue cube and stack on top of red and green cubes
    # ========================================================================== #

    # Move to a stable position first
    safe_position = np.array([0.0, 0.0, 0.2])  # High and centered
    safe_pose = sapien.Pose(safe_position, red_grasp_pose.q)
    res = planner.move_to_pose_with_screw(safe_pose)
    if not res:
        print("Failed to reach safe position")
        planner.close()
        return res
    
    # -------------------------------------------------------------------------- #
    # Phase 2.1: Grasp blue cube
    # -------------------------------------------------------------------------- #
    obb_blue = get_actor_obb(blue_cube)
    grasp_info_blue = compute_grasp_info_by_obb(
        obb_blue,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info_blue["closing"], grasp_info_blue["center"]
    blue_grasp_pose = env.agent.build_grasp_pose(approaching, closing, blue_cube.pose.sp.p)
    
    # Reach blue cube
    reach_pose_blue = blue_grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose_blue)
    if not res:
        print("Failed to reach blue cube")
        planner.close()
        return res
    
    # Grasp blue cube
    res = planner.move_to_pose_with_screw(blue_grasp_pose)
    if not res:
        print("Failed to grasp blue cube")
        planner.close()
        return res
    planner.close_gripper()

    # Lift blue cube
    lift_pose_blue = sapien.Pose([0, 0, 0.12]) * blue_grasp_pose  # Higher lift to clear other cubes
    res = planner.move_to_pose_with_screw(lift_pose_blue)
    if not res:
        print("Failed to lift blue cube")
        planner.close()
        return res

    # -------------------------------------------------------------------------- #
    # Phase 2.2: Stack blue cube on top of red and green cubes
    # -------------------------------------------------------------------------- #
    # Calculate stacking position - center point between red and green cubes, elevated
    current_red_pos = red_cube.pose.sp.p
    current_green_pos = green_cube.pose.sp.p
    
    # Position blue cube so it's on top of both red and green cubes
    stack_pos = np.zeros(3)
    stack_pos[0] = (current_red_pos[0] + current_green_pos[0]) / 2.0  # Center between cubes
    stack_pos[1] = (current_red_pos[1] + current_green_pos[1]) / 2.0  # Center between cubes
    stack_pos[2] = 0.06  # Height: table + cube height + blue cube half height (0.02 + 0.02 + 0.02)
    
    # Approach position (slightly higher for safety)
    approach_pos = stack_pos.copy()
    approach_pos[2] += 0.03  # 3cm above position
    
    approach_pose = sapien.Pose(approach_pos, blue_grasp_pose.q)
    res = planner.move_to_pose_with_screw(approach_pose)
    if not res:
        print("Failed to reach stack approach position")
        planner.close()
        return res

    # Lower to stacking position
    stack_pose = sapien.Pose(stack_pos, blue_grasp_pose.q)
    res = planner.move_to_pose_with_screw(stack_pose)
    if not res:
        print("Failed to reach stack position")
        planner.close()
        return res

    # Place blue cube on stack
    planner.open_gripper()


    # ========================================================================== #
    # STAGE 3: Pick up purple cube and place next to the arrangement
    # ========================================================================== #
    
    # -------------------------------------------------------------------------- #
    # Phase 3.1: Grasp purple cube
    # -------------------------------------------------------------------------- #
    obb_purple = get_actor_obb(purple_cube)
    grasp_info_purple = compute_grasp_info_by_obb(
        obb_purple,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    purple_grasp_pose = env.agent.build_grasp_pose(approaching, grasp_info_purple["closing"], purple_cube.pose.sp.p)
    
    # Reach purple cube
    reach_pose_purple = purple_grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose_purple)
    if not res:
        print("Failed to reach purple cube")
        planner.close()
        return res
    
    # Grasp purple cube
    res = planner.move_to_pose_with_screw(purple_grasp_pose)
    if not res:
        print("Failed to grasp purple cube")
        planner.close()
        return res
    planner.close_gripper()

    # Lift purple cube
    lift_pose_purple = sapien.Pose([0, 0, 0.12]) * purple_grasp_pose  # Higher lift to clear other cubes
    res = planner.move_to_pose_with_screw(lift_pose_purple)
    if not res:
        print("Failed to lift purple cube")
        planner.close()
        return res

    # -------------------------------------------------------------------------- #
    # Phase 3.2: Place purple cube next to the arrangement (triangle arrangement)
    # -------------------------------------------------------------------------- #
    
    red_xy   = current_red_pos[:2].astype(float)
    green_xy = current_green_pos[:2].astype(float)
 
    # Direction from red to green (XY only)
    rg = green_xy - red_xy
    d = np.linalg.norm(rg)

    if d < 1e-9:
        rg_unit = np.array([1.0, 0.0])
    else:
        rg_unit = rg / d

    # Perpendiculars: CCW = "left" of red->green in a standard right-handed XY (x right, y up).
    perp_ccw = np.array([-rg_unit[1],  rg_unit[0]])  # 90° CCW
    perp_cw  = np.array([ rg_unit[1], -rg_unit[0]])  # 90° CW

    # --- CHOOSE THE TRIANGLE YOU WANT ---
    MODE = "equilateral_by_base"   # or "fixed_radius"

    if MODE == "equilateral_by_base":
        # Equilateral triangle using the existing base length d (red<->green).
        # Altitude h = (sqrt(3)/2) * d
        h = (np.sqrt(3.0) / 2.0) * d
    else:
        # Isosceles with equal legs = cube_size (purple at distance cube_size from both cubes).
        # Altitude h = sqrt(R^2 - (d/2)^2). Clamp at 0 if base is too long.
        R = cube_size
        h = np.sqrt(max(R*R - (0.5*d)**2, 0.0))

    
    # --- PICK SIDE RELIABLY ---
    # Set ccw=True for "left" of red->green in a standard (+x right, +y up) world.
    ccw = True
    perp_unit = perp_ccw if ccw else perp_cw

    
    target_xy = random.choice([red_xy, green_xy])

    place_xy = target_xy + h * perp_unit
    
    # # Optional: nudge out a hair so faces don't kiss
    # clearance = np.array([0.0001 * perp_unit[0], 0.0001* perp_unit[1]])
    # place_xy += clearance
    # Build final 3D pose (on the table) Height: purple cube half height (0.02)
    purple_target_pos = np.array([place_xy[0], place_xy[1], 0.02])

    # Rotate the grasp pose to align with orientation
    xyz_angles = torch.tensor([0, 0, np.pi*0.5])  # Roll, Pitch, Yaw angles in radians
    newq = matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))

    purple_place_pose = sapien.Pose(p=purple_target_pos, q=lift_pose_purple.q) * sapien.Pose([0, 0, 0], newq)
    res = planner.move_to_pose_with_screw(purple_place_pose)
    if not res:
        print("Failed to place purple cube")
        planner.close()
        return res
    
    planner.open_gripper()
    planner.close()
    return res