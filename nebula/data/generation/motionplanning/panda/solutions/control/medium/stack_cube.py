import numpy as np
import sapien
from transforms3d.euler import euler2quat
import random

from nebula.benchmarks.capabilities.control.medium.stack_cube import ControlStackCubeMediumEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def ControlStackCubeMediumSolution(env: ControlStackCubeMediumEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Move the red cube next to the green cube, and then stack the blue cube on top of the red and green cubes"})
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

    # Blue cube grasp pose (compute now while robot is stable)
    obb_blue = get_actor_obb(blue_cube)
    grasp_info_blue = compute_grasp_info_by_obb(
        obb_blue,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    blue_grasp_pose = env.agent.build_grasp_pose(approaching, grasp_info_blue["closing"], blue_cube.pose.sp.p)
    
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
    planner.move_to_pose_with_screw(safe_pose)
    
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
    planner.move_to_pose_with_screw(reach_pose_blue)
    
    # Grasp blue cube
    planner.move_to_pose_with_screw(blue_grasp_pose)
    planner.close_gripper()

    # Lift blue cube
    lift_pose_blue = sapien.Pose([0, 0, 0.12]) * blue_grasp_pose  # Higher lift to clear other cubes
    planner.move_to_pose_with_screw(lift_pose_blue)

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
    approach_pos[2] += 0.03  # 3cm above final position
    
    approach_pose = sapien.Pose(approach_pos, blue_grasp_pose.q)
    planner.move_to_pose_with_screw(approach_pose)

    # Lower to final stacking position
    stack_pose = sapien.Pose(stack_pos, blue_grasp_pose.q)
    planner.move_to_pose_with_screw(stack_pose)

    # Place blue cube on stack - capture final state
    res = planner.open_gripper()
    planner.close()
    return res