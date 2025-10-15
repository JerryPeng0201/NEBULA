import numpy as np
import sapien
from transforms3d.euler import euler2quat
import random
from nebula.envs.tasks.capabilities.spatial.easy.move_cube import SpatialEasyMoveCubeEnv
from nebula.data_collection.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data_collection.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: SpatialEasyMoveCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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
    red_cube = env.red_cube    # Red cube that needs to be moved
    green_cube = env.green_cube  # Green cube as spatial reference (at center)

    # Get the target direction
    target_direction = env.target_direction

    # Pre-compute grasp pose to avoid mid-execution computation
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
    
    # Fix tensor indexing - get position as numpy array
    red_cube_pos = red_cube.pose.p[0].cpu().numpy()
    red_grasp_pose = env.agent.build_grasp_pose(approaching, closing, red_cube_pos)

    # ========================================================================== #
    # Phase 1: Reach and grasp the red cube
    # ========================================================================== #
    
    # Reach red cube
    reach_pose_red = red_grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose_red)
    
    # Grasp red cube
    planner.move_to_pose_with_screw(red_grasp_pose)
    planner.close_gripper()

    # Lift red cube
    lift_pose = sapien.Pose([0, 0, 0.08]) * red_grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # ========================================================================== #
    # Phase 2: Calculate target position and move to spatial location
    # ========================================================================== #
    
    # Get current green cube position (reference cube) - fix tensor indexing
    green_cube_pos = green_cube.pose.p[0].cpu().numpy()
    
    # Calculate target placement position based on spatial direction
    # Use environment's offset distance (0.06m = 6cm)
    placement_distance = env.target_offset_distance  # 0.06
    
    # Define spatial offset vectors relative to green cube (match environment logic)
    target_position = green_cube_pos.copy()
    
    if target_direction == "right":
        target_position[0] += placement_distance  # +x direction
    elif target_direction == "left":
        target_position[0] -= placement_distance  # -x direction
    elif target_direction == "front":
        target_position[1] -= placement_distance  # -y direction (front)
    elif target_direction == "back":
        target_position[1] += placement_distance  # +y direction (back)
    else:
        print(f"Invalid target direction: {target_direction}")
        planner.close()
        return False
    
    # Keep same height as green cube (on table surface)
    target_position[2] = green_cube_pos[2]  # Same z-level as green cube
    
    # print(f"Green cube position: {green_cube_pos}")
    # print(f"Target position ({target_direction}): {target_position}")
    # print(f"Offset distance: {placement_distance}m")

    # ========================================================================== #
    # Phase 3: Transport to target area
    # ========================================================================== #
    
    # Move to approach position above target location
    approach_position = target_position.copy()
    approach_position[2] += 0.05  # 5cm above target position for safety
    
    approach_pose = sapien.Pose(approach_position, red_grasp_pose.q)
    planner.move_to_pose_with_screw(approach_pose)

    # ========================================================================== #
    # Phase 4: Lower and place the red cube
    # ========================================================================== #
    
    # Lower to final placement position
    place_pose = sapien.Pose(target_position, red_grasp_pose.q)
    res = planner.move_to_pose_with_screw(place_pose)

    # Place the red cube
    planner.open_gripper()


    planner.close()
    return res