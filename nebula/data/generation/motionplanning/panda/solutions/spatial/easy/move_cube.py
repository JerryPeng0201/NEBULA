import numpy as np
import sapien
from transforms3d.euler import euler2quat
from nebula.benchmarks.capabilities.spatial.easy.move_cube import SpatialEasyMoveCubeEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def SpatialEasyMoveCubeSolution(env: SpatialEasyMoveCubeEnv, seed=None, debug=False, vis=False):
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

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose_red = red_grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose_red)
    
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(red_grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.08]) * red_grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Calculate target position
    # -------------------------------------------------------------------------- #
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
    
    # -------------------------------------------------------------------------- #
    # Move to approach position above target location
    # -------------------------------------------------------------------------- #
    approach_position = target_position.copy()
    approach_position[2] += 0.05  # 5cm above target position for safety
    
    approach_pose = sapien.Pose(approach_position, red_grasp_pose.q)
    planner.move_to_pose_with_screw(approach_pose)

    # -------------------------------------------------------------------------- #
    # Place
    # -------------------------------------------------------------------------- #
    place_pose = sapien.Pose(target_position, red_grasp_pose.q)
    planner.move_to_pose_with_screw(place_pose)
    res = planner.open_gripper()

    planner.close()
    return res