import numpy as np
import sapien
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def SpatialMediumPlaceContainerSolution(env, seed=None, debug=False, vis=False):
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
    
    cube = env.movable_cube
    container = env.container
    target_direction = env.target_3d_direction
    
    obb = get_actor_obb(cube)
    
    approaching = np.array([0, 0, -1])
    target_closing =  env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, cube.pose.sp.p)
    
    # -------------------------------------------------------------------------- #
    # Pick Sequence
    # -------------------------------------------------------------------------- #
    # 1. Reach above cube
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    
    # 2. Grasp
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    
    # 3. Lift
    lift_height = 0.15 if target_direction in ["top", "inside"] else 0.08
    lift_pose = sapien.Pose([0, 0, lift_height]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    
    # -------------------------------------------------------------------------- #
    # Place Planning based on Spatial Direction
    # -------------------------------------------------------------------------- #
    container_pos = container.pose.sp.p
    container_radius = env.container_radius
    container_height = env.container_height
    cube_half_size = env.cube_half_size
    
    if target_direction == "inside":
        # Place inside container - move to center and lower into bowl
        target_pos = container_pos.copy()
        target_pos[2] = container_pos[2] - container_height * 0.5
        
        # First move above container
        above_container_pose = sapien.Pose(
            [container_pos[0], container_pos[1], container_pos[2] + 0.1],
            grasp_pose.q
        )
        planner.move_to_pose_with_screw(above_container_pose)
        
        # Then lower into container
        place_pose = sapien.Pose(target_pos, grasp_pose.q)
        planner.move_to_pose_with_screw(place_pose)
        
    elif target_direction == "beside":
        # Place beside container - offset horizontally at same level
        angle = np.random.uniform(0, 2 * np.pi)
        offset_distance = container_radius + 0.03
        
        target_pos = container_pos.copy()
        target_pos[0] += offset_distance * np.cos(angle)
        target_pos[1] += offset_distance * np.sin(angle)
        target_pos[2] = container_pos[2] - container_height + cube_half_size + 0.10
        
        # Move to beside position
        place_pose = sapien.Pose(target_pos, grasp_pose.q)
        planner.move_to_pose_with_screw(place_pose)
        
    elif target_direction == "bottom":
        # Place below container (on table, under container projection)
        target_pos = container_pos.copy()
        target_pos[2] = cube_half_size + 0.01  # Just above table
        
        # Navigate around container if needed
        current_pos = env.agent.tcp.pose.p[0].cpu().numpy()
        if current_pos[2] > container_pos[2] - container_height:
            # Move to side first to avoid collision
            side_pos = target_pos.copy()
            side_pos[0] += container_radius + 0.1
            side_pos[2] = container_pos[2] - container_height - 0.05
            
            side_pose = sapien.Pose(side_pos, grasp_pose.q)
            planner.move_to_pose_with_screw(side_pose)
        
        # Move to target position below container
        place_pose = sapien.Pose(target_pos, grasp_pose.q)
        planner.move_to_pose_with_screw(place_pose)
    
    # -------------------------------------------------------------------------- #
    # Release and Retreat
    # -------------------------------------------------------------------------- #
    # Open gripper
    planner.open_gripper()
    
    # Retreat safely based on direction
    retreat_offset = [0, 0, -0.08] if target_direction != "bottom" else [0.1, 0, 0]
    retreat_pose = place_pose * sapien.Pose(retreat_offset)
    res = planner.move_to_pose_with_screw(retreat_pose)
    
    planner.close()
    return res