import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from nebula.benchmarks.capabilities.spatial.hard.build_block import SpatialHardBuildBlockEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def _get_elapsed_steps(env, fallback_steps: int = 0) -> int:
    """Get trajectory steps (Python int)"""
    try:
        if hasattr(env, "env") and hasattr(env.env, "_elapsed_steps"):
            return int(env.env._elapsed_steps)
    except Exception:
        pass
    return int(fallback_steps)

def SpatialHardBuildBlockSolution(env: SpatialHardBuildBlockEnv, seed=None, debug=False, vis=False):
    """
    Motion-planning solution for BuildBlock task with in-place stacking strategy:
    1. Leave cube1 (red) at its current position - DO NOT MOVE IT
    2. Stack cube2 (green) directly on top of cube1 wherever it is
    3. Place triangle (blue) on top of the cube tower
    Returns: [{ "success": bool, "elapsed_steps": int }]
    """

    wrapped_env = env
    raw_env = env.unwrapped
    
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
    base_env = env.unwrapped

    # Get objects
    cube1 = base_env.objects["cube1"]
    cube2 = base_env.objects["cube2"]
    triangle = base_env.objects["triangle"]

    try:
        # -------------------------------------------------------------------------- #
        # Get cube1 position and calculate cube2 target position
        # -------------------------------------------------------------------------- #
        
        # Get cube1's current position (DON'T MOVE IT)
        cube1_pos = cube1.pose.p
        if torch.is_tensor(cube1_pos):
            cube1_pos = cube1_pos.cpu().numpy()
        if len(cube1_pos.shape) > 1:
            cube1_pos = cube1_pos[0]
        
        # Calculate stacking position for cube2 - directly on top of cube1
        cube2_target_pos = np.array([
            cube1_pos[0],  # Same x as cube1
            cube1_pos[1],  # Same y as cube1
            cube1_pos[2] + base_env.cube_half_size * 2  # Stack height: cube1_top + cube2_half_height
        ])
        
        # -------------------------------------------------------------------------- #
        # Place cube2 on cube1
        # -------------------------------------------------------------------------- #
        success = place_object_at_position(
            planner, base_env, cube2,
            target_position=cube2_target_pos,
            object_name="cube2",
            description="on top of cube1"
        )
        
        if not success:
            steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
            planner.close()
            return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]

        # -------------------------------------------------------------------------- #
        # Calculate triangle target position and place on cube tower
        # -------------------------------------------------------------------------- #
        
        # Calculate triangle position - on top of the cube tower
        triangle_target_pos = np.array([
            cube1_pos[0],  # Same x as cube1
            cube1_pos[1],  # Same y as cube1
            cube1_pos[2] + base_env.cube_half_size * 4 + base_env.triangle_height/2  # Top of cube tower
        ])
        
        # Place triangle on cube tower
        success = place_object_at_position(
            planner, base_env, triangle,
            target_position=triangle_target_pos,
            object_name="triangle",
            description="on top of cube tower"
        )
        
        if not success:
            steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
            planner.close()
            return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]

        # -------------------------------------------------------------------------- #
        # Evaluate final result
        # -------------------------------------------------------------------------- #
        evaluation = env.evaluate()
        success = evaluation.get("success", False)

        steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
        planner.close()
        return [{"success": success, "elapsed_steps": steps}]

    except Exception as e:
        steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
        try:
            planner.close()
        except:
            pass
        return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]

def place_object_at_position(planner, base_env, obj, target_position, object_name, description=""):
    """
    Generic object placement function with complete pick-and-place sequence
    
    Args:
        planner: Motion planner instance
        base_env: Base environment
        obj: Object to place
        target_position: Target position [x, y, z]
        object_name: Name for debugging
        description: Additional description for logging
    
    Returns:
        bool: Success status
    """
    FINGER_LENGTH = 0.025
    
    try:
        # -------------------------------------------------------------------------- #
        # Calculate grasp pose
        # -------------------------------------------------------------------------- #
        
        # Get object's oriented bounding box
        obb = get_actor_obb(obj)
        approaching = np.array([0, 0, -1], dtype=np.float32)  # Top-down approach
        
        # Get TCP y-axis for grasp orientation
        tcp_pose_matrix = base_env.agent.tcp.pose.to_transformation_matrix()
        if torch.is_tensor(tcp_pose_matrix):
            tcp_pose_matrix = tcp_pose_matrix.cpu().numpy()
        
        if len(tcp_pose_matrix.shape) == 3 and tcp_pose_matrix.shape[0] == 1:
            tcp_pose_matrix = tcp_pose_matrix[0]
        
        tcp_y = tcp_pose_matrix[:3, 1]
        
        # Compute grasp information
        grasp_info = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=tcp_y, depth=FINGER_LENGTH
        )
        closing = grasp_info["closing"]
        
        # Get current object position
        obj_pos = obj.pose.p
        if torch.is_tensor(obj_pos):
            obj_pos = obj_pos.cpu().numpy()
        if len(obj_pos.shape) > 1:
            obj_pos = obj_pos[0]
        
        # Build grasp pose
        grasp_pose = base_env.agent.build_grasp_pose(approaching, closing, obj_pos)

        # -------------------------------------------------------------------------- #
        # Reach
        # -------------------------------------------------------------------------- #
        approach_height = 0.06
        approach_pose = grasp_pose * sapien.Pose([0, 0, -approach_height])
        res = planner.move_to_pose_with_screw(approach_pose)
        if res == -1:
            return False

        # -------------------------------------------------------------------------- #
        # Grasp
        # -------------------------------------------------------------------------- #
        res = planner.move_to_pose_with_screw(grasp_pose)
        if res == -1:
            return False
        planner.close_gripper()

        # -------------------------------------------------------------------------- #
        # Lift
        # -------------------------------------------------------------------------- #
        lift_height = 0.10 if object_name != "triangle" else 0.15
        lift_pose = sapien.Pose([0, 0, lift_height]) * grasp_pose
        res = planner.move_to_pose_with_screw(lift_pose)
        if res == -1:
            return False

        # -------------------------------------------------------------------------- #
        # Move to pre-place position
        # -------------------------------------------------------------------------- #
        hover_height = 0.10 if object_name != "triangle" else 0.15
        pre_place_pose = sapien.Pose(target_position + np.array([0, 0, hover_height]), grasp_pose.q)
        res = planner.move_to_pose_with_screw(pre_place_pose)
        if res == -1:
            return False

        # -------------------------------------------------------------------------- #
        # Place
        # -------------------------------------------------------------------------- #
        place_pose = sapien.Pose(target_position, grasp_pose.q)
        res = planner.move_to_pose_with_screw(place_pose)
        if res == -1:
            return False
        planner.open_gripper()

        # -------------------------------------------------------------------------- #
        # Retreat
        # -------------------------------------------------------------------------- #
        retreat_height = 0.12 if object_name != "triangle" else 0.18
        retreat_pose = sapien.Pose([0, 0, retreat_height]) * place_pose
        res = planner.move_to_pose_with_screw(retreat_pose)
        # Note: Don't fail on retreat - it's not critical for task success
        
        return True

    except Exception as e:
        return False