import numpy as np
import sapien
import torch

from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

from nebula.benchmarks.capabilities.spatial.easy.place_between import (
    SpatialEasyPlaceBetweenEnv
)

def _get_elapsed_steps(env, fallback_steps: int = 0) -> int:
    """Get trajectory steps (Python int)"""
    try:
        if hasattr(env, "env") and hasattr(env.env, "_elapsed_steps"):
            return int(env.env._elapsed_steps)
    except Exception:
        pass
    return int(fallback_steps)

def SpatialEasyPlaceBetweenSolution(env: SpatialEasyPlaceBetweenEnv, seed=None, debug=False, vis=False):
    """
    Motion-planning solution for PlaceBetween task:
    - Grasp the red movable cube
    - Lift
    - Move to the midpoint above blue/green reference cubes 
    - Lower and release gripper
    - Evaluate success after stabilization
    Returns: [{ "success": bool, "elapsed_steps": int }]
    """
    # Reset + increase episode step limit to avoid early 100-step recording truncation
    env.reset(seed=seed)
    if hasattr(env, "env") and hasattr(env.env, "_max_episode_steps"):
        try:
            env.env._max_episode_steps = max(1000, int(getattr(env.env, "_max_episode_steps", 100)))
        except Exception:
            env.env._max_episode_steps = 1000

    # Use official planner (internally handles batched torch actions and stepping)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    base_env = env.unwrapped  # Use base env for object/pose access, but step through wrapper to keep recording

    # Get object handles
    movable_cube = base_env.movable_cube  # Red movable cube
    blue_cube    = base_env.blue_cube
    green_cube   = base_env.green_cube

    try:
        # -------------------------------------------------------------------------- #
        # Calculate grasp pose
        # -------------------------------------------------------------------------- #
        # Use OBB for grasp info (same approach as sphere example)
        obb = get_actor_obb(movable_cube)

        approaching = np.array([0, 0, -1], dtype=np.float32)
        # Get closing direction from TCP's y-axis, let compute_grasp_info_by_obb handle orthogonality/depth
        tcp_y = base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=tcp_y,
            depth=FINGER_LENGTH,
        )
        closing = grasp_info["closing"]
        # Use object's world position as grasp center (sufficient for small cube; use grasp_info["center"] if needed)
        grasp_pose = base_env.agent.build_grasp_pose(approaching, closing, movable_cube.pose.sp.p)

        # -------------------------------------------------------------------------- #
        # Reach
        # -------------------------------------------------------------------------- #
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1:
            steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
            planner.close()
            return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]

        # -------------------------------------------------------------------------- #
        # Grasp
        # -------------------------------------------------------------------------- #
        res = planner.move_to_pose_with_screw(grasp_pose)
        if res == -1:
            steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
            planner.close()
            return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]
        planner.close_gripper()

        # -------------------------------------------------------------------------- #
        # Lift
        # -------------------------------------------------------------------------- #
        lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
        res = planner.move_to_pose_with_screw(lift_pose)
        if res == -1:
            steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
            planner.close()
            return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]

        # -------------------------------------------------------------------------- #
        # Calculate midpoint pose (keep grasp orientation, change position)
        # -------------------------------------------------------------------------- #
        blue_pos = blue_cube.pose.p
        green_pos = green_cube.pose.p
        if hasattr(blue_pos, "cpu"):  blue_pos  = blue_pos.cpu().numpy()
        if hasattr(green_pos, "cpu"): green_pos = green_pos.cpu().numpy()
        if blue_pos.ndim  == 2: blue_pos  = blue_pos[0]
        if green_pos.ndim == 2: green_pos = green_pos[0]

        midpoint = np.array([
            (blue_pos[0] + green_pos[0]) / 2.0,
            (blue_pos[1] + green_pos[1]) / 2.0,
            base_env.cube_half_size,  # Place at table height
        ], dtype=np.float32)

        # -------------------------------------------------------------------------- #
        # Move above midpoint
        # -------------------------------------------------------------------------- #
        pre_place = midpoint.copy(); pre_place[2] += 0.08
        pre_place_pose = sapien.Pose(pre_place, grasp_pose.q)
        res = planner.move_to_pose_with_screw(pre_place_pose)
        if res == -1:
            steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
            planner.close()
            return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]

        # -------------------------------------------------------------------------- #
        # Lower to place position
        # -------------------------------------------------------------------------- #
        place_pose = sapien.Pose(midpoint, grasp_pose.q)
        res = planner.move_to_pose_with_screw(place_pose)
        if res == -1:
            steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
            planner.close()
            return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]

        # -------------------------------------------------------------------------- #
        # Release
        # -------------------------------------------------------------------------- #
        planner.open_gripper()

        # -------------------------------------------------------------------------- #
        # Retreat
        # -------------------------------------------------------------------------- #
        retreat_pose = sapien.Pose([0, 0, 0.1]) * place_pose
        res = planner.move_to_pose_with_screw(retreat_pose)

        # -------------------------------------------------------------------------- #
        # Stabilize then evaluate
        # -------------------------------------------------------------------------- #
        # Wait using batched torch actions; planner is already torch batched, cleaner to use planner steps
        # Could also directly use env.step with 0 force action: planner doesn't expose "empty step" interface
        zero_action = torch.zeros((1, *env.action_space.shape), device=env.device, dtype=torch.float32)
        for _ in range(30):
            obs, r, terminated, truncated, info = env.step(zero_action)
            if terminated or truncated:
                break

        evaluation = env.evaluate()
        success = evaluation.get("success", False)

        steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
        planner.close()
        return [{"success": success, "elapsed_steps": steps}]

    except Exception as e:
        print(f"[PlaceBetween solver] Error: {e}")
        import traceback; traceback.print_exc()
        steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
        planner.close()
        return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]