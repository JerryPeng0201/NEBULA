import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from nebula.benchmarks.capabilities.spatial.medium.pick_cube import SpatialMediumPickCubeEnv
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

def SpatialMediumPickCubeSolution(env: SpatialMediumPickCubeEnv, seed=None, debug=False, vis=False):
    """
    Motion-planning solution for PickCube task with 3D spatial relations:
    - Identify target object based on spatial relation (inside/outside/on_top_of/beside)
    - Grasp the correct target object
    - Lift the object to demonstrate successful pick
    - Maintain robot stability
    Returns: [{ "success": bool, "elapsed_steps": int }]
    """
    # Reset environment
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

    # Get target object based on spatial relation
    target_object_name = base_env.target_object
    target_object = base_env.objects[target_object_name]

    try:
        # -------------------------------------------------------------------------- #
        # Calculate grasp pose
        # -------------------------------------------------------------------------- #
        obb = get_actor_obb(target_object)
        approaching = np.array([0, 0, -1], dtype=np.float32)
        tcp_y = base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=tcp_y,
            depth=FINGER_LENGTH,
        )
        closing = grasp_info["closing"]
        grasp_pose = base_env.agent.build_grasp_pose(approaching, closing, target_object.pose.sp.p)

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
        lift_pose = sapien.Pose([0, 0, 0.10]) * grasp_pose
        res = planner.move_to_pose_with_screw(lift_pose)
        if res == -1:
            steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
            planner.close()
            return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]

        # -------------------------------------------------------------------------- #
        # Stabilize then evaluate
        # -------------------------------------------------------------------------- #
        zero_action = torch.zeros((1, *env.action_space.shape), device=env.device, dtype=torch.float32)
        for _ in range(20):
            obs, r, terminated, truncated, info = env.step(zero_action)
            if terminated or truncated:
                break

        evaluation = env.evaluate()
        success = evaluation.get("success", False)

        steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
        planner.close()
        return [{"success": success, "elapsed_steps": steps}]

    except Exception as e:
        print(f"[PickCube solver] Error: {e}")
        import traceback
        traceback.print_exc()
        steps = _get_elapsed_steps(env, getattr(planner, "steps", 0))
        planner.close()
        return [{"success": torch.tensor(False, dtype=torch.bool), "elapsed_steps": steps}]