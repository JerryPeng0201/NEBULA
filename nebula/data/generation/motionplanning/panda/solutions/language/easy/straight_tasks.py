import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from nebula.benchmarks.capabilities.language.easy.straight_tasks import LanguageStraightEasyEnv
from nebula.data.generation.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from nebula.data.generation.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb
)

# ------------------------ helpers ------------------------ #

def _elapsed_steps_of(env_like):
    """Best-effort to read elapsed steps from possibly wrapped env."""
    visited = set()
    cur = env_like
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        for attr in ("_elapsed_steps", "elapsed_steps"):
            if hasattr(cur, attr):
                try:
                    return int(getattr(cur, attr))
                except Exception:
                    pass
        cur = getattr(cur, "env", None)
    return 0

# --------------------------- solver -------------------------- #

def LanguageStraightEasySolution(env: LanguageStraightEasyEnv, seed=None, debug=False, vis=False):
    """
    Language Easy Task solver: Grasp the red block/cube based on language command.
    
    Task: "{Grab/Pick/Select} red block"
    
    Strategy:
    1. Recognize the language command (different verbs for same action)
    2. Use OBB-based grasp planning for robust grasping
    3. Grasp and lift the red cube (the target is always red cube)
    4. Verify successful completion
    """
    outer_env = env
    # Reset environment - task_instruction will be set automatically during _initialize_episode
    env.reset(seed=seed)

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_u = env.unwrapped
    FINGER_LENGTH = 0.025

    try:
        # Get target object - always the red cube for this task
        target_obj = env_u.red_cube
        
        # -------------------------------------------------------------------------- #
        # Compute grasp pose using OBB
        # -------------------------------------------------------------------------- #
        
        # Get object oriented bounding box
        obb = get_actor_obb(target_obj)
        approaching = np.array([0, 0, -1])  # Approach from above
        
        # Get TCP closing direction
        target_closing = env_u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        
        # Compute grasp information
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = env_u.agent.build_grasp_pose(approaching, closing, target_obj.pose.sp.p)

        # -------------------------------------------------------------------------- #
        # Phase 1: Reach (approach position above target)
        # -------------------------------------------------------------------------- #
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        result = planner.move_to_pose_with_screw(reach_pose)
        if result == -1:
            print("Failed to reach red cube")
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]

        # -------------------------------------------------------------------------- #
        # Phase 2: Open gripper before grasping
        # -------------------------------------------------------------------------- #
        planner.open_gripper()

        # -------------------------------------------------------------------------- #
        # Phase 3: Grasp the red cube
        # -------------------------------------------------------------------------- #
        result = planner.move_to_pose_with_screw(grasp_pose)
        if result == -1:
            print("Failed to grasp red cube")
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]
        
        planner.close_gripper()
        # Allow gripper to close
        for _ in range(3):
            qpos = env_u.agent.robot.get_qpos().cpu().numpy().flatten()
            action = qpos[:env.action_space.shape[0]]
            env.step(action)

        # -------------------------------------------------------------------------- #
        # Phase 4: Lift the red cube
        # -------------------------------------------------------------------------- #
        lift_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
        result = planner.move_to_pose_with_screw(lift_pose)
        if result == -1:
            print("Failed to lift red cube")
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]

        planner.close()
        
        # Evaluate success
        info = env_u.evaluate()
        success_status = info["success"] if isinstance(info["success"], torch.Tensor) else torch.tensor(info["success"])
        
        return [{"success": success_status, "elapsed_steps": _elapsed_steps_of(outer_env)}]

    except Exception as e:
        print(f"Exception occurred: {e}")
        try:
            planner.close()
        except:
            pass
        return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]

