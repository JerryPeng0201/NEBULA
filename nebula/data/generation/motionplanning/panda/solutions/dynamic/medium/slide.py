import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from nebula.benchmarks.capabilities.dynamic.medium import DynamicMediumSlidingPickCubeEnv
from nebula.data.generation.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

# ------------------------ helpers ------------------------ #

def _np_pos(actor):
    """Return actor position as numpy (3,), handling optional batch dim."""
    p = actor.pose.p
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    if getattr(p, "ndim", 1) == 2:
        p = p[0]
    return p

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

def DynamicMediumSlidingPickCubeSolution(env: DynamicMediumSlidingPickCubeEnv, seed=None, debug=False, vis=False):
    """
    Ultra simple solver: Just pick up the cube from wherever it ends up
    """
    outer_env = env
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

    try:
        tcp_q = euler2quat(np.pi, 0, 0)
        for step in range(20):
            qpos = env_u.agent.robot.get_qpos().cpu().numpy().flatten()
            action = qpos[:env.action_space.shape[0]]
            env.step(action)

        # Get cube position
        cube_pos = _np_pos(env_u.cube)

        # Open gripper first
        planner.open_gripper()

        # Move to safe position above table center
        safe_pos = np.array([0.0, 0.0, 0.2])
        result = planner.move_to_pose_with_screw(sapien.Pose(safe_pos, tcp_q))

        if result == -1:
            home_pos = np.array([-0.1, 0.0, 0.3])
            result = planner.move_to_pose_with_screw(sapien.Pose(home_pos, tcp_q))
        # Check if cube position is reachable by constraining it to workspace
        cube_pos_safe = cube_pos.copy()
        
        # Constrain cube position to robot workspace
        cube_pos_safe[0] = max(-0.2, min(0.2, cube_pos[0]))  # X: -20cm to +20cm
        cube_pos_safe[1] = max(-0.15, min(0.15, cube_pos[1]))  # Y: -15cm to +15cm
        
        # Move above safe cube position
        above_cube = cube_pos_safe.copy()
        above_cube[2] = cube_pos_safe[2] + 0.1  # 10cm above
        
        result = planner.move_to_pose_with_screw(sapien.Pose(above_cube, tcp_q))

        if result == -1:
            above_cube[0] = max(-0.15, min(0.15, cube_pos_safe[0] * 0.7))  # Scale back towards robot
            above_cube[1] = cube_pos_safe[1] * 0.7  # Scale back towards center
            above_cube[2] = cube_pos_safe[2] + 0.12  # Higher
            
            result = planner.move_to_pose_with_screw(sapien.Pose(above_cube, tcp_q))
            
            if result == -1:
                above_cube = np.array([0.05, 0.0, 0.15])  # Very close to robot base
                result = planner.move_to_pose_with_screw(sapien.Pose(above_cube, tcp_q))

        # Use the safe cube position for grasping too
        grasp_pos = cube_pos_safe.copy()
        grasp_pos[2] = cube_pos_safe[2] + 0.025  # Just above cube
        
        result = planner.move_to_pose_with_screw(sapien.Pose(grasp_pos, tcp_q))
        
        if result == -1:
            # Try approaching from the side
            grasp_pos[0] = grasp_pos[0] - 0.03  # Slightly towards robot
            grasp_pos[2] = cube_pos_safe[2] + 0.04  # A bit higher
            
            result = planner.move_to_pose_with_screw(sapien.Pose(grasp_pos, tcp_q))

        planner.close_gripper()
        
        # Hold position briefly
        for _ in range(5):
            qpos = env_u.agent.robot.get_qpos().cpu().numpy().flatten()
            action = qpos[:env.action_space.shape[0]]
            env.step(action)

        lift_pos = cube_pos_safe.copy()
        lift_pos[2] = cube_pos_safe[2] + 0.08 # Lift 8cm up
    
        result = planner.move_to_pose_with_screw(sapien.Pose(lift_pos, tcp_q))
        
        if result == -1:
            lift_pos[2] = cube_pos_safe[2] + 0.05
            result = planner.move_to_pose_with_screw(sapien.Pose(lift_pos, tcp_q))
           
        success = False
        for check_step in range(10):
            qpos = env_u.agent.robot.get_qpos().cpu().numpy().flatten()
            action = qpos[:env.action_space.shape[0]]
            obs, reward, terminated, truncated, info = env.step(action)
            
            try:
                eval_result = env_u.evaluate()
                if "success" in eval_result:
                    success = bool(eval_result["success"])
                    if success:
                        break
            except Exception as e:
                print(f"Evaluation error: {e}")
                
            if terminated.any() if hasattr(terminated, "any") else terminated:
                break

        planner.close()
        return [{"success": torch.tensor(success), "elapsed_steps": torch.tensor(_elapsed_steps_of(outer_env))}]

    except Exception as e:
        print(f"Error in solve: {e}")
        import traceback
        traceback.print_exc()
        planner.close()
        return [{"success": torch.tensor(False), "elapsed_steps": torch.tensor(_elapsed_steps_of(outer_env))}]