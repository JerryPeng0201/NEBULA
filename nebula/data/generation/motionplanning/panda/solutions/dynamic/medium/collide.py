import numpy as np
import sapien
import torch

from nebula.benchmarks.capabilities.dynamic.medium import DynamicMediumPickCubeWithCollisionEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

# ------------------------ helpers ------------------------ #

def _np_pos(actor):
    """Return actor position as numpy (3,), handling optional batch dim."""
    p = actor.pose.p
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    if getattr(p, "ndim", 1) == 2:
        p = p[0]
    return p

def _np_quat(q):
    """Return quaternion (x,y,z,w) as numpy (4,), handling optional batch dim."""
    if hasattr(q, "cpu"):
        q = q.cpu().numpy()
    if getattr(q, "ndim", 1) == 2:
        q = q[0]
    return q

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

def DynamicMediumCollideSolution(env: DynamicMediumPickCubeWithCollisionEnv, seed=None, debug=False, vis=False):
    """
    DynamicMedium Pick Cube With Collision solver: Pick the red cube and lift it above minimum height.
    During the task, a ball will hit the cube once to test the robot's adaptability to dynamic disturbances.
    
    Strategy:
    1. Quickly grasp the cube before ball interference
    2. React to collision and regain control if needed
    3. Lift the cube above minimum height and hold steady
    """
    outer_env = env
    env.reset(seed=seed, options={"task_instruction": "Pick up the red cube and lift it."})

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
    
    # Helper to execute steps
    def execute_hold_steps(num_steps=3):
        """Execute holding steps to let simulation progress"""
        for _ in range(num_steps):
            qpos = env_u.agent.robot.get_qpos().cpu().numpy().flatten()
            action = qpos[:env.action_space.shape[0]]
            env.step(action)

    try:
        # Get target cube
        target_cube = env_u.cube
        
        if debug:
            print(f"Starting collision task solution...")
            print(f"Target cube position: {_np_pos(target_cube)}")
        
        # -------------------------------------------------------------------------- #
        # Phase 1: Quick grasp before interference
        # -------------------------------------------------------------------------- #
        
        # Get the OBB of the target cube
        obb = get_actor_obb(target_cube)
        
        # Compute grasp pose using OBB
        approaching = np.array([0, 0, -1])
        target_closing = env_u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
        )
        
        closing = grasp_info["closing"]
        center = grasp_info["center"]
        approaching = closing.copy()
        approaching[2] += 0.05  # Approach from above
        
        # Move to approach position (fast)
        result = planner.move_to_pose_with_screw(sapien.Pose(approaching, [0, 1, 0, 0]))
        if result == -1:
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]
        
        # Open gripper
        planner.open_gripper()
        
        # Move to grasp position
        result = planner.move_to_pose_with_screw(sapien.Pose(closing, [0, 1, 0, 0]))
        if result == -1:
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]
        
        # Close gripper to grasp
        planner.close_gripper()
        
        # Hold position to secure grasp
        execute_hold_steps(5)
        
        # -------------------------------------------------------------------------- #
        # Phase 2: Lift and adapt to collision
        # -------------------------------------------------------------------------- #
        
        # Check if grasped
        if not env_u.agent.is_grasping(target_cube).any():
            # Try to regrasp if lost
            planner.open_gripper()
            result = planner.move_to_pose_with_screw(sapien.Pose(closing, [0, 1, 0, 0]))
            if result == -1:
                planner.close()
                return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]
            planner.close_gripper()
            execute_hold_steps(5)
        
        # Lift the cube above minimum height
        lift_height = 0.08  # Above the 0.05m threshold
        lift_pos = closing.copy()
        lift_pos[2] = lift_height
        
        result = planner.move_to_pose_with_screw(sapien.Pose(lift_pos, [0, 1, 0, 0]))
        if result == -1:
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]
        
        # Hold steady at lift height (maintain control during and after interference)
        # Execute enough steps for ball to launch and potentially hit
        for i in range(15):
            # Check if still grasping
            if not env_u.agent.is_grasping(target_cube).any():
                # Lost grasp due to collision - try to regain
                cube_pos = _np_pos(target_cube)
                
                # Recompute grasp with updated OBB
                obb = get_actor_obb(target_cube)
                target_closing = env_u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
                
                grasp_info = compute_grasp_info_by_obb(
                    obb,
                    approaching=np.array([0, 0, -1]),
                    target_closing=target_closing,
                )
                
                closing = grasp_info["closing"]
                center = grasp_info["center"]
                
                # Open gripper
                planner.open_gripper()
                
                # Move to new grasp position
                result = planner.move_to_pose_with_screw(sapien.Pose(closing, [0, 1, 0, 0]))
                if result == -1:
                    # If can't regain, just keep trying to hold steady
                    execute_hold_steps(3)
                    continue
                
                # Close gripper
                planner.close_gripper()
                execute_hold_steps(3)
                
                # Re-lift
                lift_pos = closing.copy()
                lift_pos[2] = lift_height
                result = planner.move_to_pose_with_screw(sapien.Pose(lift_pos, [0, 1, 0, 0]))
                if result == -1:
                    execute_hold_steps(3)
                    continue
            else:
                # Maintain position - hold steady
                execute_hold_steps(3)
        
        # -------------------------------------------------------------------------- #
        # Phase 3: Verify success
        # -------------------------------------------------------------------------- #
        
        success = False
        max_check_steps = 30
        check_count = 0

        while check_count < max_check_steps and not success:
            # Hold steady position
            execute_hold_steps(2)
            
            # Check success
            try:
                eval_result = env_u.evaluate()
                if "success" in eval_result:
                    success_tensor = eval_result["success"]
                    success = bool(success_tensor.any() if hasattr(success_tensor, "any") else success_tensor)
                    if success:
                        break
            except Exception:
                pass
            
            check_count += 1

        planner.close()
        return [{"success": torch.tensor(success), "elapsed_steps": _elapsed_steps_of(outer_env)}]

    except Exception as e:
        print(f"Error in DynamicMediumCollideSolution: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        try:
            planner.close()
        except:
            pass
        return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]