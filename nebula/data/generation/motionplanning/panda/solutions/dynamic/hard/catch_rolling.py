import numpy as np
import sapien
from transforms3d.quaternions import axangle2quat, qmult
import torch

from nebula.benchmarks.capabilities.dynamic.hard import DynamicHardCatchRollingSphereEnv
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

# --------------------- motion primitives --------------------- #

def move_to_pose_with_screw_fast(planner, pose, dry_run=False, speed_multiplier=8.0):
    from nebula.data_collection.motionplanning.panda.motionplanner import to_sapien_pose
    pose = to_sapien_pose(pose)
    
    if planner.grasp_pose_visual is not None:
        planner.grasp_pose_visual.set_pose(pose)
    
    pose = sapien.Pose(p=pose.p, q=pose.q)
    fast_time_step = planner.base_env.control_timestep * speed_multiplier
    
    result = planner.planner.plan_screw(
        np.concatenate([pose.p, pose.q]),
        planner.robot.get_qpos().cpu().numpy()[0],
        time_step=fast_time_step,
        use_point_cloud=False,
    )
    
    if result["status"] != "Success":
        result = planner.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            planner.robot.get_qpos().cpu().numpy()[0],
            time_step=fast_time_step,
            use_point_cloud=False,
            planner_name="RRTConnect",
        )
    
    if result["status"] != "Success":
        return -1
    
    planner.render_wait()
    if dry_run:
        return result
    
    return planner.follow_path(result, refine_steps=1)

def _move_fast(planner, p, q, speed_multiplier=8.0):
    return move_to_pose_with_screw_fast(planner, sapien.Pose(p, q), speed_multiplier=speed_multiplier)

def _move(planner, p, q):
    return planner.move_to_pose_with_screw(sapien.Pose(p, q))

def _repeat_hold(planner, p, q, steps=3):
    pose = sapien.Pose(p, q)
    for _ in range(int(max(1, steps))):
        if planner.move_to_pose_with_screw(pose) == -1:
            return -1
    return 0

def predict_future_pos(pos, vel, horizon=0.15):
    vel_norm = np.linalg.norm(vel[:2])
    if vel_norm < 1e-4:
        return pos
    
    decay_factor = 0.85
    predicted = pos.copy()
    predicted[:2] += vel[:2] * horizon * decay_factor
    return predicted

# --------------------------- solver -------------------------- #

def DynamicHardCatchRollingSphereSolution(env: DynamicHardCatchRollingSphereEnv, seed=None, debug=False, vis=False):
    """
    DynamicHard Rolling Sphere solver: Wait for green light, then fast sphere capture.
    
    Strategy:
    1. Wait and track sphere until light turns green (25 steps)
    2. Fast capture with trajectory-aligned gripper 
    3. Place in bin and verify success
    """
    env.reset(seed=seed, options={"task_instruction": "Catch the rolling sphere into the shallow bin, but only when the light turns green."})

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_u = env.unwrapped
    
    # Helper to execute steps
    def execute_hold_steps(num_steps=3):
        """Execute holding steps to let simulation progress"""
        for _ in range(num_steps):
            qpos = env_u.agent.robot.get_qpos().cpu().numpy().flatten()
            action = qpos[:env.action_space.shape[0]]
            env.step(action)

    try:
        if debug:
            print(f"Starting catch rolling sphere task...")
            print(f"Waiting for green light...")
        
        # Phase 1: Wait for green light while tracking sphere
        LIGHT_ACTIVATION_STEPS = 25
        current_step = 0
        light_is_green = False
        
        # Keep robot at safe distance initially
        safe_pos = np.array([-0.4, 0.0, 0.15], dtype=np.float32)
        safe_q = np.array([0, 1, 0, 0], dtype=np.float32)
        planner.move_to_pose_with_screw(sapien.Pose(safe_pos, safe_q))
        
        # Wait for light activation while doing minimal tracking
        while current_step < LIGHT_ACTIVATION_STEPS:
            # Execute waiting action (hold current position)
            qpos = env_u.agent.robot.get_qpos().cpu().numpy().flatten()
            action = qpos[:env.action_space.shape[0]]
            obs, reward, terminated, truncated, info = env.step(action)
            current_step += 1
            
            # Check if light turned green early (shouldn't happen but be safe)
            eval_result = env_u.evaluate()
            if "is_light_green" in eval_result and eval_result["is_light_green"].any():
                light_is_green = True
                break
                
            # Check for early termination
            if terminated or truncated:
                break
        
        # Verify light is actually green
        eval_result = env_u.evaluate()
        if "is_light_green" in eval_result:
            light_is_green = eval_result["is_light_green"].any()
        
        if not light_is_green:
            # Light hasn't turned green, continue waiting
            max_extra_wait = 10
            wait_count = 0
            while not light_is_green and wait_count < max_extra_wait:
                qpos = env_u.agent.robot.get_qpos().cpu().numpy().flatten()
                action = qpos[:env.action_space.shape[0]]
                obs, reward, terminated, truncated, info = env.step(action)
                
                eval_result = env_u.evaluate()
                if "is_light_green" in eval_result:
                    light_is_green = eval_result["is_light_green"].any()
                    
                wait_count += 1
        
        if not light_is_green:
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]
        
        # Phase 2: Light is green - capture the rolling sphere
        if debug:
            print(f"Green light! Capturing sphere...")
        
        max_capture_attempts = 8
        sphere_captured = False

        for attempt in range(max_capture_attempts):
            # Get current sphere state
            sphere_pos = _np_pos(env_u.obj)
            sphere_vel = env_u.obj.linear_velocity.cpu().numpy().flatten()[:3]
            sphere_speed = np.linalg.norm(sphere_vel[:2])

            # Skip if sphere is too far from robot workspace
            robot_pos = env_u.agent.robot.pose.p.cpu().numpy().flatten()[:3]
            distance_to_ball = np.linalg.norm(sphere_pos[:2] - robot_pos[:2])
            if distance_to_ball > 0.7:
                # Wait a few steps for sphere to move closer
                execute_hold_steps(5)
                continue

            # Use OBB-based grasping with prediction for moving sphere
            # Get the OBB of the rolling sphere
            obb = get_actor_obb(env_u.obj)
            
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
            
            # Predict future position for moving target
            horizon = 0.08 + sphere_speed * 0.05  # Shorter horizon for better accuracy
            predicted_pos = predict_future_pos(sphere_pos, sphere_vel, horizon)
            
            # Adjust grasp position based on prediction
            closing[:2] = predicted_pos[:2]  # Update x,y with predicted position
            closing[2] = max(closing[2], 0.005)  # Ensure minimum height
            
            # Approach above the predicted position
            approach_pos = closing.copy()
            approach_pos[2] += 0.05  # Stay above sphere before descent

            result = planner.move_to_pose_with_screw(sapien.Pose(approach_pos, [0, 1, 0, 0]))
            if result == -1:
                continue

            # Open gripper before descent
            planner.open_gripper()

            # Descend to grasp position
            result = planner.move_to_pose_with_screw(sapien.Pose(closing, [0, 1, 0, 0]))
            if result == -1:
                # Try slightly higher if failed
                closing[2] = 0.01
                result = planner.move_to_pose_with_screw(sapien.Pose(closing, [0, 1, 0, 0]))
                if result == -1:
                    continue

            # Close gripper and hold
            planner.close_gripper()
            execute_hold_steps(5)

            # Check capture success with multiple validation steps
            grasp_confirmed = False
            for check_step in range(5):
                if env_u.agent.is_grasping(env_u.obj).any():
                    grasp_confirmed = True
                    break
                # Give physics engine time to update
                execute_hold_steps(1)

            if grasp_confirmed:
                # Lift to secure the grasp
                secure_pos = closing.copy()
                secure_pos[2] += 0.05
                planner.move_to_pose_with_screw(sapien.Pose(secure_pos, [0, 1, 0, 0]))
                sphere_captured = True
                break
            else:
                # Retry with next attempt
                planner.open_gripper()

        if not sphere_captured:
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]

        # Phase 3: Transport to bin
        bin_pos = _np_pos(env_u.bin)
        bin_center_pos = bin_pos.copy()
        bin_center_pos[2] = bin_pos[2] + env_u.block_half_size[0] + env_u.radius
        
        # Approach bin
        bin_approach_pos = bin_center_pos.copy()
        bin_approach_pos[2] += 0.08
        
        result = planner.move_to_pose_with_screw(sapien.Pose(bin_approach_pos, [0, 1, 0, 0]))
        if result == -1:
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]
        
        # Phase 4: Placement
        result = planner.move_to_pose_with_screw(sapien.Pose(bin_center_pos, [0, 1, 0, 0]))
        if result == -1:
            planner.close()
            return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]
        
        execute_hold_steps(4)
        
        # Phase 5: Release
        planner.open_gripper()
        
        # Retreat
        retreat_pos = bin_center_pos.copy()
        retreat_pos[2] += 0.06
        retreat_pos[0] += 0.04
        planner.move_to_pose_with_screw(sapien.Pose(retreat_pos, [0, 1, 0, 0]))
        execute_hold_steps(5)
        
        # Phase 6: Success verification
        success = False
        max_check_steps = 30
        check_count = 0

        while check_count < max_check_steps and not success:
            # Hold steady and wait for sphere to settle
            execute_hold_steps(2)
            
            # Check success from evaluation
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
        
        if debug:
            print(f"Task completed. Success: {success}, Steps: {_elapsed_steps_of(outer_env)}")
        
        return [{"success": torch.tensor(success), "elapsed_steps": _elapsed_steps_of(outer_env)}]

    except Exception as e:
        print(f"Error in catch_rolling solution: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        try:
            planner.close()
        except:
            pass
        return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(outer_env)}]