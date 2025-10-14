import numpy as np
import sapien
from transforms3d.quaternions import axangle2quat, qmult
import torch

from nebula.benchmarks.capabilities.dynamic.hard import DynamicHardRollBallEnv
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

def move_to_pose_with_screw_fast(planner, pose, dry_run=False, refine_steps=0, speed_multiplier=8.0):
    from nebula.data.generation.motionplanning.panda.motionplanner import to_sapien_pose
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
        result = planner.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            planner.robot.get_qpos().cpu().numpy()[0],
            time_step=fast_time_step,
            use_point_cloud=False,
        )
        if result["status"] != "Success":
                print(result["status"])
                planner.render_wait()
                return -1
    
    if result["status"] != "Success":
        print(f"Fast planning failed: {result['status']}")
        return -1
    
    planner.render_wait()
    if dry_run:
        return result
    
    return planner.follow_path(result, refine_steps=1)

def _move_fast(planner, p, q, speed_multiplier=8.0):
    return move_to_pose_with_screw_fast(planner, sapien.Pose(p, q), speed_multiplier=speed_multiplier)

# --------------------------- solver -------------------------- #

def _move(planner, p, q):
    """Normal speed movement function"""
    return planner.move_to_pose_with_screw(sapien.Pose(p, q))

def _repeat_hold(planner, p, q, steps=6):
    """Normal version of hold operation"""
    pose = sapien.Pose(p, q)
    for _ in range(int(max(1, steps))):
        if planner.move_to_pose_with_screw(pose) == -1:
            return -1
    return 0

def DynamicHardRollBallSolution(env: DynamicHardRollBallEnv, seed=None, debug=False, vis=False):
    """
    RollBall solver: Only uses fast movement for ball pushing
    Other steps maintain normal speed to ensure stability
    """
    env.reset(seed=seed, options={"task_instruction": "Roll a ball to a target position while avoiding a bouncing ball that crosses the workspace."})

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

    if debug:
        print(f"Starting roll ball task solution...")
    
    # Get ball and goal positions
    ball_pos = _np_pos(env_u.ball)
    goal_pos = _np_pos(env_u.goal_region)
    tcp_q = _np_quat(env_u.agent.tcp.pose.q)

    # Calculate push direction: from ball toward goal
    ball_to_goal = goal_pos[:2] - ball_pos[:2]
    push_distance = np.linalg.norm(ball_to_goal)

    if push_distance < 0.01:
        planner.close()
        return [{"success": torch.tensor(True), "elapsed_steps": _elapsed_steps_of(env)}]

    push_direction = ball_to_goal / push_distance

    behind_distance = 0.05
    tcp_behind_ball = ball_pos[:2] - push_direction * behind_distance

    # Heights
    ball_center_z = ball_pos[2]
    approach_z = ball_center_z + 0.10
    contact_z = ball_center_z

    # Waypoints
    approach_pos = np.array([tcp_behind_ball[0], tcp_behind_ball[1], approach_z], dtype=np.float32)
    start_pos = np.array([tcp_behind_ball[0], tcp_behind_ball[1], contact_z], dtype=np.float32)
    
    push_through_distance = 0.15
    push_target_2d = ball_pos[:2] + push_direction * push_through_distance
    end_pos = np.array([push_target_2d[0], push_target_2d[1], contact_z], dtype=np.float32)

    try:
        result = _move(planner, approach_pos, tcp_q)
        planner.close_gripper()
        result = _move(planner, start_pos, tcp_q)
        
        dx = ball_pos[0] - goal_pos[0]
        dy = ball_pos[1] - goal_pos[1]
        angle = np.arctan2(dx, dy)
        z_rotation_quat = axangle2quat(np.array([0, 0, 1]), angle)
        tcp_q_ = qmult(tcp_q, z_rotation_quat)
        result = _move(planner,start_pos, tcp_q_)

        result = _move_fast(planner, end_pos, tcp_q_, speed_multiplier=5)
        execute_hold_steps(3)
    
        success = False
        max_check_steps = 50
        check_count = 0

        try:
            while check_count < max_check_steps and not success:
                # Hold steady and wait for ball to settle
                execute_hold_steps(2)
                
                # Check success from evaluation
                try:
                    eval_result = env_u.evaluate()
                    if "success" in eval_result:
                        success_tensor = eval_result["success"]
                        if hasattr(success_tensor, "any"):
                            current_success = bool(success_tensor.any())
                        else:
                            current_success = bool(success_tensor)
                        
                        if current_success:
                            success = True
                            break
                            
                except Exception:
                    pass
                    
                check_count += 1
                    
        except Exception:
            success = False

        planner.close()
        
        if debug:
            print(f"Task completed. Success: {success}, Steps: {_elapsed_steps_of(env) + check_count}")

        return [{"success": torch.tensor(success), "elapsed_steps": _elapsed_steps_of(env) + check_count}]

    except Exception as e:
        print(f"Error in DynamicHardRollBallSolution: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        planner.close()
        return [{"success": torch.tensor(False), "elapsed_steps": _elapsed_steps_of(env)}]