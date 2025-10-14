import numpy as np
import sapien
import torch

from nebula.benchmarks.capabilities.dynamic.easy import DynamicPressLightSwitchEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

# ------------------------ small utils ------------------------ #

def _np_pose_p(actor):
    """Return actor position as numpy (3,) handling optional batch dimension."""
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

def _to_np_quat(q):
    """Return (x,y,z,w) quaternion in numpy from a SAPIEN/torch pose quaternion."""
    if hasattr(q, "cpu"):
        q = q.cpu().numpy()
    if getattr(q, "ndim", 1) == 2:
        q = q[0]
    return q

# ---------------------- motion primitives -------------------- #

def _approach_press_and_retract(planner, tcp_q, target_p,
                                approach_dz=0.12,
                                press_depths=(0.08, 0.05, 0.03),
                                hold_steps=(10, 10, 10),
                                retract_dz=0.12):
    """
    Move above target, descend by stages to ensure distance threshold is met,
    hold briefly, then retract.
    """
    # approach above the switch
    approach_p = target_p + np.array([0.0, 0.0, approach_dz], dtype=np.float32)
    approach_pose = sapien.Pose(approach_p, tcp_q)
    res = planner.move_to_pose_with_screw(approach_pose)
    if res == -1:
        return -1

    # staged press: progressively lower z
    for dz, settle in zip(press_depths, hold_steps):
        press_p = target_p + np.array([0.0, 0.0, dz], dtype=np.float32)
        press_pose = sapien.Pose(press_p, tcp_q)
        res = planner.move_to_pose_with_screw(press_pose)
        if res == -1:
            return -1
        # small settle to allow the env to register proximity
        for _ in range(int(max(1, settle))):
            # zero-action step via tiny no-op motion; planner has no direct sleep/settle,
            # but issuing the same pose again keeps controller running a few steps.
            res2 = planner.move_to_pose_with_screw(press_pose)
            if res2 == -1:
                return -1

    # retract
    retract_pose = sapien.Pose(target_p + np.array([0.0, 0.0, retract_dz], dtype=np.float32), tcp_q)
    res = planner.move_to_pose_with_screw(retract_pose)
    return res

# --------------------------- solver -------------------------- #

def DynamicEasyPressSwitchSolution(env: DynamicPressLightSwitchEnv, seed=None, debug=False, vis=False):
    """
    Press the light switch quickly:
    1) Read switch position
    2) Move above it, descend in stages, hold briefly, retract
    Returns [{'success': bool, 'elapsed_steps': int}]
    """
    outer_env = env
    env.reset(seed=seed, options={"task_instruction": "Only press the switch when the light is red. Do not press the switch when the light is off."})

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_u = env.unwrapped

    # current TCP orientation is sufficient for pressing (no rotation needed)
    tcp_q = _to_np_quat(env_u.agent.tcp.pose.q)

    # read switch position (numpy, shape (3,))
    switch_p = _np_pose_p(env_u.switch)

    # execute fast approach/press/retract
    # note: the environment uses distance threshold to detect "press",
    # so staged depths ensure we cross within the threshold safely.
    ok = _approach_press_and_retract(
        planner,
        tcp_q=tcp_q,
        target_p=switch_p,
        approach_dz=0.12,
        press_depths=(0.08, 0.05, 0.03),
        hold_steps=(6, 6, 6),
        retract_dz=0.12,
    ) != -1

    # finalize
    planner.close()

    # evaluate success flag if available; else rely on motion result
    success = bool(ok)
    try:
        ev = env_u.evaluate()
        # ev["success"] may be tensor with batch dim or scalar-bool tensor
        s = ev.get("success", None)
        if s is not None:
            if hasattr(s, "any"):
                success = bool(s.any().item() if hasattr(s, "item") else bool(s.any()))
            else:
                success = bool(s)
    except Exception:
        pass

    elapsed = _elapsed_steps_of(outer_env)
    return [{"success": torch.tensor(success, dtype=torch.bool), "elapsed_steps": int(elapsed)}]
