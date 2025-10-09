import numpy as np
import sapien
from transforms3d.euler import euler2quat

from nebula.benchmarks.capabilities.control.easy.push_cube import ControlPushCubeEasyEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_push_info_by_obb, get_actor_obb

def ControlPushCubeEasySolution(env: ControlPushCubeEasyEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Push the blue cube to the target region"})
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

    cube = env.obj
    target = env.goal_region

    # Get positions
    cube_pos = cube.pose.sp.p
    target_pos = target.pose.sp.p
    push_direction = target_pos - cube_pos
    push_direction[2] = 0
    push_direction = push_direction / np.linalg.norm(push_direction)

    # Close gripper for pushing
    planner.close_gripper()

    # Get OBB and compute push info
    obb = get_actor_obb(cube)
    approaching = np.array([0, 0, -1])
    target_closing = push_direction
    push_info = compute_push_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing = push_info["closing"]

    # Compute start position behind the cube
    push_height = cube_pos[2] - 0.01
    start_pos = cube_pos - push_direction * 0.08
    start_pos[2] = push_height
    push_start_pose = env.agent.build_grasp_pose(approaching, closing, start_pos)

    # Move to start position
    planner.move_to_pose_with_screw(push_start_pose)

    # Push to target (slightly before)
    push_end = target_pos - push_direction * 0.02
    push_end[2] = push_height
    push_end_pose = env.agent.build_grasp_pose(approaching, closing, push_end)
    res = planner.move_to_pose_with_screw(push_end_pose)
    

    # Move up to safe position
    final_pos = push_end + np.array([0, 0, 0.1])
    final_pose = env.agent.build_grasp_pose(approaching, closing, final_pos)
    planner.move_to_pose_with_screw(final_pose)

    planner.close()
    return res

