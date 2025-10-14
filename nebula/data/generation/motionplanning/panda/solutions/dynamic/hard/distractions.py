import numpy as np
import sapien

from nebula.benchmarks.capabilities.dynamic.hard import DynamicHardDistractorBallPickCubeEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb
)

def DynamicHardDistractorBallPickCubeSolution(env: DynamicHardDistractorBallPickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Pick up the red cube and lift it."})
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
    
    # Target the red cube (ignore the orange ball distractor)
    target_cube = env.target_cube
    
    # Get grasp pose for the red cube
    obb = get_actor_obb(target_cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, target_cube.pose.sp.p)
    
    # Reach - approach the red cube
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    
    # Grasp the red cube
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    
    # Lift the red cube
    lift_pose = sapien.Pose([0, 0, 0.12]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    
    planner.close()
    return res