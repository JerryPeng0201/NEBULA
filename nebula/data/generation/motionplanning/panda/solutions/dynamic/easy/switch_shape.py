import numpy as np
import sapien

from nebula.benchmarks.capabilities.dynamic.easy import DynamicShapeSwitchPickEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb
)

def DynamicEasySwitchShapePickSolution(env: DynamicShapeSwitchPickEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Pick the cube and lift it up"})
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
    
    # Get switch timing from environment
    switch_step = env.shape_switch_state['switch_step'][0].item()
    
    # Phase 1: Approach the currently cube-shaped object (target_cube)
    # Before shape switch, this is the cube-shaped object
    target_object = env.target_cube
    
    obb = get_actor_obb(target_object)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, target_object.pose.sp.p)
    
    # Reach - approach the cube gradually until shape switch
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    
    # Phase 2: Shape switch happened, redirect to the NEW cube-shaped object
    # After shape switch, distractor_cube is now the cube-shaped object and should be the new target
    target_object = env.distractor_cube
    
    # Recalculate grasp pose for the new cube-shaped object location
    obb = get_actor_obb(target_object)
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, target_object.pose.sp.p)
    
    # Reach new cube-shaped object (redirect to different position)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    
    # Grasp
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    
    # Lift
    lift_pose = sapien.Pose([0, 0, 0.12]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    
    planner.close()
    return res