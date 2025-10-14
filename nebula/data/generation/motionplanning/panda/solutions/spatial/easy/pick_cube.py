import numpy as np
import sapien
from transforms3d.euler import euler2quat
import random
from nebula.envs.tasks.capabilities.spatial.easy.pick_cube import SpatialEasyPickCubeEnv
from nebula.data_collection.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data_collection.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: SpatialEasyPickCubeEnv, seed=None, debug=False, vis=False):

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
    env = env.unwrapped

   
    cube = env.cubes[env.target_color]


    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, cube.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])

    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(lift_pose)
  


   
    # -------------------------------------------------------------------------- #
    # Place in target position
    # -------------------------------------------------------------------------- #
    target = env.cubes['blue']
    reach_position = target.pose.sp.p.copy()
    reach_pose = sapien.Pose(reach_position, grasp_pose.q)
    res = planner.move_to_pose_with_screw(reach_pose)
    
    planner.open_gripper()

    planner.close()
    
    return res

