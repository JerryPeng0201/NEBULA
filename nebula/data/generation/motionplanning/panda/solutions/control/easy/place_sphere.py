import numpy as np
import sapien
from transforms3d.euler import euler2quat

from nebula.benchmarks.capabilities.control.easy.place_sphere import ControlPlaceSphereEasyEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def ControlEasyPlaceSphereSolution(env: ControlPlaceSphereEasyEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={"task_instruction": "Pick up the blue sphere and place it into the bin"})
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

    
    sphere = env.obj
    bin_target = env.bin


    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(sphere)

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
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, sphere.pose.sp.p)
    
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
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
   
    # -------------------------------------------------------------------------- #
    # Reach to bin with proper height adjustment
    # -------------------------------------------------------------------------- #

    # Calculate position above bin with proper clearance
    bin_height = 0.01  # Adjust based on your bin height
    clearance = 0.01  # Safety clearance above bin
    reach_position = bin_target.pose.sp.p.copy()
    reach_position[2] += bin_height + clearance  # Adjust Z coordinate

    # Create reach pose maintaining grasp orientation
    reach_pose = sapien.Pose(reach_position, grasp_pose.q)
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Place in bin
    # -------------------------------------------------------------------------- #
    res = planner.open_gripper()

    planner.close()
    return res

