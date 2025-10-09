import numpy as np
import sapien
from transforms3d.euler import euler2quat

from nebula.benchmarks.capabilities.control.hard.place_sphere import ControlPlaceSphereHardEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb


def ControlPlaceSphereHardSolution(env: ControlPlaceSphereHardEnv, seed=None, debug=False, vis=False):
    """
    Solve the ControlPlaceSphereHardEnv by placing the sphere sequentially into:
    1. Yellow bin (center, y=0)
    2. Red bin (left, y=-0.067)
    3. Blue bin (right, y=0.067)
    
    Args:
        env: The environment instance
        seed: Random seed for reproducibility
        debug: Enable debug output
        vis: Enable visualization
    
    Returns:
        Motion planning result status
    """
    task_instruction = "Pick up the blue sphere and place it into the yellow bin, then move it to the red bin, then move it to the blue bin"
    env.reset(seed=seed, options={"task_instruction": task_instruction})
    
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

    # Get references to the sphere and bins
    sphere = env.obj
    yellow_bin = env.yellow_bin  # Center (y=0)
    red_bin = env.red_bin         # Left (y=-0.067)
    blue_bin = env.blue_bin        # Right (y=0.067)

    # ========================================================================== #
    # STAGE 1: Pick up sphere and place in YELLOW bin
    # ========================================================================== #
    
    # Get object oriented bounding box for initial grasp
    obb = get_actor_obb(sphere)
    approaching = np.array([0, 0, -1])  # Approach from above
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    # Compute initial grasp pose
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    initial_grasp_pose = env.agent.build_grasp_pose(approaching, closing, sphere.pose.sp.p)
    
    # -------------------------------------------------------------------------- #
    # Phase 1.1: Reach sphere
    # -------------------------------------------------------------------------- #
    reach_pose = initial_grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    
    # -------------------------------------------------------------------------- #
    # Phase 1.2: Grasp sphere
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(initial_grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Phase 1.3: Lift sphere
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * initial_grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Phase 1.4: Move to yellow bin and place
    # -------------------------------------------------------------------------- #
    # Correct height: bin_base + bin_thickness + sphere_radius
    yellow_target_position = yellow_bin.pose.sp.p.copy()
    yellow_target_position[2] = yellow_bin.pose.sp.p[2] + env.block_half_size[0] + env.radius
    
    yellow_place_pose = sapien.Pose(yellow_target_position, initial_grasp_pose.q)
    planner.move_to_pose_with_screw(yellow_place_pose)

    # Place in yellow bin
    planner.open_gripper()

    # ========================================================================== #
    # STAGE 2: Pick sphere from yellow bin and move to RED bin
    # ========================================================================== #
    
    # -------------------------------------------------------------------------- #
    # Phase 2.1: Reach sphere in yellow bin
    # -------------------------------------------------------------------------- #
    current_sphere_pos = sphere.pose.sp.p
    reach_pose_2 = sapien.Pose(current_sphere_pos, initial_grasp_pose.q)
    planner.move_to_pose_with_screw(reach_pose_2)
    
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Phase 2.2: Lift sphere from yellow bin
    # -------------------------------------------------------------------------- #
    lift_pose_2 = sapien.Pose([0, 0, 0.1]) * reach_pose_2
    planner.move_to_pose_with_screw(lift_pose_2)

    # -------------------------------------------------------------------------- #
    # Phase 2.3: Move to red bin and place
    # -------------------------------------------------------------------------- #
    # Correct height: bin_base + bin_thickness + sphere_radius
    red_target_position = red_bin.pose.sp.p.copy()
    red_target_position[2] = red_bin.pose.sp.p[2] + env.block_half_size[0] + env.radius
    
    red_place_pose = sapien.Pose(red_target_position, reach_pose_2.q)
    planner.move_to_pose_with_screw(red_place_pose)

    # Place in red bin
    planner.open_gripper()

    # ========================================================================== #
    # STAGE 3: Pick sphere from red bin and move to BLUE bin
    # ========================================================================== #
    
    # -------------------------------------------------------------------------- #
    # Phase 3.1: Reach sphere in red bin
    # -------------------------------------------------------------------------- #
    current_sphere_pos_3 = sphere.pose.sp.p
    reach_pose_3 = sapien.Pose(current_sphere_pos_3, initial_grasp_pose.q)
    planner.move_to_pose_with_screw(reach_pose_3)
    
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Phase 3.2: Lift sphere from red bin
    # -------------------------------------------------------------------------- #
    lift_pose_3 = sapien.Pose([0, 0, 0.1]) * reach_pose_3
    planner.move_to_pose_with_screw(lift_pose_3)

    # -------------------------------------------------------------------------- #
    # Phase 3.3: Move to blue bin and place (FINAL STAGE)
    # -------------------------------------------------------------------------- #
    # Correct height: bin_base + bin_thickness + sphere_radius
    blue_target_position = blue_bin.pose.sp.p.copy()
    blue_target_position[2] = blue_bin.pose.sp.p[2] + env.block_half_size[0] + env.radius
    
    blue_place_pose = sapien.Pose(blue_target_position, reach_pose_3.q)
    planner.move_to_pose_with_screw(blue_place_pose)

    # Place in blue bin - capture final state
    res = planner.open_gripper()

    planner.close()
    return res
