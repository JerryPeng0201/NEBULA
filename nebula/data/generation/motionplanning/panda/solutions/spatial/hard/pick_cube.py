import numpy as np
import sapien
from nebula.benchmarks.capabilities.spatial.hard.pick_cube import SpatialHardPickCubeEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def SpatialHardPickCubeSolution(env: SpatialHardPickCubeEnv, seed=None, debug=False, vis=False):
    """
    Motion-planning solution for complex spatial reasoning pick task.
    Handles nested spatial relationships (on_top_of_inside, beside_under_platform, etc.)
    """
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
    base_env = env.unwrapped
    target_object = base_env.objects[base_env.target_object_color]

    # print(env.task_instruction)

    try:
        # Calculate grasp pose
        obb = get_actor_obb(target_object)
        approaching = np.array([0, 0, -1], dtype=np.float32)
        tcp_y = base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=tcp_y,
            depth=FINGER_LENGTH,
        )
        grasp_pose = base_env.agent.build_grasp_pose(
            approaching, 
            grasp_info["closing"], 
            target_object.pose.sp.p
        )

        # Reach (approach from above in world frame)
        reach_offset = np.array([0, 0, 0.05])
        reach_pose = sapien.Pose(grasp_pose.p + reach_offset, grasp_pose.q)
        planner.move_to_pose_with_screw(reach_pose)

        # Grasp
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()

        # Lift (higher for complex scenes)
        lift_offset = np.array([0, 0, 0.12])
        lift_pose = sapien.Pose(grasp_pose.p + lift_offset, grasp_pose.q)
        res = planner.move_to_pose_with_screw(lift_pose)

        planner.close()
        return res

    except Exception as e:
        print(f"[PickCube Hard solver] Error: {e}")
        import traceback
        traceback.print_exc()
        planner.close()
        return planner.close()