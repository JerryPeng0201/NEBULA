import numpy as np
import sapien
from nebula.benchmarks.capabilities.spatial.medium.pick_cube import SpatialMediumPickCubeEnv
from nebula.data.generation.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from nebula.data.generation.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def SpatialMediumPickCubeSolution(env: SpatialMediumPickCubeEnv, seed=None, debug=False, vis=False):
    """
    Motion-planning solution for PickCube task with 3D spatial relations.
    Identifies and picks the target object based on spatial relation.
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
    target_object = base_env.objects[base_env.target_object]

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

        # Reach (approach from above)
        reach_offset = np.array([0, 0, 0.05])
        reach_pose = sapien.Pose(grasp_pose.p + reach_offset, grasp_pose.q)
        planner.move_to_pose_with_screw(reach_pose)

        # Grasp
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()

        # Lift
        lift_offset = np.array([0, 0, 0.10])
        lift_pose = sapien.Pose(grasp_pose.p + lift_offset, grasp_pose.q)
        res = planner.move_to_pose_with_screw(lift_pose)

        planner.close()
        return res

    except Exception as e:
        print(f"[PickCube solver] Error: {e}")
        import traceback
        traceback.print_exc()
        planner.close()
        return planner.close()