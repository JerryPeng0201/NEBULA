import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
import sapien.utils.viewer
import h5py
import json
import os
import sys
import tyro
from dataclasses import dataclass
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from nebula.utils import sapien_utils
from nebula.utils.wrappers.record import RecordEpisode
from nebula.core.simulation.engine import BaseEnv
import nebula.trajectory.utils as trajectory_utils

from nebula.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

import torch
from copy import deepcopy

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "ControlEasy-PlaceSphere"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"
    """The robot to use. Robot setups supported for teleop in this script are panda and panda_stick"""
    record_dir: str = "demos"
    """directory to record the demonstration data and optionally videos"""
    save_video: bool = True
    """whether to save the videos of the demonstrations after collecting them all"""
    viewer_shader: str = "default"
    """the shader to use for the viewer. 'default' is fast but lower-quality shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
    video_saving_shader: str = "default"
    """the shader to use for the videos of the demonstrations. 'minimal' is the fast shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
    subtask_idx: int = 0
    """the index of the orgnized folder for each collection."""
    use_remote: bool = False
    """whether to use the remote motion planner server. If set to True, the remote motion planner server must be running and the address must be set in the environment variable MPLIB_GRPC_ADDR"""
    only_replay: bool = False
    """whether to only replay the recorded trajectories and not collect new ones"""
    task_instruction: str = ""
    """The task instruction to use for the teleoperation."""

def parse_args() -> Args:
    return tyro.cli(Args)
 
def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop/"
    if os.name == 'nt':
        output_dir = f"{args.record_dir}\\{args.env_id}\\teleop\\"
    opts = {"task_instruction":args.task_instruction}
    if not args.only_replay:

        env = gym.make(
            args.env_id,
            obs_mode='none',
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            enable_shadow=True,
            viewer_camera_configs=dict(shader_pack=args.viewer_shader)
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory_tmp",
            save_video=False,
            info_on_video=False,
            task_name=args.env_id,
            source_type="teleoperation",
            source_desc="teleoperation via the click+drag system",
            subtask_idx=args.subtask_idx,
            record_reward=True
        )
        num_trajs = 0
        seed = random.randint(0, 10000000)
        env.reset(seed=seed,options=opts)
        while True:
            print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
            code = solve(env,args=args, debug=False, vis=True)
            if code == "quit":
                num_trajs += 1
                break
            elif code == "continue":
                seed += 1
                num_trajs += 1
                env.reset(seed=seed,options=opts)
                continue
            elif code == "restart":
                env.reset(seed=seed, options=dict(save_trajectory=False, **opts))
        h5_file_path = env._h5_file.filename
        json_file_path = env._json_path
        env.close()
        del env
    else:
        
        h5_file_path = os.path.join(output_dir,f"{args.subtask_idx}/trajectory.h5")
        json_file_path = os.path.join(output_dir,f"{args.subtask_idx}/metadata.json")
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"File {h5_file_path} does not exist. Please collect trajectories first.")
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File {json_file_path} does not exist. Please collect trajectories first.")
    
    print(f"Trajectories saved to {h5_file_path}")
    if args.save_video:
        print(f"Saving videos to {output_dir}")

        trajectory_data = h5py.File(h5_file_path)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)

        env_gym = gym.make(
            args.env_id,
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            render_mode="sensors",
            reward_mode="none",
            human_render_camera_configs=dict(shader_pack=args.video_saving_shader),
        )
        env = RecordEpisode(
                env_gym,
                output_dir=output_dir,
                trajectory_name="trajectory",
                save_video=True,
                info_on_video=False,
                save_trajectory=True,
                video_fps=30,
                subtask_idx=args.subtask_idx,
                task_name=args.env_id,
            )
        print(f"Replaying trajectories from {h5_file_path} and saving videos to {output_dir}")
        video_info_list = []
        for episode in json_data["episodes"]:
            
            traj_id = f"traj_{episode['episode_id']}"
            print(f"Replaying trajectory {traj_id}")
            data = trajectory_data[traj_id]
            env.reset(**episode["reset_kwargs"])
            env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

            env.base_env.set_state_dict(env_states_list[0])
            for action in np.array(data["actions"]):
                env.step(action)
            
            video_info_list.append(deepcopy(env.current_episode_videos))
            
        # explicitly reset the environment to ensure the next episode starts fresh, to get last video info
        env.reset(**episode["reset_kwargs"])
        # append the last video info to the list
        video_info_list.append(deepcopy(env.current_episode_videos))

        video_info_list.pop(0)  # remove the first empty video info
        for idx,episode in enumerate(json_data["episodes"]):
            episode["videos"] = video_info_list[idx]
            if "task_instruction" not in episode and args.task_instruction != "":
                # If the task instruction is not already in the episode, add it
                # This is useful for the case where the task instruction is not provided in the JSON file
                episode["task_instruction"] = args.task_instruction

        with open(json_file_path, "w") as f:
            json.dump(json_data, f, indent=2)

        trajectory_data.close()
        env.close()
        del env



def solve(env: BaseEnv,args: Args, debug=False, vis=False,):
    use_remote = args.use_remote
    grpc_addr = os.environ.get("MPLIB_GRPC_ADDR", "localhost:50051")
    
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], env.unwrapped.control_mode
    robot_has_gripper = False

    if env.unwrapped.robot_uids == "panda_stick":
        if use_remote:
            from nebula.data.generation.teleoperation.panda.remote_motionplanner import RemotePandaArmMotionPlanningSolver as RemoteStick  # or make a stick version if needed
            planner = RemoteStick(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=False,
                                  print_env_info=False, joint_acc_limits=0.5, joint_vel_limits=0.5, grpc_addr=grpc_addr)
        else:
            from nebula.data.generation.motionplanning.panda.motionplanner_stick import PandaStickMotionPlanningSolver
            planner = PandaStickMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose,
                                                     visualize_target_grasp_pose=False, print_env_info=False,
                                                     joint_acc_limits=0.5, joint_vel_limits=0.5)
    elif env.unwrapped.robot_uids in ["panda", "panda_wristcam"]:
        robot_has_gripper = True
        if use_remote:
            from nebula.data.generation.teleoperation.panda.remote_motionplanner import RemotePandaArmMotionPlanningSolver
            planner = RemotePandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose,
                                                         visualize_target_grasp_pose=False, print_env_info=False,
                                                         joint_acc_limits=0.5, joint_vel_limits=0.5, grpc_addr=grpc_addr)
        else:
            from nebula.data.generation.teleoperation.panda.local_motionplanner import PandaArmMotionPlanningSolver
            planner = PandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose,
                                                   visualize_target_grasp_pose=False, print_env_info=False,
                                                   joint_acc_limits=0.5, joint_vel_limits=0.5)

    viewer = env.render_human()

    last_checkpoint_state = None
    gripper_open = True
    def select_panda_hand():
        viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "panda_hand")._objs[0].entity)
    select_panda_hand()
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
    while True:

        transform_window.enabled = True
        # transform_window.update_ghost_objects
        # print(transform_window.ghost_objects, transform_window._gizmo_pose)
        # planner.grasp_pose_visual.set_pose(transform_window._gizmo_pose)

        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open (if there is a gripper)
            u: move the panda hand up
            j: move the panda hand down
            k: rotate the grasp pose clockwise in Yaw
            l: rotate the grasp pose counter-clockwise in Yaw
            i: rotate the grasp pose clockwise in Pitch
            o: rotate the grasp pose counter-clockwise in Pitch
            arrow_keys: move the panda hand in the direction of the arrow keys
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost panda arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data. Save trajectories and optionally videos.
            """)
            pass
        # elif viewer.window.key_press("k"):
        #     print("Saving checkpoint")
        #     last_checkpoint_state = env.get_state_dict()
        # elif viewer.window.key_press("l"):
        #     if last_checkpoint_state is not None:
        #         print("Loading previous checkpoint")
        #         env.set_state_dict(last_checkpoint_state)
        #     else:
        #         print("Could not find previous checkpoint")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g") and robot_has_gripper:
            if gripper_open:
                gripper_open = False
                _, reward, _ ,_, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _ ,_, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
        
        elif viewer.window.key_press("k"):
            select_panda_hand()
            # Rotate 1 degree Yaw the grasp pose
            xyz_angles = torch.tensor([0, 0, np.pi/90])  # Roll, Pitch, Yaw angles in radians
            newq = matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, 0],q=newq)).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("l"):
            select_panda_hand()
            # Rotate 1 degree Yaw the grasp pose
            xyz_angles = torch.tensor([0, 0, -np.pi/90])  # Roll, Pitch, Yaw angles in radians
            newq = matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, 0],q=newq)).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("i"):
            select_panda_hand()
            # Rotate 1 degree Pitch the grasp pose
            xyz_angles = torch.tensor([0, np.pi/90, 0])  # Roll, Pitch, Yaw angles in radians
            newq = matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, 0],q=newq)).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("o"):
            select_panda_hand()
            # Rotate 1 degree Pitch the grasp pose
            xyz_angles = torch.tensor([0, -np.pi/90, 0])  # Roll, Pitch, Yaw angles in radians
            newq = matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, 0],q=newq)).to_transformation_matrix()
            transform_window.update_ghost_objects()

        elif viewer.window.key_press("u"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, +0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[+0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[-0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, -0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, +0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        if execute_current_pose:
            # z-offset of end-effector gizmo to TCP position is hardcoded for the panda robot here
            if env.unwrapped.robot_uids == "panda" or env.unwrapped.robot_uids == "panda_wristcam":
                result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.1]), dry_run=True)
            elif env.unwrapped.robot_uids == "panda_stick":
                result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.15]), dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False

if __name__ == "__main__":
    main(parse_args())
