from typing import Any, Dict, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.simulation.utils import randomization
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import common, sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs import Pose
from nebula.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("Control-PlaceSphere-Hard", max_episode_steps=50)
class ControlPlaceSphereHardEnv(BaseEnv):
    """
    **Task Description:**
    Place the sphere into the specified target bin (red, yellow, or blue) based on the task instruction.

    **Randomizations:**
    - The target bin is randomly selected at the start of each episode
    - The sphere is initialized in [-0.1, -0.05] x [-0.1, 0.1]
    - The bins are in fixed positions at y = -0.067, 0, 0.067

    **Success Conditions:**
    - The sphere is placed on the top of the TARGET bin specified in the instruction. 
    - The robot remains static and the gripper is not closed at the end state.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values
    radius = 0.02  # radius of the sphere
    inner_side_half_len = 0.02  # side length of the bin's inner square
    short_side_half_size = 0.0025  # length of the shortest edge of the block
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Initialize target bin color to None - will be set during reset
        self.target_bin_color = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # Base camera configuration
        base_pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
        base_camera = CameraConfig(
            "base_camera",
            pose=base_pose,
            width=512,
            height=512,
            fov=np.pi / 2,
            near=0.01,
            far=100,
        )
        
        # Hand camera configuration
        hand_camera = CameraConfig(
            uid="hand_camera",
            pose=sapien.Pose(p=[0, 0, -0.05], q=[0, 0.7071, 0, 0.7071]),
            width=512,
            height=512,
            fov=np.pi / 2,
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["panda_hand_tcp"],
        )

        # Back right camera
        back_right_pose = sapien_utils.look_at(eye=[-0.3, -0.3, 0.3], target=[0, 0, 0.1])
        back_right_camera = CameraConfig(
            uid="back_right_camera",
            pose=back_right_pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=2.0,
        )

        # Back left camera
        back_left_pose = sapien_utils.look_at(eye=[-0.3, 0.3, 0.3], target=[0, 0, 0.1])
        back_left_camera = CameraConfig(
            uid="back_left_camera",
            pose=back_left_pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=2.0,
        )

        # Front right camera
        front_right_pose = sapien_utils.look_at(eye=[0.5, -0.5, 0.3], target=[0, 0, 0.1])
        front_right_camera = CameraConfig(
            uid="front_right_camera",
            pose=front_right_pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=2.0,
        )

        # Front left camera
        front_left_pose = sapien_utils.look_at(eye=[0.5, 0.5, 0.3], target=[0, 0, 0.1])
        front_left_camera = CameraConfig(
            uid="front_left_camera",
            pose=front_left_pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=2.0,
        )
        
        return [base_camera, hand_camera, back_right_camera, back_left_camera, front_right_camera, front_left_camera]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _build_bin(self, radius, color, color_name):
        """Build a bin with specified color and name"""
        builder = self.scene.create_actor_builder()

        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            # Create material with color
            material = sapien.render.RenderMaterial()
            material.set_base_color(color)
            builder.add_box_visual(pose, half_size, material=material)

        # Set initial pose to avoid warning (actual pose will be set in _initialize_episode)
        builder.initial_pose = sapien.Pose()
        
        # build the kinematic bin
        return builder.build_kinematic(name=f"{color_name}_bin")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the sphere
        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.radius], q=[1, 0, 0, 0])
        )

        # load the three bins - red, yellow, and blue
        self.red_bin = self._build_bin(self.radius, color=np.array([1.0, 0.0, 0.0, 1.0]), color_name="red")
        
        self.yellow_bin = self._build_bin(self.radius, color=np.array([1.0, 0.5, 0.0, 1.0]), color_name="yellow")
        
        self.blue_bin = self._build_bin(self.radius, color=np.array([0.0, 0.0, 1.0, 1.0]), color_name="blue")
        
        # Keep reference to bins for compatibility
        self.bins = [self.red_bin, self.yellow_bin, self.blue_bin]
        self.bin_colors = ["red", "yellow", "blue"]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # Extract target bin color from task instruction if available
            task_instruction = options.get("task_instruction", "")
            if "red bin" in task_instruction:
                self.target_bin_color = "red"
            elif "yellow bin" in task_instruction:
                self.target_bin_color = "yellow"
            elif "blue bin" in task_instruction:
                self.target_bin_color = "blue"
            else:
                # Fallback: randomly select target if not specified in instruction
                import random
                self.target_bin_color = random.choice(self.bin_colors)

            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # init the sphere at a fixed position in the first 1/4 zone along the x-axis
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = -0.075  # fixed x position in first 1/4 zone (middle of [-0.1, -0.05])
            xyz[..., 1] = 0.0     # fixed y position (center)
            xyz[..., 2] = self.radius  # on the table
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # init the three bins on the same line along the x-axis with fixed positions
            # Red bin positioned at y = -0.067, yellow bin at y = 0, Blue bin at y = 0.067
            
            # Red bin position (fixed)
            red_pos = torch.zeros((b, 3))
            red_pos[:, 0] = 0.05  # fixed x position
            red_pos[:, 1] = -0.067  # fixed y position for red bin
            red_pos[:, 2] = self.block_half_size[0]  # on the table
            red_bin_pose = Pose.create_from_pq(p=red_pos, q=[1, 0, 0, 0])
            self.red_bin.set_pose(red_bin_pose)

            # yellow bin position (fixed)
            yellow_pos = torch.zeros((b, 3))
            yellow_pos[:, 0] = 0.05  # fixed x position (same as red bin)
            yellow_pos[:, 1] = 0.0  # fixed y position for yellow bin (center)
            yellow_pos[:, 2] = self.block_half_size[0]  # on the table
            yellow_bin_pose = Pose.create_from_pq(p=yellow_pos, q=[1, 0, 0, 0])
            self.yellow_bin.set_pose(yellow_bin_pose)

            # Blue bin position (fixed)
            blue_pos = torch.zeros((b, 3))
            blue_pos[:, 0] = 0.05  # fixed x position (same as red bin)
            blue_pos[:, 1] = 0.067  # fixed y position for blue bin
            blue_pos[:, 2] = self.block_half_size[0]  # on the table
            blue_bin_pose = Pose.create_from_pq(p=blue_pos, q=[1, 0, 0, 0])
            self.blue_bin.set_pose(blue_bin_pose)

    def evaluate(self):
        pos_obj = self.obj.pose.p
        
        # Check if object is on each of the three bins
        is_obj_on_red_bin = self._check_obj_on_bin(pos_obj, self.red_bin.pose.p)
        is_obj_on_yellow_bin = self._check_obj_on_bin(pos_obj, self.yellow_bin.pose.p)
        is_obj_on_blue_bin = self._check_obj_on_bin(pos_obj, self.blue_bin.pose.p)
        
        # Check if object is on any bin (for general tracking)
        is_obj_on_any_bin = torch.logical_or(
            torch.logical_or(is_obj_on_red_bin, is_obj_on_yellow_bin), 
            is_obj_on_blue_bin
        )
        
        # Check if object is on the TARGET bin specifically
        if self.target_bin_color == "red":
            is_obj_on_target_bin = is_obj_on_red_bin
        elif self.target_bin_color == "yellow":
            is_obj_on_target_bin = is_obj_on_yellow_bin
        elif self.target_bin_color == "blue":
            is_obj_on_target_bin = is_obj_on_blue_bin
        else:
            # Fallback if target_bin_color is not set properly
            is_obj_on_target_bin = torch.zeros_like(is_obj_on_red_bin, dtype=torch.bool)
        
        is_obj_static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.obj)
        
        # Success is only when object is on TARGET bin, static, and not grasped
        success = is_obj_on_target_bin & is_obj_static & (~is_obj_grasped)
        
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_on_red_bin": is_obj_on_red_bin,
            "is_obj_on_yellow_bin": is_obj_on_yellow_bin,
            "is_obj_on_blue_bin": is_obj_on_blue_bin,
            "is_obj_on_any_bin": is_obj_on_any_bin,
            "is_obj_on_target_bin": is_obj_on_target_bin,
            "is_obj_static": is_obj_static,
            "target_bin_color": self.target_bin_color,
            "success": success,
        }

    def _check_obj_on_bin(self, pos_obj, pos_bin):
        """Check if object is on top of a specific bin"""
        offset = pos_obj - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
        )
        return torch.logical_and(xy_flag, z_flag)
    
    def _get_target_bin_id(self):
        """Convert target bin color to numeric ID"""
        color_to_id = {"red": 0, "yellow": 1, "blue": 2}
        return color_to_id.get(self.target_bin_color, 0)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            red_bin_pos=self.red_bin.pose.p,
            yellow_bin_pos=self.yellow_bin.pose.p,
            blue_bin_pos=self.blue_bin.pose.p,
            target_bin_color=self._get_target_bin_id(),
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward - reward for getting close to TARGET bin only
        if self.target_bin_color == "red":
            target_bin = self.red_bin
        elif self.target_bin_color == "yellow":
            target_bin = self.yellow_bin
        elif self.target_bin_color == "blue":
            target_bin = self.blue_bin
        else:
            target_bin = self.yellow_bin  # fallback

        target_bin_top_pos = target_bin.pose.p.clone()
        target_bin_top_pos[:, 2] = target_bin_top_pos[:, 2] + self.block_half_size[0] + self.radius
        obj_to_target_bin_dist = torch.linalg.norm(target_bin_top_pos - obj_pos, axis=1)
        
        place_reward = 1 - torch.tanh(5.0 * obj_to_target_bin_dist)
        reward[info["is_obj_grasped"]] = (4 + place_reward)[info["is_obj_grasped"]]

        # ungrasp and static reward - only when on target bin
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_obj_grasped = info["is_obj_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_obj_grasped
        ] = 16.0  # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
        v = torch.linalg.norm(self.obj.linear_velocity, axis=1)
        av = torch.linalg.norm(self.obj.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        robot_static_reward = self.agent.is_static(
            0.2
        )  # keep the robot static at the end state, since the sphere may spin when being placed on top
        reward[info["is_obj_on_target_bin"]] = (
            6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        )[info["is_obj_on_target_bin"]]

        # success reward - only for target bin
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward