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


@register_env("Control-PlaceSphere-Hard", max_episode_steps=100)
class ControlPlaceSphereHardEnv(BaseEnv):
    """
    **Task Description:**
    Place the sphere into three bins sequentially: yellow bin → red bin → blue bin.

    **Randomizations:**
    - The sphere is initialized at a fixed position
    - The bins are in fixed positions at y = 0 (yellow), -0.067 (red), 0.067 (blue)

    **Success Conditions:**
    - The sphere must be placed in the yellow bin first
    - Then moved to the red bin
    - Finally moved to the blue bin
    - The gripper must be open (not grasping) at each placement
    """

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
            # Yellow bin at y = 0 (center), Red bin at y = -0.067, Blue bin at y = 0.067
            
            # Yellow bin position (fixed) - center
            yellow_pos = torch.zeros((b, 3))
            yellow_pos[:, 0] = 0.05  # fixed x position
            yellow_pos[:, 1] = 0.0  # fixed y position for yellow bin (center)
            yellow_pos[:, 2] = self.block_half_size[0]  # on the table
            yellow_bin_pose = Pose.create_from_pq(p=yellow_pos, q=[1, 0, 0, 0])
            self.yellow_bin.set_pose(yellow_bin_pose)

            # Red bin position (fixed)
            red_pos = torch.zeros((b, 3))
            red_pos[:, 0] = 0.05  # fixed x position (same as yellow bin)
            red_pos[:, 1] = -0.067  # fixed y position for red bin
            red_pos[:, 2] = self.block_half_size[0]  # on the table
            red_bin_pose = Pose.create_from_pq(p=red_pos, q=[1, 0, 0, 0])
            self.red_bin.set_pose(red_bin_pose)

            # Blue bin position (fixed)
            blue_pos = torch.zeros((b, 3))
            blue_pos[:, 0] = 0.05  # fixed x position (same as yellow bin)
            blue_pos[:, 1] = 0.067  # fixed y position for blue bin
            blue_pos[:, 2] = self.block_half_size[0]  # on the table
            blue_bin_pose = Pose.create_from_pq(p=blue_pos, q=[1, 0, 0, 0])
            self.blue_bin.set_pose(blue_bin_pose)

            # Task phase tracking: 0=initial, 1=yellow placed, 2=red placed, 3=blue placed (complete)
            self.task_phase = torch.zeros(len(env_idx), dtype=torch.int32, device=self.device)
            self.has_placed_in_yellow = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
            self.has_placed_in_red = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
            self.has_placed_in_blue = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)

    def evaluate(self):
        pos_obj = self.obj.pose.p
        
        # Check if object is on each of the three bins
        is_obj_on_yellow_bin = self._check_obj_on_bin(pos_obj, self.yellow_bin.pose.p)
        is_obj_on_red_bin = self._check_obj_on_bin(pos_obj, self.red_bin.pose.p)
        is_obj_on_blue_bin = self._check_obj_on_bin(pos_obj, self.blue_bin.pose.p)
        
        is_obj_grasped = self.agent.is_grasping(self.obj)
        
        # Phase 0 → 1: Yellow bin placement
        yellow_placement_success = (self.task_phase == 0) & is_obj_on_yellow_bin & (~is_obj_grasped)
        self.has_placed_in_yellow = torch.where(
            yellow_placement_success,
            torch.ones_like(self.has_placed_in_yellow),
            self.has_placed_in_yellow
        )
        self.task_phase = torch.where(
            yellow_placement_success,
            torch.ones_like(self.task_phase),
            self.task_phase
        )
        
        # Phase 1 → 2: Red bin placement
        red_placement_success = (self.task_phase == 1) & is_obj_on_red_bin & (~is_obj_grasped)
        self.has_placed_in_red = torch.where(
            red_placement_success,
            torch.ones_like(self.has_placed_in_red),
            self.has_placed_in_red
        )
        self.task_phase = torch.where(
            red_placement_success,
            torch.full_like(self.task_phase, 2),
            self.task_phase
        )
        
        # Phase 2 → 3: Blue bin placement
        blue_placement_success = (self.task_phase == 2) & is_obj_on_blue_bin & (~is_obj_grasped)
        self.has_placed_in_blue = torch.where(
            blue_placement_success,
            torch.ones_like(self.has_placed_in_blue),
            self.has_placed_in_blue
        )
        self.task_phase = torch.where(
            blue_placement_success,
            torch.full_like(self.task_phase, 3),
            self.task_phase
        )
        
        # Success when all three placements are complete
        success = self.has_placed_in_yellow & self.has_placed_in_red & self.has_placed_in_blue
        
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_on_yellow_bin": is_obj_on_yellow_bin,
            "is_obj_on_red_bin": is_obj_on_red_bin,
            "is_obj_on_blue_bin": is_obj_on_blue_bin,
            "task_phase": self.task_phase,
            "has_placed_in_yellow": self.has_placed_in_yellow,
            "has_placed_in_red": self.has_placed_in_red,
            "has_placed_in_blue": self.has_placed_in_blue,
            "success": success,
        }

    def _check_obj_on_bin(self, pos_obj, pos_bin):
        """Check if object is on top of a specific bin"""
        offset = pos_obj - pos_bin
        # Relaxed tolerances for motion planning (20mm XY, 15mm Z)
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.020
        z_flag = (
            torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.015
        )
        return torch.logical_and(xy_flag, z_flag)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            yellow_bin_pos=self.yellow_bin.pose.p,
            red_bin_pos=self.red_bin.pose.p,
            blue_bin_pos=self.blue_bin.pose.p,
            task_phase=info["task_phase"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        
        # Base reaching reward
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))
        reward = reaching_reward.clone()
        
        # Phase 0: Reward for moving towards yellow bin
        yellow_bin_top_pos = self.yellow_bin.pose.p.clone()
        yellow_bin_top_pos[:, 2] += self.block_half_size[0] + self.radius
        obj_to_yellow_dist = torch.linalg.norm(yellow_bin_top_pos - obj_pos, axis=1)
        
        phase_0_mask = info["task_phase"] == 0
        if phase_0_mask.any():
            yellow_placement_reward = 1 - torch.tanh(5.0 * obj_to_yellow_dist)
            reward[phase_0_mask & info["is_obj_grasped"]] = (4 + yellow_placement_reward)[phase_0_mask & info["is_obj_grasped"]]
            reward[phase_0_mask & info["is_obj_on_yellow_bin"] & (~info["is_obj_grasped"])] = 10
        
        # Phase 1: Reward for moving towards red bin
        red_bin_top_pos = self.red_bin.pose.p.clone()
        red_bin_top_pos[:, 2] += self.block_half_size[0] + self.radius
        obj_to_red_dist = torch.linalg.norm(red_bin_top_pos - obj_pos, axis=1)
        
        phase_1_mask = info["task_phase"] == 1
        if phase_1_mask.any():
            red_placement_reward = 1 - torch.tanh(5.0 * obj_to_red_dist)
            reward[phase_1_mask & info["is_obj_grasped"]] = (4 + red_placement_reward)[phase_1_mask & info["is_obj_grasped"]]
            reward[phase_1_mask & info["is_obj_on_red_bin"] & (~info["is_obj_grasped"])] = 13
        
        # Phase 2: Reward for moving towards blue bin
        blue_bin_top_pos = self.blue_bin.pose.p.clone()
        blue_bin_top_pos[:, 2] += self.block_half_size[0] + self.radius
        obj_to_blue_dist = torch.linalg.norm(blue_bin_top_pos - obj_pos, axis=1)
        
        phase_2_mask = info["task_phase"] == 2
        if phase_2_mask.any():
            blue_placement_reward = 1 - torch.tanh(5.0 * obj_to_blue_dist)
            reward[phase_2_mask & info["is_obj_grasped"]] = (4 + blue_placement_reward)[phase_2_mask & info["is_obj_grasped"]]
            reward[phase_2_mask & info["is_obj_on_blue_bin"] & (~info["is_obj_grasped"])] = 16
        
        # Final success reward
        reward[info["success"]] = 20
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 20.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward