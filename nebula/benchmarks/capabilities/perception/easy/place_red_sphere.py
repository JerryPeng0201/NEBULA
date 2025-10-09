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


@register_env("Perception-PlaceRedSphere-Easy", max_episode_steps=50)
class PerceptionPlaceRedSphereEasyEnv(BaseEnv):
    """
    **Task Description:**
    Place either sphere (blue or red) into the shallow bin.

    **Randomizations:**
    - The position of the bin and the spheres are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1],
    and the spheres are initialized in [-0.1, -0.05] x [-0.1, 0.1]

    **Success Conditions:**
    - Either sphere is placed on the top of the bin. The robot remains static and the gripper is not closed at the end state.
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

    def _build_bin(self, radius):
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
            builder.add_box_visual(pose, half_size)

        # build the kinematic bin
        return builder.build_kinematic(name="bin")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the two spheres - original blue and red
        self.blue_sphere = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="blue_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.radius], q=[1, 0, 0, 0])
        )
        
        self.red_sphere = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([255, 0, 0, 255]) / 255,
            name="red_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.radius], q=[1, 0, 0, 0])
        )

        # load the bin
        self.bin = self._build_bin(self.radius)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # init the spheres in the first 1/4 zone along the x-axis (so that they don't collide the bin)
            xyz = torch.zeros((b, 3))
            xyz[..., 2] = self.radius  # on the table
            q = [1, 0, 0, 0]
            
            # Blue sphere position
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0]  # first 1/4 zone of x ([-0.1, -0.05])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.1 - 0.1)[..., 0]   # left half of y range ([-0.1, 0])
            blue_sphere_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.blue_sphere.set_pose(blue_sphere_pose)

            # Red sphere position
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0]  # first 1/4 zone of x ([-0.1, -0.05])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.1)[..., 0]         # right half of y range ([0, 0.1])
            red_sphere_pose = Pose.create_from_pq(p=xyz, q=q)
            self.red_sphere.set_pose(red_sphere_pose)

            # init the bin in the last 1/2 zone along the x-axis (so that it doesn't collide the spheres)
            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.rand((b, 1))[..., 0] * 0.1
            )  # the last 1/2 zone of x ([0, 0.1])
            pos[:, 1] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )  # spanning all possible ys
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)

    def evaluate(self):
        pos_blue_sphere = self.blue_sphere.pose.p
        pos_red_sphere = self.red_sphere.pose.p
        pos_bin = self.bin.pose.p
        
        # Check if either sphere is on the bin
        is_blue_sphere_on_bin = self._check_sphere_on_bin(pos_blue_sphere, pos_bin)
        is_red_sphere_on_bin = self._check_sphere_on_bin(pos_red_sphere, pos_bin)
        is_any_sphere_on_bin = torch.logical_or(is_blue_sphere_on_bin, is_red_sphere_on_bin)
        
        # Check if spheres are static
        is_blue_sphere_static = self.blue_sphere.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_red_sphere_static = self.red_sphere.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        # Check if spheres are grasped
        is_blue_sphere_grasped = self.agent.is_grasping(self.blue_sphere)
        is_red_sphere_grasped = self.agent.is_grasping(self.red_sphere)
        
        # Success: either sphere on bin, that sphere is static, and that sphere is not grasped
        blue_sphere_success = is_blue_sphere_on_bin & (~is_blue_sphere_grasped)
        red_sphere_success = is_red_sphere_on_bin & (~is_red_sphere_grasped)
        success = torch.logical_or(blue_sphere_success, red_sphere_success)
        
        return {
            "is_blue_sphere_grasped": is_blue_sphere_grasped,
            "is_red_sphere_grasped": is_red_sphere_grasped,
            "is_blue_sphere_on_bin": is_blue_sphere_on_bin,
            "is_red_sphere_on_bin": is_red_sphere_on_bin,
            "success": success,
        }

    def _check_sphere_on_bin(self, pos_sphere, pos_bin):
        """Check if sphere is on top of the bin"""
        offset = pos_sphere - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
        )
        return torch.logical_and(xy_flag, z_flag)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_blue_sphere_grasped=info["is_blue_sphere_grasped"],
            is_red_sphere_grasped=info["is_red_sphere_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                blue_sphere_pose=self.blue_sphere.pose.raw_pose,
                red_sphere_pose=self.red_sphere.pose.raw_pose,
                tcp_to_blue_sphere_pos=self.blue_sphere.pose.p - self.agent.tcp.pose.p,
                tcp_to_red_sphere_pos=self.red_sphere.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to either sphere
        tcp_pose = self.agent.tcp.pose.p
        blue_sphere_pos = self.blue_sphere.pose.p
        red_sphere_pos = self.red_sphere.pose.p
        
        obj_to_tcp_dist_blue = torch.linalg.norm(tcp_pose - blue_sphere_pos, axis=1)
        obj_to_tcp_dist_red = torch.linalg.norm(tcp_pose - red_sphere_pos, axis=1)
        obj_to_tcp_dist = torch.minimum(obj_to_tcp_dist_blue, obj_to_tcp_dist_red)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward
        bin_top_pos = self.bin.pose.p.clone()
        bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.block_half_size[0] + self.radius
        
        blue_sphere_to_bin_dist = torch.linalg.norm(bin_top_pos - blue_sphere_pos, axis=1)
        red_sphere_to_bin_dist = torch.linalg.norm(bin_top_pos - red_sphere_pos, axis=1)
        
        # Reward based on whichever sphere is grasped
        is_blue_grasped = info["is_blue_sphere_grasped"]
        is_red_grasped = info["is_red_sphere_grasped"]
        
        blue_place_reward = 1 - torch.tanh(5.0 * blue_sphere_to_bin_dist)
        red_place_reward = 1 - torch.tanh(5.0 * red_sphere_to_bin_dist)
        
        reward[is_blue_grasped] = (4 + blue_place_reward)[is_blue_grasped]
        reward[is_red_grasped] = (4 + red_place_reward)[is_red_grasped]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_any_sphere_grasped = torch.logical_or(is_blue_grasped, is_red_grasped)
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_any_sphere_grasped
        ] = 16.0  # give ungrasp a bigger reward
        
        # Static reward for whichever sphere is on the bin
        blue_v = torch.linalg.norm(self.blue_sphere.linear_velocity, axis=1)
        blue_av = torch.linalg.norm(self.blue_sphere.angular_velocity, axis=1)
        blue_static_reward = 1 - torch.tanh(blue_v * 10 + blue_av)
        
        red_v = torch.linalg.norm(self.red_sphere.linear_velocity, axis=1)
        red_av = torch.linalg.norm(self.red_sphere.angular_velocity, axis=1)
        red_static_reward = 1 - torch.tanh(red_v * 10 + red_av)
        
        robot_static_reward = self.agent.is_static(0.2)
        
        is_blue_on_bin = info["is_blue_sphere_on_bin"]
        is_red_on_bin = info["is_red_sphere_on_bin"]
        is_any_on_bin = torch.logical_or(is_blue_on_bin, is_red_on_bin)
        
        # Use the static reward for whichever sphere is on the bin
        static_reward = torch.where(is_blue_on_bin, blue_static_reward, red_static_reward)
        
        reward[is_any_on_bin] = (
            6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        )[is_any_on_bin]

        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward