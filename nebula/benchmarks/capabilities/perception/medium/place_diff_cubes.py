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


@register_env("Perception-PlaceDiffCubes-Medium", max_episode_steps=50)
class PerceptionPlaceDiffCubesMediumEnv(BaseEnv):
    """
    **Task Description:**
    Place any cube (red, blue, green, or large yellow cube) into the shallow bin.

    **Randomizations:**
    - The position of the bin and the cubes are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1],
    and the cubes are initialized at fixed positions in [-0.1, -0.05] x [-0.1, 0.1]

    **Success Conditions:**
    - Any cube is placed on the top of the bin. The robot remains static and the gripper is not closed at the end state.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # cube sizes
    small_cube_size = 0.015  # half_size for small cubes (3cm total)
    large_cube_size = 0.025  # half_size for large cube (5cm total)
    
    # bin dimensions adjusted for cubes
    inner_side_half_len = 0.035  # adjusted for cube sizes
    short_side_half_size = 0.0025
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]

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

    def _build_bin(self):
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

        # load the four cubes with different colors
        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.small_cube_size,
            color=np.array([255, 0, 0, 255]) / 255,  # red
            name="red_cube",
            body_type="dynamic",
        )
        
        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=self.small_cube_size,
            color=np.array([0, 0, 255, 255]) / 255,  # blue
            name="blue_cube",
            body_type="dynamic",
        )
        
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.small_cube_size,
            color=np.array([0, 180, 0, 255]) / 255,  # green
            name="green_cube",
            body_type="dynamic",
        )
        
        self.large_yellow_cube = actors.build_cube(
            self.scene,
            half_size=self.large_cube_size,
            color=np.array([255, 255, 0, 255]) / 255,  # yellow
            name="large_yellow_cube",
            body_type="dynamic",
        )

        # load the bin
        self.bin = self._build_bin()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            q = [1, 0, 0, 0]
            
            # Red cube position (fixed)
            xyz[..., 0] = -0.08  # fixed x position
            xyz[..., 1] = -0.09  # fixed y position  
            xyz[..., 2] = self.small_cube_size  # on the table
            red_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.red_cube.set_pose(red_cube_pose)

            # Blue cube position (fixed)
            xyz[..., 0] = -0.08  # same x
            xyz[..., 1] = -0.03  # spaced y position
            xyz[..., 2] = self.small_cube_size  # on the table
            blue_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.blue_cube.set_pose(blue_cube_pose)

            # Green cube position (fixed)
            xyz[..., 0] = -0.08  # same x
            xyz[..., 1] = 0.03   # spaced y position
            xyz[..., 2] = self.small_cube_size  # on the table
            green_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.green_cube.set_pose(green_cube_pose)

            # Large yellow cube position (fixed)
            xyz[..., 0] = -0.08  # same x
            xyz[..., 1] = 0.09   # spaced y position
            xyz[..., 2] = self.large_cube_size  # on the table
            large_yellow_cube_pose = Pose.create_from_pq(p=xyz, q=q)
            self.large_yellow_cube.set_pose(large_yellow_cube_pose)

            # init the bin in the last 1/2 zone along the x-axis (so that it doesn't collide the cubes)
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
        pos_red_cube = self.red_cube.pose.p
        pos_blue_cube = self.blue_cube.pose.p
        pos_green_cube = self.green_cube.pose.p
        pos_large_yellow_cube = self.large_yellow_cube.pose.p
        pos_bin = self.bin.pose.p
        
        # Check if any cube is on the bin
        is_red_cube_on_bin = self._check_object_on_bin(pos_red_cube, pos_bin, self.small_cube_size)
        is_blue_cube_on_bin = self._check_object_on_bin(pos_blue_cube, pos_bin, self.small_cube_size)
        is_green_cube_on_bin = self._check_object_on_bin(pos_green_cube, pos_bin, self.small_cube_size)
        is_large_yellow_cube_on_bin = self._check_object_on_bin(pos_large_yellow_cube, pos_bin, self.large_cube_size)
        is_any_cube_on_bin = torch.logical_or(
            torch.logical_or(
                torch.logical_or(is_red_cube_on_bin, is_blue_cube_on_bin),
                is_green_cube_on_bin
            ),
            is_large_yellow_cube_on_bin
        )
        
        # Check if cubes are static
        is_red_cube_static = self.red_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_blue_cube_static = self.blue_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_green_cube_static = self.green_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_large_yellow_cube_static = self.large_yellow_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        # Check if cubes are grasped
        is_red_cube_grasped = self.agent.is_grasping(self.red_cube)
        is_blue_cube_grasped = self.agent.is_grasping(self.blue_cube)
        is_green_cube_grasped = self.agent.is_grasping(self.green_cube)
        is_large_yellow_cube_grasped = self.agent.is_grasping(self.large_yellow_cube)
        
        # Success: any cube on bin, that cube is static, and that cube is not grasped
        red_cube_success = is_red_cube_on_bin & is_red_cube_static & (~is_red_cube_grasped)
        blue_cube_success = is_blue_cube_on_bin & is_blue_cube_static & (~is_blue_cube_grasped)
        green_cube_success = is_green_cube_on_bin & is_green_cube_static & (~is_green_cube_grasped)
        large_yellow_cube_success = is_large_yellow_cube_on_bin & is_large_yellow_cube_static & (~is_large_yellow_cube_grasped)
        success = torch.logical_or(
            torch.logical_or(
                torch.logical_or(red_cube_success, blue_cube_success),
                green_cube_success
            ),
            large_yellow_cube_success
        )
        
        return {
            "is_red_cube_grasped": is_red_cube_grasped,
            "is_blue_cube_grasped": is_blue_cube_grasped,
            "is_green_cube_grasped": is_green_cube_grasped,
            "is_large_yellow_cube_grasped": is_large_yellow_cube_grasped,
            "is_red_cube_on_bin": is_red_cube_on_bin,
            "is_blue_cube_on_bin": is_blue_cube_on_bin,
            "is_green_cube_on_bin": is_green_cube_on_bin,
            "is_large_yellow_cube_on_bin": is_large_yellow_cube_on_bin,
            "is_red_cube_static": is_red_cube_static,
            "is_blue_cube_static": is_blue_cube_static,
            "is_green_cube_static": is_green_cube_static,
            "is_large_yellow_cube_static": is_large_yellow_cube_static,
            "success": success,
        }

    def _check_object_on_bin(self, pos_object, pos_bin, object_height):
        """Check if object is on top of the bin"""
        offset = pos_object - pos_bin
        # Lenient xy check for object placement
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.02
        z_flag = (
            torch.abs(offset[..., 2] - object_height - self.block_half_size[0]) <= 0.015
        )
        return torch.logical_and(xy_flag, z_flag)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_red_cube_grasped=info["is_red_cube_grasped"],
            is_blue_cube_grasped=info["is_blue_cube_grasped"],
            is_green_cube_grasped=info["is_green_cube_grasped"],
            is_large_yellow_cube_grasped=info["is_large_yellow_cube_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
                blue_cube_pose=self.blue_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                large_yellow_cube_pose=self.large_yellow_cube.pose.raw_pose,
                tcp_to_red_cube_pos=self.red_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_blue_cube_pos=self.blue_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_green_cube_pos=self.green_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_large_yellow_cube_pos=self.large_yellow_cube.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to any cube
        tcp_pose = self.agent.tcp.pose.p
        red_cube_pos = self.red_cube.pose.p
        blue_cube_pos = self.blue_cube.pose.p
        green_cube_pos = self.green_cube.pose.p
        large_yellow_cube_pos = self.large_yellow_cube.pose.p
        
        obj_to_tcp_dist_red = torch.linalg.norm(tcp_pose - red_cube_pos, axis=1)
        obj_to_tcp_dist_blue = torch.linalg.norm(tcp_pose - blue_cube_pos, axis=1)
        obj_to_tcp_dist_green = torch.linalg.norm(tcp_pose - green_cube_pos, axis=1)
        obj_to_tcp_dist_yellow = torch.linalg.norm(tcp_pose - large_yellow_cube_pos, axis=1)
        obj_to_tcp_dist = torch.minimum(
            torch.minimum(
                torch.minimum(obj_to_tcp_dist_red, obj_to_tcp_dist_blue),
                obj_to_tcp_dist_green
            ),
            obj_to_tcp_dist_yellow
        )
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward
        bin_top_pos_small = self.bin.pose.p.clone()
        bin_top_pos_small[:, 2] = bin_top_pos_small[:, 2] + self.block_half_size[0] + self.small_cube_size
        
        bin_top_pos_large = self.bin.pose.p.clone()
        bin_top_pos_large[:, 2] = bin_top_pos_large[:, 2] + self.block_half_size[0] + self.large_cube_size
        
        red_cube_to_bin_dist = torch.linalg.norm(bin_top_pos_small - red_cube_pos, axis=1)
        blue_cube_to_bin_dist = torch.linalg.norm(bin_top_pos_small - blue_cube_pos, axis=1)
        green_cube_to_bin_dist = torch.linalg.norm(bin_top_pos_small - green_cube_pos, axis=1)
        large_yellow_cube_to_bin_dist = torch.linalg.norm(bin_top_pos_large - large_yellow_cube_pos, axis=1)
        
        # Reward based on whichever cube is grasped
        is_red_cube_grasped = info["is_red_cube_grasped"]
        is_blue_cube_grasped = info["is_blue_cube_grasped"]
        is_green_cube_grasped = info["is_green_cube_grasped"]
        is_large_yellow_cube_grasped = info["is_large_yellow_cube_grasped"]
        
        red_cube_place_reward = 1 - torch.tanh(5.0 * red_cube_to_bin_dist)
        blue_cube_place_reward = 1 - torch.tanh(5.0 * blue_cube_to_bin_dist)
        green_cube_place_reward = 1 - torch.tanh(5.0 * green_cube_to_bin_dist)
        large_yellow_cube_place_reward = 1 - torch.tanh(5.0 * large_yellow_cube_to_bin_dist)
        
        reward[is_red_cube_grasped] = (4 + red_cube_place_reward)[is_red_cube_grasped]
        reward[is_blue_cube_grasped] = (4 + blue_cube_place_reward)[is_blue_cube_grasped]
        reward[is_green_cube_grasped] = (4 + green_cube_place_reward)[is_green_cube_grasped]
        reward[is_large_yellow_cube_grasped] = (4 + large_yellow_cube_place_reward)[is_large_yellow_cube_grasped]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_any_cube_grasped = torch.logical_or(
            torch.logical_or(
                torch.logical_or(is_red_cube_grasped, is_blue_cube_grasped),
                is_green_cube_grasped
            ),
            is_large_yellow_cube_grasped
        )
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_any_cube_grasped
        ] = 16.0  # give ungrasp a bigger reward
        
        # Static reward for whichever cube is on the bin
        red_cube_v = torch.linalg.norm(self.red_cube.linear_velocity, axis=1)
        red_cube_av = torch.linalg.norm(self.red_cube.angular_velocity, axis=1)
        red_cube_static_reward = 1 - torch.tanh(red_cube_v * 10 + red_cube_av)
        
        blue_cube_v = torch.linalg.norm(self.blue_cube.linear_velocity, axis=1)
        blue_cube_av = torch.linalg.norm(self.blue_cube.angular_velocity, axis=1)
        blue_cube_static_reward = 1 - torch.tanh(blue_cube_v * 10 + blue_cube_av)
        
        green_cube_v = torch.linalg.norm(self.green_cube.linear_velocity, axis=1)
        green_cube_av = torch.linalg.norm(self.green_cube.angular_velocity, axis=1)
        green_cube_static_reward = 1 - torch.tanh(green_cube_v * 10 + green_cube_av)
        
        large_yellow_cube_v = torch.linalg.norm(self.large_yellow_cube.linear_velocity, axis=1)
        large_yellow_cube_av = torch.linalg.norm(self.large_yellow_cube.angular_velocity, axis=1)
        large_yellow_cube_static_reward = 1 - torch.tanh(large_yellow_cube_v * 10 + large_yellow_cube_av)
        
        robot_static_reward = self.agent.is_static(0.2)
        
        is_red_cube_on_bin = info["is_red_cube_on_bin"]
        is_blue_cube_on_bin = info["is_blue_cube_on_bin"]
        is_green_cube_on_bin = info["is_green_cube_on_bin"]
        is_large_yellow_cube_on_bin = info["is_large_yellow_cube_on_bin"]
        is_any_on_bin = torch.logical_or(
            torch.logical_or(
                torch.logical_or(is_red_cube_on_bin, is_blue_cube_on_bin),
                is_green_cube_on_bin
            ),
            is_large_yellow_cube_on_bin
        )
        
        # Use the static reward for whichever cube is on the bin
        static_reward = torch.where(
            is_red_cube_on_bin, red_cube_static_reward,
            torch.where(
                is_blue_cube_on_bin, blue_cube_static_reward,
                torch.where(is_green_cube_on_bin, green_cube_static_reward, large_yellow_cube_static_reward)
            )
        )
        
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