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


@register_env("Perception-PlaceRightCubes-Hard", max_episode_steps=50)
class PerceptionPlaceRightCubesHardEnv(BaseEnv):
    """
    **Task Description:**
    Place the small cube into the shallow bin. Only the small cube fits in the bin - the medium and large cubes are too big.

    **Randomizations:**
    - The position of the bin and the cubes are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1],
    and the cubes are initialized at fixed positions in [-0.1, -0.05] x [-0.1, 0.1]

    **Success Conditions:**
    - Only the small cube can be successfully placed in the bin. The robot remains static and the gripper is not closed at the end state.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # cube sizes
    small_cube_size = 0.012   # half_size for small cube (2.4cm total) - fits in bin
    medium_cube_size = 0.020  # half_size for medium cube (4cm total) - too big for bin
    large_cube_size = 0.030   # half_size for large cube (6cm total) - too big for bin
    
    # bin dimensions - small enough that only small cube fits
    inner_side_half_len = 0.018  # small bin opening (3.6cm total)
    short_side_half_size = 0.002
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

        # load the three cubes with different sizes and colors
        self.small_cube = actors.build_cube(
            self.scene,
            half_size=self.small_cube_size,
            color=np.array([0, 255, 0, 255]) / 255,  # green - fits in bin
            name="small_cube",
            body_type="dynamic",
        )
        
        self.medium_cube = actors.build_cube(
            self.scene,
            half_size=self.medium_cube_size,
            color=np.array([255, 255, 0, 255]) / 255,  # yellow - too big for bin
            name="medium_cube",
            body_type="dynamic",
        )
        
        self.large_cube = actors.build_cube(
            self.scene,
            half_size=self.large_cube_size,
            color=np.array([255, 0, 0, 255]) / 255,  # red - too big for bin
            name="large_cube",
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
            
            # Object positions (3 cubes in a single line with proper spacing)
            # All objects on same x-coordinate, spaced in y-direction
            
            # Small cube (green) - fits in bin
            xyz[..., 0] = -0.08  # fixed x position for all cubes
            xyz[..., 1] = -0.08  # leftmost position
            xyz[..., 2] = self.small_cube_size
            small_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.small_cube.set_pose(small_cube_pose)

            # Medium cube (yellow) - too big for bin
            xyz[..., 0] = -0.08
            xyz[..., 1] = 0.0    # center position
            xyz[..., 2] = self.medium_cube_size
            medium_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.medium_cube.set_pose(medium_cube_pose)

            # Large cube (red) - too big for bin
            xyz[..., 0] = -0.08
            xyz[..., 1] = 0.08   # rightmost position
            xyz[..., 2] = self.large_cube_size
            large_cube_pose = Pose.create_from_pq(p=xyz, q=q)
            self.large_cube.set_pose(large_cube_pose)

            # init the bin in the target zone
            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.rand((b, 1))[..., 0] * 0.1
            )  # x range [0, 0.1]
            pos[:, 1] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )  # y range [-0.1, 0.1]
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)

    def evaluate(self):
        pos_small_cube = self.small_cube.pose.p
        pos_medium_cube = self.medium_cube.pose.p
        pos_large_cube = self.large_cube.pose.p
        pos_bin = self.bin.pose.p
        
        # Check if any cube is on the bin (but only small cube should fit)
        is_small_cube_on_bin = self._check_cube_on_bin(pos_small_cube, pos_bin, self.small_cube_size)
        is_medium_cube_on_bin = self._check_cube_on_bin(pos_medium_cube, pos_bin, self.medium_cube_size, fits_in_bin=False)
        is_large_cube_on_bin = self._check_cube_on_bin(pos_large_cube, pos_bin, self.large_cube_size, fits_in_bin=False)
        
        # Check if cubes are static
        is_small_cube_static = self.small_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_medium_cube_static = self.medium_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_large_cube_static = self.large_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        # Check if cubes are grasped
        is_small_cube_grasped = self.agent.is_grasping(self.small_cube)
        is_medium_cube_grasped = self.agent.is_grasping(self.medium_cube)
        is_large_cube_grasped = self.agent.is_grasping(self.large_cube)
        
        # Success: ONLY small cube can succeed (others are too big for bin)
        small_cube_success = is_small_cube_on_bin & is_small_cube_static & (~is_small_cube_grasped)
        # Medium and large cubes cannot succeed as they don't fit in the bin
        success = small_cube_success
        
        return {
            "is_small_cube_grasped": is_small_cube_grasped,
            "is_medium_cube_grasped": is_medium_cube_grasped,
            "is_large_cube_grasped": is_large_cube_grasped,
            "is_small_cube_on_bin": is_small_cube_on_bin,
            "is_medium_cube_on_bin": is_medium_cube_on_bin,
            "is_large_cube_on_bin": is_large_cube_on_bin,
            "is_small_cube_static": is_small_cube_static,
            "is_medium_cube_static": is_medium_cube_static,
            "is_large_cube_static": is_large_cube_static,
            "success": success,
        }

    def _check_cube_on_bin(self, pos_cube, pos_bin, cube_size, fits_in_bin=True):
        """Check if cube is on top of the bin"""
        offset = pos_cube - pos_bin
        
        if fits_in_bin:
            # For small cube that fits in bin - standard check
            xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.01
            z_flag = (
                torch.abs(offset[..., 2] - cube_size - self.block_half_size[0]) <= 0.01
            )
        else:
            # For medium/large cubes that don't fit - they can't be successfully placed
            # Make the check impossible to pass
            xy_flag = torch.zeros_like(offset[..., 0], dtype=torch.bool)
            z_flag = torch.zeros_like(offset[..., 0], dtype=torch.bool)
        
        return torch.logical_and(xy_flag, z_flag)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_small_cube_grasped=info["is_small_cube_grasped"],
            is_medium_cube_grasped=info["is_medium_cube_grasped"],
            is_large_cube_grasped=info["is_large_cube_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                small_cube_pose=self.small_cube.pose.raw_pose,
                medium_cube_pose=self.medium_cube.pose.raw_pose,
                large_cube_pose=self.large_cube.pose.raw_pose,
                tcp_to_small_cube_pos=self.small_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_medium_cube_pos=self.medium_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_large_cube_pos=self.large_cube.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to any cube
        tcp_pose = self.agent.tcp.pose.p
        small_cube_pos = self.small_cube.pose.p
        medium_cube_pos = self.medium_cube.pose.p
        large_cube_pos = self.large_cube.pose.p
        
        obj_to_tcp_dist_small = torch.linalg.norm(tcp_pose - small_cube_pos, axis=1)
        obj_to_tcp_dist_medium = torch.linalg.norm(tcp_pose - medium_cube_pos, axis=1)
        obj_to_tcp_dist_large = torch.linalg.norm(tcp_pose - large_cube_pos, axis=1)
        obj_to_tcp_dist = torch.minimum(
            torch.minimum(obj_to_tcp_dist_small, obj_to_tcp_dist_medium),
            obj_to_tcp_dist_large
        )
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward - only give placement reward for small cube (others can't fit)
        bin_top_pos_small = self.bin.pose.p.clone()
        bin_top_pos_small[:, 2] = bin_top_pos_small[:, 2] + self.block_half_size[0] + self.small_cube_size
        
        small_cube_to_bin_dist = torch.linalg.norm(bin_top_pos_small - small_cube_pos, axis=1)
        
        # Reward based on whichever cube is grasped
        is_small_cube_grasped = info["is_small_cube_grasped"]
        is_medium_cube_grasped = info["is_medium_cube_grasped"]
        is_large_cube_grasped = info["is_large_cube_grasped"]
        
        # Only small cube gets placement reward (others can't fit in bin)
        small_cube_place_reward = 1 - torch.tanh(5.0 * small_cube_to_bin_dist)
        reward[is_small_cube_grasped] = (4 + small_cube_place_reward)[is_small_cube_grasped]
        
        # Medium and large cubes get reduced reward since they can't complete the task
        reward[is_medium_cube_grasped] = 3.0
        reward[is_large_cube_grasped] = 3.0

        # ungrasp and static reward - only for small cube success
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_any_cube_grasped = torch.logical_or(
            torch.logical_or(is_small_cube_grasped, is_medium_cube_grasped),
            is_large_cube_grasped
        )
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_any_cube_grasped
        ] = 16.0  # give ungrasp a bigger reward
        
        # Static reward for small cube (only one that can succeed)
        small_cube_v = torch.linalg.norm(self.small_cube.linear_velocity, axis=1)
        small_cube_av = torch.linalg.norm(self.small_cube.angular_velocity, axis=1)
        small_cube_static_reward = 1 - torch.tanh(small_cube_v * 10 + small_cube_av)
        
        robot_static_reward = self.agent.is_static(0.2)
        
        is_small_cube_on_bin = info["is_small_cube_on_bin"]
        
        # Only small cube can be successfully placed
        reward[is_small_cube_on_bin] = (
            6 + (ungrasp_reward + small_cube_static_reward + robot_static_reward) / 3.0
        )[is_small_cube_on_bin]

        # success reward - only for small cube
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward